import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import os
import datetime
import matplotlib.pyplot as plt
import json
from scipy.optimize import minimize

sys.path.append(os.path.join(os.path.dirname(__file__), '../env'))
from cced_crystallization_env import make_protein_crystallization_spec, ProteinCrystallizationBaseEnv

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG = {
    # Learning parameters
    'lr_policy': 0.0001,        # 降低学习率
    'lr_q': 0.0003,
    'gamma': 0.99,              # 提高折扣因子
    'batch_size': 256,          # 增大批量
    'buffer_limit': 200000,     # 增大缓冲区
    'tau': 0.002,               # 软更新系数
    
    # Environment parameters
    'batch_size_env': 2,
    'horizon': 25,
    'seed': 42,
    
    # Network architecture - 增强版
    'latent_dim': 32,           # 增大潜在维度
    'hidden_dim': 512,          # 增大隐藏层
    'num_layers': 3,            # 更多层
    
    # Training parameters
    'num_of_episodes': 500,
    'memory_threshold': 2000,
    'training_iterations': 10,   # 增加训练迭代
    'policy_delay': 2,          # TD3: 延迟策略更新
    'print_interval': 10,
    
    # Exploration parameters
    'noise_std': 0.2,
    'noise_decay': 0.998,
    'noise_min': 0.02,
    'target_noise_std': 0.2,    # TD3: 目标策略噪声
    'target_noise_clip': 0.5,
    
    # Auxiliary losses
    'projection_loss_weight': 0.5,   # 投影损失权重
    'entropy_weight': 0.01,          # 熵正则化
    
    # Priority Experience Replay
    'use_per': True,
    'per_alpha': 0.6,
    'per_beta_start': 0.4,
    'per_beta_end': 1.0,
    
    # Output parameters
    'plot_training_curve': True,
    'save_models': True,
}


class SumTree:
    """优先经验回放的 SumTree 数据结构"""
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0
    
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def total(self):
        return self.tree[0]
    
    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write = (self.write + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1
    
    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)
    
    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """优先经验回放缓冲区"""
    def __init__(self, capacity, alpha=0.6, device='cpu'):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.device = device
        self.max_priority = 1.0
        self.epsilon = 1e-6
    
    def put(self, transition):
        self.tree.add(self.max_priority ** self.alpha, transition)
    
    def sample(self, n, beta=0.4):
        indices = []
        priorities = []
        batch = []
        
        segment = self.tree.total() / n
        
        for i in range(n):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, p, data = self.tree.get(s)
            indices.append(idx)
            priorities.append(p)
            batch.append(data)
        
        # 计算重要性采样权重
        sampling_probs = np.array(priorities) / self.tree.total()
        weights = (self.tree.n_entries * sampling_probs) ** (-beta)
        weights = weights / weights.max()
        
        # 解包批次
        s_lst, a_soft_lst, a_proj_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], [], []
        for transition in batch:
            s, a_soft, a_proj, r, s_prime, done = transition
            s_lst.append(s)
            a_soft_lst.append(a_soft)
            a_proj_lst.append(a_proj)
            r_lst.append(r)
            s_prime_lst.append(s_prime)
            done_mask_lst.append([0.0 if done else 1.0])
        
        return (
            torch.FloatTensor(np.array(s_lst)).to(self.device),
            torch.FloatTensor(np.array(a_soft_lst)).to(self.device),
            torch.FloatTensor(np.array(a_proj_lst)).to(self.device),
            torch.FloatTensor(np.array(r_lst)).unsqueeze(1).to(self.device),
            torch.FloatTensor(np.array(s_prime_lst)).to(self.device),
            torch.FloatTensor(np.array(done_mask_lst)).to(self.device),
            torch.FloatTensor(weights).to(self.device),
            indices
        )
    
    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(idx, priority)
    
    def size(self):
        return self.tree.n_entries


class PolicyNetwork(nn.Module):
    """增强版策略网络：更深更宽"""
    def __init__(self, state_dim, n_protocols, n_droplets, param_dim, hidden_dim, latent_dim, num_layers=3):
        super().__init__()
        self.n_protocols = n_protocols
        self.n_droplets = n_droplets
        self.param_dim = param_dim
        
        # 深层编码器
        encoder_layers = [nn.Linear(state_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            encoder_layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            ])
        encoder_layers.append(nn.Linear(hidden_dim, latent_dim))
        encoder_layers.append(nn.Tanh())
        self.encoder = nn.Sequential(*encoder_layers)
        
        # 深层解码器
        decoder_layers = [
            nn.Linear(latent_dim + state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        ]
        for _ in range(num_layers - 1):
            decoder_layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            ])
        self.decoder = nn.Sequential(*decoder_layers)
        
        # 输出头
        self.discrete_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_droplets * n_protocols)
        )
        self.continuous_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_droplets * param_dim)
        )
        
    def forward(self, state, return_logits=False):
        batch = state.dim() > 1
        if not batch:
            state = state.unsqueeze(0)
        
        bs = state.shape[0]
        z = self.encoder(state)
        h = self.decoder(torch.cat([z, state], dim=-1))
        
        X_logits = self.discrete_head(h).view(bs, self.n_droplets, self.n_protocols)
        X_soft = F.softmax(X_logits, dim=-1)
        u_raw = torch.sigmoid(self.continuous_head(h)).view(bs, self.n_droplets, self.param_dim)
        
        if not batch:
            X_soft = X_soft.squeeze(0)
            X_logits = X_logits.squeeze(0)
            u_raw = u_raw.squeeze(0)
        
        if return_logits:
            return X_soft, u_raw, X_logits
        return X_soft, u_raw
    
    def get_action_flat(self, state):
        X_soft, u_raw = self.forward(state)
        batch = X_soft.dim() > 2
        if not batch:
            X_soft = X_soft.unsqueeze(0)
            u_raw = u_raw.unsqueeze(0)
        
        bs = X_soft.shape[0]
        a_flat = torch.cat([X_soft.view(bs, -1), u_raw.view(bs, -1)], dim=-1)
        
        if not batch:
            a_flat = a_flat.squeeze(0)
        return a_flat
    
    def get_entropy(self, state):
        """计算策略熵"""
        X_soft, u_raw, X_logits = self.forward(state, return_logits=True)
        
        # 离散动作熵
        batch = X_soft.dim() > 2
        if not batch:
            X_soft = X_soft.unsqueeze(0)
            X_logits = X_logits.unsqueeze(0)
        
        discrete_entropy = -(X_soft * F.log_softmax(X_logits, dim=-1)).sum(dim=-1).mean()
        
        return discrete_entropy


class TwinQNetwork(nn.Module):
    """双 Q 网络 (TD3 style)"""
    def __init__(self, state_dim, action_dim, hidden_dim, num_layers=3):
        super().__init__()
        
        # Q1 网络
        q1_layers = [nn.Linear(state_dim + action_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            q1_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()])
        q1_layers.append(nn.Linear(hidden_dim, 1))
        self.q1 = nn.Sequential(*q1_layers)
        
        # Q2 网络
        q2_layers = [nn.Linear(state_dim + action_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            q2_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()])
        q2_layers.append(nn.Linear(hidden_dim, 1))
        self.q2 = nn.Sequential(*q2_layers)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1(x), self.q2(x)
    
    def q1_forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1(x)


class FastProjector:
    """快速投影器"""
    def __init__(self, spec):
        self.spec = spec
        self.K = spec.K
        self.R = spec.R
        self.d = spec.R + 2
        self.B = spec.batch_size
        
    def project(self, X_soft, u_raw):
        X_soft_np = X_soft.detach().cpu().numpy()
        u_raw_np = u_raw.detach().cpu().numpy()
        
        k_vec = np.argmax(X_soft_np, axis=-1).astype(int)
        u_mat = np.zeros((self.B, self.d))
        
        for j in range(self.B):
            u_mat[j] = self._project_continuous(k_vec[j], u_raw_np[j])
        
        return k_vec, u_mat
    
    def _project_continuous(self, k, u_raw):
        spec = self.spec
        R = self.R
        d = self.d
        
        p_target = u_raw[:R] * spec.p_max
        T_target = spec.T_bounds[0] + u_raw[R] * (spec.T_bounds[1] - spec.T_bounds[0])
        tau_target = spec.tau_bounds[0] + u_raw[R+1] * (spec.tau_bounds[1] - spec.tau_bounds[0])
        u_target = np.concatenate([p_target, [T_target, tau_target]])
        
        def objective(u):
            return 0.5 * np.sum((u - u_target)**2)
        
        def gradient(u):
            return u - u_target
        
        proto = spec.protocols[k]
        
        cons = [{
            'type': 'eq',
            'fun': lambda u: np.sum(u[:R]) - 1.0,
            'jac': lambda u: np.concatenate([np.ones(R), np.zeros(d - R)])
        }]
        
        if proto.G is not None and proto.G.size > 0:
            for i in range(len(proto.h)):
                row = proto.G[i].copy()
                hi = float(proto.h[i])
                cons.append({
                    'type': 'ineq',
                    'fun': lambda u, row=row, hi=hi: hi - np.dot(row, u),
                    'jac': lambda u, row=row: -row
                })
        
        bounds = [(0.0, spec.p_max)] * R + [tuple(spec.T_bounds), tuple(spec.tau_bounds)]
        
        x0 = u_target.copy()
        p = np.clip(x0[:R], 1e-6, spec.p_max)
        p = p / np.sum(p)
        x0[:R] = p
        x0[R] = np.clip(x0[R], spec.T_bounds[0], spec.T_bounds[1])
        x0[R+1] = np.clip(x0[R+1], spec.tau_bounds[0], spec.tau_bounds[1])
        
        result = minimize(objective, x0, method='SLSQP', jac=gradient,
                         constraints=cons, bounds=bounds,
                         options={'ftol': 1e-8, 'maxiter': 100, 'disp': False})
        
        if result.success:
            return result.x
        else:
            return self._default_feasible(k)
    
    def _default_feasible(self, k):
        spec = self.spec
        R = self.R
        p = np.ones(R) / R
        T = np.mean(spec.T_bounds)
        tau = np.mean(spec.tau_bounds)
        return np.concatenate([p, [T, tau]])


class OrnsteinUhlenbeckNoise:
    """OU 噪声"""
    def __init__(self, size, theta=0.15, sigma=0.2):
        self.theta = theta
        self.sigma = sigma
        self.size = size
        self.reset()

    def __call__(self):
        dx = self.theta * (-self.x) + self.sigma * np.random.randn(self.size)
        self.x = self.x + dx
        return self.x
    
    def reset(self):
        self.x = np.zeros(self.size)


def flatten_action(k_vec, u_mat, n_protocols):
    B = len(k_vec)
    k_onehot = np.zeros((B, n_protocols))
    for j in range(B):
        k_onehot[j, k_vec[j]] = 1.0
    return np.concatenate([k_onehot.flatten(), u_mat.flatten()])


def main(config=None):
    if config is None:
        config = CONFIG
    
    print(f"Using device: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Environment setup
    spec = make_protein_crystallization_spec(
        seed=config['seed'],
        batch_size=config['batch_size_env'],
        horizon=config['horizon']
    )
    env = ProteinCrystallizationBaseEnv(spec)
    
    state_dim = env.observation_space.shape[0]
    n_protocols = spec.K
    n_droplets = spec.batch_size
    param_dim = spec.R + 2
    action_dim = n_droplets * n_protocols + n_droplets * param_dim
    
    # Initialize networks
    policy = PolicyNetwork(
        state_dim, n_protocols, n_droplets, param_dim,
        config['hidden_dim'], config['latent_dim'], config['num_layers']
    ).to(DEVICE)
    policy_target = PolicyNetwork(
        state_dim, n_protocols, n_droplets, param_dim,
        config['hidden_dim'], config['latent_dim'], config['num_layers']
    ).to(DEVICE)
    policy_target.load_state_dict(policy.state_dict())
    
    q_net = TwinQNetwork(state_dim, action_dim, config['hidden_dim'], config['num_layers']).to(DEVICE)
    q_target = TwinQNetwork(state_dim, action_dim, config['hidden_dim'], config['num_layers']).to(DEVICE)
    q_target.load_state_dict(q_net.state_dict())
    
    projector = FastProjector(spec)
    
    # Optimizers
    policy_optimizer = optim.Adam(policy.parameters(), lr=config['lr_policy'])
    q_optimizer = optim.Adam(q_net.parameters(), lr=config['lr_q'])
    
    # Memory
    if config['use_per']:
        memory = PrioritizedReplayBuffer(config['buffer_limit'], config['per_alpha'], DEVICE)
    else:
        memory = PrioritizedReplayBuffer(config['buffer_limit'], 0.0, DEVICE)
    
    score_record = []
    best_quality_record = []
    total_violations = 0
    total_steps = 0
    update_count = 0
    
    # Noise
    noise_dim = n_droplets * n_protocols + n_droplets * param_dim
    ou_noise = OrnsteinUhlenbeckNoise(noise_dim, sigma=config['noise_std'])
    noise_scale = config['noise_std']
    
    # PER beta annealing
    beta = config['per_beta_start']
    beta_increment = (config['per_beta_end'] - config['per_beta_start']) / config['num_of_episodes']
    
    
    for n_epi in range(config['num_of_episodes']):
        s, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0.0
        epi_violations = 0
        ou_noise.reset()
        
        while not (done or truncated):
            s_tensor = torch.FloatTensor(s).to(DEVICE)
            
            with torch.no_grad():
                X_soft, u_raw = policy(s_tensor)
            
            # Add exploration noise
            a_flat = policy.get_action_flat(s_tensor).detach().cpu().numpy()
            noise = ou_noise() * noise_scale
            a_flat_noisy = np.clip(a_flat + noise, 0, 1)
            
            X_soft_noisy = a_flat_noisy[:n_droplets * n_protocols].reshape(n_droplets, n_protocols)
            u_raw_noisy = a_flat_noisy[n_droplets * n_protocols:].reshape(n_droplets, param_dim)
            
            X_soft_t = torch.FloatTensor(X_soft_noisy).to(DEVICE)
            u_raw_t = torch.FloatTensor(u_raw_noisy).to(DEVICE)
            k_vec, u_mat = projector.project(X_soft_t, u_raw_t)
            
            total_steps += config['batch_size_env']
            
            action = {"k": k_vec, "u": u_mat}
            s_prime, r, done, truncated, info = env.step(action)
            
            a_soft_flat = a_flat_noisy
            a_proj_flat = flatten_action(k_vec, u_mat, n_protocols)
            memory.put((s, a_soft_flat, a_proj_flat, r, s_prime, done))
            
            s = s_prime
            episode_reward += r
        
        score_record.append(episode_reward)
        best_quality_record.append(info.get('best_quality', 0))
        noise_scale = max(config['noise_min'], noise_scale * config['noise_decay'])
        beta = min(config['per_beta_end'], beta + beta_increment)
        
        # Training
        if memory.size() > config['memory_threshold']:
            for _ in range(config['training_iterations']):
                update_count += 1
                
                # Sample with priority
                (s_batch, a_soft_batch, a_proj_batch, r_batch, 
                 s_prime_batch, done_mask, weights, indices) = memory.sample(config['batch_size'], beta)
                
                # ===== Update Q networks =====
                with torch.no_grad():
                    # TD3: Add noise to target action
                    a_next = policy_target.get_action_flat(s_prime_batch)
                    noise = torch.randn_like(a_next) * config['target_noise_std']
                    noise = noise.clamp(-config['target_noise_clip'], config['target_noise_clip'])
                    a_next = (a_next + noise).clamp(0, 1)
                    
                    # TD3: Use minimum of two Q-values
                    q1_next, q2_next = q_target(s_prime_batch, a_next)
                    q_next = torch.min(q1_next, q2_next)
                    target = r_batch + config['gamma'] * q_next * done_mask
                
                q1, q2 = q_net(s_batch, a_proj_batch)
                
                # Weighted TD error for PER
                td_error1 = (q1 - target).abs().detach().cpu().numpy().flatten()
                td_error2 = (q2 - target).abs().detach().cpu().numpy().flatten()
                td_errors = (td_error1 + td_error2) / 2
                
                q_loss = (weights * ((q1 - target)**2 + (q2 - target)**2)).mean()
                
                q_optimizer.zero_grad()
                q_loss.backward()
                torch.nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)
                q_optimizer.step()
                
                # Update priorities
                memory.update_priorities(indices, td_errors)
                
                # ===== Delayed Policy Update (TD3) =====
                if update_count % config['policy_delay'] == 0:
                    # Get current policy actions
                    X_soft_curr, u_raw_curr = policy(s_batch)
                    bs = s_batch.shape[0]
                    a_soft_curr = torch.cat([
                        X_soft_curr.view(bs, -1),
                        u_raw_curr.view(bs, -1)
                    ], dim=-1)
                    
                    # Policy gradient loss (maximize Q1)
                    q1_val = q_net.q1_forward(s_batch, a_soft_curr)
                    policy_loss = -q1_val.mean()
                    
                    # Auxiliary loss 1: Projection loss (soft action -> projected action)
                    proj_loss = F.mse_loss(a_soft_curr, a_proj_batch)
                    
                    # Auxiliary loss 2: Entropy regularization
                    entropy = policy.get_entropy(s_batch)
                    
                    # Total policy loss
                    total_policy_loss = (
                        policy_loss 
                        + config['projection_loss_weight'] * proj_loss 
                        - config['entropy_weight'] * entropy
                    )
                    
                    policy_optimizer.zero_grad()
                    total_policy_loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                    policy_optimizer.step()
                    
                    # Soft update targets
                    for p_targ, p in zip(q_target.parameters(), q_net.parameters()):
                        p_targ.data.copy_(p_targ.data * (1 - config['tau']) + p.data * config['tau'])
                    for p_targ, p in zip(policy_target.parameters(), policy.parameters()):
                        p_targ.data.copy_(p_targ.data * (1 - config['tau']) + p.data * config['tau'])
        
        if n_epi % config['print_interval'] == 0 and n_epi != 0:
            avg_score = np.mean(score_record[-config['print_interval']:])
            cvr = total_violations / max(1, total_steps)
            print(f"Episode {n_epi}: Avg Score = {avg_score:.4f}, "
                  f"Best Quality = {info.get('best_quality', 0):.4f}, "
                  f"CVR = {cvr:.6f}, Noise = {noise_scale:.4f}")
    
    print(f"\n{'='*70}")
    print(f"LIRL Enhanced Training Complete!")
    print(f"Final Avg Score: {np.mean(score_record[-50:]):.4f}")
    print(f"Max Score: {max(score_record):.4f}")
    print(f"{'='*70}")
    
    policy.cpu()
    q_net.cpu()
    
    return score_record, best_quality_record, {'policy': policy, 'q_net': q_net}


def save_results(score_records, models, config):
    if not config['save_models']:
        return
        
    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_base_dir = "/home/one/LIRL-CPS-main/Protein_Crystallization_Screening/exp"
    save_dir = os.path.join(exp_base_dir, f"lirl_enhanced_{now_str}")
    os.makedirs(save_dir, exist_ok=True)

    np.save(os.path.join(save_dir, "scores.npy"), score_records)
    
    torch.save(models['policy'].state_dict(), os.path.join(save_dir, "policy.pth"))
    torch.save(models['q_net'].state_dict(), os.path.join(save_dir, "q_net.pth"))
    
    config_to_save = {k: v for k, v in config.items() if not callable(v)}
    config_to_save['device'] = str(DEVICE)
    with open(os.path.join(save_dir, "config.json"), 'w') as f:
        json.dump(config_to_save, f, indent=2)
    
    if config['plot_training_curve']:
        plt.figure(figsize=(10, 6))
        plt.plot(score_records, label='Episode Reward', alpha=0.7)
        window = min(20, len(score_records) // 5) if len(score_records) > 10 else 1
        if window > 1:
            moving_avg = np.convolve(score_records, np.ones(window)/window, mode='valid')
            plt.plot(range(window-1, len(score_records)), moving_avg, 'r-', 
                    linewidth=2, label=f'Moving Avg (window={window})')
        plt.title("LIRL Enhanced Training Curve")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, "training_curve.png"), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Training curve saved to: {os.path.join(save_dir, 'training_curve.png')}")
    
    print(f"Results saved to: {save_dir}")


if __name__ == "__main__":
    score_record, models = main(CONFIG)
    save_results(score_record, models, CONFIG)
