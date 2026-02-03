"""
CPO (Constrained Policy Optimization) for CityFlow Multi-Intersection Traffic Signal Control

CPO 是一种基于信任域的约束强化学习算法，能够在保证约束满足的同时优化策略。

核心思想：
1. 使用信任域方法（类似 TRPO）进行策略优化
2. 在每次更新时求解约束优化问题：
   max_θ E[A(s,a)]
   s.t. KL(π_θ || π_old) ≤ δ  (信任域约束)
        C(π_θ) ≤ d            (安全约束)

3. 使用共轭梯度 + 线搜索求解
4. 通过投影确保约束满足

参考论文：
- Achiam et al. "Constrained Policy Optimization" (2017)
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import scipy.optimize

# 添加环境路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "env"))
from cityflow_multi_env import CityFlowMultiIntersectionEnv, get_default_config

# =======================
# GPU 设备检测
# =======================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[DEVICE] 使用设备: {DEVICE}", flush=True)
if torch.cuda.is_available():
    print(f"[DEVICE] GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"[DEVICE] CUDA 版本: {torch.version.cuda}", flush=True)

# =======================
# HYPERPARAMETERS CONFIG
# =======================
CONFIG = {
    # CPO parameters
    'lr_critic': 0.001,
    'gamma': 0.99,
    'lambda_gae': 0.95,
    'delta': 0.01,           # KL 信任域半径
    'cost_limit': 100.0,     # 约束阈值（每 episode 最大违反次数）
    'cost_gamma': 0.99,
    'damping': 0.1,          # Fisher 矩阵阻尼系数
    'max_kl': 0.01,          # 最大 KL 散度
    'line_search_coef': 0.9, # 线搜索衰减系数
    'line_search_max_iter': 10,
    'cg_iters': 10,          # 共轭梯度迭代次数
    
    # Value function
    'value_iters': 5,
    
    # Training parameters
    'batch_size': 64,
    'rollout_length': 360,
    'num_of_episodes': 200,
    
    # Environment parameters
    'episode_length': 3600,
    'ctrl_interval': 10,
    'min_green': 10,
    'min_duration': 10,
    'max_duration': 60,
    
    # Network architecture
    'hidden_dim1': 256,
    'hidden_dim2': 128,
    
    # Continuous action parameters
    'log_std_init': -0.5,
    'log_std_min': -2.0,
    'log_std_max': 0.5,
    
    # Output parameters
    'print_interval': 10,
    'save_models': True,
    'output_dir': './outputs/cpo_cityflow',
}


class RolloutBuffer:
    """CPO 的 Rollout Buffer"""
    def __init__(self):
        self.states = []
        self.discrete_actions = []
        self.continuous_actions = []
        self.rewards = []
        self.costs = []
        self.dones = []
        self.values = []
        self.cost_values = []
        self.disc_log_probs = []
        self.cont_log_probs = []
        
    def store(self, state, disc_action, cont_action, disc_log_prob, cont_log_prob,
              reward, cost, done, value, cost_value):
        self.states.append(state)
        self.discrete_actions.append(disc_action)
        self.continuous_actions.append(cont_action)
        self.disc_log_probs.append(disc_log_prob)
        self.cont_log_probs.append(cont_log_prob)
        self.rewards.append(reward)
        self.costs.append(cost)
        self.dones.append(done)
        self.values.append(value)
        self.cost_values.append(cost_value)
    
    def clear(self):
        self.states = []
        self.discrete_actions = []
        self.continuous_actions = []
        self.rewards = []
        self.costs = []
        self.dones = []
        self.values = []
        self.cost_values = []
        self.disc_log_probs = []
        self.cont_log_probs = []
    
    def compute_gae(self, last_value, last_cost_value, gamma, cost_gamma, lambda_gae):
        """计算 GAE 优势"""
        rewards = np.array(self.rewards)
        costs = np.array(self.costs)
        dones = np.array(self.dones)
        values = np.array(self.values + [last_value])
        cost_values = np.array(self.cost_values + [last_cost_value])
        
        # 奖励 GAE
        advantages = np.zeros_like(rewards)
        last_gae = 0
        for t in reversed(range(len(rewards))):
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * values[t + 1] * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + gamma * lambda_gae * next_non_terminal * last_gae
        returns = advantages + values[:-1]
        
        # 约束代价 GAE
        cost_advantages = np.zeros_like(costs)
        last_cost_gae = 0
        for t in reversed(range(len(costs))):
            next_non_terminal = 1.0 - dones[t]
            delta = costs[t] + cost_gamma * cost_values[t + 1] * next_non_terminal - cost_values[t]
            cost_advantages[t] = last_cost_gae = delta + cost_gamma * lambda_gae * next_non_terminal * last_cost_gae
        cost_returns = cost_advantages + cost_values[:-1]
        
        return returns, advantages, cost_returns, cost_advantages
    
    def get_tensors(self, device):
        """转换为张量"""
        return {
            'states': torch.FloatTensor(np.array(self.states)).to(device),
            'disc_actions': torch.LongTensor(np.array(self.discrete_actions)).to(device),
            'cont_actions': torch.FloatTensor(np.array(self.continuous_actions)).to(device),
            'disc_log_probs': torch.FloatTensor(np.array(self.disc_log_probs)).to(device),
            'cont_log_probs': torch.FloatTensor(np.array(self.cont_log_probs)).to(device),
        }


class CPOActorCritic(nn.Module):
    """
    CPO Actor-Critic 网络
    
    Actor: 输出离散和连续动作的分布
    Reward Critic: 估计奖励价值
    Cost Critic: 估计约束代价价值
    """
    def __init__(self, state_size: int, num_intersections: int, num_phases: int,
                 hidden_dim1: int = 256, hidden_dim2: int = 128,
                 log_std_init: float = -0.5, log_std_min: float = -2.0, log_std_max: float = 0.5):
        super(CPOActorCritic, self).__init__()
        
        self.num_intersections = num_intersections
        self.num_phases = num_phases
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # ========== Actor ==========
        self.actor_shared = nn.Sequential(
            nn.Linear(state_size, hidden_dim1),
            nn.Tanh(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.Tanh()
        )
        
        # 离散动作头
        self.discrete_heads = nn.ModuleList([
            nn.Linear(hidden_dim2, num_phases) for _ in range(num_intersections)
        ])
        
        # 连续动作参数
        self.continuous_mean = nn.Linear(hidden_dim2, num_intersections)
        self.log_std = nn.Parameter(torch.ones(num_intersections) * log_std_init)
        
        # ========== Reward Critic ==========
        self.reward_critic = nn.Sequential(
            nn.Linear(state_size, hidden_dim1),
            nn.Tanh(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.Tanh(),
            nn.Linear(hidden_dim2, 1)
        )
        
        # ========== Cost Critic ==========
        self.cost_critic = nn.Sequential(
            nn.Linear(state_size, hidden_dim1),
            nn.Tanh(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.Tanh(),
            nn.Linear(hidden_dim2, 1)
        )
    
    def forward_actor(self, state):
        """Actor 前向传播"""
        actor_features = self.actor_shared(state)
        
        # 离散动作 logits
        discrete_logits = [head(actor_features) for head in self.discrete_heads]
        
        # 连续动作参数
        cont_mean = torch.sigmoid(self.continuous_mean(actor_features))
        cont_log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        cont_std = cont_log_std.exp()
        
        return discrete_logits, cont_mean, cont_std
    
    def forward_critics(self, state):
        """Critic 前向传播"""
        reward_value = self.reward_critic(state)
        cost_value = self.cost_critic(state)
        return reward_value.squeeze(-1), cost_value.squeeze(-1)
    
    def get_action(self, state, deterministic=False):
        """获取动作"""
        discrete_logits, cont_mean, cont_std = self.forward_actor(state)
        
        # 离散动作
        disc_actions = []
        disc_log_probs = []
        disc_entropies = []
        
        for logits in discrete_logits:
            dist = Categorical(logits=logits)
            if deterministic:
                action = logits.argmax(dim=-1)
            else:
                action = dist.sample()
            disc_actions.append(action)
            disc_log_probs.append(dist.log_prob(action))
            disc_entropies.append(dist.entropy())
        
        disc_actions = torch.stack(disc_actions, dim=-1)
        disc_log_probs = torch.stack(disc_log_probs, dim=-1).sum(dim=-1)
        disc_entropy = torch.stack(disc_entropies, dim=-1).mean(dim=-1)
        
        # 连续动作
        cont_dist = Normal(cont_mean, cont_std)
        if deterministic:
            cont_actions = cont_mean
        else:
            cont_actions = cont_dist.sample()
            cont_actions = torch.clamp(cont_actions, 0, 1)
        
        cont_log_probs = cont_dist.log_prob(cont_actions).sum(dim=-1)
        cont_entropy = cont_dist.entropy().mean(dim=-1)
        
        return disc_actions, cont_actions, disc_log_probs, cont_log_probs, disc_entropy + cont_entropy
    
    def evaluate_actions(self, state, disc_actions, cont_actions):
        """评估给定动作的对数概率"""
        discrete_logits, cont_mean, cont_std = self.forward_actor(state)
        
        # 离散动作
        disc_log_probs = []
        disc_entropies = []
        
        for i, logits in enumerate(discrete_logits):
            dist = Categorical(logits=logits)
            disc_log_probs.append(dist.log_prob(disc_actions[:, i]))
            disc_entropies.append(dist.entropy())
        
        disc_log_probs = torch.stack(disc_log_probs, dim=-1).sum(dim=-1)
        disc_entropy = torch.stack(disc_entropies, dim=-1).mean(dim=-1)
        
        # 连续动作
        cont_dist = Normal(cont_mean, cont_std)
        cont_log_probs = cont_dist.log_prob(cont_actions).sum(dim=-1)
        cont_entropy = cont_dist.entropy().mean(dim=-1)
        
        return disc_log_probs, cont_log_probs, disc_entropy + cont_entropy
    
    def get_kl_divergence(self, state, old_disc_logits, old_cont_mean, old_cont_std):
        """计算新旧策略的 KL 散度"""
        new_disc_logits, new_cont_mean, new_cont_std = self.forward_actor(state)
        
        # 离散动作 KL
        disc_kl = 0
        for old_logits, new_logits in zip(old_disc_logits, new_disc_logits):
            old_probs = F.softmax(old_logits, dim=-1)
            new_log_probs = F.log_softmax(new_logits, dim=-1)
            old_log_probs = F.log_softmax(old_logits, dim=-1)
            disc_kl += (old_probs * (old_log_probs - new_log_probs)).sum(dim=-1).mean()
        
        # 连续动作 KL (两个高斯分布)
        cont_kl = (
            torch.log(new_cont_std / old_cont_std) +
            (old_cont_std.pow(2) + (old_cont_mean - new_cont_mean).pow(2)) / (2 * new_cont_std.pow(2)) - 0.5
        ).sum(dim=-1).mean()
        
        return disc_kl + cont_kl


class CPOAgent:
    """
    CPO 智能体
    
    使用信任域方法进行约束策略优化
    """
    def __init__(self, env: CityFlowMultiIntersectionEnv, config: Dict = None, device=None):
        self.config = config or CONFIG.copy()
        self.env = env
        self.device = device or DEVICE
        
        self.state_size = env.observation_space.shape[0]
        self.num_intersections = env.num_intersections
        self.num_phases = env.num_phases
        self.min_duration = env.min_duration
        self.max_duration = env.max_duration
        
        print(f"[CPO Agent] 初始化:")
        print(f"  设备: {self.device}")
        print(f"  状态维度: {self.state_size}")
        print(f"  路口数量: {self.num_intersections}")
        print(f"  每个路口相位数: {self.num_phases}")
        print(f"  绿灯时长范围: [{self.min_duration}, {self.max_duration}]秒")
        print(f"  约束阈值 (cost_limit): {self.config['cost_limit']}")
        print(f"  信任域半径 (delta): {self.config['delta']}")
        
        # 创建网络
        self.actor_critic = CPOActorCritic(
            state_size=self.state_size,
            num_intersections=self.num_intersections,
            num_phases=self.num_phases,
            hidden_dim1=self.config['hidden_dim1'],
            hidden_dim2=self.config['hidden_dim2'],
            log_std_init=self.config['log_std_init'],
            log_std_min=self.config['log_std_min'],
            log_std_max=self.config['log_std_max']
        ).to(self.device)
        
        # Critic 优化器
        self.critic_optimizer = optim.Adam(
            list(self.actor_critic.reward_critic.parameters()) + 
            list(self.actor_critic.cost_critic.parameters()),
            lr=self.config['lr_critic']
        )
        
        # Buffer
        self.buffer = RolloutBuffer()
        
    def select_action(self, state: np.ndarray, deterministic: bool = False):
        """选择动作"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            disc_actions, cont_actions, disc_log_prob, cont_log_prob, _ = \
                self.actor_critic.get_action(state_tensor, deterministic)
            
            reward_value, cost_value = self.actor_critic.forward_critics(state_tensor)
            
            return (
                disc_actions.squeeze(0).cpu().numpy(),
                cont_actions.squeeze(0).cpu().numpy(),
                disc_log_prob.item(),
                cont_log_prob.item(),
                reward_value.item(),
                cost_value.item()
            )
    
    def convert_to_env_action(self, discrete_actions: np.ndarray, continuous_actions: np.ndarray) -> np.ndarray:
        """转换为环境动作"""
        env_action = np.zeros(self.num_intersections * 2, dtype=np.int64)
        
        for i in range(self.num_intersections):
            phase = int(discrete_actions[i])
            phase = np.clip(phase, 0, self.num_phases - 1)
            env_action[i * 2] = phase
            
            duration = self.min_duration + continuous_actions[i] * (self.max_duration - self.min_duration)
            duration_idx = int(round(duration - self.min_duration))
            duration_idx = np.clip(duration_idx, 0, self.max_duration - self.min_duration)
            env_action[i * 2 + 1] = duration_idx
        
        return env_action
    
    def store_transition(self, state, disc_action, cont_action, disc_log_prob, cont_log_prob,
                         reward, cost, done, value, cost_value):
        """存储经验"""
        self.buffer.store(state, disc_action, cont_action, disc_log_prob, cont_log_prob,
                         reward, cost, done, value, cost_value)
    
    def _flat_grad(self, y, x, retain_graph=False, create_graph=False):
        """计算展平的梯度"""
        if create_graph:
            retain_graph = True
        
        g = torch.autograd.grad(y, x, retain_graph=retain_graph, create_graph=create_graph)
        g = torch.cat([t.view(-1) for t in g])
        return g
    
    def _hessian_vector_product(self, kl, params, v, damping=0.1):
        """计算 Hessian-向量积: H @ v"""
        kl_grad = self._flat_grad(kl, params, retain_graph=True, create_graph=True)
        kl_grad_v = (kl_grad * v).sum()
        hvp = self._flat_grad(kl_grad_v, params, retain_graph=True)
        return hvp + damping * v
    
    def _conjugate_gradient(self, kl, params, b, nsteps=10, residual_tol=1e-10):
        """共轭梯度法求解 H @ x = b"""
        x = torch.zeros_like(b)
        r = b.clone()
        p = b.clone()
        rdotr = torch.dot(r, r)
        
        for _ in range(nsteps):
            hvp = self._hessian_vector_product(kl, params, p, self.config['damping'])
            alpha = rdotr / (torch.dot(p, hvp) + 1e-8)
            x += alpha * p
            r -= alpha * hvp
            new_rdotr = torch.dot(r, r)
            
            if new_rdotr < residual_tol:
                break
            
            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr
        
        return x
    
    def _set_params(self, params, flat_params):
        """设置网络参数"""
        idx = 0
        for p in params:
            numel = p.numel()
            p.data.copy_(flat_params[idx:idx + numel].view(p.shape))
            idx += numel
    
    def _get_flat_params(self, params):
        """获取展平的参数"""
        return torch.cat([p.view(-1) for p in params])
    
    def update(self, episode_total_cost: float) -> Dict[str, float]:
        """
        CPO 更新
        
        1. 更新 Critic
        2. 使用信任域方法更新 Actor
        """
        # 获取最后状态价值
        with torch.no_grad():
            last_state = torch.FloatTensor(self.buffer.states[-1]).unsqueeze(0).to(self.device)
            last_reward_value, last_cost_value = self.actor_critic.forward_critics(last_state)
        
        # 计算 GAE
        returns, advantages, cost_returns, cost_advantages = self.buffer.compute_gae(
            last_reward_value.item(), last_cost_value.item(),
            self.config['gamma'], self.config['cost_gamma'], self.config['lambda_gae']
        )
        
        # 转换为张量
        data = self.buffer.get_tensors(self.device)
        states = data['states']
        disc_actions = data['disc_actions']
        cont_actions = data['cont_actions']
        old_disc_log_probs = data['disc_log_probs']
        old_cont_log_probs = data['cont_log_probs']
        
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        cost_returns_tensor = torch.FloatTensor(cost_returns).to(self.device)
        cost_advantages_tensor = torch.FloatTensor(cost_advantages).to(self.device)
        
        # 标准化优势
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        # ========== 更新 Critic ==========
        for _ in range(self.config['value_iters']):
            reward_values, cost_values = self.actor_critic.forward_critics(states)
            
            reward_value_loss = F.mse_loss(reward_values, returns_tensor)
            cost_value_loss = F.mse_loss(cost_values, cost_returns_tensor)
            value_loss = reward_value_loss + cost_value_loss
            
            self.critic_optimizer.zero_grad()
            value_loss.backward()
            self.critic_optimizer.step()
        
        # ========== CPO Actor 更新 ==========
        # 保存旧策略参数
        with torch.no_grad():
            old_disc_logits, old_cont_mean, old_cont_std = self.actor_critic.forward_actor(states)
            old_disc_logits = [logits.clone() for logits in old_disc_logits]
            old_cont_mean = old_cont_mean.clone()
            old_cont_std = old_cont_std.clone()
        
        # 获取 Actor 参数
        actor_params = list(self.actor_critic.actor_shared.parameters()) + \
                       list(self.actor_critic.discrete_heads.parameters()) + \
                       list(self.actor_critic.continuous_mean.parameters()) + \
                       [self.actor_critic.log_std]
        
        # 计算奖励目标梯度
        new_disc_log_probs, new_cont_log_probs, entropy = \
            self.actor_critic.evaluate_actions(states, disc_actions, cont_actions)
        
        ratio = torch.exp((new_disc_log_probs + new_cont_log_probs) - 
                         (old_disc_log_probs + old_cont_log_probs))
        
        # 奖励目标
        reward_objective = (ratio * advantages_tensor).mean()
        reward_grad = self._flat_grad(reward_objective, actor_params, retain_graph=True)
        
        # 约束目标（期望约束代价）
        cost_objective = (ratio * cost_advantages_tensor).mean()
        cost_grad = self._flat_grad(cost_objective, actor_params, retain_graph=True)
        
        # KL 散度
        kl = self.actor_critic.get_kl_divergence(states, old_disc_logits, old_cont_mean, old_cont_std)
        
        # 使用共轭梯度计算搜索方向
        # 对于简化版 CPO，我们使用近似方法
        search_dir = self._conjugate_gradient(kl, actor_params, reward_grad, self.config['cg_iters'])
        
        # 计算步长
        shs = 0.5 * torch.dot(search_dir, self._hessian_vector_product(kl, actor_params, search_dir, self.config['damping']))
        max_step = torch.sqrt(self.config['delta'] / (shs + 1e-8))
        
        # 约束调整
        # 如果违反约束，需要调整步长方向
        cost_violation = episode_total_cost - self.config['cost_limit']
        
        if cost_violation > 0:
            # 约束被违反，需要在约束方向上投影
            cost_search_dir = self._conjugate_gradient(kl, actor_params, cost_grad, self.config['cg_iters'])
            
            # 混合搜索方向
            alpha = min(1.0, cost_violation / (cost_grad.norm() + 1e-8))
            search_dir = search_dir - alpha * cost_search_dir
        
        # 线搜索
        old_params = self._get_flat_params(actor_params)
        expected_improvement = torch.dot(reward_grad, search_dir)
        
        step_frac = 1.0
        for _ in range(self.config['line_search_max_iter']):
            new_params = old_params + step_frac * max_step * search_dir
            self._set_params(actor_params, new_params)
            
            with torch.no_grad():
                new_disc_log_probs, new_cont_log_probs, _ = \
                    self.actor_critic.evaluate_actions(states, disc_actions, cont_actions)
                new_ratio = torch.exp((new_disc_log_probs + new_cont_log_probs) - 
                                     (old_disc_log_probs + old_cont_log_probs))
                new_objective = (new_ratio * advantages_tensor).mean()
                new_kl = self.actor_critic.get_kl_divergence(states, old_disc_logits, old_cont_mean, old_cont_std)
            
            improvement = new_objective - reward_objective
            
            if improvement > 0 and new_kl < self.config['max_kl']:
                break
            
            step_frac *= self.config['line_search_coef']
        else:
            # 线搜索失败，恢复旧参数
            self._set_params(actor_params, old_params)
        
        # 清空 buffer
        self.buffer.clear()
        
        return {
            'reward_objective': reward_objective.item(),
            'cost_objective': cost_objective.item(),
            'value_loss': value_loss.item(),
            'kl': kl.item() if hasattr(kl, 'item') else kl,
            'cost_violation': cost_violation,
        }
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'actor_critic': self.actor_critic.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'config': self.config,
        }, path)
        print(f"模型已保存: {path}")
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint['actor_critic'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        print(f"模型已加载: {path} (设备: {self.device})")


def train(config: Dict = None, cityflow_config_path: str = None):
    """训练 CPO 智能体"""
    config = config or CONFIG.copy()
    
    if cityflow_config_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cityflow_config_path = os.path.join(script_dir, "../examples/City_3_5/config.json")
    
    env_config = get_default_config(cityflow_config_path)
    env_config["episode_length"] = config['episode_length']
    env_config["ctrl_interval"] = config['ctrl_interval']
    env_config["min_green"] = config['min_green']
    env_config["min_duration"] = config['min_duration']
    env_config["max_duration"] = config['max_duration']
    env_config["verbose_violations"] = False
    env_config["log_violations"] = True
    
    env = CityFlowMultiIntersectionEnv(env_config)
    
    print(f"\n{'='*60}")
    print("CPO for CityFlow Traffic Signal Control")
    print(f"{'='*60}")
    
    agent = CPOAgent(env, config)
    
    output_dir = config.get('output_dir', './outputs/cpo_cityflow')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # 训练记录
    episode_rewards = []
    episode_travel_times = []
    episode_violations = []
    episode_kls = []
    
    print(f"\n训练开始，输出目录: {run_dir}", flush=True)
    print(f"总 Episodes: {config['num_of_episodes']}", flush=True)
    print(f"约束阈值: {config['cost_limit']}", flush=True)
    print(f"信任域半径: {config['delta']}", flush=True)
    print(f"打印间隔: 每 {config['print_interval']} episodes\n", flush=True)
    
    import time
    train_start_time = time.time()
    
    for n_epi in range(config['num_of_episodes']):
        episode_start_time = time.time()
        state, info = env.reset()
        done = False
        episode_reward = 0
        episode_cost = 0
        step = 0
        
        prev_total_violations = 0
        total_steps = config['episode_length'] // config['ctrl_interval']
        
        while not done:
            disc_actions, cont_actions, disc_log_prob, cont_log_prob, value, cost_value = \
                agent.select_action(state)
            
            env_action = agent.convert_to_env_action(disc_actions, cont_actions)
            next_state, reward, terminated, truncated, info = env.step(env_action)
            done = terminated or truncated
            
            # 计算步进代价
            current_violations = sum(info.get('total_violations', {}).values())
            step_cost = current_violations - prev_total_violations
            prev_total_violations = current_violations
            
            agent.store_transition(state, disc_actions, cont_actions, disc_log_prob, cont_log_prob,
                                  reward, step_cost, done, value, cost_value)
            
            episode_reward += reward
            episode_cost += step_cost
            state = next_state
            step += 1
            
            if step % 100 == 0 or done:
                progress = step / total_steps
                bar_len = 20
                filled = int(bar_len * progress)
                bar = '█' * filled + '░' * (bar_len - filled)
                print(f"\r  [{bar}] {progress*100:5.1f}% | Step {step}/{total_steps} | "
                      f"R={episode_reward:.0f} | C={episode_cost:.0f}", end="", flush=True)
        
        # CPO 更新
        losses = agent.update(episode_cost)
        
        # 统计
        episode_rewards.append(episode_reward)
        avg_travel_time = info.get('average_travel_time', 0)
        episode_travel_times.append(avg_travel_time)
        
        total_viol = sum(info.get('total_violations', {}).values())
        episode_violations.append(total_viol)
        episode_kls.append(losses.get('kl', 0))
        
        episode_time = time.time() - episode_start_time
        
        constraint_status = "✓" if total_viol <= config['cost_limit'] else "✗"
        
        print(f"\n{constraint_status} Episode {n_epi+1}/{config['num_of_episodes']} 完成 | "
              f"Reward={episode_reward:.1f} | Cost={total_viol:.0f}/{config['cost_limit']:.0f} | "
              f"KL={losses.get('kl', 0):.4f} | Time={episode_time:.1f}s", flush=True)
        
        if (n_epi + 1) % config['print_interval'] == 0:
            avg_reward = np.mean(episode_rewards[-config['print_interval']:])
            avg_tt = np.mean(episode_travel_times[-config['print_interval']:])
            avg_viol = np.mean(episode_violations[-config['print_interval']:])
            avg_kl = np.mean(episode_kls[-config['print_interval']:])
            elapsed = time.time() - train_start_time
            
            constraint_satisfied = sum(1 for v in episode_violations[-config['print_interval']:] 
                                       if v <= config['cost_limit'])
            satisfaction_rate = constraint_satisfied / config['print_interval'] * 100
            
            print(f"\n{'─'*60}")
            print(f"📊 Episode {n_epi+1}/{config['num_of_episodes']} 统计 (耗时: {elapsed/60:.1f}分钟)")
            print(f"   平均奖励: {avg_reward:.1f}")
            print(f"   平均行程时间: {avg_tt:.1f}s")
            print(f"   平均约束违反: {avg_viol:.1f} (阈值: {config['cost_limit']})")
            print(f"   约束满足率: {satisfaction_rate:.1f}%")
            print(f"   平均 KL: {avg_kl:.4f}")
            print(f"{'─'*60}\n", flush=True)
    
    # 保存
    if config.get('save_models', True):
        model_path = os.path.join(run_dir, "cpo_final.pt")
        agent.save(model_path)
    
    import json
    stats_path = os.path.join(run_dir, "training_stats.json")
    with open(stats_path, 'w') as f:
        json.dump({
            'episode_rewards': episode_rewards,
            'episode_travel_times': episode_travel_times,
            'episode_violations': episode_violations,
            'episode_kls': episode_kls,
        }, f, indent=2)
    print(f"训练统计已保存: {stats_path}")
    
    final_satisfaction = sum(1 for v in episode_violations[-20:] if v <= config['cost_limit']) / min(20, len(episode_violations)) * 100
    print(f"\n最终约束满足率 (后20 episodes): {final_satisfaction:.1f}%")
    
    env.close()
    
    return {
        'agent': agent,
        'episode_rewards': episode_rewards,
        'episode_violations': episode_violations,
        'run_dir': run_dir,
    }


def evaluate(model_path: str, cityflow_config_path: str = None, n_episodes: int = 5, render: bool = True):
    """评估模型"""
    checkpoint = torch.load(model_path, map_location=DEVICE)
    config = checkpoint.get('config', CONFIG)
    
    if cityflow_config_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cityflow_config_path = os.path.join(script_dir, "../examples/City_3_5/config.json")
    
    env_config = get_default_config(cityflow_config_path)
    env_config["episode_length"] = config.get('episode_length', 3600)
    env_config["ctrl_interval"] = config.get('ctrl_interval', 10)
    env_config["min_green"] = config.get('min_green', 10)
    env_config["min_duration"] = config.get('min_duration', 10)
    env_config["max_duration"] = config.get('max_duration', 60)
    env_config["verbose_violations"] = render
    
    env = CityFlowMultiIntersectionEnv(env_config, render_mode="human" if render else None)
    
    agent = CPOAgent(env, config)
    agent.load(model_path)
    
    print(f"\n{'='*60}")
    print("CPO 模型评估")
    print(f"{'='*60}")
    print(f"模型路径: {model_path}")
    print(f"约束阈值: {config.get('cost_limit', 100)}")
    print(f"评估 Episodes: {n_episodes}\n")
    
    episode_rewards = []
    episode_travel_times = []
    episode_violations = []
    
    for ep in range(n_episodes):
        state, info = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            disc_actions, cont_actions, _, _, _, _ = agent.select_action(state, deterministic=True)
            env_action = agent.convert_to_env_action(disc_actions, cont_actions)
            state, reward, terminated, truncated, info = env.step(env_action)
            done = terminated or truncated
            episode_reward += reward
            
            if render:
                env.render()
        
        episode_rewards.append(episode_reward)
        avg_travel_time = info.get('average_travel_time', 0)
        episode_travel_times.append(avg_travel_time)
        
        total_viol = sum(info.get('total_violations', {}).values())
        episode_violations.append(total_viol)
        
        constraint_status = "✓" if total_viol <= config.get('cost_limit', 100) else "✗"
        print(f"{constraint_status} Episode {ep+1}/{n_episodes}: "
              f"Reward={episode_reward:.1f}, AvgTT={avg_travel_time:.1f}s, "
              f"Violations={total_viol}")
        
        if render:
            env.print_intersection_flow_summary()
            env.print_violation_summary()
    
    env.close()
    
    satisfaction_rate = sum(1 for v in episode_violations if v <= config.get('cost_limit', 100)) / n_episodes * 100
    
    print(f"\n{'='*60}")
    print("评估结果")
    print(f"{'='*60}")
    print(f"平均奖励: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"平均行程时间: {np.mean(episode_travel_times):.2f}s")
    print(f"平均约束违反: {np.mean(episode_violations):.2f}")
    print(f"约束满足率: {satisfaction_rate:.1f}%")
    
    return {
        'episode_rewards': episode_rewards,
        'episode_violations': episode_violations,
        'satisfaction_rate': satisfaction_rate,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="CPO for CityFlow Traffic Signal Control")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "evaluate"])
    parser.add_argument("--config", type=str, default="../examples/City_3_5/config.json")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--episode-length", type=int, default=3600)
    parser.add_argument("--cost-limit", type=float, default=100.0, help="约束阈值")
    parser.add_argument("--delta", type=float, default=0.01, help="信任域半径")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if not os.path.isabs(args.config):
        config_path = os.path.join(script_dir, args.config)
    else:
        config_path = args.config
    
    if args.mode == "train":
        config = CONFIG.copy()
        config['num_of_episodes'] = args.episodes
        config['episode_length'] = args.episode_length
        config['cost_limit'] = args.cost_limit
        config['delta'] = args.delta
        
        results = train(config=config, cityflow_config_path=config_path)
        
        print("\n训练完成!")
        print(f"最终平均奖励: {np.mean(results['episode_rewards'][-10:]):.2f}")
        print(f"最终平均约束违反: {np.mean(results['episode_violations'][-10:]):.2f}")
        
    elif args.mode == "evaluate":
        if args.model is None:
            print("错误: evaluate 模式需要指定 --model 参数")
            sys.exit(1)
        
        if not os.path.isabs(args.model):
            model_path = os.path.join(script_dir, args.model)
        else:
            model_path = args.model
        
        results = evaluate(model_path, cityflow_config_path=config_path, n_episodes=5)
    
    print("\n完成!")

