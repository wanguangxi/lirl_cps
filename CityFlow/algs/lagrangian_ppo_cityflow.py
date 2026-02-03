"""
Lagrangian-PPO (Constrained PPO) for CityFlow Multi-Intersection Traffic Signal Control

基于拉格朗日松弛的约束强化学习算法：
- 使用拉格朗日乘子将约束优化转化为无约束优化
- 目标函数: max_π L(π, λ) = J(π) - λ * (C(π) - d)
  - J(π): 期望回报
  - C(π): 期望约束代价（约束违反次数）
  - d: 约束阈值
  - λ: 拉格朗日乘子

算法流程：
1. 收集轨迹，计算回报和约束代价
2. 更新策略（最大化拉格朗日目标）
3. 更新拉格朗日乘子（对偶梯度上升）

参考论文：
- Tessler et al. "Reward Constrained Policy Optimization" (2019)
- Ray et al. "Benchmarking Safe Exploration in Deep Reinforcement Learning" (2019)
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
    # PPO parameters
    'lr_actor': 0.0003,
    'lr_critic': 0.001,
    'gamma': 0.99,
    'lambda_gae': 0.95,
    'epsilon_clip': 0.2,
    'entropy_coef': 0.01,
    'value_coef': 0.5,
    'max_grad_norm': 0.5,
    
    # Lagrangian parameters
    'lr_lambda': 0.01,           # 拉格朗日乘子学习率
    'lambda_init': 0.1,          # 初始拉格朗日乘子
    'lambda_max': 10.0,          # 最大拉格朗日乘子
    'cost_limit': 100.0,         # 约束阈值（每 episode 允许的最大约束违反次数）
    'cost_gamma': 0.99,          # 约束代价折扣因子
    
    # Training parameters
    'batch_size': 64,
    'n_epochs': 10,
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
    'output_dir': './outputs/lagrangian_ppo_cityflow',
}


class ConstrainedRolloutBuffer:
    """存储 rollout 数据，包含约束代价"""
    def __init__(self):
        self.states = []
        self.discrete_actions = []
        self.continuous_actions = []
        self.discrete_log_probs = []
        self.continuous_log_probs = []
        self.rewards = []
        self.costs = []           # 约束代价（违反次数）
        self.dones = []
        self.values = []
        self.cost_values = []     # 约束代价的价值估计
        
    def store(self, state, disc_action, cont_action, disc_log_prob, cont_log_prob, 
              reward, cost, done, value, cost_value):
        self.states.append(state)
        self.discrete_actions.append(disc_action)
        self.continuous_actions.append(cont_action)
        self.discrete_log_probs.append(disc_log_prob)
        self.continuous_log_probs.append(cont_log_prob)
        self.rewards.append(reward)
        self.costs.append(cost)
        self.dones.append(done)
        self.values.append(value)
        self.cost_values.append(cost_value)
    
    def clear(self):
        self.states = []
        self.discrete_actions = []
        self.continuous_actions = []
        self.discrete_log_probs = []
        self.continuous_log_probs = []
        self.rewards = []
        self.costs = []
        self.dones = []
        self.values = []
        self.cost_values = []
    
    def compute_returns_and_advantages(self, last_value, last_cost_value, gamma, cost_gamma, lambda_gae):
        """计算奖励和约束代价的 GAE"""
        rewards = np.array(self.rewards)
        costs = np.array(self.costs)
        dones = np.array(self.dones)
        values = np.array(self.values + [last_value])
        cost_values = np.array(self.cost_values + [last_cost_value])
        
        # 奖励的 GAE
        advantages = np.zeros_like(rewards)
        last_gae = 0
        for t in reversed(range(len(rewards))):
            next_non_terminal = 1.0 - dones[t]
            next_value = values[t + 1]
            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + gamma * lambda_gae * next_non_terminal * last_gae
        returns = advantages + values[:-1]
        
        # 约束代价的 GAE
        cost_advantages = np.zeros_like(costs)
        last_cost_gae = 0
        for t in reversed(range(len(costs))):
            next_non_terminal = 1.0 - dones[t]
            next_cost_value = cost_values[t + 1]
            delta = costs[t] + cost_gamma * next_cost_value * next_non_terminal - cost_values[t]
            cost_advantages[t] = last_cost_gae = delta + cost_gamma * lambda_gae * next_non_terminal * last_cost_gae
        cost_returns = cost_advantages + cost_values[:-1]
        
        return returns, advantages, cost_returns, cost_advantages
    
    def get_batches(self, returns, advantages, cost_returns, cost_advantages, batch_size, device):
        """生成训练批次"""
        n_samples = len(self.states)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        states = torch.FloatTensor(np.array(self.states)).to(device)
        disc_actions = torch.LongTensor(np.array(self.discrete_actions)).to(device)
        cont_actions = torch.FloatTensor(np.array(self.continuous_actions)).to(device)
        disc_log_probs = torch.FloatTensor(np.array(self.discrete_log_probs)).to(device)
        cont_log_probs = torch.FloatTensor(np.array(self.continuous_log_probs)).to(device)
        returns_tensor = torch.FloatTensor(returns).to(device)
        advantages_tensor = torch.FloatTensor(advantages).to(device)
        cost_returns_tensor = torch.FloatTensor(cost_returns).to(device)
        cost_advantages_tensor = torch.FloatTensor(cost_advantages).to(device)
        
        # 标准化优势
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        for start in range(0, n_samples, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]
            
            yield (
                states[batch_indices],
                disc_actions[batch_indices],
                cont_actions[batch_indices],
                disc_log_probs[batch_indices],
                cont_log_probs[batch_indices],
                returns_tensor[batch_indices],
                advantages_tensor[batch_indices],
                cost_returns_tensor[batch_indices],
                cost_advantages_tensor[batch_indices]
            )


class LagrangianActorCritic(nn.Module):
    """
    拉格朗日 Actor-Critic 网络
    
    包含：
    - Actor: 输出离散动作概率和连续动作参数
    - Reward Critic: 估计奖励价值 V_r(s)
    - Cost Critic: 估计约束代价价值 V_c(s)
    """
    def __init__(self, state_size: int, num_intersections: int, num_phases: int,
                 hidden_dim1: int = 256, hidden_dim2: int = 128,
                 log_std_init: float = -0.5, log_std_min: float = -2.0, log_std_max: float = 0.5):
        super(LagrangianActorCritic, self).__init__()
        
        self.num_intersections = num_intersections
        self.num_phases = num_phases
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # ========== 共享特征提取 ==========
        self.shared_fc = nn.Sequential(
            nn.Linear(state_size, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU()
        )
        
        # ========== Actor (离散动作) ==========
        self.discrete_heads = nn.ModuleList([
            nn.Linear(hidden_dim2, num_phases) for _ in range(num_intersections)
        ])
        
        # ========== Actor (连续动作) ==========
        self.continuous_mean = nn.Linear(hidden_dim2, num_intersections)
        self.log_std = nn.Parameter(torch.ones(num_intersections) * log_std_init)
        
        # ========== Reward Critic ==========
        self.reward_critic = nn.Sequential(
            nn.Linear(hidden_dim2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # ========== Cost Critic ==========
        self.cost_critic = nn.Sequential(
            nn.Linear(hidden_dim2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, state):
        """前向传播"""
        shared_features = self.shared_fc(state)
        
        # 离散动作 logits
        discrete_logits = [head(shared_features) for head in self.discrete_heads]
        
        # 连续动作参数
        cont_mean = torch.sigmoid(self.continuous_mean(shared_features))
        cont_log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        cont_std = cont_log_std.exp()
        
        # 价值估计
        reward_value = self.reward_critic(shared_features)
        cost_value = self.cost_critic(shared_features)
        
        return discrete_logits, cont_mean, cont_std, reward_value, cost_value
    
    def get_action_and_value(self, state, discrete_actions=None, continuous_actions=None):
        """获取动作、对数概率和价值"""
        discrete_logits, cont_mean, cont_std, reward_value, cost_value = self.forward(state)
        
        # 离散动作
        disc_log_probs = []
        disc_entropies = []
        sampled_disc_actions = []
        
        for i, logits in enumerate(discrete_logits):
            dist = Categorical(logits=logits)
            
            if discrete_actions is None:
                action = dist.sample()
            else:
                action = discrete_actions[:, i]
            
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
            
            sampled_disc_actions.append(action)
            disc_log_probs.append(log_prob)
            disc_entropies.append(entropy)
        
        disc_actions_tensor = torch.stack(sampled_disc_actions, dim=1)
        disc_log_probs_tensor = torch.stack(disc_log_probs, dim=1).sum(dim=1)
        disc_entropy = torch.stack(disc_entropies, dim=1).mean(dim=1)
        
        # 连续动作
        cont_dist = Normal(cont_mean, cont_std)
        
        if continuous_actions is None:
            cont_actions_tensor = cont_dist.sample()
            cont_actions_tensor = torch.clamp(cont_actions_tensor, 0, 1)
        else:
            cont_actions_tensor = continuous_actions
        
        cont_log_probs_tensor = cont_dist.log_prob(cont_actions_tensor).sum(dim=1)
        cont_entropy = cont_dist.entropy().mean(dim=1)
        
        total_entropy = disc_entropy + cont_entropy
        
        return (disc_actions_tensor, cont_actions_tensor, 
                disc_log_probs_tensor, cont_log_probs_tensor,
                reward_value.squeeze(-1), cost_value.squeeze(-1), total_entropy)
    
    def get_values(self, state):
        """获取奖励和代价价值"""
        shared_features = self.shared_fc(state)
        reward_value = self.reward_critic(shared_features).squeeze(-1)
        cost_value = self.cost_critic(shared_features).squeeze(-1)
        return reward_value, cost_value


class LagrangianPPOAgent:
    """
    Lagrangian-PPO 智能体
    
    使用拉格朗日乘子处理约束：
    - 目标: max_π [J(π) - λ * (C(π) - d)]
    - 对偶更新: λ = max(0, λ + α * (C(π) - d))
    """
    def __init__(self, env: CityFlowMultiIntersectionEnv, config: Dict = None, device=None):
        self.config = config or CONFIG.copy()
        self.env = env
        self.device = device or DEVICE
        
        # 环境参数
        self.state_size = env.observation_space.shape[0]
        self.num_intersections = env.num_intersections
        self.num_phases = env.num_phases
        self.min_duration = env.min_duration
        self.max_duration = env.max_duration
        
        print(f"[Lagrangian-PPO Agent] 初始化:")
        print(f"  设备: {self.device}")
        print(f"  状态维度: {self.state_size}")
        print(f"  路口数量: {self.num_intersections}")
        print(f"  每个路口相位数: {self.num_phases}")
        print(f"  绿灯时长范围: [{self.min_duration}, {self.max_duration}]秒")
        print(f"  约束阈值 (cost_limit): {self.config['cost_limit']}")
        print(f"  初始 λ: {self.config['lambda_init']}")
        
        # 创建网络
        self.actor_critic = LagrangianActorCritic(
            state_size=self.state_size,
            num_intersections=self.num_intersections,
            num_phases=self.num_phases,
            hidden_dim1=self.config['hidden_dim1'],
            hidden_dim2=self.config['hidden_dim2'],
            log_std_init=self.config['log_std_init'],
            log_std_min=self.config['log_std_min'],
            log_std_max=self.config['log_std_max']
        ).to(self.device)
        
        # 优化器
        self.optimizer = optim.Adam([
            {'params': self.actor_critic.shared_fc.parameters(), 'lr': self.config['lr_actor']},
            {'params': self.actor_critic.discrete_heads.parameters(), 'lr': self.config['lr_actor']},
            {'params': self.actor_critic.continuous_mean.parameters(), 'lr': self.config['lr_actor']},
            {'params': [self.actor_critic.log_std], 'lr': self.config['lr_actor']},
            {'params': self.actor_critic.reward_critic.parameters(), 'lr': self.config['lr_critic']},
            {'params': self.actor_critic.cost_critic.parameters(), 'lr': self.config['lr_critic']},
        ])
        
        # 拉格朗日乘子（可学习）
        self.lagrangian_multiplier = nn.Parameter(
            torch.tensor([self.config['lambda_init']], dtype=torch.float32, device=self.device)
        )
        self.lambda_optimizer = optim.Adam([self.lagrangian_multiplier], lr=self.config['lr_lambda'])
        
        # Rollout buffer
        self.buffer = ConstrainedRolloutBuffer()
        
        # 记录
        self.episode_costs = []
        
    def get_lambda(self):
        """获取当前拉格朗日乘子（确保非负）"""
        return torch.clamp(self.lagrangian_multiplier, 0, self.config['lambda_max']).item()
    
    def select_action(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float, float, float]:
        """选择动作"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            disc_actions, cont_actions, disc_log_prob, cont_log_prob, reward_value, cost_value, _ = \
                self.actor_critic.get_action_and_value(state_tensor)
            
            return (
                disc_actions.squeeze(0).cpu().numpy(),
                cont_actions.squeeze(0).cpu().numpy(),
                disc_log_prob.item(),
                cont_log_prob.item(),
                reward_value.item(),
                cost_value.item()
            )
    
    def convert_to_env_action(self, discrete_actions: np.ndarray, continuous_actions: np.ndarray) -> np.ndarray:
        """转换为环境动作格式"""
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
        """存储转换"""
        self.buffer.store(state, disc_action, cont_action, disc_log_prob, cont_log_prob, 
                         reward, cost, done, value, cost_value)
    
    def update(self, episode_total_cost: float) -> Dict[str, float]:
        """
        Lagrangian-PPO 更新
        
        1. 更新策略（考虑拉格朗日惩罚）
        2. 更新拉格朗日乘子
        """
        # 获取最后状态的价值
        with torch.no_grad():
            last_state = torch.FloatTensor(self.buffer.states[-1]).unsqueeze(0).to(self.device)
            last_reward_value, last_cost_value = self.actor_critic.get_values(last_state)
            last_reward_value = last_reward_value.item()
            last_cost_value = last_cost_value.item()
        
        # 计算回报和优势
        returns, advantages, cost_returns, cost_advantages = self.buffer.compute_returns_and_advantages(
            last_reward_value, last_cost_value, 
            self.config['gamma'], self.config['cost_gamma'], self.config['lambda_gae']
        )
        
        # PPO 更新
        total_policy_loss = 0
        total_reward_value_loss = 0
        total_cost_value_loss = 0
        total_entropy_loss = 0
        n_updates = 0
        
        current_lambda = self.get_lambda()
        
        for epoch in range(self.config['n_epochs']):
            for batch in self.buffer.get_batches(returns, advantages, cost_returns, cost_advantages, 
                                                  self.config['batch_size'], self.device):
                (states, disc_actions, cont_actions, old_disc_log_probs, old_cont_log_probs,
                 batch_returns, batch_advantages, batch_cost_returns, batch_cost_advantages) = batch
                
                # 获取当前策略
                _, _, new_disc_log_probs, new_cont_log_probs, reward_values, cost_values, entropy = \
                    self.actor_critic.get_action_and_value(states, disc_actions, cont_actions)
                
                # 计算比率
                disc_ratio = torch.exp(new_disc_log_probs - old_disc_log_probs)
                cont_ratio = torch.exp(new_cont_log_probs - old_cont_log_probs)
                ratio = disc_ratio * cont_ratio
                
                # ========== 拉格朗日目标 ==========
                # 奖励优势
                surr1_reward = ratio * batch_advantages
                surr2_reward = torch.clamp(ratio, 1 - self.config['epsilon_clip'], 
                                           1 + self.config['epsilon_clip']) * batch_advantages
                reward_loss = -torch.min(surr1_reward, surr2_reward).mean()
                
                # 约束代价优势（乘以拉格朗日乘子）
                surr1_cost = ratio * batch_cost_advantages
                surr2_cost = torch.clamp(ratio, 1 - self.config['epsilon_clip'], 
                                         1 + self.config['epsilon_clip']) * batch_cost_advantages
                cost_loss = torch.min(surr1_cost, surr2_cost).mean()  # 最小化代价
                
                # 拉格朗日目标: max [reward - λ * cost]
                # 等价于 min [-reward + λ * cost]
                policy_loss = reward_loss + current_lambda * cost_loss
                
                # Value losses
                reward_value_loss = F.mse_loss(reward_values, batch_returns)
                cost_value_loss = F.mse_loss(cost_values, batch_cost_returns)
                
                # Entropy
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (policy_loss + 
                       self.config['value_coef'] * (reward_value_loss + cost_value_loss) + 
                       self.config['entropy_coef'] * entropy_loss)
                
                # 更新网络
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.config['max_grad_norm'])
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_reward_value_loss += reward_value_loss.item()
                total_cost_value_loss += cost_value_loss.item()
                total_entropy_loss += entropy_loss.item()
                n_updates += 1
        
        # ========== 更新拉格朗日乘子 ==========
        # 对偶梯度上升: λ = λ + α * (C(π) - d)
        cost_violation = episode_total_cost - self.config['cost_limit']
        
        self.lambda_optimizer.zero_grad()
        # 梯度 = (C(π) - d)，最小化 -λ * (C(π) - d) 等价于最大化 λ * (C(π) - d)
        lambda_loss = -self.lagrangian_multiplier * cost_violation
        lambda_loss.backward()
        self.lambda_optimizer.step()
        
        # 确保 λ 非负
        with torch.no_grad():
            self.lagrangian_multiplier.data.clamp_(0, self.config['lambda_max'])
        
        # 清空 buffer
        self.buffer.clear()
        
        return {
            'policy_loss': total_policy_loss / n_updates,
            'reward_value_loss': total_reward_value_loss / n_updates,
            'cost_value_loss': total_cost_value_loss / n_updates,
            'entropy_loss': total_entropy_loss / n_updates,
            'lambda': self.get_lambda(),
            'cost_violation': cost_violation,
        }
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'actor_critic': self.actor_critic.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lagrangian_multiplier': self.lagrangian_multiplier.data,
            'lambda_optimizer': self.lambda_optimizer.state_dict(),
            'config': self.config,
        }, path)
        print(f"模型已保存: {path}")
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint['actor_critic'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.lagrangian_multiplier.data = checkpoint['lagrangian_multiplier']
        self.lambda_optimizer.load_state_dict(checkpoint['lambda_optimizer'])
        print(f"模型已加载: {path} (设备: {self.device})")


def train(config: Dict = None, cityflow_config_path: str = None):
    """训练 Lagrangian-PPO 智能体"""
    config = config or CONFIG.copy()
    
    # CityFlow 配置
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
    print("Lagrangian-PPO for CityFlow Traffic Signal Control")
    print(f"{'='*60}")
    
    agent = LagrangianPPOAgent(env, config)
    
    # 输出目录
    output_dir = config.get('output_dir', './outputs/lagrangian_ppo_cityflow')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # 训练记录
    episode_rewards = []
    episode_travel_times = []
    episode_throughputs = []
    episode_violations = []
    episode_lambdas = []
    episode_policy_losses = []
    
    print(f"\n训练开始，输出目录: {run_dir}", flush=True)
    print(f"总 Episodes: {config['num_of_episodes']}", flush=True)
    print(f"约束阈值: {config['cost_limit']} (每 episode 最大约束违反)", flush=True)
    print(f"初始 λ: {config['lambda_init']}", flush=True)
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
        step_costs = []
        
        total_steps = config['episode_length'] // config['ctrl_interval']
        
        # 记录上一步的累计违反，用于计算步进增量
        prev_total_violations = 0
        
        while not done:
            # 选择动作
            disc_actions, cont_actions, disc_log_prob, cont_log_prob, value, cost_value = \
                agent.select_action(state)
            
            # 转换并执行
            env_action = agent.convert_to_env_action(disc_actions, cont_actions)
            next_state, reward, terminated, truncated, info = env.step(env_action)
            done = terminated or truncated
            
            # 获取这一步的约束代价（通过累计违反的增量计算）
            current_total_violations = sum(info.get('total_violations', {}).values())
            step_cost = current_total_violations - prev_total_violations
            prev_total_violations = current_total_violations
            step_costs.append(step_cost)
            
            # 存储经验
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
                      f"R={episode_reward:.0f} | C={episode_cost:.0f} | λ={agent.get_lambda():.3f}", 
                      end="", flush=True)
        
        # 更新
        losses = agent.update(episode_cost)
        
        # 统计
        episode_rewards.append(episode_reward)
        avg_travel_time = info.get('average_travel_time', 0)
        episode_travel_times.append(avg_travel_time)
        
        flow_stats = info.get('intersection_flow', {})
        total_throughput = sum(s.get('throughput', 0) for s in flow_stats.values())
        episode_throughputs.append(total_throughput)
        
        env_violations = info.get('total_violations', {})
        total_viol = sum(env_violations.values()) if env_violations else 0
        episode_violations.append(total_viol)
        episode_lambdas.append(losses['lambda'])
        episode_policy_losses.append(losses['policy_loss'])
        
        episode_time = time.time() - episode_start_time
        
        # 约束满足状态
        constraint_status = "✓" if total_viol <= config['cost_limit'] else "✗"
        
        print(f"\n{constraint_status} Episode {n_epi+1}/{config['num_of_episodes']} 完成 | "
              f"Reward={episode_reward:.1f} | Cost={total_viol:.0f}/{config['cost_limit']:.0f} | "
              f"λ={losses['lambda']:.3f} | Time={episode_time:.1f}s", flush=True)
        
        # 详细统计
        if (n_epi + 1) % config['print_interval'] == 0:
            avg_reward = np.mean(episode_rewards[-config['print_interval']:])
            avg_tt = np.mean(episode_travel_times[-config['print_interval']:])
            avg_viol = np.mean(episode_violations[-config['print_interval']:])
            avg_lambda = np.mean(episode_lambdas[-config['print_interval']:])
            elapsed = time.time() - train_start_time
            
            # 约束满足率
            constraint_satisfied = sum(1 for v in episode_violations[-config['print_interval']:] 
                                       if v <= config['cost_limit'])
            satisfaction_rate = constraint_satisfied / config['print_interval'] * 100
            
            print(f"\n{'─'*60}")
            print(f"📊 Episode {n_epi+1}/{config['num_of_episodes']} 统计 (耗时: {elapsed/60:.1f}分钟)")
            print(f"   平均奖励: {avg_reward:.1f}")
            print(f"   平均行程时间: {avg_tt:.1f}s")
            print(f"   平均约束违反: {avg_viol:.1f} (阈值: {config['cost_limit']})")
            print(f"   约束满足率: {satisfaction_rate:.1f}%")
            print(f"   平均 λ: {avg_lambda:.3f}")
            print(f"   Policy Loss: {losses['policy_loss']:.4f}")
            print(f"{'─'*60}\n", flush=True)
    
    # 保存
    if config.get('save_models', True):
        model_path = os.path.join(run_dir, "lagrangian_ppo_final.pt")
        agent.save(model_path)
    
    import json
    stats_path = os.path.join(run_dir, "training_stats.json")
    with open(stats_path, 'w') as f:
        json.dump({
            'episode_rewards': episode_rewards,
            'episode_travel_times': episode_travel_times,
            'episode_throughputs': episode_throughputs,
            'episode_violations': episode_violations,
            'episode_lambdas': episode_lambdas,
            'episode_policy_losses': episode_policy_losses,
            'config': {k: v for k, v in config.items() if not callable(v)},
        }, f, indent=2)
    print(f"训练统计已保存: {stats_path}")
    
    # 最终统计
    final_satisfaction = sum(1 for v in episode_violations[-20:] if v <= config['cost_limit']) / 20 * 100
    print(f"\n最终约束满足率 (后20 episodes): {final_satisfaction:.1f}%")
    print(f"最终 λ: {agent.get_lambda():.3f}")
    
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
    
    agent = LagrangianPPOAgent(env, config)
    agent.load(model_path)
    
    print(f"\n{'='*60}")
    print("Lagrangian-PPO 模型评估")
    print(f"{'='*60}")
    print(f"模型路径: {model_path}")
    print(f"当前 λ: {agent.get_lambda():.3f}")
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
            disc_actions, cont_actions, _, _, _, _ = agent.select_action(state)
            env_action = agent.convert_to_env_action(disc_actions, cont_actions)
            state, reward, terminated, truncated, info = env.step(env_action)
            done = terminated or truncated
            episode_reward += reward
            
            if render:
                env.render()
        
        episode_rewards.append(episode_reward)
        avg_travel_time = info.get('average_travel_time', 0)
        episode_travel_times.append(avg_travel_time)
        
        env_violations = info.get('total_violations', {})
        total_viol = sum(env_violations.values()) if env_violations else 0
        episode_violations.append(total_viol)
        
        constraint_status = "✓" if total_viol <= config.get('cost_limit', 100) else "✗"
        print(f"{constraint_status} Episode {ep+1}/{n_episodes}: "
              f"Reward={episode_reward:.1f}, "
              f"AvgTT={avg_travel_time:.1f}s, "
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
    
    parser = argparse.ArgumentParser(description="Lagrangian-PPO for CityFlow Traffic Signal Control")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "evaluate"])
    parser.add_argument("--config", type=str, default="../examples/City_3_5/config.json")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--episode-length", type=int, default=3600)
    parser.add_argument("--cost-limit", type=float, default=100.0, help="约束阈值（每episode最大违反次数）")
    parser.add_argument("--lambda-init", type=float, default=0.1, help="初始拉格朗日乘子")
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
        config['lambda_init'] = args.lambda_init
        
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

