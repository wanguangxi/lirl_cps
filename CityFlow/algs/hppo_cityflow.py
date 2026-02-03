"""
HPPO (Hybrid Proximal Policy Optimization) for CityFlow Multi-Intersection Traffic Signal Control

适用于混合动作空间：
- 离散动作：相位选择 (每个路口选择 0 ~ num_phases-1)
- 连续参数：每个相位对应的绿灯时长 (min_duration ~ max_duration)

HPPO 核心思想：
1. Actor 网络同时输出离散动作的概率分布和连续参数
2. 使用 PPO 的 clipped objective 进行稳定训练
3. 结合 GAE (Generalized Advantage Estimation) 降低方差
4. 原生支持混合动作空间

参考论文：
- Schulman et al. "Proximal Policy Optimization Algorithms" (2017)
- Fan et al. "Hybrid Actor-Critic Reinforcement Learning in Parameterized Action Space" (2019)
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
    'lambda_gae': 0.95,      # GAE lambda
    'epsilon_clip': 0.2,     # PPO clipping parameter
    'entropy_coef': 0.01,    # Entropy bonus coefficient
    'value_coef': 0.5,       # Value loss coefficient
    'max_grad_norm': 0.5,    # Gradient clipping
    
    # Training parameters
    'batch_size': 64,
    'n_epochs': 10,          # PPO epochs per update
    'rollout_length': 360,   # Steps per rollout (1 episode)
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
    'log_std_init': -0.5,    # Initial log std for continuous actions
    'log_std_min': -2.0,
    'log_std_max': 0.5,
    
    # Output parameters
    'print_interval': 10,
    'save_models': True,
    'output_dir': './outputs/hppo_cityflow',
}


class RolloutBuffer:
    """存储 rollout 数据用于 PPO 更新"""
    def __init__(self):
        self.states = []
        self.discrete_actions = []
        self.continuous_actions = []
        self.discrete_log_probs = []
        self.continuous_log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        
    def store(self, state, disc_action, cont_action, disc_log_prob, cont_log_prob, reward, done, value):
        self.states.append(state)
        self.discrete_actions.append(disc_action)
        self.continuous_actions.append(cont_action)
        self.discrete_log_probs.append(disc_log_prob)
        self.continuous_log_probs.append(cont_log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
    
    def clear(self):
        self.states = []
        self.discrete_actions = []
        self.continuous_actions = []
        self.discrete_log_probs = []
        self.continuous_log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
    
    def compute_returns_and_advantages(self, last_value, gamma, lambda_gae):
        """计算 GAE 优势和回报"""
        rewards = np.array(self.rewards)
        dones = np.array(self.dones)
        values = np.array(self.values + [last_value])
        
        # GAE
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t + 1]
            
            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + gamma * lambda_gae * next_non_terminal * last_gae
        
        returns = advantages + values[:-1]
        
        return returns, advantages
    
    def get_batches(self, returns, advantages, batch_size, device):
        """生成训练批次"""
        n_samples = len(self.states)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        # 转换为张量
        states = torch.FloatTensor(np.array(self.states)).to(device)
        disc_actions = torch.LongTensor(np.array(self.discrete_actions)).to(device)
        cont_actions = torch.FloatTensor(np.array(self.continuous_actions)).to(device)
        disc_log_probs = torch.FloatTensor(np.array(self.discrete_log_probs)).to(device)
        cont_log_probs = torch.FloatTensor(np.array(self.continuous_log_probs)).to(device)
        returns_tensor = torch.FloatTensor(returns).to(device)
        advantages_tensor = torch.FloatTensor(advantages).to(device)
        
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
                advantages_tensor[batch_indices]
            )


class HybridActorCritic(nn.Module):
    """
    混合 Actor-Critic 网络
    
    Actor 输出：
    - 每个路口的相位选择概率 (离散)
    - 每个路口的绿灯时长参数 (连续，均值和标准差)
    
    Critic 输出：
    - 状态价值 V(s)
    """
    def __init__(self, state_size: int, num_intersections: int, num_phases: int,
                 hidden_dim1: int = 256, hidden_dim2: int = 128,
                 log_std_init: float = -0.5, log_std_min: float = -2.0, log_std_max: float = 0.5):
        super(HybridActorCritic, self).__init__()
        
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
        # 每个路口的相位选择概率
        self.discrete_heads = nn.ModuleList([
            nn.Linear(hidden_dim2, num_phases) for _ in range(num_intersections)
        ])
        
        # ========== Actor (连续动作) ==========
        # 每个路口的绿灯时长参数 (均值)
        self.continuous_mean = nn.Linear(hidden_dim2, num_intersections)
        
        # 可学习的 log_std
        self.log_std = nn.Parameter(torch.ones(num_intersections) * log_std_init)
        
        # ========== Critic ==========
        self.critic_fc = nn.Sequential(
            nn.Linear(hidden_dim2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, state):
        """前向传播，返回策略分布参数和状态价值"""
        shared_features = self.shared_fc(state)
        
        # 离散动作概率
        discrete_logits = []
        for head in self.discrete_heads:
            logits = head(shared_features)
            discrete_logits.append(logits)
        
        # 连续动作参数
        cont_mean = torch.sigmoid(self.continuous_mean(shared_features))  # [0, 1]
        cont_log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        cont_std = cont_log_std.exp()
        
        # 状态价值
        value = self.critic_fc(shared_features)
        
        return discrete_logits, cont_mean, cont_std, value
    
    def get_action_and_value(self, state, discrete_actions=None, continuous_actions=None):
        """
        获取动作、对数概率和价值
        
        如果提供了动作，则计算给定动作的对数概率（用于训练）
        否则采样新动作（用于收集数据）
        """
        discrete_logits, cont_mean, cont_std, value = self.forward(state)
        
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
        disc_log_probs_tensor = torch.stack(disc_log_probs, dim=1).sum(dim=1)  # 所有路口的对数概率之和
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
        
        # 总熵
        total_entropy = disc_entropy + cont_entropy
        
        return (disc_actions_tensor, cont_actions_tensor, 
                disc_log_probs_tensor, cont_log_probs_tensor,
                value.squeeze(-1), total_entropy)
    
    def get_value(self, state):
        """仅获取状态价值"""
        shared_features = self.shared_fc(state)
        value = self.critic_fc(shared_features)
        return value.squeeze(-1)


class HPPOAgent:
    """
    HPPO 智能体 - 用于 CityFlow 多路口交通信号控制
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
        
        print(f"[HPPO Agent] 初始化:")
        print(f"  设备: {self.device}")
        print(f"  状态维度: {self.state_size}")
        print(f"  路口数量: {self.num_intersections}")
        print(f"  每个路口相位数: {self.num_phases}")
        print(f"  绿灯时长范围: [{self.min_duration}, {self.max_duration}]秒")
        
        # 创建网络
        self.actor_critic = HybridActorCritic(
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
            {'params': self.actor_critic.critic_fc.parameters(), 'lr': self.config['lr_critic']},
        ])
        
        # Rollout buffer
        self.buffer = RolloutBuffer()
        
    def select_action(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
        """
        选择动作
        
        Returns:
            discrete_actions: 每个路口的相位选择
            continuous_actions: 每个路口的绿灯时长 (归一化 [0,1])
            disc_log_prob: 离散动作的对数概率
            cont_log_prob: 连续动作的对数概率
            value: 状态价值
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            disc_actions, cont_actions, disc_log_prob, cont_log_prob, value, _ = \
                self.actor_critic.get_action_and_value(state_tensor)
            
            return (
                disc_actions.squeeze(0).cpu().numpy(),
                cont_actions.squeeze(0).cpu().numpy(),
                disc_log_prob.item(),
                cont_log_prob.item(),
                value.item()
            )
    
    def convert_to_env_action(self, discrete_actions: np.ndarray, continuous_actions: np.ndarray) -> np.ndarray:
        """
        将动作转换为环境格式
        
        Args:
            discrete_actions: 相位选择 (num_intersections,)
            continuous_actions: 归一化的绿灯时长 [0,1] (num_intersections,)
        
        Returns:
            env_action: [phase_0, duration_idx_0, ...]
        """
        env_action = np.zeros(self.num_intersections * 2, dtype=np.int64)
        
        for i in range(self.num_intersections):
            # 相位
            phase = int(discrete_actions[i])
            phase = np.clip(phase, 0, self.num_phases - 1)
            env_action[i * 2] = phase
            
            # 时长：从 [0,1] 映射到 [min_duration, max_duration]
            duration = self.min_duration + continuous_actions[i] * (self.max_duration - self.min_duration)
            duration_idx = int(round(duration - self.min_duration))
            duration_idx = np.clip(duration_idx, 0, self.max_duration - self.min_duration)
            env_action[i * 2 + 1] = duration_idx
        
        return env_action
    
    def store_transition(self, state, disc_action, cont_action, disc_log_prob, cont_log_prob, reward, done, value):
        """存储转换"""
        self.buffer.store(state, disc_action, cont_action, disc_log_prob, cont_log_prob, reward, done, value)
    
    def update(self) -> Dict[str, float]:
        """
        PPO 更新
        
        Returns:
            losses: 各项损失的字典
        """
        # 计算最后一个状态的价值
        with torch.no_grad():
            last_state = torch.FloatTensor(self.buffer.states[-1]).unsqueeze(0).to(self.device)
            last_value = self.actor_critic.get_value(last_state).item()
        
        # 计算回报和优势
        returns, advantages = self.buffer.compute_returns_and_advantages(
            last_value, self.config['gamma'], self.config['lambda_gae']
        )
        
        # PPO 多轮更新
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        n_updates = 0
        
        for epoch in range(self.config['n_epochs']):
            for batch in self.buffer.get_batches(returns, advantages, self.config['batch_size'], self.device):
                states, disc_actions, cont_actions, old_disc_log_probs, old_cont_log_probs, batch_returns, batch_advantages = batch
                
                # 获取当前策略的动作概率和价值
                _, _, new_disc_log_probs, new_cont_log_probs, values, entropy = \
                    self.actor_critic.get_action_and_value(states, disc_actions, cont_actions)
                
                # 计算比率
                disc_ratio = torch.exp(new_disc_log_probs - old_disc_log_probs)
                cont_ratio = torch.exp(new_cont_log_probs - old_cont_log_probs)
                ratio = disc_ratio * cont_ratio  # 混合比率
                
                # Clipped surrogate objective
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.config['epsilon_clip'], 1 + self.config['epsilon_clip']) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values, batch_returns)
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (policy_loss + 
                       self.config['value_coef'] * value_loss + 
                       self.config['entropy_coef'] * entropy_loss)
                
                # 更新
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.config['max_grad_norm'])
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
                n_updates += 1
        
        # 清空 buffer
        self.buffer.clear()
        
        return {
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'entropy_loss': total_entropy_loss / n_updates,
        }
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'actor_critic': self.actor_critic.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config,
        }, path)
        print(f"模型已保存: {path}")
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint['actor_critic'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"模型已加载: {path} (设备: {self.device})")


def train(config: Dict = None, cityflow_config_path: str = None):
    """训练 HPPO 智能体"""
    config = config or CONFIG.copy()
    
    # 设置 CityFlow 配置路径
    if cityflow_config_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cityflow_config_path = os.path.join(script_dir, "../examples/City_3_5/config.json")
    
    # 创建环境配置
    env_config = get_default_config(cityflow_config_path)
    env_config["episode_length"] = config['episode_length']
    env_config["ctrl_interval"] = config['ctrl_interval']
    env_config["min_green"] = config['min_green']
    env_config["min_duration"] = config['min_duration']
    env_config["max_duration"] = config['max_duration']
    env_config["verbose_violations"] = False
    env_config["log_violations"] = True
    
    # 创建环境
    env = CityFlowMultiIntersectionEnv(env_config)
    
    print(f"\n{'='*60}")
    print("HPPO for CityFlow Traffic Signal Control")
    print(f"{'='*60}")
    
    # 创建智能体
    agent = HPPOAgent(env, config)
    
    # 创建输出目录
    output_dir = config.get('output_dir', './outputs/hppo_cityflow')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # 训练记录
    episode_rewards = []
    episode_travel_times = []
    episode_throughputs = []
    episode_violations = []
    episode_policy_losses = []
    episode_value_losses = []
    
    print(f"\n训练开始，输出目录: {run_dir}", flush=True)
    print(f"总 Episodes: {config['num_of_episodes']}", flush=True)
    print(f"每 Episode 步数: {config['episode_length'] // config['ctrl_interval']}", flush=True)
    print(f"PPO epochs: {config['n_epochs']}", flush=True)
    print(f"打印间隔: 每 {config['print_interval']} episodes\n", flush=True)
    
    import time
    train_start_time = time.time()
    
    for n_epi in range(config['num_of_episodes']):
        episode_start_time = time.time()
        state, info = env.reset()
        done = False
        episode_reward = 0
        step = 0
        
        total_steps = config['episode_length'] // config['ctrl_interval']
        
        while not done:
            # 选择动作
            disc_actions, cont_actions, disc_log_prob, cont_log_prob, value = agent.select_action(state)
            
            # 转换为环境动作
            env_action = agent.convert_to_env_action(disc_actions, cont_actions)
            
            # 执行动作
            next_state, reward, terminated, truncated, info = env.step(env_action)
            done = terminated or truncated
            
            # 存储经验
            agent.store_transition(state, disc_actions, cont_actions, disc_log_prob, cont_log_prob, reward, done, value)
            
            episode_reward += reward
            state = next_state
            step += 1
            
            # 每100步打印进度
            if step % 100 == 0 or done:
                progress = step / total_steps
                bar_len = 20
                filled = int(bar_len * progress)
                bar = '█' * filled + '░' * (bar_len - filled)
                print(f"\r  [{bar}] {progress*100:5.1f}% | Step {step}/{total_steps} | "
                      f"R={episode_reward:.0f}", end="", flush=True)
        
        # PPO 更新
        losses = agent.update()
        
        # 记录统计
        episode_rewards.append(episode_reward)
        avg_travel_time = info.get('average_travel_time', 0)
        episode_travel_times.append(avg_travel_time)
        
        flow_stats = info.get('intersection_flow', {})
        total_throughput = sum(s.get('throughput', 0) for s in flow_stats.values())
        episode_throughputs.append(total_throughput)
        
        env_violations = info.get('total_violations', {})
        total_viol = sum(env_violations.values()) if env_violations else 0
        episode_violations.append(total_viol)
        
        episode_policy_losses.append(losses['policy_loss'])
        episode_value_losses.append(losses['value_loss'])
        
        episode_time = time.time() - episode_start_time
        
        # 打印 Episode 结果
        print(f"\n✓ Episode {n_epi+1}/{config['num_of_episodes']} 完成 | "
              f"Reward={episode_reward:.1f} | AvgTT={avg_travel_time:.0f}s | "
              f"Violations={total_viol} | Time={episode_time:.1f}s", flush=True)
        
        # 详细统计
        if (n_epi + 1) % config['print_interval'] == 0:
            avg_reward = np.mean(episode_rewards[-config['print_interval']:])
            avg_tt = np.mean(episode_travel_times[-config['print_interval']:])
            avg_tp = np.mean(episode_throughputs[-config['print_interval']:])
            avg_viol = np.mean(episode_violations[-config['print_interval']:])
            avg_policy_loss = np.mean(episode_policy_losses[-config['print_interval']:])
            avg_value_loss = np.mean(episode_value_losses[-config['print_interval']:])
            elapsed = time.time() - train_start_time
            
            print(f"\n{'─'*60}")
            print(f"📊 Episode {n_epi+1}/{config['num_of_episodes']} 统计 (耗时: {elapsed/60:.1f}分钟)")
            print(f"   平均奖励: {avg_reward:.1f}")
            print(f"   平均行程时间: {avg_tt:.1f}s")
            print(f"   总吞吐量: {avg_tp:.0f}")
            print(f"   约束违反: {avg_viol:.1f}")
            print(f"   Policy Loss: {avg_policy_loss:.4f}, Value Loss: {avg_value_loss:.4f}")
            print(f"{'─'*60}\n", flush=True)
    
    # 保存模型
    if config.get('save_models', True):
        model_path = os.path.join(run_dir, "hppo_cityflow_final.pt")
        agent.save(model_path)
    
    # 保存训练曲线
    import json
    stats_path = os.path.join(run_dir, "training_stats.json")
    with open(stats_path, 'w') as f:
        json.dump({
            'episode_rewards': episode_rewards,
            'episode_travel_times': episode_travel_times,
            'episode_throughputs': episode_throughputs,
            'episode_violations': episode_violations,
            'episode_policy_losses': episode_policy_losses,
            'episode_value_losses': episode_value_losses,
        }, f, indent=2)
    print(f"训练统计已保存: {stats_path}")
    
    # 打印最终违反统计
    if episode_violations:
        print(f"\n约束违反总计（环境检测）: {sum(episode_violations)}")
    
    env.close()
    
    return {
        'agent': agent,
        'episode_rewards': episode_rewards,
        'episode_travel_times': episode_travel_times,
        'run_dir': run_dir,
    }


def evaluate(model_path: str, cityflow_config_path: str = None, n_episodes: int = 5, render: bool = True):
    """评估训练好的模型"""
    # 加载模型
    checkpoint = torch.load(model_path, map_location=DEVICE)
    config = checkpoint.get('config', CONFIG)
    
    # 设置 CityFlow 配置路径
    if cityflow_config_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cityflow_config_path = os.path.join(script_dir, "../examples/City_3_5/config.json")
    
    # 创建环境
    env_config = get_default_config(cityflow_config_path)
    env_config["episode_length"] = config.get('episode_length', 3600)
    env_config["ctrl_interval"] = config.get('ctrl_interval', 10)
    env_config["min_green"] = config.get('min_green', 10)
    env_config["min_duration"] = config.get('min_duration', 10)
    env_config["max_duration"] = config.get('max_duration', 60)
    env_config["verbose_violations"] = render
    
    env = CityFlowMultiIntersectionEnv(env_config, render_mode="human" if render else None)
    
    # 创建智能体并加载权重
    agent = HPPOAgent(env, config)
    agent.load(model_path)
    
    print(f"\n{'='*60}")
    print("HPPO 模型评估")
    print(f"{'='*60}")
    print(f"模型路径: {model_path}")
    print(f"评估 Episodes: {n_episodes}\n")
    
    episode_rewards = []
    episode_travel_times = []
    episode_violations = []
    
    for ep in range(n_episodes):
        state, info = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # 选择动作（评估时不需要存储）
            disc_actions, cont_actions, _, _, _ = agent.select_action(state)
            
            # 转换为环境动作
            env_action = agent.convert_to_env_action(disc_actions, cont_actions)
            
            # 执行动作
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
        
        print(f"Episode {ep+1}/{n_episodes}: "
              f"Reward={episode_reward:.1f}, "
              f"AvgTravelTime={avg_travel_time:.1f}s, "
              f"Violations={total_viol}")
        
        if render:
            env.print_intersection_flow_summary()
            env.print_violation_summary()
    
    env.close()
    
    print(f"\n{'='*60}")
    print("评估结果")
    print(f"{'='*60}")
    print(f"平均奖励: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"平均行程时间: {np.mean(episode_travel_times):.2f}s")
    print(f"平均约束违反: {np.mean(episode_violations):.2f}")
    
    return {
        'episode_rewards': episode_rewards,
        'episode_travel_times': episode_travel_times,
        'episode_violations': episode_violations,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="HPPO for CityFlow Traffic Signal Control")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "evaluate"],
                       help="运行模式")
    parser.add_argument("--config", type=str, default="../examples/City_3_5/config.json",
                       help="CityFlow 配置文件路径")
    parser.add_argument("--model", type=str, default=None,
                       help="模型文件路径 (evaluate 模式需要)")
    parser.add_argument("--episodes", type=int, default=200,
                       help="训练 episodes 数")
    parser.add_argument("--episode-length", type=int, default=3600,
                       help="每个 episode 的仿真时长（秒）")
    parser.add_argument("--min-duration", type=int, default=10,
                       help="最小绿灯时长（秒）")
    parser.add_argument("--max-duration", type=int, default=60,
                       help="最大绿灯时长（秒）")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子")
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # 获取脚本目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 处理配置文件路径
    if not os.path.isabs(args.config):
        config_path = os.path.join(script_dir, args.config)
    else:
        config_path = args.config
    
    if args.mode == "train":
        # 更新配置
        config = CONFIG.copy()
        config['num_of_episodes'] = args.episodes
        config['episode_length'] = args.episode_length
        config['min_duration'] = args.min_duration
        config['max_duration'] = args.max_duration
        
        results = train(config=config, cityflow_config_path=config_path)
        
        print("\n训练完成!")
        print(f"最终平均奖励: {np.mean(results['episode_rewards'][-10:]):.2f}")
        print(f"最终平均行程时间: {np.mean(results['episode_travel_times'][-10:]):.2f}s")
        
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

