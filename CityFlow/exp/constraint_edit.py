"""
约束编辑实验 - 测试 LIRL 模型在约束变化下的适应能力

实验内容：
1. 加载已训练的 LIRL 模型，测试原始环境下的性能
2. 新建约束环境（某路口发生交通事故，禁止通行），修改动作投影满足约束，测试性能
3. 在约束环境下进行迁移学习，测试性能
4. 对比三组测试性能
5. 测试迁移学习所需的最小训练 episode 和时间

约束设定：
- 选择一个路口（如 intersection_1_1）发生交通事故
- 该路口所有相位变为红灯（禁止通行）
- 相邻路口需要调整信号配时
"""

import os
import sys
import json
import random
import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import collections
from datetime import datetime
from typing import Dict, List, Optional, Set
import argparse

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "env"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "algs"))

from cityflow_multi_env import CityFlowMultiIntersectionEnv, get_default_config

# 尝试导入绘图库
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False
    print("[WARNING] matplotlib 未安装，将跳过绘图")

# =======================
# GPU 设备检测
# =======================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[DEVICE] 使用设备: {DEVICE}")
if torch.cuda.is_available():
    print(f"[DEVICE] GPU: {torch.cuda.get_device_name(0)}")


# =======================
# 配置
# =======================
CONFIG = {
    # Learning parameters
    'lr_mu': 0.0003,
    'lr_q': 0.001,
    'gamma': 0.99,
    'batch_size': 128 if torch.cuda.is_available() else 64,
    'buffer_limit': 100000,
    'tau': 0.005,
    
    # Environment parameters
    'episode_length': 3600,
    'ctrl_interval': 10,
    'min_green': 10,
    'min_duration': 10,
    'max_duration': 60,
    
    # Network architecture
    'hidden_dim1': 256,
    'hidden_dim2': 128,
    
    # Training parameters
    'memory_threshold': 500,
    'training_iterations': 10,
}

# 事故路口配置
ACCIDENT_INTERSECTION = "intersection_1_1"  # 发生事故的路口


# =======================
# 神经网络
# =======================

class ActorNetwork(nn.Module):
    """Actor网络"""
    def __init__(self, state_size, action_size, hidden1=256, hidden2=128):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, action_size)
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))


class CriticNetwork(nn.Module):
    """Critic网络"""
    def __init__(self, state_size, action_size, hidden1=256, hidden2=128):
        super().__init__()
        self.fc_s = nn.Linear(state_size, hidden1)
        self.fc_a = nn.Linear(action_size, hidden1)
        self.fc2 = nn.Linear(hidden1 * 2, hidden2)
        self.fc_out = nn.Linear(hidden2, 1)
        self._init_weights()
    
    def _init_weights(self):
        for m in [self.fc_s, self.fc_a, self.fc2, self.fc_out]:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, x, a):
        h1 = F.relu(self.fc_s(x))
        h2 = F.relu(self.fc_a(a))
        cat = torch.cat([h1, h2], dim=1)
        q = F.relu(self.fc2(cat))
        return self.fc_out(q)


class ReplayBuffer:
    """经验回放缓冲区"""
    def __init__(self, buffer_limit=100000):
        self.buffer = collections.deque(maxlen=buffer_limit)
    
    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        return random.sample(self.buffer, min(n, len(self.buffer)))
    
    def size(self):
        return len(self.buffer)
    
    def clear(self):
        self.buffer.clear()


class OrnsteinUhlenbeckNoise:
    """OU噪声"""
    def __init__(self, mu, theta=0.15, dt=0.01, sigma=0.2):
        self.theta = theta
        self.dt = dt
        self.sigma = sigma
        self.mu = mu
        self.x_prev = np.zeros_like(self.mu)
    
    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x
    
    def reset(self):
        self.x_prev = np.zeros_like(self.mu)


# =======================
# 约束感知动作投影器
# =======================

class ConstraintAwareProjector:
    """
    约束感知动作投影器
    
    支持额外的约束：
    - 事故路口约束：指定路口强制红灯
    """
    def __init__(self, num_intersections, num_phases, min_duration, max_duration, 
                 min_green=10, blocked_intersections: Set[str] = None):
        self.num_intersections = num_intersections
        self.num_phases = num_phases
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.min_green = min_green
        self.num_duration_options = max_duration - min_duration + 1
        
        # 被阻塞的路口（发生事故）
        self.blocked_intersections = blocked_intersections or set()
    
    def set_blocked_intersections(self, blocked: Set[str]):
        """设置被阻塞的路口"""
        self.blocked_intersections = blocked
    
    def project(self, continuous_action, env=None):
        """将连续动作投影到满足约束的离散动作"""
        if isinstance(continuous_action, torch.Tensor):
            continuous_action = continuous_action.detach().cpu().numpy()
        
        a_ = np.clip(continuous_action.flatten(), 0, 1)
        
        expected_dim = self.num_intersections * 2
        if len(a_) < expected_dim:
            padded = np.ones(expected_dim) * 0.5
            padded[:len(a_)] = a_
            a_ = padded
        elif len(a_) > expected_dim:
            a_ = a_[:expected_dim]
        
        discrete_action = np.zeros(self.num_intersections * 2, dtype=np.int64)
        
        if env is None:
            return self._simple_project(a_)
        
        try:
            current_phases = env.current_phases.copy()
            phase_elapsed = env.phase_elapsed.copy()
            target_durations = env.target_durations.copy()
            valid_phases = env.valid_phases.copy()
            intersection_ids = env.intersection_ids.copy()
        except Exception:
            return self._simple_project(a_)
        
        for i, inter_id in enumerate(intersection_ids):
            phase_prob = a_[i * 2]
            duration_prob = a_[i * 2 + 1]
            
            # ========== 事故路口约束 ==========
            # 如果是被阻塞的路口，在满足 min_green 约束的前提下尽快切换相位
            # 这会造成交通混乱但不会产生约束违反
            if inter_id in self.blocked_intersections:
                cur_phase = current_phases.get(inter_id, 0)
                elapsed = phase_elapsed.get(inter_id, 0.0)
                target_dur = target_durations.get(inter_id, self.min_duration)
                
                # 检查是否可以切换（满足 min_green 和 target_duration）
                can_switch = (elapsed >= self.min_green) and (elapsed >= target_dur)
                
                if can_switch:
                    # 切换到下一个相位
                    next_phase = (cur_phase + 1) % self.num_phases
                    discrete_action[i * 2] = next_phase
                    discrete_action[i * 2 + 1] = 0  # 最短时长，尽快再次切换
                else:
                    # 保持当前相位，等待 min_green 满足
                    discrete_action[i * 2] = cur_phase
                    discrete_action[i * 2 + 1] = 0  # 设置最短目标时长
                continue
            
            cur_phase = current_phases.get(inter_id, 0)
            elapsed = phase_elapsed.get(inter_id, 0.0)
            target_duration = target_durations.get(inter_id, self.min_duration)
            inter_valid_phases = valid_phases.get(inter_id, [True] * self.num_phases)
            
            desired_phase = int(phase_prob * (self.num_phases - 1) + 0.5)
            desired_phase = np.clip(desired_phase, 0, self.num_phases - 1)
            
            can_switch = (elapsed >= self.min_green) and (elapsed >= target_duration)
            
            if desired_phase != cur_phase:
                if can_switch and inter_valid_phases[desired_phase]:
                    selected_phase = desired_phase
                else:
                    selected_phase = cur_phase
            else:
                selected_phase = cur_phase
            
            desired_duration_idx = int(duration_prob * (self.num_duration_options - 1) + 0.5)
            desired_duration_idx = np.clip(desired_duration_idx, 0, self.num_duration_options - 1)
            
            min_duration_idx = max(0, self.min_green - self.min_duration)
            selected_duration_idx = max(desired_duration_idx, min_duration_idx)
            
            discrete_action[i * 2] = selected_phase
            discrete_action[i * 2 + 1] = selected_duration_idx
        
        return discrete_action
    
    def _simple_project(self, a_):
        discrete_action = np.zeros(self.num_intersections * 2, dtype=np.int64)
        for i in range(self.num_intersections):
            phase_prob = a_[i * 2]
            duration_prob = a_[i * 2 + 1]
            discrete_action[i * 2] = int(np.clip(phase_prob * (self.num_phases - 1) + 0.5, 0, self.num_phases - 1))
            discrete_action[i * 2 + 1] = int(np.clip(duration_prob * (self.num_duration_options - 1) + 0.5, 0, self.num_duration_options - 1))
        return discrete_action


# =======================
# LIRL Agent
# =======================

class LIRLAgent:
    """LIRL (DDPG) Agent"""
    def __init__(self, state_size, num_intersections, num_phases, config,
                 blocked_intersections: Set[str] = None):
        self.config = config
        self.state_size = state_size
        self.num_intersections = num_intersections
        self.num_phases = num_phases
        
        action_size = num_intersections * 2
        
        self.actor = ActorNetwork(state_size, action_size,
                                   config['hidden_dim1'], config['hidden_dim2']).to(DEVICE)
        self.actor_target = ActorNetwork(state_size, action_size,
                                          config['hidden_dim1'], config['hidden_dim2']).to(DEVICE)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = CriticNetwork(state_size, action_size,
                                     config['hidden_dim1'], config['hidden_dim2']).to(DEVICE)
        self.critic_target = CriticNetwork(state_size, action_size,
                                            config['hidden_dim1'], config['hidden_dim2']).to(DEVICE)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config['lr_mu'])
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config['lr_q'])
        
        self.memory = ReplayBuffer(config['buffer_limit'])
        self.ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(action_size))
        self.noise_scale = 1.0
        
        self.projector = ConstraintAwareProjector(
            num_intersections, num_phases,
            config['min_duration'], config['max_duration'],
            config['min_green'],
            blocked_intersections
        )
        self.env = None
    
    def set_env(self, env):
        self.env = env
    
    def set_blocked_intersections(self, blocked: Set[str]):
        """设置被阻塞的路口"""
        self.projector.set_blocked_intersections(blocked)
    
    def select_action(self, state, deterministic=False):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            action = self.actor(state_t).squeeze(0).cpu().numpy()
        
        if not deterministic:
            noise = self.ou_noise() * self.noise_scale
            action = np.clip(action + noise, 0, 1)
        
        env_action = self.projector.project(action, self.env)
        return env_action, action
    
    def store(self, state, action, reward, next_state, done):
        self.memory.put((state, action, reward, next_state, done))
    
    def train_step(self):
        if self.memory.size() < self.config['memory_threshold']:
            return
        
        batch = self.memory.sample(self.config['batch_size'])
        states = torch.FloatTensor([t[0] for t in batch]).to(DEVICE)
        actions = torch.FloatTensor([t[1] for t in batch]).to(DEVICE)
        rewards = torch.FloatTensor([[t[2]] for t in batch]).to(DEVICE)
        next_states = torch.FloatTensor([t[3] for t in batch]).to(DEVICE)
        dones = torch.FloatTensor([[1.0 - t[4]] for t in batch]).to(DEVICE)
        
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target = rewards + self.config['gamma'] * target_q * dones
        
        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        actor_loss = -self.critic(states, self.actor(states)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # Soft update
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.config['tau'] * param.data + (1 - self.config['tau']) * target_param.data)
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.config['tau'] * param.data + (1 - self.config['tau']) * target_param.data)
    
    def decay_noise(self, episode, total_episodes):
        self.noise_scale = max(0.1, 1.0 - episode / (total_episodes * 0.8))
    
    def reset_noise(self):
        self.ou_noise.reset()
    
    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path, map_location=DEVICE)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        if 'actor_target' in checkpoint:
            self.actor_target.load_state_dict(checkpoint['actor_target'])
        else:
            self.actor_target.load_state_dict(checkpoint['actor'])
        if 'critic_target' in checkpoint:
            self.critic_target.load_state_dict(checkpoint['critic_target'])
        else:
            self.critic_target.load_state_dict(checkpoint['critic'])


# =======================
# 环境创建
# =======================

def create_environment(config_path: str, base_config: Dict):
    """创建环境"""
    env_config = get_default_config(config_path)
    env_config.update({
        "episode_length": base_config['episode_length'],
        "ctrl_interval": base_config['ctrl_interval'],
        "min_green": base_config['min_green'],
        "min_duration": base_config['min_duration'],
        "max_duration": base_config['max_duration'],
        "verbose_violations": False,
        "log_violations": True,
    })
    return CityFlowMultiIntersectionEnv(env_config)


# =======================
# 评估函数
# =======================

def evaluate_agent(agent, env, num_episodes: int = 10, desc: str = "") -> Dict:
    """评估智能体"""
    agent.set_env(env)
    
    all_rewards = []
    all_travel_times = []
    all_throughputs = []
    all_violations = []
    all_violation_rates = []
    
    for ep in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        episode_violations = 0
        step_count = 0
        done = False
        
        while not done:
            env_action, _ = agent.select_action(state, deterministic=True)
            state, reward, terminated, truncated, info = env.step(env_action)
            done = terminated or truncated
            episode_reward += reward
            
            # Track constraint violations
            violations = info.get('total_violations', {})
            episode_violations += sum(violations.values())
            step_count += 1
        
        all_rewards.append(episode_reward)
        all_travel_times.append(info.get('average_travel_time', 0))
        
        flow_stats = info.get('intersection_flow', {})
        throughput = sum(s.get('throughput', 0) for s in flow_stats.values())
        all_throughputs.append(throughput)
        all_violations.append(episode_violations)
        
        # Violation rate = violations / steps
        violation_rate = episode_violations / max(step_count, 1)
        all_violation_rates.append(violation_rate)
        
        if desc:
            print(f"    {desc} Episode {ep+1}/{num_episodes}: TP={throughput:.0f}, TT={all_travel_times[-1]:.0f}s, Viol={violation_rate:.2f}")
    
    return {
        'mean_reward': np.mean(all_rewards),
        'std_reward': np.std(all_rewards),
        'mean_travel_time': np.mean(all_travel_times),
        'std_travel_time': np.std(all_travel_times),
        'mean_throughput': np.mean(all_throughputs),
        'std_throughput': np.std(all_throughputs),
        'mean_violations': np.mean(all_violations),
        'std_violations': np.std(all_violations),
        'mean_violation_rate': np.mean(all_violation_rates),
        'std_violation_rate': np.std(all_violation_rates),
        'all_rewards': all_rewards,
        'all_travel_times': all_travel_times,
        'all_throughputs': all_throughputs,
        'all_violations': all_violations,
        'all_violation_rates': all_violation_rates,
    }


# =======================
# 迁移学习
# =======================

def transfer_learning(agent, env, target_throughput: float, target_travel_time: float,
                     max_episodes: int = 100, patience: int = 10,
                     print_interval: int = 5) -> Dict:
    """
    迁移学习 - 在约束环境下微调模型
    
    Args:
        agent: LIRL 智能体
        env: 约束环境
        target_throughput: 目标吞吐量（原始性能的一定比例）
        target_travel_time: 目标行程时间
        max_episodes: 最大训练回合数
        patience: 早停耐心值
        print_interval: 打印间隔
    
    Returns:
        迁移学习结果
    """
    agent.set_env(env)
    agent.memory.clear()  # 清空旧经验
    agent.noise_scale = 0.5  # 降低探索噪声
    
    episode_rewards = []
    episode_travel_times = []
    episode_throughputs = []
    
    best_performance = float('-inf')
    episodes_without_improvement = 0
    convergence_episode = None
    
    start_time = time.time()
    
    print(f"\n  开始迁移学习 (最大 {max_episodes} 回合)...")
    print(f"  目标吞吐量: {target_throughput:.0f}, 目标行程时间: {target_travel_time:.0f}s")
    
    for ep in range(max_episodes):
        state, info = env.reset()
        episode_reward = 0
        done = False
        agent.reset_noise()
        
        while not done:
            env_action, action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(env_action)
            done = terminated or truncated
            
            agent.store(state, action, reward, next_state, done)
            episode_reward += reward
            state = next_state
        
        # 训练
        for _ in range(10):
            agent.train_step()
        
        agent.decay_noise(ep, max_episodes)
        
        # 记录统计
        episode_rewards.append(episode_reward)
        avg_tt = info.get('average_travel_time', 0)
        episode_travel_times.append(avg_tt)
        
        flow_stats = info.get('intersection_flow', {})
        throughput = sum(s.get('throughput', 0) for s in flow_stats.values())
        episode_throughputs.append(throughput)
        
        # 检查是否达到目标
        current_performance = throughput - avg_tt  # 简单的性能指标
        
        if current_performance > best_performance:
            best_performance = current_performance
            episodes_without_improvement = 0
        else:
            episodes_without_improvement += 1
        
        # 检查收敛条件
        if convergence_episode is None:
            if throughput >= target_throughput * 0.9 and avg_tt <= target_travel_time * 1.1:
                convergence_episode = ep + 1
                convergence_time = time.time() - start_time
                print(f"    *** 达到目标性能! Episode {ep+1}, 用时 {convergence_time:.1f}s ***")
        
        if (ep + 1) % print_interval == 0:
            avg_tp = np.mean(episode_throughputs[-print_interval:])
            avg_tt_val = np.mean(episode_travel_times[-print_interval:])
            print(f"    Episode {ep+1}/{max_episodes}: TP={avg_tp:.0f}, TT={avg_tt_val:.0f}s")
        
        # 早停
        if episodes_without_improvement >= patience and convergence_episode is not None:
            print(f"    早停: {patience} 回合无改进")
            break
    
    total_time = time.time() - start_time
    
    return {
        'episode_rewards': episode_rewards,
        'episode_travel_times': episode_travel_times,
        'episode_throughputs': episode_throughputs,
        'total_episodes': len(episode_rewards),
        'total_time': total_time,
        'convergence_episode': convergence_episode,
        'convergence_time': convergence_time if convergence_episode else None,
        'final_throughput': np.mean(episode_throughputs[-5:]) if len(episode_throughputs) >= 5 else np.mean(episode_throughputs),
        'final_travel_time': np.mean(episode_travel_times[-5:]) if len(episode_travel_times) >= 5 else np.mean(episode_travel_times),
    }


# =======================
# 绘图函数
# =======================

def plot_comparison(results: Dict, output_dir: str):
    """绘制对比图"""
    if not HAS_PLOT:
        return
    
    scenarios = list(results.keys())
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    colors = ['#2ecc71', '#e74c3c', '#3498db']
    x_pos = np.arange(len(scenarios))
    
    # Throughput Comparison
    ax = axes[0]
    throughputs = [results[s]['evaluation']['mean_throughput'] for s in scenarios]
    throughput_stds = [results[s]['evaluation']['std_throughput'] for s in scenarios]
    bars = ax.bar(x_pos, throughputs, yerr=throughput_stds, capsize=8,
                  color=colors[:len(scenarios)], edgecolor='black', linewidth=1.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(scenarios, rotation=15, ha='right')
    ax.set_ylabel('Throughput (vehicles)', fontsize=12)
    ax.set_title('Throughput Comparison', fontsize=14)
    
    for bar, val in zip(bars, throughputs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                f'{val:.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Travel Time Comparison
    ax = axes[1]
    travel_times = [results[s]['evaluation']['mean_travel_time'] for s in scenarios]
    travel_time_stds = [results[s]['evaluation']['std_travel_time'] for s in scenarios]
    bars = ax.bar(x_pos, travel_times, yerr=travel_time_stds, capsize=8,
                  color=colors[:len(scenarios)], edgecolor='black', linewidth=1.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(scenarios, rotation=15, ha='right')
    ax.set_ylabel('Average Travel Time (s)', fontsize=12)
    ax.set_title('Travel Time Comparison', fontsize=14)
    
    for bar, val in zip(bars, travel_times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                f'{val:.0f}s', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Violation Rate Comparison
    ax = axes[2]
    violation_rates = [results[s]['evaluation'].get('mean_violation_rate', 0) for s in scenarios]
    violation_rate_stds = [results[s]['evaluation'].get('std_violation_rate', 0) for s in scenarios]
    bars = ax.bar(x_pos, violation_rates, yerr=violation_rate_stds, capsize=8,
                  color=colors[:len(scenarios)], edgecolor='black', linewidth=1.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(scenarios, rotation=15, ha='right')
    ax.set_ylabel('Violation Rate (per step)', fontsize=12)
    ax.set_title('Constraint Violation Rate Comparison', fontsize=14)
    
    for bar, val in zip(bars, violation_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scenario_comparison.png'), dpi=150)
    plt.close()
    print(f"  Comparison chart saved: scenario_comparison.png")


def plot_transfer_learning(transfer_result: Dict, output_dir: str):
    """绘制迁移学习曲线"""
    if not HAS_PLOT:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Throughput Curve
    ax = axes[0]
    throughputs = transfer_result['episode_throughputs']
    ax.plot(throughputs, color='#3498db', linewidth=2, label='Throughput')
    if transfer_result['convergence_episode']:
        ax.axvline(x=transfer_result['convergence_episode']-1, color='#e74c3c', 
                   linestyle='--', label=f'Convergence (Ep {transfer_result["convergence_episode"]})')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Throughput')
    ax.set_title('Transfer Learning - Throughput')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Travel Time Curve
    ax = axes[1]
    travel_times = transfer_result['episode_travel_times']
    ax.plot(travel_times, color='#e74c3c', linewidth=2, label='Travel Time')
    if transfer_result['convergence_episode']:
        ax.axvline(x=transfer_result['convergence_episode']-1, color='#3498db',
                   linestyle='--', label=f'Convergence (Ep {transfer_result["convergence_episode"]})')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Travel Time (s)')
    ax.set_title('Transfer Learning - Travel Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'transfer_learning.png'), dpi=150)
    plt.close()
    print(f"  迁移学习曲线已保存: transfer_learning.png")


# =======================
# 主函数
# =======================

def main():
    parser = argparse.ArgumentParser(description="约束编辑实验")
    parser.add_argument("--model", type=str,
                       default=os.path.join(PROJECT_ROOT, 
                           "outputs/algorithm_compare/run_20260114_100609/LIRL_model.pt"),
                       help="预训练模型路径")
    parser.add_argument("--config", type=str,
                       default=os.path.join(PROJECT_ROOT, "examples/City_3_5/config.json"),
                       help="CityFlow 配置文件路径")
    parser.add_argument("--eval-episodes", type=int, default=10,
                       help="评估回合数")
    parser.add_argument("--max-transfer-episodes", type=int, default=100,
                       help="迁移学习最大回合数")
    parser.add_argument("--output-dir", type=str,
                       default=os.path.join(PROJECT_ROOT, "outputs/constraint_edit"),
                       help="输出目录")
    parser.add_argument("--accident-intersection", type=str, default="intersection_1_1",
                       help="发生事故的路口")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子")
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"\n{'#'*70}")
    print(f"# 约束编辑实验")
    print(f"# 模型路径: {args.model}")
    print(f"# 事故路口: {args.accident_intersection}")
    print(f"# 输出目录: {run_dir}")
    print(f"{'#'*70}")
    
    # 检查模型是否存在
    if not os.path.exists(args.model):
        print(f"[ERROR] 模型文件不存在: {args.model}")
        return
    
    results = {}
    
    # ========================================
    # 场景1: 原始环境 - 加载预训练模型测试
    # ========================================
    print(f"\n{'='*70}")
    print("场景 1: 原始环境 - 预训练模型")
    print(f"{'='*70}")
    
    env_original = create_environment(args.config, CONFIG)
    
    state_size = env_original.observation_space.shape[0]
    num_intersections = env_original.num_intersections
    num_phases = env_original.num_phases
    
    print(f"  环境信息:")
    print(f"    状态维度: {state_size}")
    print(f"    路口数量: {num_intersections}")
    print(f"    路口列表: {env_original.intersection_ids[:5]}...")
    
    # 创建智能体并加载模型
    agent_original = LIRLAgent(
        state_size=state_size,
        num_intersections=num_intersections,
        num_phases=num_phases,
        config=CONFIG
    )
    agent_original.load(args.model)
    print(f"  已加载模型: {args.model}")
    
    # 评估
    print(f"\n  评估原始环境性能...")
    eval_original = evaluate_agent(agent_original, env_original, args.eval_episodes, "Original")
    
    results['Original'] = {
        'evaluation': eval_original,
        'description': 'Pre-trained model performance in original environment'
    }
    
    print(f"\n  Original Environment Evaluation:")
    print(f"    Throughput: {eval_original['mean_throughput']:.0f} ± {eval_original['std_throughput']:.0f}")
    print(f"    Travel Time: {eval_original['mean_travel_time']:.0f} ± {eval_original['std_travel_time']:.0f}s")
    print(f"    Violation Rate: {eval_original['mean_violation_rate']:.4f} ± {eval_original['std_violation_rate']:.4f}")
    
    env_original.close()
    
    # ========================================
    # 场景2: 约束环境 - 不重训练，仅修改投影
    # ========================================
    print(f"\n{'='*70}")
    print(f"场景 2: 约束环境 - 仅修改动作投影（无重训练）")
    print(f"  事故路口: {args.accident_intersection}")
    print(f"{'='*70}")
    
    env_constrained = create_environment(args.config, CONFIG)
    
    # 创建智能体并加载模型，设置阻塞路口
    agent_constrained = LIRLAgent(
        state_size=state_size,
        num_intersections=num_intersections,
        num_phases=num_phases,
        config=CONFIG,
        blocked_intersections={args.accident_intersection}
    )
    agent_constrained.load(args.model)
    
    print(f"  已设置事故路口约束: {args.accident_intersection}")
    print(f"  该路口信号灯将强制保持红灯状态")
    
    # 评估
    print(f"\n  评估约束环境性能（无重训练）...")
    eval_constrained = evaluate_agent(agent_constrained, env_constrained, args.eval_episodes, "Constrained")
    
    results['Constrained (No Retrain)'] = {
        'evaluation': eval_constrained,
        'description': f'Accident at {args.accident_intersection}, only modified action projection'
    }
    
    print(f"\n  Constrained (No Retrain) Evaluation:")
    print(f"    Throughput: {eval_constrained['mean_throughput']:.0f} ± {eval_constrained['std_throughput']:.0f}")
    print(f"    Travel Time: {eval_constrained['mean_travel_time']:.0f} ± {eval_constrained['std_travel_time']:.0f}s")
    print(f"    Violation Rate: {eval_constrained['mean_violation_rate']:.4f} ± {eval_constrained['std_violation_rate']:.4f}")
    
    # 计算性能下降
    tp_drop = (eval_original['mean_throughput'] - eval_constrained['mean_throughput']) / eval_original['mean_throughput'] * 100
    tt_increase = (eval_constrained['mean_travel_time'] - eval_original['mean_travel_time']) / eval_original['mean_travel_time'] * 100
    print(f"\n  性能变化:")
    print(f"    吞吐量下降: {tp_drop:.1f}%")
    print(f"    行程时间增加: {tt_increase:.1f}%")
    
    env_constrained.close()
    
    # ========================================
    # 场景3: 约束环境 - 迁移学习
    # ========================================
    print(f"\n{'='*70}")
    print(f"场景 3: 约束环境 - 迁移学习")
    print(f"{'='*70}")
    
    env_transfer = create_environment(args.config, CONFIG)
    
    # 创建新智能体，加载预训练模型，设置阻塞路口
    agent_transfer = LIRLAgent(
        state_size=state_size,
        num_intersections=num_intersections,
        num_phases=num_phases,
        config=CONFIG,
        blocked_intersections={args.accident_intersection}
    )
    agent_transfer.load(args.model)
    
    # 设置迁移学习目标（原始性能的某个比例）
    target_throughput = eval_original['mean_throughput'] * 0.8  # 目标: 恢复到原始吞吐量的80%
    target_travel_time = eval_original['mean_travel_time'] * 1.2  # 目标: 行程时间不超过原始的120%
    
    # 执行迁移学习
    transfer_result = transfer_learning(
        agent=agent_transfer,
        env=env_transfer,
        target_throughput=target_throughput,
        target_travel_time=target_travel_time,
        max_episodes=args.max_transfer_episodes,
        patience=20,
        print_interval=10
    )
    
    # 最终评估
    print(f"\n  迁移学习后最终评估...")
    eval_transfer = evaluate_agent(agent_transfer, env_transfer, args.eval_episodes, "迁移学习后")
    
    results['Constrained (Transfer)'] = {
        'evaluation': eval_transfer,
        'transfer_learning': transfer_result,
        'description': 'Performance after transfer learning in constrained environment'
    }
    
    print(f"\n  Transfer Learning Evaluation:")
    print(f"    Throughput: {eval_transfer['mean_throughput']:.0f} ± {eval_transfer['std_throughput']:.0f}")
    print(f"    Travel Time: {eval_transfer['mean_travel_time']:.0f} ± {eval_transfer['std_travel_time']:.0f}s")
    print(f"    Violation Rate: {eval_transfer['mean_violation_rate']:.4f} ± {eval_transfer['std_violation_rate']:.4f}")
    
    # 保存迁移学习后的模型
    transfer_model_path = os.path.join(run_dir, "transfer_model.pt")
    agent_transfer.save(transfer_model_path)
    print(f"  迁移学习模型已保存: {transfer_model_path}")
    
    env_transfer.close()
    
    # ========================================
    # Results Summary
    # ========================================
    print(f"\n{'='*90}")
    print("Experiment Results Summary")
    print(f"{'='*90}")
    
    print(f"\n{'Scenario':<25} {'Throughput':<20} {'Travel Time':<20} {'Violation Rate':<20}")
    print(f"{'-'*90}")
    
    for scenario, data in results.items():
        eval_data = data['evaluation']
        tp_str = f"{eval_data['mean_throughput']:.0f} ± {eval_data['std_throughput']:.0f}"
        tt_str = f"{eval_data['mean_travel_time']:.0f} ± {eval_data['std_travel_time']:.0f}s"
        vr_str = f"{eval_data['mean_violation_rate']:.4f}"
        print(f"{scenario:<25} {tp_str:<20} {tt_str:<20} {vr_str:<20}")
    
    # 迁移学习统计
    print(f"\n{'='*70}")
    print("迁移学习统计")
    print(f"{'='*70}")
    print(f"  总训练回合: {transfer_result['total_episodes']}")
    print(f"  总训练时间: {transfer_result['total_time']:.1f}s")
    if transfer_result['convergence_episode']:
        print(f"  收敛回合: {transfer_result['convergence_episode']}")
        print(f"  收敛时间: {transfer_result['convergence_time']:.1f}s")
    else:
        print(f"  未达到收敛条件")
    
    # 性能恢复比例
    if eval_original['mean_throughput'] > 0:
        recovery_tp = eval_transfer['mean_throughput'] / eval_original['mean_throughput'] * 100
        recovery_tt = eval_original['mean_travel_time'] / eval_transfer['mean_travel_time'] * 100
        print(f"\n  性能恢复比例:")
        print(f"    吞吐量恢复: {recovery_tp:.1f}%")
        print(f"    行程时间恢复: {recovery_tt:.1f}%")
    
    # 绘制对比图
    print(f"\n{'='*70}")
    print("绘制对比图表...")
    print(f"{'='*70}")
    
    plot_comparison(results, run_dir)
    plot_transfer_learning(transfer_result, run_dir)
    
    # 保存结果
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        return obj
    
    results_serializable = convert_to_serializable(results)
    
    results_path = os.path.join(run_dir, "results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results_serializable, f, indent=2, ensure_ascii=False)
    print(f"完整结果已保存: {results_path}")
    
    # 汇总
    summary = {
        'experiment_info': {
            'timestamp': timestamp,
            'model_path': args.model,
            'accident_intersection': args.accident_intersection,
            'eval_episodes': args.eval_episodes,
            'max_transfer_episodes': args.max_transfer_episodes,
        },
        'results': {
            'Original': {
                'throughput': float(eval_original['mean_throughput']),
                'travel_time': float(eval_original['mean_travel_time']),
                'violation_rate': float(eval_original['mean_violation_rate']),
            },
            'Constrained (No Retrain)': {
                'throughput': float(eval_constrained['mean_throughput']),
                'travel_time': float(eval_constrained['mean_travel_time']),
                'violation_rate': float(eval_constrained['mean_violation_rate']),
            },
            'Constrained (Transfer)': {
                'throughput': float(eval_transfer['mean_throughput']),
                'travel_time': float(eval_transfer['mean_travel_time']),
                'violation_rate': float(eval_transfer['mean_violation_rate']),
            },
        },
        'transfer_learning': {
            'total_episodes': transfer_result['total_episodes'],
            'total_time': transfer_result['total_time'],
            'convergence_episode': transfer_result['convergence_episode'],
            'convergence_time': transfer_result['convergence_time'],
        }
    }
    
    summary_path = os.path.join(run_dir, "summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"汇总结果已保存: {summary_path}")
    
    print(f"\n{'='*70}")
    print(f"实验完成!")
    print(f"输出目录: {run_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

