"""
算法对比实验程序

对比 5 种强化学习算法在 CityFlow 交通信号控制任务上的表现：
1. LIRL (DDPG-based)
2. PDQN (Parameterized Deep Q-Network)
3. HPPO (Hybrid Proximal Policy Optimization)
4. Lagrangian-PPO (Constrained PPO with Lagrangian relaxation)
5. CPO (Constrained Policy Optimization)

实验内容：
1. 每种算法训练 500 个回合，绘制学习曲线对比
2. 训练完成后保存训练好的策略模型
3. 每种算法测试 10 个回合，统计平均吞吐量、平均行程时间、约束违反率
4. 绘制热力图对比
5. 所有实验结果保存为 JSON 文件
"""

import os
import sys
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal
import collections
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import argparse
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "env"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "algs"))

from cityflow_multi_env import CityFlowMultiIntersectionEnv, get_default_config

# 尝试导入绘图库
try:
    import matplotlib
    matplotlib.use('Agg')  # 非交互式后端
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False
    print("[WARNING] matplotlib/seaborn 未安装，将跳过绘图")

# =======================
# GPU 设备检测
# =======================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[DEVICE] 使用设备: {DEVICE}")
if torch.cuda.is_available():
    print(f"[DEVICE] GPU: {torch.cuda.get_device_name(0)}")


# =======================
# 统一的网络架构配置
# =======================
UNIFIED_CONFIG = {
    # Network architecture (统一)
    'hidden_dim1': 256,
    'hidden_dim2': 128,
    
    # Learning parameters
    'gamma': 0.99,
    'batch_size': 128 if torch.cuda.is_available() else 64,
    
    # Environment parameters
    'episode_length': 3600,
    'ctrl_interval': 10,
    'min_green': 10,
    'min_duration': 10,
    'max_duration': 60,
    
    # Constraint parameters
    'cost_limit': 100.0,
}


# =======================
# 共用的神经网络组件
# =======================

class BaseActorNetwork(nn.Module):
    """基础 Actor 网络 - 用于 DDPG/LIRL"""
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


class BaseCriticNetwork(nn.Module):
    """基础 Critic 网络"""
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


class HybridActorCritic(nn.Module):
    """混合动作空间的 Actor-Critic 网络 - 用于 HPPO, Lagrangian-PPO, CPO"""
    def __init__(self, state_size, num_intersections, num_phases, 
                 hidden1=256, hidden2=128, log_std_init=-0.5):
        super().__init__()
        self.num_intersections = num_intersections
        self.num_phases = num_phases
        
        # 共享特征提取层
        self.shared = nn.Sequential(
            nn.Linear(state_size, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU()
        )
        
        # 离散动作头：每个路口的相位选择
        self.discrete_head = nn.Linear(hidden2, num_intersections * num_phases)
        
        # 连续动作头：每个路口的绿灯时长参数
        self.continuous_mean = nn.Linear(hidden2, num_intersections)
        self.log_std = nn.Parameter(torch.ones(num_intersections) * log_std_init)
        
        # Value head
        self.value_head = nn.Linear(hidden2, 1)
        
        # Cost value head (用于约束RL)
        self.cost_value_head = nn.Linear(hidden2, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        features = self.shared(x)
        
        # 离散动作 logits
        discrete_logits = self.discrete_head(features)
        discrete_logits = discrete_logits.view(-1, self.num_intersections, self.num_phases)
        
        # 连续动作均值
        continuous_mean = torch.sigmoid(self.continuous_mean(features))
        
        # Value
        value = self.value_head(features)
        cost_value = self.cost_value_head(features)
        
        return discrete_logits, continuous_mean, value, cost_value
    
    def get_action(self, state, deterministic=False):
        with torch.no_grad():
            discrete_logits, continuous_mean, value, cost_value = self.forward(state)
            
            # 离散动作
            discrete_probs = F.softmax(discrete_logits, dim=-1)
            if deterministic:
                discrete_actions = discrete_probs.argmax(dim=-1)
            else:
                dist = Categorical(probs=discrete_probs)
                discrete_actions = dist.sample()
            
            # 连续动作
            if deterministic:
                continuous_actions = continuous_mean
            else:
                std = torch.exp(self.log_std.clamp(-2.0, 0.5))
                dist = Normal(continuous_mean, std)
                continuous_actions = dist.sample()
                continuous_actions = torch.clamp(continuous_actions, 0, 1)
            
            return discrete_actions, continuous_actions, value, cost_value


class QNetwork(nn.Module):
    """Q 网络 - 用于 PDQN"""
    def __init__(self, state_size, num_intersections, num_phases, 
                 hidden1=256, hidden2=128):
        super().__init__()
        self.num_intersections = num_intersections
        self.num_phases = num_phases
        
        # 输入: state + 连续参数 (每个路口一个时长参数)
        input_size = state_size + num_intersections
        
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        # 输出: 每个路口每个相位的 Q 值
        self.fc3 = nn.Linear(hidden2, num_intersections * num_phases)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward(self, state, params):
        x = torch.cat([state, params], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values.view(-1, self.num_intersections, self.num_phases)


class ParameterNetwork(nn.Module):
    """参数网络 - 用于 PDQN"""
    def __init__(self, state_size, num_intersections, hidden1=256, hidden2=128):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, num_intersections)
        self._init_weights()
    
    def _init_weights(self):
        for m in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # 使用 sigmoid 将输出限制在 [0, 1]
        return torch.sigmoid(self.fc3(x))


# =======================
# 经验回放缓冲区
# =======================

class ReplayBuffer:
    """通用经验回放缓冲区"""
    def __init__(self, buffer_limit=100000):
        self.buffer = collections.deque(maxlen=buffer_limit)
    
    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        return random.sample(self.buffer, min(n, len(self.buffer)))
    
    def size(self):
        return len(self.buffer)


class RolloutBuffer:
    """Rollout 缓冲区 (on-policy)"""
    def __init__(self):
        self.clear()
    
    def clear(self):
        self.states = []
        self.discrete_actions = []
        self.continuous_actions = []
        self.rewards = []
        self.costs = []
        self.dones = []
        self.values = []
        self.cost_values = []
        self.log_probs_discrete = []
        self.log_probs_continuous = []
    
    def store(self, state, disc_action, cont_action, reward, cost, done, 
              value, cost_value, log_prob_d, log_prob_c):
        self.states.append(state)
        self.discrete_actions.append(disc_action)
        self.continuous_actions.append(cont_action)
        self.rewards.append(reward)
        self.costs.append(cost)
        self.dones.append(done)
        self.values.append(value)
        self.cost_values.append(cost_value)
        self.log_probs_discrete.append(log_prob_d)
        self.log_probs_continuous.append(log_prob_c)
    
    def get_batch(self, gamma=0.99, lambda_gae=0.95, device=DEVICE):
        states = torch.FloatTensor(np.array(self.states)).to(device)
        disc_actions = torch.LongTensor(np.array(self.discrete_actions)).to(device)
        cont_actions = torch.FloatTensor(np.array(self.continuous_actions)).to(device)
        
        # 计算 GAE
        rewards = np.array(self.rewards)
        values = np.array(self.values + [0])
        dones = np.array(self.dones)
        
        advantages = np.zeros_like(rewards)
        last_gae = 0
        for t in reversed(range(len(rewards))):
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * values[t+1] * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + gamma * lambda_gae * next_non_terminal * last_gae
        returns = advantages + values[:-1]
        
        # 对 costs 计算类似的 advantage
        costs = np.array(self.costs)
        cost_values = np.array(self.cost_values + [0])
        cost_advantages = np.zeros_like(costs)
        last_cost_gae = 0
        for t in reversed(range(len(costs))):
            next_non_terminal = 1.0 - dones[t]
            delta = costs[t] + gamma * cost_values[t+1] * next_non_terminal - cost_values[t]
            cost_advantages[t] = last_cost_gae = delta + gamma * lambda_gae * next_non_terminal * last_cost_gae
        cost_returns = cost_advantages + cost_values[:-1]
        
        return {
            'states': states,
            'discrete_actions': disc_actions,
            'continuous_actions': cont_actions,
            'advantages': torch.FloatTensor(advantages).to(device),
            'returns': torch.FloatTensor(returns).to(device),
            'cost_advantages': torch.FloatTensor(cost_advantages).to(device),
            'cost_returns': torch.FloatTensor(cost_returns).to(device),
            'old_log_probs_d': torch.FloatTensor(self.log_probs_discrete).to(device),
            'old_log_probs_c': torch.FloatTensor(self.log_probs_continuous).to(device),
        }


# =======================
# 动作转换器
# =======================

class ActionConverter:
    """将网络输出转换为环境动作"""
    def __init__(self, num_intersections, num_phases, min_duration, max_duration):
        self.num_intersections = num_intersections
        self.num_phases = num_phases
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.num_duration_options = max_duration - min_duration + 1
    
    def convert(self, discrete_actions, continuous_params):
        """
        Args:
            discrete_actions: shape (num_intersections,) 离散相位
            continuous_params: shape (num_intersections,) 归一化时长参数 [0, 1]
        Returns:
            env_action: shape (num_intersections * 2,) 环境动作格式
        """
        if isinstance(discrete_actions, torch.Tensor):
            discrete_actions = discrete_actions.cpu().numpy()
        if isinstance(continuous_params, torch.Tensor):
            continuous_params = continuous_params.cpu().numpy()
        
        discrete_actions = np.atleast_1d(discrete_actions).flatten()
        continuous_params = np.atleast_1d(continuous_params).flatten()
        
        env_action = np.zeros(self.num_intersections * 2, dtype=np.int64)
        
        for i in range(self.num_intersections):
            # 离散相位
            phase = int(discrete_actions[i]) if i < len(discrete_actions) else 0
            env_action[i * 2] = np.clip(phase, 0, self.num_phases - 1)
            
            # 连续时长 -> 时长索引
            param = continuous_params[i] if i < len(continuous_params) else 0.5
            duration_idx = int(param * (self.num_duration_options - 1) + 0.5)
            env_action[i * 2 + 1] = np.clip(duration_idx, 0, self.num_duration_options - 1)
        
        return env_action


class ConstraintAwareProjector:
    """
    约束感知动作投影器 - 用于 LIRL 算法
    
    确保输出的动作满足：
    1. 最小绿灯时间约束
    2. 目标时长约束
    3. 相位有效性约束
    """
    def __init__(self, num_intersections, num_phases, min_duration, max_duration, min_green=10):
        self.num_intersections = num_intersections
        self.num_phases = num_phases
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.min_green = min_green
        self.num_duration_options = max_duration - min_duration + 1
        self.duration_options = list(range(min_duration, max_duration + 1))
    
    def project(self, continuous_action, env=None):
        """
        将连续动作投影到满足约束的离散动作
        
        Args:
            continuous_action: numpy array, shape (num_intersections * 2,)
                              每个路口2个值 [phase_prob, duration_prob] 都在 [0, 1]
            env: CityFlowMultiIntersectionEnv 环境对象（用于获取当前状态）
        
        Returns:
            discrete_action: numpy array, shape (num_intersections * 2,)
        """
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
        
        # 如果没有环境对象，使用简单投影
        if env is None:
            return self._simple_project(a_)
        
        # 获取环境状态
        try:
            current_phases = env.current_phases.copy()
            phase_elapsed = env.phase_elapsed.copy()
            target_durations = env.target_durations.copy()
            valid_phases = env.valid_phases.copy()
            intersection_ids = env.intersection_ids.copy()
        except Exception:
            return self._simple_project(a_)
        
        # 对每个路口进行约束感知投影
        for i, inter_id in enumerate(intersection_ids):
            phase_prob = a_[i * 2]
            duration_prob = a_[i * 2 + 1]
            
            cur_phase = current_phases.get(inter_id, 0)
            elapsed = phase_elapsed.get(inter_id, 0.0)
            target_duration = target_durations.get(inter_id, self.min_duration)
            inter_valid_phases = valid_phases.get(inter_id, [True] * self.num_phases)
            
            # 确定网络期望的相位
            desired_phase = int(phase_prob * (self.num_phases - 1) + 0.5)
            desired_phase = np.clip(desired_phase, 0, self.num_phases - 1)
            
            # 约束检查：是否可以切换相位
            can_switch = (elapsed >= self.min_green) and (elapsed >= target_duration)
            
            if desired_phase != cur_phase:
                if can_switch and inter_valid_phases[desired_phase]:
                    # 满足约束，可以切换
                    selected_phase = desired_phase
                else:
                    # 不满足约束，保持当前相位
                    selected_phase = cur_phase
            else:
                # 保持当前相位
                selected_phase = cur_phase
            
            # 时长参数
            desired_duration_idx = int(duration_prob * (self.num_duration_options - 1) + 0.5)
            desired_duration_idx = np.clip(desired_duration_idx, 0, self.num_duration_options - 1)
            
            # 确保时长不小于最小绿灯时间
            min_duration_idx = max(0, self.min_green - self.min_duration)
            selected_duration_idx = max(desired_duration_idx, min_duration_idx)
            
            discrete_action[i * 2] = selected_phase
            discrete_action[i * 2 + 1] = selected_duration_idx
        
        return discrete_action
    
    def _simple_project(self, a_):
        """简单投影（无环境状态时使用）"""
        discrete_action = np.zeros(self.num_intersections * 2, dtype=np.int64)
        
        for i in range(self.num_intersections):
            phase_prob = a_[i * 2]
            duration_prob = a_[i * 2 + 1]
            
            phase_idx = int(phase_prob * (self.num_phases - 1) + 0.5)
            discrete_action[i * 2] = np.clip(phase_idx, 0, self.num_phases - 1)
            
            duration_idx = int(duration_prob * (self.num_duration_options - 1) + 0.5)
            discrete_action[i * 2 + 1] = np.clip(duration_idx, 0, self.num_duration_options - 1)
        
        return discrete_action


# =======================
# 算法实现
# =======================

class LIRLAgent:
    """LIRL (DDPG) Agent - 使用约束感知动作投影"""
    def __init__(self, state_size, num_intersections, num_phases, config):
        self.name = "LIRL"
        self.config = config
        self.num_intersections = num_intersections
        self.num_phases = num_phases
        
        action_size = num_intersections * 2
        
        self.actor = BaseActorNetwork(state_size, action_size, 
                                       config['hidden_dim1'], config['hidden_dim2']).to(DEVICE)
        self.actor_target = BaseActorNetwork(state_size, action_size,
                                              config['hidden_dim1'], config['hidden_dim2']).to(DEVICE)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = BaseCriticNetwork(state_size, action_size,
                                         config['hidden_dim1'], config['hidden_dim2']).to(DEVICE)
        self.critic_target = BaseCriticNetwork(state_size, action_size,
                                                config['hidden_dim1'], config['hidden_dim2']).to(DEVICE)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.0003)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.001)
        
        self.memory = ReplayBuffer()
        self.noise_scale = 1.0
        
        # 使用约束感知投影器
        self.projector = ConstraintAwareProjector(
            num_intersections, num_phases,
            config['min_duration'], config['max_duration'],
            config['min_green']
        )
        self.env = None  # 环境引用，用于约束感知投影
    
    def set_env(self, env):
        """设置环境引用，用于约束感知投影"""
        self.env = env
    
    def select_action(self, state, deterministic=False):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            action = self.actor(state_t).squeeze(0).cpu().numpy()
        
        if not deterministic:
            noise = np.random.normal(0, 0.1, action.shape) * self.noise_scale
            action = np.clip(action + noise, 0, 1)
        
        # 使用约束感知投影（如果有环境引用）
        env_action = self.projector.project(action, self.env)
        return env_action, action
    
    def store(self, state, action, reward, next_state, done):
        self.memory.put((state, action, reward, next_state, done))
    
    def train_step(self):
        if self.memory.size() < 500:
            return
        
        batch = self.memory.sample(self.config['batch_size'])
        states = torch.FloatTensor([t[0] for t in batch]).to(DEVICE)
        actions = torch.FloatTensor([t[1] for t in batch]).to(DEVICE)
        rewards = torch.FloatTensor([[t[2]] for t in batch]).to(DEVICE)
        next_states = torch.FloatTensor([t[3] for t in batch]).to(DEVICE)
        dones = torch.FloatTensor([[1.0 - t[4]] for t in batch]).to(DEVICE)
        
        # Critic update
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target = rewards + self.config['gamma'] * target_q * dones
        
        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Actor update
        actor_loss = -self.critic(states, self.actor(states)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Soft update
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(0.005 * param.data + 0.995 * target_param.data)
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(0.005 * param.data + 0.995 * target_param.data)
    
    def decay_noise(self, episode, total_episodes):
        self.noise_scale = max(0.1, 1.0 - episode / (total_episodes * 0.8))
    
    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path, map_location=DEVICE)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])


class PDQNAgent:
    """PDQN Agent"""
    def __init__(self, state_size, num_intersections, num_phases, config):
        self.name = "PDQN"
        self.config = config
        self.num_intersections = num_intersections
        self.num_phases = num_phases
        
        self.q_net = QNetwork(state_size, num_intersections, num_phases,
                               config['hidden_dim1'], config['hidden_dim2']).to(DEVICE)
        self.q_net_target = QNetwork(state_size, num_intersections, num_phases,
                                      config['hidden_dim1'], config['hidden_dim2']).to(DEVICE)
        self.q_net_target.load_state_dict(self.q_net.state_dict())
        
        self.param_net = ParameterNetwork(state_size, num_intersections,
                                           config['hidden_dim1'], config['hidden_dim2']).to(DEVICE)
        self.param_net_target = ParameterNetwork(state_size, num_intersections,
                                                  config['hidden_dim1'], config['hidden_dim2']).to(DEVICE)
        self.param_net_target.load_state_dict(self.param_net.state_dict())
        
        self.q_optimizer = optim.Adam(self.q_net.parameters(), lr=0.001)
        self.param_optimizer = optim.Adam(self.param_net.parameters(), lr=0.0003)
        
        self.memory = ReplayBuffer()
        self.epsilon = 1.0
        
        self.converter = ActionConverter(num_intersections, num_phases,
                                          config['min_duration'], config['max_duration'])
    
    def select_action(self, state, deterministic=False):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            params = self.param_net(state_t).squeeze(0)
            q_values = self.q_net(state_t, params.unsqueeze(0)).squeeze(0)
        
        if not deterministic and random.random() < self.epsilon:
            disc_actions = np.random.randint(0, self.num_phases, self.num_intersections)
        else:
            disc_actions = q_values.argmax(dim=-1).cpu().numpy()
        
        cont_params = params.cpu().numpy()
        if not deterministic:
            cont_params = np.clip(cont_params + np.random.normal(0, 0.1, cont_params.shape), 0, 1)
        
        env_action = self.converter.convert(disc_actions, cont_params)
        return env_action, (disc_actions, cont_params)
    
    def store(self, state, action, reward, next_state, done):
        disc_actions, cont_params = action
        self.memory.put((state, disc_actions, cont_params, reward, next_state, done))
    
    def train_step(self):
        if self.memory.size() < 500:
            return
        
        batch = self.memory.sample(self.config['batch_size'])
        states = torch.FloatTensor([t[0] for t in batch]).to(DEVICE)
        disc_actions = torch.LongTensor([t[1] for t in batch]).to(DEVICE)
        cont_params = torch.FloatTensor([t[2] for t in batch]).to(DEVICE)
        rewards = torch.FloatTensor([[t[3]] for t in batch]).to(DEVICE)
        next_states = torch.FloatTensor([t[4] for t in batch]).to(DEVICE)
        dones = torch.FloatTensor([[1.0 - t[5]] for t in batch]).to(DEVICE)
        
        # Q-network update
        with torch.no_grad():
            next_params = self.param_net_target(next_states)
            next_q_values = self.q_net_target(next_states, next_params)
            next_q_max = next_q_values.max(dim=-1)[0].sum(dim=-1, keepdim=True)
            target_q = rewards + self.config['gamma'] * next_q_max * dones
        
        current_q_values = self.q_net(states, cont_params)
        current_q = current_q_values.gather(2, disc_actions.unsqueeze(-1)).squeeze(-1).sum(dim=-1, keepdim=True)
        q_loss = F.mse_loss(current_q, target_q)
        
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        
        # Parameter network update
        params = self.param_net(states)
        q_values = self.q_net(states, params)
        param_loss = -q_values.max(dim=-1)[0].sum(dim=-1).mean()
        
        self.param_optimizer.zero_grad()
        param_loss.backward()
        self.param_optimizer.step()
        
        # Soft update
        for param, target_param in zip(self.q_net.parameters(), self.q_net_target.parameters()):
            target_param.data.copy_(0.005 * param.data + 0.995 * target_param.data)
        for param, target_param in zip(self.param_net.parameters(), self.param_net_target.parameters()):
            target_param.data.copy_(0.005 * param.data + 0.995 * target_param.data)
    
    def decay_epsilon(self, episode, total_episodes):
        self.epsilon = max(0.05, 1.0 - episode / (total_episodes * 0.8))
    
    def save(self, path):
        torch.save({
            'q_net': self.q_net.state_dict(),
            'param_net': self.param_net.state_dict(),
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path, map_location=DEVICE)
        self.q_net.load_state_dict(checkpoint['q_net'])
        self.param_net.load_state_dict(checkpoint['param_net'])


class HPPOAgent:
    """HPPO Agent"""
    def __init__(self, state_size, num_intersections, num_phases, config):
        self.name = "HPPO"
        self.config = config
        self.num_intersections = num_intersections
        self.num_phases = num_phases
        
        self.network = HybridActorCritic(state_size, num_intersections, num_phases,
                                          config['hidden_dim1'], config['hidden_dim2']).to(DEVICE)
        self.optimizer = optim.Adam(self.network.parameters(), lr=0.0003)
        
        self.buffer = RolloutBuffer()
        
        self.converter = ActionConverter(num_intersections, num_phases,
                                          config['min_duration'], config['max_duration'])
    
    def select_action(self, state, deterministic=False):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            discrete_logits, continuous_mean, value, cost_value = self.network(state_t)
            
            discrete_probs = F.softmax(discrete_logits, dim=-1).squeeze(0)
            
            if deterministic:
                disc_actions = discrete_probs.argmax(dim=-1).cpu().numpy()
                cont_actions = continuous_mean.squeeze(0).cpu().numpy()
            else:
                dist_d = Categorical(probs=discrete_probs)
                disc_actions = dist_d.sample().cpu().numpy()
                
                std = torch.exp(self.network.log_std.clamp(-2.0, 0.5))
                dist_c = Normal(continuous_mean.squeeze(0), std)
                cont_actions = torch.clamp(dist_c.sample(), 0, 1).cpu().numpy()
        
        env_action = self.converter.convert(disc_actions, cont_actions)
        
        # 计算 log prob
        if not deterministic:
            log_prob_d = dist_d.log_prob(torch.LongTensor(disc_actions).to(DEVICE)).sum().item()
            log_prob_c = dist_c.log_prob(torch.FloatTensor(cont_actions).to(DEVICE)).sum().item()
        else:
            log_prob_d = 0
            log_prob_c = 0
        
        return env_action, (disc_actions, cont_actions, value.item(), cost_value.item(), log_prob_d, log_prob_c)
    
    def store(self, state, action_info, reward, cost, done):
        disc_actions, cont_actions, value, cost_value, log_prob_d, log_prob_c = action_info
        self.buffer.store(state, disc_actions, cont_actions, reward, cost, done,
                          value, cost_value, log_prob_d, log_prob_c)
    
    def train_step(self):
        if len(self.buffer.states) < 10:
            return
        
        batch = self.buffer.get_batch(self.config['gamma'], 0.95, DEVICE)
        
        # Normalize advantages
        advantages = batch['advantages']
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for _ in range(10):  # PPO epochs
            discrete_logits, continuous_mean, values, cost_values = self.network(batch['states'])
            
            # Discrete action loss
            discrete_probs = F.softmax(discrete_logits, dim=-1)
            dist_d = Categorical(probs=discrete_probs)
            new_log_probs_d = dist_d.log_prob(batch['discrete_actions']).sum(dim=-1)
            
            # Continuous action loss
            std = torch.exp(self.network.log_std.clamp(-2.0, 0.5))
            dist_c = Normal(continuous_mean, std)
            new_log_probs_c = dist_c.log_prob(batch['continuous_actions']).sum(dim=-1)
            
            # PPO ratio and clipped objective
            ratio = torch.exp(new_log_probs_d + new_log_probs_c - 
                             batch['old_log_probs_d'] - batch['old_log_probs_c'])
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(values.squeeze(-1), batch['returns'])
            
            # Entropy bonus
            entropy = dist_d.entropy().mean() + dist_c.entropy().mean()
            
            loss = actor_loss + 0.5 * value_loss - 0.01 * entropy
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.optimizer.step()
        
        self.buffer.clear()
    
    def save(self, path):
        torch.save({'network': self.network.state_dict()}, path)
    
    def load(self, path):
        checkpoint = torch.load(path, map_location=DEVICE)
        self.network.load_state_dict(checkpoint['network'])


class LagrangianPPOAgent(HPPOAgent):
    """Lagrangian-PPO Agent"""
    def __init__(self, state_size, num_intersections, num_phases, config):
        super().__init__(state_size, num_intersections, num_phases, config)
        self.name = "Lagrangian-PPO"
        
        self.lagrangian_lambda = 0.1
        self.cost_limit = config.get('cost_limit', 100.0)
        self.lr_lambda = 0.01
    
    def train_step(self):
        if len(self.buffer.states) < 10:
            return
        
        batch = self.buffer.get_batch(self.config['gamma'], 0.95, DEVICE)
        
        # Normalize advantages
        advantages = batch['advantages']
        cost_advantages = batch['cost_advantages']
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        avg_cost = batch['cost_returns'].mean().item()
        
        for _ in range(10):
            discrete_logits, continuous_mean, values, cost_values = self.network(batch['states'])
            
            discrete_probs = F.softmax(discrete_logits, dim=-1)
            dist_d = Categorical(probs=discrete_probs)
            new_log_probs_d = dist_d.log_prob(batch['discrete_actions']).sum(dim=-1)
            
            std = torch.exp(self.network.log_std.clamp(-2.0, 0.5))
            dist_c = Normal(continuous_mean, std)
            new_log_probs_c = dist_c.log_prob(batch['continuous_actions']).sum(dim=-1)
            
            ratio = torch.exp(new_log_probs_d + new_log_probs_c - 
                             batch['old_log_probs_d'] - batch['old_log_probs_c'])
            
            # Lagrangian objective
            surr1 = ratio * (advantages - self.lagrangian_lambda * cost_advantages)
            surr2 = torch.clamp(ratio, 0.8, 1.2) * (advantages - self.lagrangian_lambda * cost_advantages)
            actor_loss = -torch.min(surr1, surr2).mean()
            
            value_loss = F.mse_loss(values.squeeze(-1), batch['returns'])
            cost_value_loss = F.mse_loss(cost_values.squeeze(-1), batch['cost_returns'])
            
            entropy = dist_d.entropy().mean() + dist_c.entropy().mean()
            
            loss = actor_loss + 0.5 * value_loss + 0.5 * cost_value_loss - 0.01 * entropy
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.optimizer.step()
        
        # Update Lagrangian multiplier
        self.lagrangian_lambda = max(0, self.lagrangian_lambda + self.lr_lambda * (avg_cost - self.cost_limit))
        self.lagrangian_lambda = min(self.lagrangian_lambda, 10.0)
        
        self.buffer.clear()


class CPOAgent(HPPOAgent):
    """CPO Agent (Simplified)"""
    def __init__(self, state_size, num_intersections, num_phases, config):
        super().__init__(state_size, num_intersections, num_phases, config)
        self.name = "CPO"
        self.cost_limit = config.get('cost_limit', 100.0)
    
    def train_step(self):
        if len(self.buffer.states) < 10:
            return
        
        batch = self.buffer.get_batch(self.config['gamma'], 0.95, DEVICE)
        
        advantages = batch['advantages']
        cost_advantages = batch['cost_advantages']
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        avg_cost = batch['cost_returns'].mean().item()
        
        # 动态调整成本权重
        cost_weight = max(0, (avg_cost - self.cost_limit) / max(self.cost_limit, 1))
        cost_weight = min(cost_weight, 1.0)
        
        for _ in range(10):
            discrete_logits, continuous_mean, values, cost_values = self.network(batch['states'])
            
            discrete_probs = F.softmax(discrete_logits, dim=-1)
            dist_d = Categorical(probs=discrete_probs)
            new_log_probs_d = dist_d.log_prob(batch['discrete_actions']).sum(dim=-1)
            
            std = torch.exp(self.network.log_std.clamp(-2.0, 0.5))
            dist_c = Normal(continuous_mean, std)
            new_log_probs_c = dist_c.log_prob(batch['continuous_actions']).sum(dim=-1)
            
            ratio = torch.exp(new_log_probs_d + new_log_probs_c - 
                             batch['old_log_probs_d'] - batch['old_log_probs_c'])
            
            # CPO-style objective with cost penalty
            adjusted_advantages = advantages - cost_weight * cost_advantages
            surr1 = ratio * adjusted_advantages
            surr2 = torch.clamp(ratio, 0.8, 1.2) * adjusted_advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            value_loss = F.mse_loss(values.squeeze(-1), batch['returns'])
            cost_value_loss = F.mse_loss(cost_values.squeeze(-1), batch['cost_returns'])
            
            entropy = dist_d.entropy().mean() + dist_c.entropy().mean()
            
            loss = actor_loss + 0.5 * value_loss + 0.5 * cost_value_loss - 0.01 * entropy
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.optimizer.step()
        
        self.buffer.clear()


# =======================
# 训练与评估函数
# =======================

def create_environment(config_path: str, config: Dict) -> CityFlowMultiIntersectionEnv:
    """创建 CityFlow 环境"""
    env_config = get_default_config(config_path)
    env_config.update({
        "episode_length": config['episode_length'],
        "ctrl_interval": config['ctrl_interval'],
        "min_green": config['min_green'],
        "min_duration": config['min_duration'],
        "max_duration": config['max_duration'],
        "verbose_violations": False,
        "log_violations": True,
    })
    return CityFlowMultiIntersectionEnv(env_config)


def train_agent(agent, env, num_episodes: int, print_interval: int = 50) -> Dict:
    """训练智能体"""
    print(f"\n{'='*60}")
    print(f"训练算法: {agent.name}")
    print(f"总回合数: {num_episodes}")
    print(f"{'='*60}")
    
    # 为 LIRL 设置环境引用（用于约束感知投影）
    if hasattr(agent, 'set_env'):
        agent.set_env(env)
    
    episode_rewards = []
    episode_travel_times = []
    episode_throughputs = []
    episode_violations = []
    
    for ep in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        episode_cost = 0
        done = False
        
        while not done:
            env_action, action_info = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(env_action)
            done = terminated or truncated
            
            # 获取约束违反作为 cost
            violations = info.get('total_violations', {})
            cost = sum(violations.values())
            episode_cost += cost
            
            # 存储经验
            if hasattr(agent, 'store'):
                if agent.name in ["HPPO", "Lagrangian-PPO", "CPO"]:
                    agent.store(state, action_info, reward, cost, done)
                else:
                    agent.store(state, action_info, reward, next_state, done)
            
            episode_reward += reward
            state = next_state
        
        # 训练
        if hasattr(agent, 'train_step'):
            for _ in range(10):
                agent.train_step()
        
        # 衰减探索参数
        if hasattr(agent, 'decay_noise'):
            agent.decay_noise(ep, num_episodes)
        if hasattr(agent, 'decay_epsilon'):
            agent.decay_epsilon(ep, num_episodes)
        
        # 记录统计
        episode_rewards.append(episode_reward)
        avg_tt = info.get('average_travel_time', 0)
        episode_travel_times.append(avg_tt)
        
        flow_stats = info.get('intersection_flow', {})
        throughput = sum(s.get('throughput', 0) for s in flow_stats.values())
        episode_throughputs.append(throughput)
        episode_violations.append(episode_cost)
        
        if (ep + 1) % print_interval == 0:
            avg_reward = np.mean(episode_rewards[-print_interval:])
            avg_tt_val = np.mean(episode_travel_times[-print_interval:])
            avg_tp = np.mean(episode_throughputs[-print_interval:])
            avg_viol = np.mean(episode_violations[-print_interval:])
            print(f"  Episode {ep+1}/{num_episodes}: R={avg_reward:.0f}, "
                  f"TT={avg_tt_val:.0f}s, TP={avg_tp:.0f}, Viol={avg_viol:.0f}")
    
    return {
        'episode_rewards': episode_rewards,
        'episode_travel_times': episode_travel_times,
        'episode_throughputs': episode_throughputs,
        'episode_violations': episode_violations,
    }


def evaluate_agent(agent, env, num_episodes: int = 10) -> Dict:
    """评估智能体"""
    print(f"\n  评估 {agent.name} ({num_episodes} 回合)...")
    
    # 为 LIRL 设置环境引用（用于约束感知投影）
    if hasattr(agent, 'set_env'):
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
            
            violations = info.get('total_violations', {})
            episode_violations += sum(violations.values())
            
            episode_reward += reward
            step_count += 1
        
        all_rewards.append(episode_reward)
        all_travel_times.append(info.get('average_travel_time', 0))
        
        flow_stats = info.get('intersection_flow', {})
        throughput = sum(s.get('throughput', 0) for s in flow_stats.values())
        all_throughputs.append(throughput)
        all_violations.append(episode_violations)
        
        # 违反率 = 违反次数 / 步数
        violation_rate = episode_violations / max(step_count, 1)
        all_violation_rates.append(violation_rate)
    
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
    }


# =======================
# 绘图函数
# =======================

def plot_learning_curves(results: Dict, output_dir: str):
    """绘制学习曲线"""
    if not HAS_PLOT:
        print("[INFO] 跳过学习曲线绘制")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(results)))
    
    metrics = [
        ('episode_rewards', 'Episode Reward', 'Cumulative Reward'),
        ('episode_travel_times', 'Average Travel Time (s)', 'Average Travel Time'),
        ('episode_throughputs', 'Throughput', 'Throughput'),
        ('episode_violations', 'Constraint Violations', 'Constraint Violations'),
    ]
    
    for ax, (metric_key, ylabel, title) in zip(axes.flatten(), metrics):
        for i, (alg_name, data) in enumerate(results.items()):
            values = data['training'][metric_key]
            # 平滑曲线
            window = min(20, len(values) // 10)
            if window > 1:
                smoothed = np.convolve(values, np.ones(window)/window, mode='valid')
                x = np.arange(window-1, len(values))
            else:
                smoothed = values
                x = np.arange(len(values))
            ax.plot(x, smoothed, label=alg_name, color=colors[i], linewidth=2)
        
        ax.set_xlabel('Episode')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'learning_curves.png'), dpi=150)
    plt.close()
    print(f"  学习曲线已保存: learning_curves.png")


def plot_evaluation_heatmaps(results: Dict, output_dir: str):
    """绘制评估结果热力图"""
    if not HAS_PLOT:
        print("[INFO] 跳过热力图绘制")
        return
    
    algorithms = list(results.keys())
    metrics = ['mean_throughput', 'mean_travel_time', 'mean_violation_rate']
    metric_names = ['Mean Throughput', 'Mean Travel Time (s)', 'Violation Rate']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for ax, metric, name in zip(axes, metrics, metric_names):
        values = [[results[alg]['evaluation'][metric]] for alg in algorithms]
        values = np.array(values)
        
        # 创建热力图数据
        sns.heatmap(values, annot=True, fmt='.2f', cmap='RdYlGn_r' if 'violation' in metric or 'time' in metric else 'RdYlGn',
                    xticklabels=[''], yticklabels=algorithms, ax=ax, cbar=True)
        ax.set_title(name)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'evaluation_heatmap.png'), dpi=150)
    plt.close()
    print(f"  评估热力图已保存: evaluation_heatmap.png")


def plot_comparison_bars(results: Dict, output_dir: str):
    """绘制对比柱状图"""
    if not HAS_PLOT:
        print("[INFO] 跳过柱状图绘制")
        return
    
    algorithms = list(results.keys())
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 吞吐量对比
    ax = axes[0]
    values = [results[alg]['evaluation']['mean_throughput'] for alg in algorithms]
    errors = [results[alg]['evaluation']['std_throughput'] for alg in algorithms]
    bars = ax.bar(algorithms, values, yerr=errors, capsize=5, color=plt.cm.Set2(np.linspace(0, 1, len(algorithms))))
    ax.set_ylabel('Throughput')
    ax.set_title('Throughput Comparison')
    ax.tick_params(axis='x', rotation=45)
    
    # Travel Time Comparison
    ax = axes[1]
    values = [results[alg]['evaluation']['mean_travel_time'] for alg in algorithms]
    errors = [results[alg]['evaluation']['std_travel_time'] for alg in algorithms]
    bars = ax.bar(algorithms, values, yerr=errors, capsize=5, color=plt.cm.Set2(np.linspace(0, 1, len(algorithms))))
    ax.set_ylabel('Travel Time (s)')
    ax.set_title('Travel Time Comparison')
    ax.tick_params(axis='x', rotation=45)
    
    # Violation Rate Comparison
    ax = axes[2]
    values = [results[alg]['evaluation']['mean_violation_rate'] for alg in algorithms]
    errors = [results[alg]['evaluation']['std_violation_rate'] for alg in algorithms]
    bars = ax.bar(algorithms, values, yerr=errors, capsize=5, color=plt.cm.Set2(np.linspace(0, 1, len(algorithms))))
    ax.set_ylabel('Violation Rate')
    ax.set_title('Violation Rate Comparison')
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_bars.png'), dpi=150)
    plt.close()
    print(f"  对比柱状图已保存: comparison_bars.png")


# =======================
# 主函数
# =======================

def main():
    parser = argparse.ArgumentParser(description="算法对比实验")
    parser.add_argument("--config", type=str, 
                       default=os.path.join(PROJECT_ROOT, "examples/City_3_5/config.json"),
                       help="CityFlow 配置文件路径")
    parser.add_argument("--train-episodes", type=int, default=300,
                       help="训练回合数")
    parser.add_argument("--eval-episodes", type=int, default=10,
                       help="评估回合数")
    parser.add_argument("--output-dir", type=str,
                       default=os.path.join(PROJECT_ROOT, "outputs/algorithm_compare"),
                       help="输出目录")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子")
    parser.add_argument("--algorithms", type=str, nargs="+",
                       default=["LIRL", "PDQN", "HPPO", "Lagrangian-PPO", "CPO"],
                       help="要对比的算法")
    
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
    print(f"# 算法对比实验")
    print(f"# 输出目录: {run_dir}")
    print(f"# 算法: {', '.join(args.algorithms)}")
    print(f"# 训练回合: {args.train_episodes}")
    print(f"# 评估回合: {args.eval_episodes}")
    print(f"{'#'*70}")
    
    # 创建环境获取参数
    env = create_environment(args.config, UNIFIED_CONFIG)
    state_size = env.observation_space.shape[0]
    num_intersections = env.num_intersections
    num_phases = env.num_phases
    
    print(f"\n环境信息:")
    print(f"  状态维度: {state_size}")
    print(f"  路口数量: {num_intersections}")
    print(f"  相位数量: {num_phases}")
    
    env.close()
    
    # 定义算法
    algorithm_classes = {
        "LIRL": LIRLAgent,
        "PDQN": PDQNAgent,
        "HPPO": HPPOAgent,
        "Lagrangian-PPO": LagrangianPPOAgent,
        "CPO": CPOAgent,
    }
    
    results = {}
    
    # 训练和评估每种算法
    for alg_name in args.algorithms:
        if alg_name not in algorithm_classes:
            print(f"[WARNING] 未知算法: {alg_name}, 跳过")
            continue
        
        # 创建新环境
        env = create_environment(args.config, UNIFIED_CONFIG)
        
        # 创建智能体
        agent = algorithm_classes[alg_name](
            state_size=state_size,
            num_intersections=num_intersections,
            num_phases=num_phases,
            config=UNIFIED_CONFIG
        )
        
        # 训练
        training_results = train_agent(
            agent=agent,
            env=env,
            num_episodes=args.train_episodes,
            print_interval=max(1, args.train_episodes // 10)
        )
        
        # 保存模型
        model_path = os.path.join(run_dir, f"{alg_name}_model.pt")
        agent.save(model_path)
        print(f"  模型已保存: {model_path}")
        
        # 评估
        eval_results = evaluate_agent(agent, env, args.eval_episodes)
        
        env.close()
        
        results[alg_name] = {
            'training': training_results,
            'evaluation': eval_results,
            'model_path': model_path,
        }
        
        print(f"\n  {alg_name} 评估结果:")
        print(f"    平均吞吐量: {eval_results['mean_throughput']:.1f} ± {eval_results['std_throughput']:.1f}")
        print(f"    平均行程时间: {eval_results['mean_travel_time']:.1f} ± {eval_results['std_travel_time']:.1f} s")
        print(f"    约束违反率: {eval_results['mean_violation_rate']:.4f} ± {eval_results['std_violation_rate']:.4f}")
    
    # 绘制图表
    print(f"\n{'='*60}")
    print("绘制对比图表...")
    print(f"{'='*60}")
    
    plot_learning_curves(results, run_dir)
    plot_evaluation_heatmaps(results, run_dir)
    plot_comparison_bars(results, run_dir)
    
    # 保存结果到 JSON
    # 转换 numpy 数组为列表
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
    
    # 保存完整结果
    results_path = os.path.join(run_dir, "results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results_serializable, f, indent=2, ensure_ascii=False)
    print(f"完整结果已保存: {results_path}")
    
    # 保存汇总结果
    summary = {
        'experiment_info': {
            'timestamp': timestamp,
            'algorithms': args.algorithms,
            'train_episodes': args.train_episodes,
            'eval_episodes': args.eval_episodes,
            'state_size': state_size,
            'num_intersections': num_intersections,
            'num_phases': num_phases,
        },
        'evaluation_summary': {}
    }
    
    for alg_name, data in results.items():
        eval_data = data['evaluation']
        summary['evaluation_summary'][alg_name] = {
            'mean_throughput': float(eval_data['mean_throughput']),
            'std_throughput': float(eval_data['std_throughput']),
            'mean_travel_time': float(eval_data['mean_travel_time']),
            'std_travel_time': float(eval_data['std_travel_time']),
            'mean_violation_rate': float(eval_data['mean_violation_rate']),
            'std_violation_rate': float(eval_data['std_violation_rate']),
        }
    
    summary_path = os.path.join(run_dir, "summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"汇总结果已保存: {summary_path}")
    
    # 打印最终对比表格
    print(f"\n{'='*70}")
    print("算法对比结果汇总")
    print(f"{'='*70}")
    print(f"{'算法':<20} {'平均吞吐量':<15} {'平均行程时间':<15} {'约束违反率':<15}")
    print(f"{'-'*70}")
    for alg_name in args.algorithms:
        if alg_name in results:
            eval_data = results[alg_name]['evaluation']
            print(f"{alg_name:<20} "
                  f"{eval_data['mean_throughput']:<15.1f} "
                  f"{eval_data['mean_travel_time']:<15.1f} "
                  f"{eval_data['mean_violation_rate']:<15.4f}")
    print(f"{'='*70}")
    
    print(f"\n实验完成!")
    print(f"输出目录: {run_dir}")


if __name__ == "__main__":
    main()

