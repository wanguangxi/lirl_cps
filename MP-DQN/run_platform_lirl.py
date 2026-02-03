#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LIRL Agent for Platform Environment
====================================
基于LIRL算法在Platform环境上训练

LIRL (Learning with Integer and Real-valued Actions via Lagrangian relaxation)
使用动作投影函数将连续网络输出映射到混合离散-连续动作空间
"""

import os
import click
import time
import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
import gym_platform
from gym.wrappers import Monitor
from common import ClickPythonLiteralOption
from common.wrappers import ScaledParameterisedActionWrapper
from common.platform_domain import PlatformFlattenedActionWrapper
from common.wrappers import ScaledStateWrapper

# 尝试导入scipy，如果不可用则使用numpy实现
try:
    from scipy.optimize import minimize, linear_sum_assignment
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available, using numpy fallback implementations")


# =======================
# 设备配置
# =======================
device = torch.device("cpu")


# =======================
# 动作投影函数
# =======================
class ActionProjection:
    """
    LIRL动作投影类
    将网络输出的连续动作投影到有效的离散-连续混合动作空间
    
    Platform环境动作空间:
    - 3个离散动作: run(0), hop(1), leap(2)
    - 连续参数: 每个动作各需要1个参数
    """
    
    def __init__(self, action_space, use_qp=True):
        """
        Args:
            action_space: 环境的动作空间
            use_qp: 是否使用QP求解连续参数投影
        """
        self.use_qp = use_qp
        self.num_actions = 3  # run, hop, leap
        
        # 动作参数的维度 - Platform环境中每个动作都有1个参数
        self.action_param_sizes = [1, 1, 1]  # run:1, hop:1, leap:1
        self.action_param_offsets = [0, 1, 2, 3]  # 累计偏移量
        
        # 参数范围 (假设已经被ScaledParameterisedActionWrapper缩放到[-1,1])
        self.param_min = -1.0
        self.param_max = 1.0
        
        # 计时统计
        self.reset_timings()
    
    def reset_timings(self):
        """重置计时统计"""
        self.discrete_selection_times = []
        self.qp_times = []
        self.total_projection_times = []
    
    def project(self, action_probs, action_params, record_timing=True):
        """
        将网络输出投影到有效动作空间
        
        Args:
            action_probs: 离散动作的概率分布 [3]
            action_params: 连续动作参数 [3]
            record_timing: 是否记录时间
        
        Returns:
            discrete_action: 选择的离散动作索引
            continuous_params: 投影后的连续参数
            timing_info: 计时信息字典
        """
        total_start = time.perf_counter()
        
        # 转换为numpy
        if isinstance(action_probs, torch.Tensor):
            action_probs = action_probs.detach().cpu().numpy()
        if isinstance(action_params, torch.Tensor):
            action_params = action_params.detach().cpu().numpy()
        
        action_probs = np.asarray(action_probs).flatten()
        action_params = np.asarray(action_params).flatten()
        
        # 1. 离散动作选择 (使用匈牙利算法的思想)
        discrete_start = time.perf_counter()
        discrete_action = self._select_discrete_action(action_probs)
        discrete_time = time.perf_counter() - discrete_start
        
        # 2. 连续参数投影 (使用QP)
        qp_start = time.perf_counter()
        projected_params = self._project_continuous_params(action_params, discrete_action)
        qp_time = time.perf_counter() - qp_start
        
        total_time = time.perf_counter() - total_start
        
        # 记录时间
        if record_timing:
            self.discrete_selection_times.append(discrete_time)
            self.qp_times.append(qp_time)
            self.total_projection_times.append(total_time)
        
        timing_info = {
            'discrete_selection_time': discrete_time,
            'qp_time': qp_time,
            'total_time': total_time
        }
        
        return discrete_action, projected_params, timing_info
    
    def _select_discrete_action(self, action_probs):
        """
        选择离散动作
        使用基于代价矩阵的方法（简化的匈牙利算法思想）
        
        对于Platform环境，我们构建一个代价矩阵来选择最优动作
        """
        # 构建代价矩阵: 代价 = 1 - 概率 (最大化概率 = 最小化代价)
        # 这里简化为直接选择概率最高的动作
        # 在更复杂的场景中，可以考虑动作之间的约束
        
        cost_matrix = np.zeros((1, self.num_actions))
        for i in range(self.num_actions):
            cost_matrix[0, i] = 1.0 - action_probs[i]
        
        if SCIPY_AVAILABLE:
            # 使用匈牙利算法求解 (对于单行矩阵，等价于argmin)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            return col_ind[0]
        else:
            # Numpy fallback: 对于单行代价矩阵，argmin等价于匈牙利算法
            return np.argmin(cost_matrix[0])
    
    def _project_continuous_params(self, action_params, discrete_action):
        """
        将连续参数投影到有效范围
        使用QP (Quadratic Programming) 求解最近的可行解
        
        目标: min ||x - v||^2
        约束: param_min <= x <= param_max
        
        Args:
            action_params: 原始连续参数 [3]
            discrete_action: 选择的离散动作
        
        Returns:
            投影后的参数（仅返回对应动作所需的参数）
        """
        # 获取当前动作对应的参数
        start_idx = self.action_param_offsets[discrete_action]
        end_idx = self.action_param_offsets[discrete_action + 1]
        params_for_action = action_params[start_idx:end_idx]
        
        if self.use_qp:
            projected = self._solve_qp(params_for_action)
        else:
            # 简单clip
            projected = np.clip(params_for_action, self.param_min, self.param_max)
        
        return projected
    
    def _solve_qp(self, v):
        """
        使用QP求解参数投影
        
        目标函数: min 0.5 * ||x - v||^2
        约束: param_min <= x <= param_max
        
        对于简单的盒式约束(box constraints)，最优解就是将v投影到[min,max]范围内
        即 x* = clip(v, min, max)
        
        Args:
            v: 原始参数向量
        
        Returns:
            投影后的参数向量
        """
        v = np.asarray(v, dtype=np.float64)
        n = len(v)
        
        if n == 0:
            return v
        
        if SCIPY_AVAILABLE and self.use_qp:
            # 使用scipy优化器（支持更复杂的约束）
            # 目标函数
            def objective(x):
                return 0.5 * np.sum((x - v) ** 2)
            
            # 梯度
            def gradient(x):
                return x - v
            
            # 边界约束
            bounds = [(self.param_min, self.param_max) for _ in range(n)]
            
            # 初始点 (clip到可行域内)
            x0 = np.clip(v, self.param_min, self.param_max)
            
            try:
                result = minimize(
                    objective,
                    x0,
                    method='L-BFGS-B',
                    jac=gradient,
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-8}
                )
                
                if result.success:
                    return result.x
                else:
                    return np.clip(v, self.param_min, self.param_max)
            except Exception:
                return np.clip(v, self.param_min, self.param_max)
        else:
            # Numpy fallback: 对于盒式约束，clip就是最优解
            # 数学证明: min ||x-v||^2 s.t. a<=x<=b 的解是 x*=clip(v,a,b)
            return np.clip(v, self.param_min, self.param_max)
    
    def project_with_constraints(self, action_probs, action_params, 
                                  action_mask=None, param_constraints=None):
        """
        带额外约束的动作投影
        
        Args:
            action_probs: 离散动作概率
            action_params: 连续参数
            action_mask: 可选的动作掩码 [3], True表示该动作可用
            param_constraints: 可选的参数约束字典
        
        Returns:
            discrete_action, continuous_params, timing_info
        """
        # 应用动作掩码
        if action_mask is not None:
            masked_probs = action_probs.copy()
            masked_probs[~action_mask] = -np.inf
            action_probs = masked_probs
        
        return self.project(action_probs, action_params)
    
    def get_timing_statistics(self):
        """获取计时统计信息"""
        stats = {}
        
        if self.discrete_selection_times:
            stats['discrete_selection'] = {
                'mean': np.mean(self.discrete_selection_times),
                'std': np.std(self.discrete_selection_times),
                'min': np.min(self.discrete_selection_times),
                'max': np.max(self.discrete_selection_times),
                'count': len(self.discrete_selection_times)
            }
        
        if self.qp_times:
            stats['qp'] = {
                'mean': np.mean(self.qp_times),
                'std': np.std(self.qp_times),
                'min': np.min(self.qp_times),
                'max': np.max(self.qp_times),
                'count': len(self.qp_times)
            }
        
        if self.total_projection_times:
            stats['total_projection'] = {
                'mean': np.mean(self.total_projection_times),
                'std': np.std(self.total_projection_times),
                'min': np.min(self.total_projection_times),
                'max': np.max(self.total_projection_times),
                'count': len(self.total_projection_times)
            }
        
        return stats
    
    def print_timing_summary(self):
        """打印计时摘要"""
        stats = self.get_timing_statistics()
        
        print("\n" + "="*60)
        print("Action Projection Timing Summary")
        print("="*60)
        
        if 'discrete_selection' in stats:
            ds = stats['discrete_selection']
            print(f"\n离散动作选择 (Hungarian):")
            print(f"  平均: {ds['mean']*1000:.4f} ms")
            print(f"  标准差: {ds['std']*1000:.4f} ms")
        
        if 'qp' in stats:
            qp = stats['qp']
            print(f"\nQP求解 (连续参数投影):")
            print(f"  平均: {qp['mean']*1000:.4f} ms")
            print(f"  标准差: {qp['std']*1000:.4f} ms")
        
        if 'total_projection' in stats:
            tp = stats['total_projection']
            print(f"\n总投影时间:")
            print(f"  平均: {tp['mean']*1000:.4f} ms")
            print(f"  标准差: {tp['std']*1000:.4f} ms")
            print(f"  调用次数: {tp['count']}")
        
        print("="*60)


# =======================
# 神经网络定义
# =======================
class ReplayBuffer:
    """经验回放缓冲区"""
    def __init__(self, buffer_limit):
        self.buffer = collections.deque(maxlen=buffer_limit)
    
    def put(self, transition):
        self.buffer.append(transition)
        
    def sample(self, n):
        actual_n = min(n, len(self.buffer))
        if actual_n <= 0:
            raise ValueError(f"Buffer is empty or n={n} is invalid")
        
        mini_batch = random.sample(self.buffer, actual_n)
        
        s_arr = np.array([t[0] for t in mini_batch], dtype=np.float32)
        a_arr = np.array([t[1] for t in mini_batch], dtype=np.float32)
        r_arr = np.array([t[2] for t in mini_batch], dtype=np.float32)
        s_prime_arr = np.array([t[3] for t in mini_batch], dtype=np.float32)
        done_arr = np.array([[0.0 if t[4] else 1.0] for t in mini_batch], dtype=np.float32)
        
        s_tensor = torch.from_numpy(s_arr).to(device)
        a_tensor = torch.from_numpy(a_arr).to(device)
        r_tensor = torch.from_numpy(r_arr).to(device)
        s_prime_tensor = torch.from_numpy(s_prime_arr).to(device)
        done_tensor = torch.from_numpy(done_arr).to(device)
        
        return s_tensor, a_tensor, r_tensor, s_prime_tensor, done_tensor
    
    def size(self):
        return len(self.buffer)


class MuNet(nn.Module):
    """Actor网络 - 输出连续动作"""
    def __init__(self, state_size, action_size, hidden_layers=(128,)):
        super(MuNet, self).__init__()
        self.layers = nn.ModuleList()
        
        # 构建隐藏层
        last_size = state_size
        for hidden_size in hidden_layers:
            self.layers.append(nn.Linear(last_size, hidden_size))
            last_size = hidden_size
        
        # 输出层：3个动作的选择概率 + 3个连续参数
        self.action_output = nn.Linear(last_size, 3)  # 3个离散动作
        self.param_output = nn.Linear(last_size, action_size)  # 连续参数

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        
        # 动作概率（用于选择离散动作）
        action_probs = F.softmax(self.action_output(x), dim=-1)
        # 连续参数（用tanh映射到[-1,1]，后续会缩放到实际范围）
        action_params = torch.tanh(self.param_output(x))
        
        return action_probs, action_params


class QNet(nn.Module):
    """Critic网络 - 评估Q值"""
    def __init__(self, state_size, action_size, hidden_layers=(128,)):
        super(QNet, self).__init__()
        # 输入：状态 + 3个动作概率 + 连续参数
        input_size = state_size + 3 + action_size
        
        self.layers = nn.ModuleList()
        last_size = input_size
        for hidden_size in hidden_layers:
            self.layers.append(nn.Linear(last_size, hidden_size))
            last_size = hidden_size
        
        self.output_layer = nn.Linear(last_size, 1)

    def forward(self, state, action_probs, action_params):
        x = torch.cat([state, action_probs, action_params], dim=-1)
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.output_layer(x)


class OrnsteinUhlenbeckNoise:
    """OU噪声用于探索"""
    def __init__(self, size, mu=0., theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.size = size
        self.reset()

    def reset(self):
        self.x_prev = np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) + \
            self.sigma * np.random.randn(self.size)
        self.x_prev = x
        return x


# =======================
# LIRL Agent
# =======================
class LIRLAgent:
    """LIRL Agent for Platform Environment"""
    
    def __init__(self, state_size, action_param_size, action_space=None, 
                 hidden_layers=(128,), lr_actor=1e-3, lr_critic=1e-3, 
                 gamma=0.9, tau=0.005, buffer_size=10000, batch_size=128, 
                 use_action_projection=True, seed=None):
        
        self.state_size = state_size
        self.action_param_size = action_param_size  # 3: run(1) + hop(1) + leap(1)
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.use_action_projection = use_action_projection
        
        # 设置随机种子
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        # 初始化网络
        self.mu = MuNet(state_size, action_param_size, hidden_layers).to(device)
        self.mu_target = MuNet(state_size, action_param_size, hidden_layers).to(device)
        self.mu_target.load_state_dict(self.mu.state_dict())
        
        self.q = QNet(state_size, action_param_size, hidden_layers).to(device)
        self.q_target = QNet(state_size, action_param_size, hidden_layers).to(device)
        self.q_target.load_state_dict(self.q.state_dict())
        
        # 优化器
        self.mu_optimizer = optim.Adam(self.mu.parameters(), lr=lr_actor)
        self.q_optimizer = optim.Adam(self.q.parameters(), lr=lr_critic)
        
        # 经验回放
        self.memory = ReplayBuffer(buffer_size)
        
        # 探索噪声
        self.noise = OrnsteinUhlenbeckNoise(action_param_size)
        
        # 动作投影器
        self.action_projector = ActionProjection(action_space, use_qp=True)
        
        # 动作参数范围（Platform环境）
        # run: param in [-1,1]
        # hop: param in [-1,1]
        # leap: param in [-1,1]
        self.action_param_min = np.array([-1., -1., -1.])
        self.action_param_max = np.array([1., 1., 1.])
        
        # Epsilon for exploration
        self.epsilon = 1.0
        self.epsilon_final = 0.01
        self.epsilon_decay = 0.9995
        
    def act(self, state, add_noise=True):
        """
        选择动作
        
        使用LIRL的动作投影函数将网络输出映射到有效动作空间
        """
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).float().to(device)
            if len(state_tensor.shape) == 1:
                state_tensor = state_tensor.unsqueeze(0)
            
            action_probs, action_params = self.mu(state_tensor)
            action_probs = action_probs.cpu().numpy().squeeze()
            action_params = action_params.cpu().numpy().squeeze()
        
        # 添加探索噪声
        if add_noise:
            if np.random.random() < self.epsilon:
                # Epsilon-greedy: 随机选择动作和参数
                action_probs = np.random.dirichlet(np.ones(3))  # 随机概率分布
                action_params = np.random.uniform(-1, 1, self.action_param_size)
            else:
                # 添加OU噪声到参数
                action_params = action_params + self.noise() * 0.1
                action_params = np.clip(action_params, -1, 1)
        
        # 使用动作投影函数
        if self.use_action_projection:
            action, act_param, timing_info = self.action_projector.project(
                action_probs, action_params, record_timing=True
            )
        else:
            # 不使用投影，直接选择
            action = np.argmax(action_probs)
            # 根据选择的动作提取对应的参数
            act_param = np.clip(action_params[action:action+1], -1, 1)
        
        return action, act_param, action_probs, action_params
    
    def step(self, state, action_probs, action_params, reward, next_state, done):
        """存储经验并学习"""
        # 合并动作表示
        combined_action = np.concatenate([action_probs, action_params])
        self.memory.put((state, combined_action, reward, next_state, done))
    
    def learn(self):
        """从经验中学习"""
        if self.memory.size() < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # 分离动作概率和参数
        action_probs = actions[:, :3]
        action_params = actions[:, 3:]
        
        # 更新Critic
        with torch.no_grad():
            next_action_probs, next_action_params = self.mu_target(next_states)
            target_q = self.q_target(next_states, next_action_probs, next_action_params)
            target = rewards.unsqueeze(1) + self.gamma * target_q * dones
        
        current_q = self.q(states, action_probs, action_params)
        q_loss = F.mse_loss(current_q, target)
        
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        
        # 更新Actor
        pred_action_probs, pred_action_params = self.mu(states)
        actor_loss = -self.q(states, pred_action_probs, pred_action_params).mean()
        
        self.mu_optimizer.zero_grad()
        actor_loss.backward()
        self.mu_optimizer.step()
        
        # 软更新目标网络
        self._soft_update(self.mu, self.mu_target)
        self._soft_update(self.q, self.q_target)
    
    def _soft_update(self, source, target):
        """软更新目标网络"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
    
    def decay_epsilon(self):
        """衰减epsilon"""
        self.epsilon = max(self.epsilon_final, self.epsilon * self.epsilon_decay)
    
    def reset_noise(self):
        """重置噪声"""
        self.noise.reset()
    
    def start_episode(self):
        """开始新的episode"""
        self.reset_noise()
    
    def end_episode(self):
        """结束episode"""
        self.decay_epsilon()
    
    def get_projection_stats(self):
        """获取动作投影的统计信息"""
        return self.action_projector.get_timing_statistics()
    
    def print_projection_summary(self):
        """打印动作投影的统计摘要"""
        self.action_projector.print_timing_summary()
    
    def save_models(self, prefix):
        """保存模型"""
        torch.save(self.mu.state_dict(), '{}_mu.pth'.format(prefix))
        torch.save(self.q.state_dict(), '{}_q.pth'.format(prefix))
    
    def __str__(self):
        return (f"LIRL Agent (Platform)\n"
                f"State size: {self.state_size}\n"
                f"Action param size: {self.action_param_size}\n"
                f"Gamma: {self.gamma}\n"
                f"Tau: {self.tau}\n"
                f"Batch size: {self.batch_size}\n"
                f"Use action projection: {self.use_action_projection}\n"
                f"Epsilon: {self.epsilon:.4f}\n"
                f"Memory size: {self.memory.size()}")


def pad_action(act, act_param):
    """将动作转换为环境需要的格式"""
    params = [np.zeros((1,), dtype=np.float32), np.zeros((1,), dtype=np.float32), np.zeros((1,), dtype=np.float32)]
    params[act][:] = act_param
    return (act, params)


def evaluate(env, agent, episodes=1000):
    """评估agent"""
    returns = []
    timesteps = []
    for _ in range(episodes):
        state, _ = env.reset()
        terminal = False
        t = 0
        total_reward = 0.
        while not terminal:
            t += 1
            state = np.array(state, dtype=np.float32, copy=False)
            act, act_param, _, _ = agent.act(state, add_noise=False)
            action = pad_action(act, act_param)
            (state, _), reward, terminal, _ = env.step(action)
            total_reward += reward
        timesteps.append(t)
        returns.append(total_reward)
    return np.array(returns)


@click.command()
@click.option('--seed', default=1, help='Random seed.', type=int)
@click.option('--episodes', default=20000, help='Number of episodes.', type=int)
@click.option('--evaluation-episodes', default=1000, help='Episodes over which to evaluate after training.', type=int)
@click.option('--batch-size', default=128, help='Minibatch size.', type=int)
@click.option('--gamma', default=0.9, help='Discount factor.', type=float)
@click.option('--initial-memory-threshold', default=500, help='Number of transitions required to start learning.', type=int)
@click.option('--replay-memory-size', default=10000, help='Replay memory size in transitions.', type=int)
@click.option('--tau', default=0.005, help='Soft target network update averaging factor.', type=float)
@click.option('--learning-rate-actor', default=1e-3, help="Actor network learning rate.", type=float)
@click.option('--learning-rate-critic', default=1e-3, help="Critic network learning rate.", type=float)
@click.option('--scale-actions', default=True, help="Scale actions.", type=bool)
@click.option('--use-action-projection', default=True, help="Use LIRL action projection (Hungarian + QP).", type=bool)
@click.option('--save-dir', default="results/platform", help='Model save directory.', type=str)
@click.option('--save-freq', default=0, help='How often to save models (0 = never).', type=int)
@click.option('--layers', default="[128,]", help='Hidden layer sizes.', cls=ClickPythonLiteralOption)
@click.option('--render-freq', default=100, help='How often to render / save frames of an episode.', type=int)
@click.option('--visualise', default=True, help="Render game states.", type=bool)
@click.option('--title', default="LIRL", help="Prefix of output files", type=str)
def run(seed, episodes, evaluation_episodes, batch_size, gamma, initial_memory_threshold,
        replay_memory_size, tau, learning_rate_actor, learning_rate_critic,
        scale_actions, use_action_projection, layers, save_dir, save_freq, 
        render_freq, visualise, title):
    
    if save_freq > 0 and save_dir:
        save_dir_full = os.path.join(save_dir, title + "{}".format(str(seed)))
        os.makedirs(save_dir_full, exist_ok=True)
    
    if visualise:
        assert render_freq > 0
    
    # 创建环境
    env = gym.make('Platform-v0')
    
    # 记录初始参数用于初始化
    initial_params_ = [3., 10., 400.]
    if scale_actions:
        for a in range(env.action_space.spaces[0].n):
            initial_params_[a] = 2. * (initial_params_[a] - env.action_space.spaces[1].spaces[a].low) / (
                        env.action_space.spaces[1].spaces[a].high - env.action_space.spaces[1].spaces[a].low) - 1.
    
    env = ScaledStateWrapper(env)
    env = PlatformFlattenedActionWrapper(env)
    if scale_actions:
        env = ScaledParameterisedActionWrapper(env)
    
    # 创建保存目录
    dir = os.path.join(save_dir, title)
    env = Monitor(env, directory=os.path.join(dir, str(seed)), video_callable=False, write_upon_reset=False, force=True)
    
    print(env.action_space)
    print(env.observation_space)
    
    # 设置随机种子
    env.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    
    # 获取状态和动作空间大小
    state_size = env.observation_space.spaces[0].shape[0]
    action_param_size = 3  # run(1) + hop(1) + leap(1)
    
    print(f"State size: {state_size}")
    print(f"Action param size: {action_param_size}")
    
    # 创建agent
    agent = LIRLAgent(
        state_size=state_size,
        action_param_size=action_param_size,
        action_space=env.action_space,
        hidden_layers=layers,
        lr_actor=learning_rate_actor,
        lr_critic=learning_rate_critic,
        gamma=gamma,
        tau=tau,
        buffer_size=replay_memory_size,
        batch_size=batch_size,
        use_action_projection=use_action_projection,
        seed=seed
    )
    
    print(agent)
    
    max_steps = 250
    total_reward = 0.
    returns = []
    start_time = time.time()
    
    for i in range(episodes):
        if save_freq > 0 and save_dir and i % save_freq == 0:
            agent.save_models(os.path.join(save_dir, title + str(seed), str(i)))
        
        state, _ = env.reset()
        state = np.array(state, dtype=np.float32, copy=False)
        
        if visualise and i % render_freq == 0:
            env.render()
        
        agent.start_episode()
        episode_reward = 0.
        
        for j in range(max_steps):
            # 选择动作
            act, act_param, action_probs, action_params = agent.act(state)
            action = pad_action(act, act_param)
            
            # 执行动作
            ret = env.step(action)
            (next_state, steps), reward, terminal, _ = ret
            next_state = np.array(next_state, dtype=np.float32, copy=False)
            
            # 存储经验
            agent.step(state, action_probs, action_params, reward, next_state, terminal)
            
            # 学习
            if agent.memory.size() >= initial_memory_threshold:
                agent.learn()
            
            state = next_state
            episode_reward += reward
            
            if visualise and i % render_freq == 0:
                env.render()
            
            if terminal:
                break
        
        agent.end_episode()
        
        returns.append(episode_reward)
        total_reward += episode_reward
        
        if i % 100 == 0:
            print('{0:5s} R:{1:.4f} r100:{2:.4f}'.format(
                str(i), total_reward / (i + 1), np.array(returns[-100:]).mean()))
    
    end_time = time.time()
    print("Took %.2f seconds" % (end_time - start_time))
    env.close()
    
    if save_freq > 0 and save_dir:
        agent.save_models(os.path.join(save_dir, title + str(seed), str(i)))
    
    print(agent)
    
    # 打印动作投影统计信息
    if use_action_projection:
        agent.print_projection_summary()
    
    # 保存结果
    returns = env.get_episode_rewards()
    print("Ave. return =", sum(returns) / len(returns))
    print("Ave. last 100 episode return =", sum(returns[-100:]) / 100.)
    
    np.save(os.path.join(dir, title + "{}".format(str(seed))), returns)
    
    # 保存模型
    torch.save(agent.mu.state_dict(), os.path.join(dir, 'mu_{}.pth'.format(seed)))
    torch.save(agent.q.state_dict(), os.path.join(dir, 'q_{}.pth'.format(seed)))
    
    # 评估
    if evaluation_episodes > 0:
        print("Evaluating agent over {} episodes".format(evaluation_episodes))
        agent.epsilon = 0.  # 关闭探索
        evaluation_returns = evaluate(env, agent, evaluation_episodes)
        print("Ave. evaluation return =", sum(evaluation_returns) / len(evaluation_returns))
        np.save(os.path.join(dir, title + "{}e".format(str(seed))), evaluation_returns)


if __name__ == '__main__':
    run()

