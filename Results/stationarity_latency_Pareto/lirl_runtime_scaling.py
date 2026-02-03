#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LIRL Runtime Scaling Experiment
================================
实验目的：测试LIRL算法在不同规模下的运行时间性能

实验设置：
1) num_of_jobs=100, num_of_robots=5    (小规模)
2) num_of_jobs=100, num_of_robots=50   (中规模)
3) num_of_jobs=1000, num_of_robots=500 (大规模)

测试指标：
- 策略网络前向推理时间
- 匈牙利算法执行时间
- QP求解时间
- 总决策时间
"""

import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import time
import os
import sys
import datetime
import json
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment, minimize

# 添加环境路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../env'))
try:
    import importlib
    ENV = importlib.import_module('env')
    if not hasattr(ENV, 'Env'):
        ENV = importlib.import_module('env.env')
except Exception:
    from env import env as ENV


# =======================
# 实验配置
# =======================
EXPERIMENT_CONFIGS = {
    'small': {
        'name': 'Small Scale (100 jobs, 5 robots)',
        'num_of_jobs': 100,
        'num_of_robots': 5,
    },
    'medium': {
        'name': 'Medium Scale (100 jobs, 50 robots)',
        'num_of_jobs': 100,
        'num_of_robots': 50,
    },
    'large': {
        'name': 'Large Scale (1000 jobs, 100 robots)',
        'num_of_jobs': 1000,
        'num_of_robots': 100,
    }
}

# 基础训练配置
BASE_CONFIG = {
    # Learning parameters
    'lr_mu': 0.0005,
    'lr_q': 0.001,
    'gamma': 0.98,
    'batch_size': 128,
    'buffer_limit': 1000000,
    'tau': 0.005,
    
    # Environment parameters
    'alpha': 0.5,
    'beta': 0.5,
    'num_of_episodes': 500,
    
    # Network architecture
    'hidden_dim1': 128,
    'hidden_dim2': 64,
    'critic_hidden': 32,
    
    # Training parameters
    'memory_threshold': 500,
    'training_iterations': 20,
    'noise_params': {'theta': 0.1, 'dt': 0.05, 'sigma': 0.1},
    
    # Seeds
    'seed': 42,
    
    # Output
    'print_interval': 20,
    'save_models': True,
    
    # 大规模优化参数
    'use_fast_action_selection': True,  # 大规模时使用快速动作选择
    'max_steps_per_episode': None,      # 限制每episode最大步数（None表示不限制）
}

# 针对不同规模的优化配置
SCALE_SPECIFIC_CONFIG = {
    'small': {
        'training_iterations': 20,
        'max_steps_per_episode': None,
        'use_fast_action_selection': False,
    },
    'medium': {
        'training_iterations': 20,
        'max_steps_per_episode': None,
        'use_fast_action_selection': False,
    },
    'large': {
        'training_iterations': 20,           # 减少训练迭代
        'max_steps_per_episode': None,       # 限制每episode步数
        'use_fast_action_selection': False,  # 使用top-k匈牙利算法
        'max_hungarian_size': 1000,            # 限制匈牙利算法规模为top-50
    }
}

# 运行时间测试配置
TIMING_CONFIG = {
    'num_timing_episodes': 10,      # 测试的episode数（增加以获得更稳定结果）
    'warmup_steps': 500,            # 预热步数（增加以充分预热CUDA核函数）
    'network_warmup_iterations': 1000,  # 网络专属预热迭代次数
}


# =======================
# GPU设备配置
# =======================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_device():
    """获取当前设备"""
    return DEVICE

def warmup_gpu():
    """
    全局GPU预热函数
    在运行任何实验之前调用，确保：
    1. CUDA核函数已编译（JIT）
    2. cuDNN算法已自动调优
    3. GPU内存池已初始化
    4. GPU频率已提升到最高
    """
    if not torch.cuda.is_available():
        print("GPU不可用，跳过预热")
        return
    
    print("正在预热GPU...")
    device = DEVICE
    
    # 1. 基础矩阵运算预热
    dummy_sizes = [128, 256, 512, 1024, 2048]
    for size in dummy_sizes:
        dummy = torch.randn(size, size, device=device)
        for _ in range(10):
            _ = torch.mm(dummy, dummy)
    
    # 2. 神经网络层预热（模拟不同输入大小）
    for in_features in [1000, 2000, 5000, 10000]:
        layer = torch.nn.Linear(in_features, 128).to(device)
        dummy_input = torch.randn(1, in_features, device=device)
        for _ in range(20):
            _ = layer(dummy_input)
    
    # 3. 同步确保所有操作完成
    torch.cuda.synchronize()
    
    # 4. 清理缓存
    torch.cuda.empty_cache()
    
    print(f"GPU预热完成 (Device: {torch.cuda.get_device_name(0)})")

# =======================
# 神经网络定义
# =======================
class ReplayBuffer():
    """经验回放缓冲区 - 优化GPU传输"""
    def __init__(self, buffer_limit, device=None):
        self.buffer = collections.deque(maxlen=buffer_limit)
        self.device = device if device else DEVICE
    
    def put(self, transition):
        self.buffer.append(transition)
        
    def sample(self, n):
        # 安全检查：确保不会采样超过buffer大小
        actual_n = min(n, len(self.buffer))
        if actual_n <= 0:
            raise ValueError(f"Buffer is empty or n={n} is invalid")
        
        mini_batch = random.sample(self.buffer, actual_n)
        
        # 使用numpy预分配数组，避免多次append
        s_arr = np.array([t[0] for t in mini_batch], dtype=np.float32)
        a_arr = np.array([t[1] for t in mini_batch], dtype=np.float32)
        r_arr = np.array([t[2] for t in mini_batch], dtype=np.float32)
        s_prime_arr = np.array([t[3] for t in mini_batch], dtype=np.float32)
        done_arr = np.array([[0.0 if t[4] else 1.0] for t in mini_batch], dtype=np.float32)
        
        # 直接转换到GPU（一次性传输）
        s_tensor = torch.from_numpy(s_arr).to(self.device)
        a_tensor = torch.from_numpy(a_arr).to(self.device)
        r_tensor = torch.from_numpy(r_arr).to(self.device)
        s_prime_tensor = torch.from_numpy(s_prime_arr).to(self.device)
        done_tensor = torch.from_numpy(done_arr).to(self.device)
        
        return s_tensor, a_tensor, r_tensor, s_prime_tensor, done_tensor
    
    def size(self):
        return len(self.buffer)


class MuNet(nn.Module):
    def __init__(self, state_size, action_size, config):
        super(MuNet, self).__init__()
        self.fc1 = nn.Linear(state_size, config['hidden_dim1'])
        self.fc2 = nn.Linear(config['hidden_dim1'], config['hidden_dim2'])
        self.fc_mu = nn.Linear(config['hidden_dim2'], action_size)
        self.outlayer = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.outlayer(self.fc_mu(x)) 
        return mu


class QNet(nn.Module):
    def __init__(self, state_size, action_size, config):
        super(QNet, self).__init__()
        self.fc_s = nn.Linear(state_size, config['hidden_dim2'])
        self.fc_a = nn.Linear(action_size, config['hidden_dim2'])
        self.fc_q = nn.Linear(config['hidden_dim2'] * 2, config['critic_hidden'])
        self.fc_out = nn.Linear(config['critic_hidden'], 1)

    def forward(self, x, a):
        h1 = F.relu(self.fc_s(x))
        h2 = F.relu(self.fc_a(a))
        cat = torch.cat([h1, h2], dim=1)
        q = F.relu(self.fc_q(cat))
        q = self.fc_out(q)
        return q


class OrnsteinUhlenbeckNoise:
    def __init__(self, mu, config):
        self.theta = config['noise_params']['theta']
        self.dt = config['noise_params']['dt']
        self.sigma = config['noise_params']['sigma']
        self.mu = mu
        self.x_prev = np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x


# =======================
# 动作投影函数（带计时功能）
# =======================
class ActionProjectionWithTiming:
    """带计时功能的动作投影类"""
    
    def __init__(self):
        self.reset_timings()
    
    def reset_timings(self):
        """重置计时器"""
        self.hungarian_times = []
        self.qp_times = []
        self.total_projection_times = []
    
    def get_valid_jobs_and_robots(self, env):
        """获取有效的作业和机器人 - 优化版本"""
        # 使用numpy加速
        valid_robots = np.where(env.robot_state == 1)[0].tolist()
        
        # 优化：使用task_state数组快速判断
        # task_state: 每5个元素对应一个job的5个操作
        valid_jobs = []
        task_state = env.task_state
        num_jobs = env.num_of_jobs
        
        for job_id in range(num_jobs):
            # 检查该job的5个操作是否全部完成
            start_idx = job_id * 5
            end_idx = start_idx + 5
            if not np.all(task_state[start_idx:end_idx] == 1):
                valid_jobs.append(job_id)
        
        return valid_jobs, valid_robots
    
    def fast_action_selection(self, env, a_):
        """
        快速动作选择 - 跳过匈牙利算法，使用贪心策略
        适用于大规模问题，牺牲一点最优性换取速度
        """
        valid_jobs, valid_robots = self.get_valid_jobs_and_robots(env)
        
        if len(valid_jobs) == 0 or len(valid_robots) == 0:
            return [0, 0, a_[2] if len(a_) > 2 else 0.0]
        
        job_preference = a_[0]
        robot_preference = a_[1]
        
        # 直接根据网络输出选择最接近的job和robot
        num_jobs = len(env.task_set)
        num_robots = len(env.robot_state)
        
        # 计算目标索引
        target_job = int(job_preference * num_jobs)
        target_robot = int(robot_preference * num_robots)
        
        # 在有效集合中找最接近的
        job_id = min(valid_jobs, key=lambda x: abs(x - target_job))
        robot_id = min(valid_robots, key=lambda x: abs(x - target_robot))
        
        # 连续参数直接clip
        param = float(np.clip(a_[2] if len(a_) > 2 else 0.0, 0.0, 1.0))
        
        return [job_id, robot_id, param]
    
    def build_cost_matrix(self, env, valid_jobs, valid_robots, a_):
        """构建代价矩阵 - 优化版本，使用向量化操作"""
        job_preference = a_[0]
        robot_preference = a_[1]
        
        n_jobs = len(valid_jobs)
        n_robots = len(valid_robots)
        
        # 大规模时使用简化的代价计算（仅基于偏好）
        if n_jobs * n_robots > 10000:
            # 向量化计算偏好代价
            job_indices = np.array(valid_jobs)
            robot_indices = np.array(valid_robots)
            
            job_costs = np.abs(job_preference - job_indices / len(env.task_set))
            robot_costs = np.abs(robot_preference - robot_indices / len(env.robot_state))
            
            # 广播计算代价矩阵
            cost_matrix = job_costs[:, np.newaxis] + robot_costs[np.newaxis, :]
            return cost_matrix
        
        # 小规模时使用完整的代价计算
        cost_matrix = np.zeros((n_jobs, n_robots))
        
        for i, job_id in enumerate(valid_jobs):
            current_job = env.task_set[job_id]
            current_op_idx = 0
            for op_idx, task in enumerate(current_job):
                if not task.state:
                    current_op_idx = op_idx
                    break
            current_task = current_job[current_op_idx] if current_op_idx < len(current_job) else current_job[0]
            
            for j, robot_id in enumerate(valid_robots):
                try:
                    C_duration = getattr(current_task, 'duration', 1.0)
                    robot_idle_time = max(0, env.current_time - env.robot_timeline[robot_id])
                    E_duration = getattr(current_task, 'energy', C_duration * 0.8)
                    
                    time_cost_factor = 1.0 / (C_duration + 1e-6)
                    energy_cost_factor = 1.0 / (E_duration + robot_idle_time * 5.0 + 1e-6)
                    
                    preference_cost = abs(job_preference - (job_id / len(env.task_set))) + \
                                    abs(robot_preference - (robot_id / len(env.robot_state)))
                    
                    env_cost = -(env.alpha * time_cost_factor + env.beta * energy_cost_factor)
                    cost_matrix[i, j] = preference_cost + env_cost
                except (AttributeError, IndexError):
                    job_cost = abs(job_preference - (job_id / len(env.task_set)))
                    robot_cost = abs(robot_preference - (robot_id / len(env.robot_state)))
                    cost_matrix[i, j] = job_cost + robot_cost
        
        return cost_matrix
    
    def solve_hungarian(self, cost_matrix):
        """使用匈牙利算法求解最优分配（带计时）"""
        start_time = time.perf_counter()
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        hungarian_time = time.perf_counter() - start_time
        self.hungarian_times.append(hungarian_time)
        return row_ind, col_ind, hungarian_time
    
    def solve_qp(self, v, A, b):
        """求解二次规划问题（带计时）"""
        start_time = time.perf_counter()
        
        try:
            v = np.asarray(v, dtype=np.float64)
            A = np.asarray(A, dtype=np.float64)
            b = np.asarray(b, dtype=np.float64)
            
            def objective(x):
                x = np.asarray(x, dtype=np.float64)
                return 0.5 * np.sum((x - v)**2)
            
            def constraint_fun(x):
                x = np.asarray(x, dtype=np.float64)
                return b - A @ x
                
            constraints = {'type': 'ineq', 'fun': constraint_fun}
            x0 = np.clip(v, 0.0, 1.0).astype(np.float64)
            
            result = minimize(objective, x0, constraints=constraints, method='SLSQP')
            
            qp_time = time.perf_counter() - start_time
            self.qp_times.append(qp_time)
            
            if result.success:
                return float(result.x[0]) if len(result.x) > 0 else float(v[0]), qp_time
            else:
                return float(np.clip(v[0], 0.0, 1.0)), qp_time
                
        except Exception as e:
            qp_time = time.perf_counter() - start_time
            self.qp_times.append(qp_time)
            v_safe = np.asarray(v, dtype=np.float64)
            return float(np.clip(v_safe[0] if len(v_safe) > 0 else 0.0, 0.0, 1.0)), qp_time
    
    def project(self, env, a, record_timing=True, max_hungarian_size=100):
        """
        执行动作投影（带完整计时）
        
        Args:
            env: 环境对象
            a: 动作向量
            record_timing: 是否记录时间
            max_hungarian_size: 匈牙利算法的最大规模限制（用于加速大规模问题）
        
        返回: [job_id, robot_id, param], timing_info
        """
        total_start = time.perf_counter()
        
        a_ = a.detach().cpu().numpy()
        
        # 获取有效作业和机器人
        valid_jobs, valid_robots = self.get_valid_jobs_and_robots(env)
        
        if len(valid_jobs) == 0 or len(valid_robots) == 0:
            total_time = time.perf_counter() - total_start
            if record_timing:
                self.total_projection_times.append(total_time)
            return [0, 0, a_[2] if len(a_) > 2 else 0.0], {
                'hungarian_time': 0.0,
                'qp_time': 0.0,
                'total_time': total_time
            }
        
        job_preference = a_[0]
        robot_preference = a_[1]
        
        # 大规模优化：限制参与匈牙利算法的规模
        if len(valid_jobs) > max_hungarian_size or len(valid_robots) > max_hungarian_size:
            # 基于偏好选择top-k的作业和机器人
            job_scores = [abs(job_preference - (j / len(env.task_set))) for j in valid_jobs]
            robot_scores = [abs(robot_preference - (r / len(env.robot_state))) for r in valid_robots]
            
            # 选择偏好分数最低的（最接近网络输出的）
            k_jobs = min(max_hungarian_size, len(valid_jobs))
            k_robots = min(max_hungarian_size, len(valid_robots))
            
            top_job_indices = np.argsort(job_scores)[:k_jobs]
            top_robot_indices = np.argsort(robot_scores)[:k_robots]
            
            selected_jobs = [valid_jobs[i] for i in top_job_indices]
            selected_robots = [valid_robots[i] for i in top_robot_indices]
        else:
            selected_jobs = valid_jobs
            selected_robots = valid_robots
        
        # 构建代价矩阵
        cost_matrix = self.build_cost_matrix(env, selected_jobs, selected_robots, a_)
        
        # 匈牙利算法
        row_ind, col_ind, hungarian_time = self.solve_hungarian(cost_matrix)
        
        # 选择分配结果
        if len(row_ind) > 0:
            job_id = selected_jobs[row_ind[0]]
            robot_id = selected_robots[col_ind[0]]
        else:
            job_id = selected_jobs[0]
            robot_id = selected_robots[0]
        
        # QP求解连续参数
        qp_time = 0.0
        if len(a_) > 2:
            v = a_[2:]
            n_params = len(v)
            A = np.vstack([np.eye(n_params), -np.eye(n_params)]).astype(np.float64)
            b = np.hstack([np.ones(n_params), np.zeros(n_params)]).astype(np.float64)
            param, qp_time = self.solve_qp(v, A, b)
        else:
            param = 0.0
        
        total_time = time.perf_counter() - total_start
        if record_timing:
            self.total_projection_times.append(total_time)
        
        timing_info = {
            'hungarian_time': hungarian_time,
            'qp_time': qp_time,
            'total_time': total_time
        }
        
        return [job_id, robot_id, param], timing_info
    
    def get_timing_statistics(self):
        """获取计时统计信息"""
        stats = {}
        
        if self.hungarian_times:
            stats['hungarian'] = {
                'mean': np.mean(self.hungarian_times),
                'std': np.std(self.hungarian_times),
                'min': np.min(self.hungarian_times),
                'max': np.max(self.hungarian_times),
                'total': np.sum(self.hungarian_times),
                'count': len(self.hungarian_times)
            }
        
        if self.qp_times:
            stats['qp'] = {
                'mean': np.mean(self.qp_times),
                'std': np.std(self.qp_times),
                'min': np.min(self.qp_times),
                'max': np.max(self.qp_times),
                'total': np.sum(self.qp_times),
                'count': len(self.qp_times)
            }
        
        if self.total_projection_times:
            stats['total_projection'] = {
                'mean': np.mean(self.total_projection_times),
                'std': np.std(self.total_projection_times),
                'min': np.min(self.total_projection_times),
                'max': np.max(self.total_projection_times),
                'total': np.sum(self.total_projection_times),
                'count': len(self.total_projection_times)
            }
        
        return stats


# =======================
# 训练函数
# =======================
def train_step(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer, config, device='cpu', scaler=None):
    """单步训练 - GPU优化版本，支持混合精度"""
    # 数据已经在GPU上（由ReplayBuffer处理）
    s, a, r, s_prime, done_mask = memory.sample(config['batch_size'])
    
    # 是否使用混合精度
    use_amp = scaler is not None and device.type == 'cuda'
    
    # Critic更新
    with torch.no_grad():
        next_action = mu_target(s_prime)
        target_q = q_target(s_prime, next_action)
        target = r.unsqueeze(1) + config['gamma'] * target_q * done_mask
    
    if use_amp:
        with torch.amp.autocast('cuda'):
            current_q = q(s, a)
            q_loss = F.smooth_l1_loss(current_q, target)
        
        q_optimizer.zero_grad()
        scaler.scale(q_loss).backward()
        scaler.step(q_optimizer)
        
        # Actor更新
        with torch.amp.autocast('cuda'):
            mu_loss = -q(s, mu(s)).mean()
        
        mu_optimizer.zero_grad()
        scaler.scale(mu_loss).backward()
        scaler.step(mu_optimizer)
        scaler.update()
    else:
        current_q = q(s, a)
        q_loss = F.smooth_l1_loss(current_q, target)
        
        q_optimizer.zero_grad()
        q_loss.backward()
        q_optimizer.step()
        
        # Actor更新
        mu_loss = -q(s, mu(s)).mean()
        
        mu_optimizer.zero_grad()
        mu_loss.backward()
        mu_optimizer.step()


def soft_update(net, net_target, tau):
    """软更新目标网络"""
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)


def train_policy(config, experiment_name):
    """
    训练策略网络
    
    Args:
        config: 训练配置
        experiment_name: 实验名称 ('small', 'medium', 'large')
    
    Returns:
        训练好的模型和训练记录
    """
    # 合并规模特定配置
    if experiment_name in SCALE_SPECIFIC_CONFIG:
        config = {**config, **SCALE_SPECIFIC_CONFIG[experiment_name]}
    
    print(f"\n{'='*60}")
    print(f"Training Policy for: {EXPERIMENT_CONFIGS[experiment_name]['name']}")
    print(f"Jobs: {config['num_of_jobs']}, Robots: {config['num_of_robots']}")
    print(f"Episodes: {config['num_of_episodes']}")
    print(f"Fast action selection: {config.get('use_fast_action_selection', False)}")
    print(f"Max steps per episode: {config.get('max_steps_per_episode', 'unlimited')}")
    print(f"Training iterations: {config.get('training_iterations', 20)}")
    print(f"{'='*60}")
    
    # 设置随机种子
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    
    # 检测GPU并配置
    device = DEVICE
    print(f"Using device: {device}")
    
    # 混合精度训练的GradScaler
    scaler = None
    if device.type == 'cuda':
        torch.cuda.manual_seed(config['seed'])
        torch.backends.cudnn.benchmark = True  # 启用cudnn自动优化
        scaler = torch.amp.GradScaler('cuda')  # 启用混合精度训练
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"Mixed precision training: Enabled")
    
    # 创建环境
    env = ENV.Env(config['num_of_jobs'], config['num_of_robots'], config['alpha'], config['beta'])
    state_size = len(env.state)
    action_size = len(env.action)
    
    print(f"State size: {state_size}, Action size: {action_size}")
    
    # 初始化网络并移到GPU
    q = QNet(state_size, action_size, config).to(device)
    q_target = QNet(state_size, action_size, config).to(device)
    q_target.load_state_dict(q.state_dict())
    
    mu = MuNet(state_size, action_size, config).to(device)
    mu_target = MuNet(state_size, action_size, config).to(device)
    mu_target.load_state_dict(mu.state_dict())
    
    # 优化器
    mu_optimizer = optim.Adam(mu.parameters(), lr=config['lr_mu'])
    q_optimizer = optim.Adam(q.parameters(), lr=config['lr_q'])
    
    # 训练组件 - 使用GPU加速的ReplayBuffer
    memory = ReplayBuffer(config['buffer_limit'], device=device)
    action_projector = ActionProjectionWithTiming()
    
    # 配置参数
    use_fast_action = config.get('use_fast_action_selection', False)
    max_steps = config.get('max_steps_per_episode', None)
    training_iterations = config.get('training_iterations', 20)
    
    # GPU优化：根据显存大小调整batch_size
    batch_size = config['batch_size']
    if device.type == 'cuda':
        # 获取GPU显存大小（GB）
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        # 根据显存大小和状态维度动态调整batch_size
        if gpu_memory_gb >= 20:  # 大显存GPU (如TITAN RTX)
            batch_size = min(batch_size * 8, 2048)
        elif gpu_memory_gb >= 10:
            batch_size = min(batch_size * 4, 1024)
        else:
            batch_size = min(batch_size * 2, 512)
        
        # 确保 memory_threshold 大于 batch_size
        memory_threshold = max(config['memory_threshold'], batch_size + 100)
        config = {**config, 'batch_size': batch_size, 'memory_threshold': memory_threshold}
        print(f"GPU batch size: {batch_size} (GPU memory: {gpu_memory_gb:.1f} GB)")
        print(f"Memory threshold: {memory_threshold}")
    
    # 训练记录
    score_record = []
    
    # 获取匈牙利算法规模限制
    max_hungarian_size = config.get('max_hungarian_size', 100)
    print(f"Max Hungarian size: {max_hungarian_size}")
    
    # 预分配GPU张量用于推理（减少内存分配开销）
    s_buffer = torch.zeros(state_size, dtype=torch.float32, device=device)
    
    # 训练循环
    training_start_time = time.time()
    
    for n_epi in range(config['num_of_episodes']):
        s = env.reset()
        done = False
        episode_reward = 0.0
        step_count = 0
        
        while not done:
            # 网络前向传播 - 优化版本
            with torch.no_grad():
                # 直接复制到预分配的GPU缓冲区
                s_buffer.copy_(torch.from_numpy(s.astype(np.float32)))
                a = mu(s_buffer.unsqueeze(0)).squeeze(0)
                a = torch.clamp(a, 0, 1)
                a_np = a.cpu().numpy()
            
            # 动作选择（根据配置使用快速或top-k匈牙利方法）
            if use_fast_action:
                action = action_projector.fast_action_selection(env, a_np)
            else:
                action, _ = action_projector.project(env, a, record_timing=False, max_hungarian_size=max_hungarian_size)
            
            s_prime, r, done = env.step(action)
            
            memory.put((s, a_np, r, s_prime, done))
            s = s_prime
            episode_reward += r
            step_count += 1
            
            # 检查是否达到最大步数
            if max_steps is not None and step_count >= max_steps:
                break
        
        score_record.append(episode_reward)
        
        # 训练更新
        if memory.size() > config['memory_threshold']:
            for _ in range(training_iterations):
                train_step(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer, config, device, scaler)
                soft_update(mu, mu_target, config['tau'])
                soft_update(q, q_target, config['tau'])
        
        if n_epi % config['print_interval'] == 0 and n_epi != 0:
            avg_score = np.mean(score_record[-config['print_interval']:])
            elapsed = time.time() - training_start_time
            eps_per_sec = n_epi / elapsed if elapsed > 0 else 0
            remaining = (config['num_of_episodes'] - n_epi) / eps_per_sec if eps_per_sec > 0 else 0
            print(f"Episode {n_epi}/{config['num_of_episodes']}: Avg Score = {avg_score:.4f} | "
                  f"Speed: {eps_per_sec:.2f} ep/s | ETA: {remaining/60:.1f} min")
    
    training_time = time.time() - training_start_time
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Final average score (last 20 episodes): {np.mean(score_record[-20:]):.4f}")
    
    return {
        'mu': mu,
        'mu_target': mu_target,
        'q': q,
        'q_target': q_target,
        'score_record': score_record,
        'training_time': training_time,
        'state_size': state_size,
        'action_size': action_size
    }


# =======================
# 运行时间测试函数
# =======================
def test_runtime(config, mu, experiment_name):
    """
    测试策略运行时间（使用CPU进行推理计时）
    
    Args:
        config: 配置
        mu: 训练好的策略网络
        experiment_name: 实验名称
    
    Returns:
        运行时间统计
    """
    print(f"\n{'='*60}")
    print(f"Testing Runtime for: {EXPERIMENT_CONFIGS[experiment_name]['name']}")
    print(f"Jobs: {config['num_of_jobs']}, Robots: {config['num_of_robots']}")
    print(f"{'='*60}")
    
    # =====================
    # 将模型移到CPU进行推理计时
    # =====================
    cpu_device = torch.device('cpu')
    mu_cpu = MuNet(
        state_size=mu.fc1.in_features,
        action_size=mu.fc_mu.out_features,
        config=BASE_CONFIG
    ).to(cpu_device)
    mu_cpu.load_state_dict(mu.state_dict())
    mu_cpu.eval()
    
    print(f"Runtime testing device: CPU (model copied from {next(mu.parameters()).device})")
    
    # 创建环境
    env = ENV.Env(config['num_of_jobs'], config['num_of_robots'], config['alpha'], config['beta'])
    
    # 创建带计时的动作投影器
    action_projector = ActionProjectionWithTiming()
    
    # 记录网络前向传播时间
    network_forward_times = []
    total_decision_times = []
    
    # =====================
    # 阶段1：CPU网络预热（让CPU缓存和分支预测器稳定）
    # =====================
    network_warmup_iters = TIMING_CONFIG.get('network_warmup_iterations', 1000)
    print(f"Phase 1: CPU network warmup ({network_warmup_iters} iterations)...")
    
    # 获取该网络的state_size
    state_size = env.state.shape[0] if hasattr(env.state, 'shape') else len(env.state)
    
    # 创建固定的dummy输入，专门用于预热该网络
    dummy_state = torch.randn(1, state_size, dtype=torch.float32, device=cpu_device)
    
    # 进行大量前向传播，预热CPU缓存
    for _ in range(network_warmup_iters):
        with torch.no_grad():
            _ = mu_cpu(dummy_state)
    
    # =====================
    # 阶段2：环境交互预热（包含完整的动作选择流程）
    # =====================
    print(f"Phase 2: Environment warmup ({TIMING_CONFIG['warmup_steps']} steps)...")
    s = env.reset()
    for _ in range(TIMING_CONFIG['warmup_steps']):
        with torch.no_grad():
            s_tensor = torch.from_numpy(s.astype(np.float32))
            a = mu_cpu(s_tensor)
            a = torch.clamp(a, 0, 1)
            a_np = a.numpy()
        action = action_projector.fast_action_selection(env, a_np)
        s_prime, _, done = env.step(action)
        if done:
            s = env.reset()
        else:
            s = s_prime
    
    print("Warmup completed.")
    
    # 重置计时器
    action_projector.reset_timings()
    
    # 正式测试
    print(f"Running timing tests ({TIMING_CONFIG['num_timing_episodes']} episodes)...")
    
    for episode in range(TIMING_CONFIG['num_timing_episodes']):
        s = env.reset()
        done = False
        step = 0
        
        while not done:
            # 总决策时间开始
            decision_start = time.perf_counter()
            
            # 1. 网络前向传播计时（CPU上直接计时，无需同步）
            network_start = time.perf_counter()
            with torch.no_grad():
                s_tensor = torch.from_numpy(s.astype(np.float32))
                a = mu_cpu(s_tensor)
                a = torch.clamp(a, 0, 1)
                a_np = a.numpy()
            network_time = time.perf_counter() - network_start
            network_forward_times.append(network_time)
            
            # 2. 动作投影（包含top-k匈牙利算法和QP）
            max_hungarian_size = config.get('max_hungarian_size', 100)
            action, timing_info = action_projector.project(env, torch.from_numpy(a_np), record_timing=True, max_hungarian_size=max_hungarian_size)
            
            # 总决策时间
            total_decision_time = time.perf_counter() - decision_start
            total_decision_times.append(total_decision_time)
            
            # 执行动作
            s_prime, _, done = env.step(action)
            s = s_prime
            step += 1
        
        print(f"  Episode {episode + 1}: {step} steps")
    
    # 汇总统计
    projection_stats = action_projector.get_timing_statistics()
    
    timing_results = {
        'experiment_name': experiment_name,
        'config_name': EXPERIMENT_CONFIGS[experiment_name]['name'],
        'num_of_jobs': config['num_of_jobs'],
        'num_of_robots': config['num_of_robots'],
        
        'network_forward': {
            'mean': np.mean(network_forward_times),
            'std': np.std(network_forward_times),
            'min': np.min(network_forward_times),
            'max': np.max(network_forward_times),
            'count': len(network_forward_times)
        },
        
        'hungarian': projection_stats.get('hungarian', {}),
        'qp': projection_stats.get('qp', {}),
        'total_projection': projection_stats.get('total_projection', {}),
        
        'total_decision': {
            'mean': np.mean(total_decision_times),
            'std': np.std(total_decision_times),
            'min': np.min(total_decision_times),
            'max': np.max(total_decision_times),
            'count': len(total_decision_times)
        }
    }
    
    return timing_results


def print_timing_results(timing_results):
    """打印运行时间结果"""
    print(f"\n{'='*70}")
    print(f"Runtime Results: {timing_results['config_name']}")
    print(f"Jobs: {timing_results['num_of_jobs']}, Robots: {timing_results['num_of_robots']}")
    print(f"{'='*70}")
    
    print(f"\n1. 策略网络前向推理时间:")
    net = timing_results['network_forward']
    print(f"   平均: {net['mean']*1000:.4f} ms")
    print(f"   标准差: {net['std']*1000:.4f} ms")
    print(f"   最小: {net['min']*1000:.4f} ms")
    print(f"   最大: {net['max']*1000:.4f} ms")
    
    print(f"\n2. 匈牙利算法执行时间:")
    hung = timing_results['hungarian']
    if hung:
        print(f"   平均: {hung['mean']*1000:.4f} ms")
        print(f"   标准差: {hung['std']*1000:.4f} ms")
        print(f"   最小: {hung['min']*1000:.4f} ms")
        print(f"   最大: {hung['max']*1000:.4f} ms")
    
    print(f"\n3. QP求解时间:")
    qp = timing_results['qp']
    if qp:
        print(f"   平均: {qp['mean']*1000:.4f} ms")
        print(f"   标准差: {qp['std']*1000:.4f} ms")
        print(f"   最小: {qp['min']*1000:.4f} ms")
        print(f"   最大: {qp['max']*1000:.4f} ms")
    
    print(f"\n4. 总决策时间:")
    total = timing_results['total_decision']
    print(f"   平均: {total['mean']*1000:.4f} ms")
    print(f"   标准差: {total['std']*1000:.4f} ms")
    print(f"   最小: {total['min']*1000:.4f} ms")
    print(f"   最大: {total['max']*1000:.4f} ms")
    
    # 时间占比分析
    print(f"\n5. 时间占比分析:")
    total_mean = total['mean']
    if total_mean > 0:
        net_pct = (net['mean'] / total_mean) * 100
        hung_pct = (hung['mean'] / total_mean) * 100 if hung else 0
        qp_pct = (qp['mean'] / total_mean) * 100 if qp else 0
        other_pct = 100 - net_pct - hung_pct - qp_pct
        
        print(f"   网络前向: {net_pct:.2f}%")
        print(f"   匈牙利算法: {hung_pct:.2f}%")
        print(f"   QP求解: {qp_pct:.2f}%")
        print(f"   其他: {other_pct:.2f}%")


def save_results(all_results, save_dir):
    """保存所有实验结果"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存JSON格式的结果
    results_json = {
        'timestamp': datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        'runtime_test_device': 'CPU',  # 标记运行时间测试使用CPU
        'experiments': {}
    }
    
    for exp_name, result in all_results.items():
        # 保存模型
        model_dir = os.path.join(save_dir, exp_name)
        os.makedirs(model_dir, exist_ok=True)
        
        torch.save(result['training']['mu'].state_dict(), 
                   os.path.join(model_dir, 'mu.pth'))
        torch.save(result['training']['q'].state_dict(), 
                   os.path.join(model_dir, 'q.pth'))
        
        # 保存训练曲线
        np.save(os.path.join(model_dir, 'score_record.npy'), 
                result['training']['score_record'])
        
        # 添加到JSON结果
        timing = result['timing']
        results_json['experiments'][exp_name] = {
            'config': {
                'num_of_jobs': timing['num_of_jobs'],
                'num_of_robots': timing['num_of_robots']
            },
            'training_time': result['training']['training_time'],
            'timing': {
                'network_forward_ms': timing['network_forward']['mean'] * 1000,
                'hungarian_ms': timing['hungarian']['mean'] * 1000 if timing['hungarian'] else 0,
                'qp_ms': timing['qp']['mean'] * 1000 if timing['qp'] else 0,
                'total_decision_ms': timing['total_decision']['mean'] * 1000
            }
        }
    
    # 保存JSON
    with open(os.path.join(save_dir, 'results_summary.json'), 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\nResults saved to: {save_dir}")
    return save_dir


def plot_comparison(all_results, save_dir):
    """绘制对比图"""
    
    # 提取数据
    exp_names = list(all_results.keys())
    labels = [EXPERIMENT_CONFIGS[name]['name'] for name in exp_names]
    
    network_times = []
    hungarian_times = []
    qp_times = []
    total_times = []
    
    for name in exp_names:
        timing = all_results[name]['timing']
        network_times.append(timing['network_forward']['mean'] * 1000)
        hungarian_times.append(timing['hungarian']['mean'] * 1000 if timing['hungarian'] else 0)
        qp_times.append(timing['qp']['mean'] * 1000 if timing['qp'] else 0)
        total_times.append(timing['total_decision']['mean'] * 1000)
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    x = np.arange(len(labels))
    width = 0.6
    
    # 1. 总决策时间对比
    ax1 = axes[0, 0]
    bars1 = ax1.bar(x, total_times, width, color=['#3498db', '#e74c3c', '#2ecc71'])
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('Total Decision Time Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Small\n(100×5)', 'Medium\n(100×50)', 'Large\n(1000×500)'])
    ax1.bar_label(bars1, fmt='%.2f')
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. 各组件时间堆叠图
    ax2 = axes[0, 1]
    bars_net = ax2.bar(x, network_times, width, label='Network Forward', color='#3498db')
    bars_hung = ax2.bar(x, hungarian_times, width, bottom=network_times, label='Hungarian', color='#e74c3c')
    bottom_qp = [n + h for n, h in zip(network_times, hungarian_times)]
    bars_qp = ax2.bar(x, qp_times, width, bottom=bottom_qp, label='QP', color='#2ecc71')
    
    ax2.set_ylabel('Time (ms)')
    ax2.set_title('Time Breakdown by Component')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Small\n(100×5)', 'Medium\n(100×50)', 'Large\n(1000×500)'])
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. 匈牙利算法时间对比
    ax3 = axes[1, 0]
    bars3 = ax3.bar(x, hungarian_times, width, color=['#3498db', '#e74c3c', '#2ecc71'])
    ax3.set_ylabel('Time (ms)')
    ax3.set_title('Hungarian Algorithm Time')
    ax3.set_xticks(x)
    ax3.set_xticklabels(['Small\n(100×5)', 'Medium\n(100×50)', 'Large\n(1000×500)'])
    ax3.bar_label(bars3, fmt='%.2f')
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. 训练曲线对比
    ax4 = axes[1, 1]
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    for i, name in enumerate(exp_names):
        scores = all_results[name]['training']['score_record']
        # 平滑处理
        window = 10
        smoothed = np.convolve(scores, np.ones(window)/window, mode='valid')
        ax4.plot(smoothed, color=colors[i], label=labels[i], alpha=0.8)
    
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Score')
    ax4.set_title('Training Curves Comparison')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'runtime_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Comparison plot saved to: {os.path.join(save_dir, 'runtime_comparison.png')}")


def print_summary_table(all_results):
    """打印汇总表格"""
    print("\n" + "="*100)
    print("Runtime Results (in milliseconds):")
    print(f"{'Scale':<15} {'Jobs×Robots':<12} {'Policy Net':<16} {'Hungarian':<16} {'QP':<16} {'Total':<16}")
    print("-"*100)
    
    for name in ['small', 'medium', 'large']:
        if name in all_results:
            timing = all_results[name]['timing']
            config_str = f"{timing['num_of_jobs']}×{timing['num_of_robots']}"
            
            # 获取mean和std
            net_mean = timing['network_forward']['mean'] * 1000
            net_std = timing['network_forward']['std'] * 1000
            
            hung_mean = timing['hungarian']['mean'] * 1000 if timing['hungarian'] else 0
            hung_std = timing['hungarian']['std'] * 1000 if timing['hungarian'] else 0
            
            qp_mean = timing['qp']['mean'] * 1000 if timing['qp'] else 0
            qp_std = timing['qp']['std'] * 1000 if timing['qp'] else 0
            
            total_mean = timing['total_decision']['mean'] * 1000
            total_std = timing['total_decision']['std'] * 1000
            
            # 格式化为 mean±std
            net_str = f"{net_mean:.4f}±{net_std:.4f}"
            hung_str = f"{hung_mean:.4f}±{hung_std:.4f}"
            qp_str = f"{qp_mean:.4f}±{qp_std:.4f}"
            total_str = f"{total_mean:.4f}±{total_std:.4f}"
            
            print(f"{name:<15} {config_str:<12} {net_str:<16} {hung_str:<16} {qp_str:<16} {total_str:<16}")
    
    print("="*100)
    
    # 计算规模增长比
    if 'small' in all_results and 'large' in all_results:
        small_total = all_results['small']['timing']['total_decision']['mean']
        large_total = all_results['large']['timing']['total_decision']['mean']
        scale_factor = large_total / small_total if small_total > 0 else 0
        
        print(f"\nScaling Factor (Large/Small): {scale_factor:.2f}x")
        print(f"Problem Size Increase: {(1000*100)/(100*5):.0f}x")


# =======================
# 主函数
# =======================
def main():
    """主函数"""
    print("="*80)
    print("LIRL Runtime Scaling Experiment (CPU Runtime Test)")
    print("="*80)
    print(f"Training device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    print(f"Runtime test device: CPU")  # 标记运行时间测试使用CPU
    print(f"Experiment Configurations:")
    for name, cfg in EXPERIMENT_CONFIGS.items():
        print(f"  {name}: {cfg['name']}")
    print("="*80)
    
    # 全局GPU预热（解决首个实验因GPU冷启动导致的时间偏长问题）
    warmup_gpu()
    
    # 创建保存目录（保存在exp文件夹下，CPU版本）
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_folder = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(exp_folder, f"lirl_runtime_scaling_cpu_{timestamp}")
    
    # 存储所有结果
    all_results = {}
    
    # 对每种配置进行训练和测试
    for exp_name in ['small', 'medium', 'large']:
        print(f"\n\n{'#'*80}")
        print(f"# Experiment: {EXPERIMENT_CONFIGS[exp_name]['name']}")
        print(f"{'#'*80}")
        
        # 合并配置（包括规模特定配置）
        config = BASE_CONFIG.copy()
        config.update(EXPERIMENT_CONFIGS[exp_name])
        if exp_name in SCALE_SPECIFIC_CONFIG:
            config.update(SCALE_SPECIFIC_CONFIG[exp_name])
        
        # 训练策略
        training_result = train_policy(config, exp_name)
        
        # 测试运行时间
        timing_result = test_runtime(config, training_result['mu'], exp_name)
        
        # 打印结果
        print_timing_results(timing_result)
        
        # 存储结果
        all_results[exp_name] = {
            'training': training_result,
            'timing': timing_result,
            'config': config
        }
    
    # 打印汇总表格
    print_summary_table(all_results)
    
    # 保存结果
    saved_dir = save_results(all_results, save_dir)
    
    # 绘制对比图
    try:
        plot_comparison(all_results, saved_dir)
    except Exception as e:
        print(f"Warning: Could not plot comparison: {e}")
    
    print(f"\n{'='*80}")
    print("Experiment completed!")
    print(f"Results saved to: {saved_dir}")
    print(f"{'='*80}")
    
    return all_results


if __name__ == "__main__":
    results = main()

