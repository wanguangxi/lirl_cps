"""
LIRL Runtime Scaling Experiment
================================
实验目的：测试LIRL算法在不同规模充电站系统下的运行时间可扩展性

实验设置：
1. n_stations=5 (小规模)
2. n_stations=50 (中规模)  
3. n_stations=100 (大规模)

测试指标：
- 策略网络前向推理时间
- 匈牙利算法执行时间
- QP求解执行时间
- 每次决策的总体时间
"""

import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import math
import time
import os
import sys
import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import linear_sum_assignment
from datetime import datetime, timedelta
import json
import csv

# 添加环境路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../env'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../alg'))
from ev import EVChargingEnv

# 尝试导入cvxpy用于QP求解（可选）
try:
    import cvxpy as cp
    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False
    print("Warning: cvxpy not installed, QP timing will use simplified version")


# =======================
# EXPERIMENT CONFIGURATIONS
# =======================
# 注意：arrival_rate 是泊松分布参数，表示每步平均到达车辆数
# 为了让不同规模的实验有意义，需要根据充电桩数量调整到达率
# 目标：让充电桩利用率在 60-80% 左右
EXPERIMENT_CONFIGS = {
    'small': {
        'n_stations': 5,
        'p_max': 150.0,
        'arrival_rate': 0.5,   # 每步0.5辆，适合5个充电桩
        'num_of_episodes': 500,
        'name': 'small_scale_5_stations',
        'seed_offset': 0,      # 不同实验使用不同种子偏移
    },
    'medium': {
        'n_stations': 50,
        'p_max': 150.0,
        'arrival_rate': 0.8,   # 每步0.8辆，适合50个充电桩（会有更多并发）
        'num_of_episodes': 500,
        'name': 'medium_scale_50_stations',
        'seed_offset': 1000,
    },
    'large': {
        'n_stations': 100,
        'p_max': 150.0,
        'arrival_rate': 0.95,  # 每步0.95辆，接近满载运行
        'num_of_episodes': 500,
        'name': 'large_scale_100_stations',
        'seed_offset': 2000,
    }
}

# 基础配置
BASE_CONFIG = {
    'lr_mu': 0.0005,
    'lr_q': 0.001,
    'gamma': 0.98,
    'batch_size': 128,
    'buffer_limit': 1000000,
    'tau': 0.005,
    'hidden_dim1': 128,
    'hidden_dim2': 64,
    'critic_hidden': 32,
    'memory_threshold': 500,
    'training_iterations': 20,
    'noise_params': {'theta': 0.1, 'dt': 0.05, 'sigma': 0.1},
    'seed': 42,
    'max_test_steps': 288,
    'print_interval': 20,
    'save_models': True,
    'num_timing_runs': 100,  # 运行时间测试的重复次数
}


# =======================
# NETWORK DEFINITIONS
# =======================
class ReplayBuffer():
    def __init__(self, buffer_limit=1000000):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)
        
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append(r)
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0 
            done_mask_lst.append([done_mask])
        
        s_lst_= torch.FloatTensor(np.array(s_lst))
        a_lst_= torch.tensor(np.array(a_lst), dtype=torch.float)
        r_lst_= torch.tensor(np.array(r_lst), dtype=torch.float)
        s_prime_lst_ = torch.tensor(np.array(s_prime_lst), dtype=torch.float)
        done_mask_lst_ = torch.tensor(np.array(done_mask_lst), dtype=torch.float)

        return s_lst_, a_lst_, r_lst_, s_prime_lst_, done_mask_lst_
    
    def size(self):
        return len(self.buffer)


class MuNet(nn.Module):
    def __init__(self, state_size, action_size, hidden_dim1=128, hidden_dim2=64):
        super(MuNet, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc_mu = nn.Linear(hidden_dim2, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.fc_mu(x)
        if len(mu.shape) == 1:
            mu = mu.unsqueeze(0)
        mu[:, :2] = torch.sigmoid(mu[:, :2])
        mu[:, 2:] = torch.sigmoid(mu[:, 2:])
        return mu


class QNet(nn.Module):
    def __init__(self, state_size, action_size, hidden_dim=64, critic_hidden=32):
        super(QNet, self).__init__()
        self.fc_s = nn.Linear(state_size, hidden_dim)
        self.fc_a = nn.Linear(action_size, hidden_dim)
        self.fc_q = nn.Linear(hidden_dim * 2, critic_hidden)
        self.fc_out = nn.Linear(critic_hidden, 1)

    def forward(self, x, a):
        h1 = F.relu(self.fc_s(x))
        h2 = F.relu(self.fc_a(a))
        cat = torch.cat([h1, h2], dim=1)
        q = F.relu(self.fc_q(cat))
        q = self.fc_out(q)
        return q


class OrnsteinUhlenbeckNoise:
    def __init__(self, mu, theta=0.1, dt=0.05, sigma=0.1):
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


# =======================
# TIMING UTILITIES
# =======================
class TimingResult:
    """存储运行时间测量结果"""
    def __init__(self):
        self.policy_network_times = []
        self.hungarian_times = []
        self.qp_times = []
        self.total_decision_times = []
        
    def add_timing(self, policy_time, hungarian_time, qp_time, total_time):
        self.policy_network_times.append(policy_time)
        self.hungarian_times.append(hungarian_time)
        self.qp_times.append(qp_time)
        self.total_decision_times.append(total_time)
    
    def get_statistics(self):
        return {
            'policy_network': {
                'mean': np.mean(self.policy_network_times) * 1000,  # 转换为ms
                'std': np.std(self.policy_network_times) * 1000,
                'min': np.min(self.policy_network_times) * 1000,
                'max': np.max(self.policy_network_times) * 1000,
            },
            'hungarian': {
                'mean': np.mean(self.hungarian_times) * 1000,
                'std': np.std(self.hungarian_times) * 1000,
                'min': np.min(self.hungarian_times) * 1000,
                'max': np.max(self.hungarian_times) * 1000,
            },
            'qp': {
                'mean': np.mean(self.qp_times) * 1000,
                'std': np.std(self.qp_times) * 1000,
                'min': np.min(self.qp_times) * 1000,
                'max': np.max(self.qp_times) * 1000,
            },
            'total': {
                'mean': np.mean(self.total_decision_times) * 1000,
                'std': np.std(self.total_decision_times) * 1000,
                'min': np.min(self.total_decision_times) * 1000,
                'max': np.max(self.total_decision_times) * 1000,
            }
        }


def action_projection_with_timing(env, a, measure_timing=False):
    """
    LIRL动作投影函数 - 串联架构（特征权重学习版本）
    
    ┌─────────────────────────────────────────────────────────────────┐
    │                    策略网络 (Policy Net)                         │
    │                  输出: [wait_weight, energy_weight, power_pref]  │
    │                                                                  │
    │  wait_weight:   等待时间权重 (0-1)，越高越优先长等待车辆         │
    │  energy_weight: 能量需求权重 (0-1)，越高越优先高能量需求车辆     │
    │  power_pref:    功率偏好 (0-1)，映射到[50, p_max]               │
    └─────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                    匈牙利算法 (Hungarian)                        │
    │  功能: 基于策略权重构建代价矩阵，求解最优分配                    │
    │                                                                  │
    │  代价矩阵:                                                       │
    │    cost[i,j] = -wait_weight * wait_score                        │
    │              - energy_weight * energy_score                      │
    │              - urgency_bonus (紧急情况固定优先)                  │
    │                                                                  │
    │  策略网络学习如何平衡等待时间vs能量需求的权重                    │
    └─────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                    QP求解器 (Quadratic Programming)              │
    │  功能: 约束功率 power_pref 到可行域 [50, p_max]                  │
    └─────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                    最终动作                                       │
    │                  {station_id, vehicle_id, power}                 │
    └─────────────────────────────────────────────────────────────────┘
    
    返回: (action, timing_dict) if measure_timing else action
    """
    timing = {'policy': 0, 'hungarian': 0, 'qp': 0, 'total': 0}
    
    total_start = time.perf_counter()
    
    # ========== Step 0: 处理网络输出 ==========
    a_ = a.detach().cpu().numpy()
    if len(a_.shape) > 1:
        a_ = a_.squeeze()
    
    if len(a_) < 3:
        if len(a_) == 1:
            a_ = np.array([a_[0], 0.5, 0.5])
        elif len(a_) == 2:
            a_ = np.array([a_[0], a_[1], 0.5])
    
    # 网络输出解析（新设计：特征权重学习）:
    # a_[0]: wait_weight (等待时间权重, 0-1)
    # a_[1]: energy_weight (能量需求权重, 0-1)  
    # a_[2]: power_pref (功率偏好, 0-1)
    wait_weight = a_[0]
    energy_weight = a_[1]
    power_pref = a_[2]
    
    # ========== Step 1: 获取可行动作空间 ==========
    # 获取可用充电桩列表
    valid_stations = [i for i in range(env.n_stations) if env.station_status[i] == 1]
    
    # 获取有效车辆列表（未充电且未充满）
    valid_vehicles = []
    for i in range(env.max_vehicles):
        if env.vehicles[i] is not None:
            vehicle = env.vehicles[i]
            if not vehicle['charging'] and not vehicle['fully_charged']:
                valid_vehicles.append(i)
    
    # 如果没有可用资源，返回默认动作
    if len(valid_stations) == 0 or len(valid_vehicles) == 0:
        safe_vehicle_id = 0
        for i in range(env.max_vehicles):
            if env.vehicles[i] is not None:
                safe_vehicle_id = i
                break
        
        action = {
            'station_id': valid_stations[0] if valid_stations else 0,
            'vehicle_id': safe_vehicle_id,
            'power': np.array([100.0], dtype=np.float32)
        }
        timing['total'] = time.perf_counter() - total_start
        if measure_timing:
            return action, timing
        return action
    
    # ========== Step 2: 匈牙利算法 - 基于策略权重的最优分配 ==========
    hungarian_start = time.perf_counter()
    
    n_vehicles = len(valid_vehicles)
    n_stations = len(valid_stations)
    cost_matrix = np.zeros((n_vehicles, n_stations))
    
    # 预计算所有车辆的特征用于归一化
    wait_times = []
    energy_needs = []
    for vehicle_id in valid_vehicles:
        vehicle = env.vehicles[vehicle_id]
        if vehicle is not None:
            wait_times.append(vehicle['wait_time'])
            energy_needs.append(vehicle['energy_required'] - vehicle['energy_charged'])
    
    max_wait_time = max(wait_times) if wait_times else 1
    max_energy_need = max(energy_needs) if energy_needs else 1
    max_wait_time = max(max_wait_time, 1)  # 防止除零
    max_energy_need = max(max_energy_need, 1)
    
    for i, vehicle_id in enumerate(valid_vehicles):
        vehicle = env.vehicles[vehicle_id]
        if vehicle is None:
            continue
        
        # 车辆特征
        wait_time = vehicle['wait_time']
        energy_needed = vehicle['energy_required'] - vehicle['energy_charged']
        
        # 归一化特征分数 (0-1范围)
        wait_score = wait_time / max_wait_time
        energy_score = energy_needed / max_energy_need
        
        for j, station_id in enumerate(valid_stations):
            # ====== 策略网络权重调制的成本 ======
            # 成本越低越好，所以使用负值
            # 策略网络学习如何平衡 wait_weight 和 energy_weight
            weighted_cost = -(wait_weight * wait_score * 100.0 + 
                             energy_weight * energy_score * 100.0)
            
            # ====== 紧急情况固定优先（安全约束）======
            urgency_bonus = 0.0
            if wait_time >= env.max_wait_time - 1:
                urgency_bonus = -200.0  # 即将超时，强制最高优先
            elif wait_time >= env.max_wait_time - 3:
                urgency_bonus = -100.0  # 接近超时，高优先
            
            # ====== 总成本 ======
            cost_matrix[i, j] = weighted_cost + urgency_bonus
    
    # 求解最优分配（最小化总成本）
    try:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        if len(row_ind) > 0:
            vehicle_id = valid_vehicles[row_ind[0]]
            station_id = valid_stations[col_ind[0]]
        else:
            vehicle_id = valid_vehicles[0]
            station_id = valid_stations[0]
    except Exception:
        vehicle_id = valid_vehicles[0]
        station_id = valid_stations[0]
    
    hungarian_time = time.perf_counter() - hungarian_start
    timing['hungarian'] = hungarian_time
    
    # ========== Step 3: QP求解器 - 约束功率 ==========
    qp_start = time.perf_counter()
    
    # 目标功率 = 50 + power_pref * 100 (映射到50-150 kW)
    target_power = 50.0 + power_pref * 100.0
    
    if HAS_CVXPY:
        # 使用QP求解器约束功率
        # min ||power - target_power||²
        # s.t. 50 ≤ power ≤ p_max
        try:
            p = cp.Variable(1)
            objective = cp.Minimize(cp.square(p - target_power))
            constraints = [p >= 50.0, p <= env.p_max]
            prob = cp.Problem(objective, constraints)
            prob.solve(solver=cp.OSQP, verbose=False)
            power = float(p.value[0]) if p.value is not None else target_power
        except:
            power = np.clip(target_power, 50.0, env.p_max)
    else:
        # 简单投影
        power = np.clip(target_power, 50.0, env.p_max)
    
    qp_time = time.perf_counter() - qp_start
    timing['qp'] = qp_time
    
    # ========== Step 4: 安全检查 ==========
    # 确保最终动作有效
    if station_id >= env.n_stations or env.station_status[station_id] != 1:
        for alt_station in range(env.n_stations):
            if env.station_status[alt_station] == 1:
                station_id = alt_station
                break
    
    if (vehicle_id >= env.max_vehicles or env.vehicles[vehicle_id] is None or 
        env.vehicles[vehicle_id]['charging'] or env.vehicles[vehicle_id]['fully_charged']):
        for alt_vehicle in range(env.max_vehicles):
            if (env.vehicles[alt_vehicle] is not None and
                not env.vehicles[alt_vehicle]['charging'] and
                not env.vehicles[alt_vehicle]['fully_charged']):
                vehicle_id = alt_vehicle
                break
    
    # ========== 构建最终动作 ==========
    action = {
        'station_id': station_id,
        'vehicle_id': vehicle_id,
        'power': np.array([power], dtype=np.float32)
    }
    
    timing['total'] = time.perf_counter() - total_start
    
    if measure_timing:
        return action, timing
    return action


# =======================
# TRAINING FUNCTIONS
# =======================
def train_step(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer, config):
    """
    执行一次训练更新
    返回: (q_loss_value, mu_loss_value) 用于监控训练进度
    """
    s, a, r, s_prime, done_mask = memory.sample(config['batch_size'])
    
    if len(a.shape) == 1:
        a = a.unsqueeze(1)
    
    # Critic更新
    target = torch.unsqueeze(r, dim=1) + config['gamma'] * q_target(s_prime, mu_target(s_prime)).mul(done_mask)
    q_loss = F.smooth_l1_loss(q(s, a), target.detach())
    q_optimizer.zero_grad()
    q_loss.backward()
    q_optimizer.step()
    
    # Actor更新
    mu_loss = -q(s, mu(s)).mean()
    mu_optimizer.zero_grad()
    mu_loss.backward()
    mu_optimizer.step()
    
    return q_loss.item(), mu_loss.item()


def soft_update(net, net_target, tau):
    """软更新目标网络"""
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)


def train_policy(config, exp_config, save_dir):
    """
    训练策略网络
    
    返回: (mu, training_stats, env)
        - mu: 训练好的策略网络
        - training_stats: 包含score, q_loss, mu_loss的统计字典
        - env: 环境实例
    """
    print(f"\n{'='*60}")
    print(f"Training: {exp_config['name']}")
    print(f"n_stations: {exp_config['n_stations']}")
    print(f"arrival_rate: {exp_config['arrival_rate']}")
    print(f"{'='*60}")
    
    # 设置随机种子（每个实验使用不同种子以确保差异性）
    seed_offset = exp_config.get('seed_offset', 0)
    seed = config['seed'] + seed_offset
    print(f"Using seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 创建环境
    env = EVChargingEnv(
        n_stations=exp_config['n_stations'],
        p_max=exp_config['p_max'],
        arrival_rate=exp_config['arrival_rate']
    )
    
    state_size = env.observation_space.shape[0]
    action_size = 3
    
    # 创建网络
    q = QNet(state_size, action_size, config['hidden_dim2'], config['critic_hidden'])
    q_target = QNet(state_size, action_size, config['hidden_dim2'], config['critic_hidden'])
    q_target.load_state_dict(q.state_dict())
    
    mu = MuNet(state_size, action_size, config['hidden_dim1'], config['hidden_dim2'])
    mu_target = MuNet(state_size, action_size, config['hidden_dim1'], config['hidden_dim2'])
    mu_target.load_state_dict(mu.state_dict())
    
    # 优化器
    mu_optimizer = optim.Adam(mu.parameters(), lr=config['lr_mu'])
    q_optimizer = optim.Adam(q.parameters(), lr=config['lr_q'])
    
    # 训练组件
    ou_noise = OrnsteinUhlenbeckNoise(
        mu=np.zeros(action_size),
        theta=config['noise_params']['theta'],
        dt=config['noise_params']['dt'],
        sigma=config['noise_params']['sigma']
    )
    memory = ReplayBuffer(config['buffer_limit'])
    
    # 训练统计记录
    score_record = []
    q_loss_record = []
    mu_loss_record = []
    action_stats_record = []  # 记录策略网络输出统计
    
    for n_epi in range(exp_config['num_of_episodes']):
        s = env.reset()
        done = False
        episode_reward = 0
        episode_steps = 0
        episode_actions = []  # 收集本episode的动作
        
        while not done:
            # 策略网络前向推理
            a = mu(torch.from_numpy(s).float())
            
            # 添加探索噪声（噪声衰减）
            noise_scale = max(0.1, 1.0 - n_epi / (exp_config['num_of_episodes'] * 0.8))
            noise = torch.from_numpy(ou_noise()).float()
            a = a + noise * noise_scale
            a = torch.clamp(a, 0, 1)
            a = a.squeeze() if len(a.shape) > 1 else a
            
            # 记录策略输出
            episode_actions.append(a.detach().numpy().copy())
            
            # 动作投影（匈牙利算法 + QP）
            action = action_projection_with_timing(env, a, measure_timing=False)
            s_prime, r, done, info = env.step(action)
            
            # 存储经验
            a_store = a.detach().numpy()
            memory.put((s, a_store, r, s_prime, done))
            
            s = s_prime
            episode_reward += r
            episode_steps += 1
        
        score_record.append(episode_reward)
        
        # 计算本episode的动作统计（新设计：特征权重学习）
        if episode_actions:
            actions_array = np.array(episode_actions)
            action_stats = {
                'wait_weight_mean': np.mean(actions_array[:, 0]),      # 等待时间权重
                'wait_weight_std': np.std(actions_array[:, 0]),
                'energy_weight_mean': np.mean(actions_array[:, 1]),    # 能量需求权重
                'energy_weight_std': np.std(actions_array[:, 1]),
                'power_pref_mean': np.mean(actions_array[:, 2]),       # 功率偏好
                'power_pref_std': np.std(actions_array[:, 2]),
            }
            action_stats_record.append(action_stats)
        
        # 训练更新
        episode_q_losses = []
        episode_mu_losses = []
        
        if memory.size() > config['memory_threshold']:
            for _ in range(config['training_iterations']):
                q_loss, mu_loss = train_step(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer, config)
                episode_q_losses.append(q_loss)
                episode_mu_losses.append(mu_loss)
                soft_update(mu, mu_target, config['tau'])
                soft_update(q, q_target, config['tau'])
        
        # 记录平均损失
        q_loss_record.append(np.mean(episode_q_losses) if episode_q_losses else 0)
        mu_loss_record.append(np.mean(episode_mu_losses) if episode_mu_losses else 0)
        
        # 周期性打印
        if n_epi % config['print_interval'] == 0 and n_epi != 0:
            avg_score = np.mean(score_record[-config['print_interval']:])
            avg_q_loss = np.mean(q_loss_record[-config['print_interval']:]) if q_loss_record else 0
            avg_mu_loss = np.mean(mu_loss_record[-config['print_interval']:]) if mu_loss_record else 0
            
            # 获取最近的动作统计（特征权重）
            recent_actions = action_stats_record[-config['print_interval']:] if action_stats_record else []
            if recent_actions:
                avg_wait_weight = np.mean([s['wait_weight_mean'] for s in recent_actions])
                avg_energy_weight = np.mean([s['energy_weight_mean'] for s in recent_actions])
                avg_power_pref = np.mean([s['power_pref_mean'] for s in recent_actions])
            else:
                avg_wait_weight = 0.5
                avg_energy_weight = 0.5
                avg_power_pref = 0.5
            
            print(f"Episode {n_epi}: Score={avg_score:.2f}, Q_loss={avg_q_loss:.4f}, "
                  f"Mu_loss={avg_mu_loss:.4f}, Wait_w={avg_wait_weight:.3f}, "
                  f"Energy_w={avg_energy_weight:.3f}, Power={avg_power_pref:.3f}")
    
    # 保存模型
    model_save_dir = os.path.join(save_dir, exp_config['name'])
    os.makedirs(model_save_dir, exist_ok=True)
    
    torch.save(mu.state_dict(), os.path.join(model_save_dir, 'mu.pth'))
    torch.save(mu_target.state_dict(), os.path.join(model_save_dir, 'mu_target.pth'))
    torch.save(q.state_dict(), os.path.join(model_save_dir, 'q.pth'))
    torch.save(q_target.state_dict(), os.path.join(model_save_dir, 'q_target.pth'))
    
    # 构建训练统计
    training_stats = {
        'score_record': score_record,
        'q_loss_record': q_loss_record,
        'mu_loss_record': mu_loss_record,
        'action_stats_record': action_stats_record,
    }
    
    # 保存训练曲线数据
    np.save(os.path.join(model_save_dir, 'score_record.npy'), np.array(score_record))
    np.save(os.path.join(model_save_dir, 'q_loss_record.npy'), np.array(q_loss_record))
    np.save(os.path.join(model_save_dir, 'mu_loss_record.npy'), np.array(mu_loss_record))
    
    print(f"Model saved to: {model_save_dir}")
    print(f"Final average score (last 20 episodes): {np.mean(score_record[-20:]):.4f}")
    print(f"Final Q-loss (last 20 episodes): {np.mean(q_loss_record[-20:]):.6f}")
    print(f"Final Mu-loss (last 20 episodes): {np.mean(mu_loss_record[-20:]):.6f}")
    
    return mu, training_stats, env


def test_runtime(mu, env, config, exp_config, num_runs=100):
    """测试运行时间"""
    print(f"\n{'='*60}")
    print(f"Runtime Testing: {exp_config['name']}")
    print(f"n_stations: {exp_config['n_stations']}")
    print(f"Number of timing runs: {num_runs}")
    print(f"{'='*60}")
    
    mu.eval()
    timing_result = TimingResult()
    
    # 重置环境
    s = env.reset()
    
    for run in range(num_runs):
        # 确保环境有有效状态
        if env._count_valid_vehicles() == 0 or env._count_available_stations() == 0:
            s = env.reset()
        
        # 策略网络前向推理计时
        policy_start = time.perf_counter()
        with torch.no_grad():
            a = mu(torch.from_numpy(s).float())
            a = torch.clamp(a, 0, 1)
            if len(a.shape) > 1 and a.shape[0] == 1:
                a = a.squeeze(0)
        policy_time = time.perf_counter() - policy_start
        
        # 动作投影（包含匈牙利算法和QP）计时
        action, timing = action_projection_with_timing(env, a, measure_timing=True)
        
        # 记录时间
        total_time = policy_time + timing['hungarian'] + timing['qp']
        timing_result.add_timing(
            policy_time=policy_time,
            hungarian_time=timing['hungarian'],
            qp_time=timing['qp'],
            total_time=total_time
        )
        
        # 执行动作以更新环境状态
        try:
            s_prime, r, done, info = env.step(action)
            s = s_prime
            if done:
                s = env.reset()
        except:
            s = env.reset()
    
    stats = timing_result.get_statistics()
    
    print(f"\nTiming Results (in milliseconds):")
    print(f"  Policy Network Forward:  {stats['policy_network']['mean']:.4f} ± {stats['policy_network']['std']:.4f} ms")
    print(f"  Hungarian Algorithm:     {stats['hungarian']['mean']:.4f} ± {stats['hungarian']['std']:.4f} ms")
    print(f"  QP Solver:               {stats['qp']['mean']:.4f} ± {stats['qp']['std']:.4f} ms")
    print(f"  Total Decision Time:     {stats['total']['mean']:.4f} ± {stats['total']['std']:.4f} ms")
    
    return timing_result, stats


def run_all_experiments(save_dir):
    """运行所有实验"""
    print(f"\n{'='*80}")
    print(f"LIRL Runtime Scaling Experiment")
    print(f"{'='*80}")
    print(f"Experiment configurations:")
    for name, cfg in EXPERIMENT_CONFIGS.items():
        print(f"  {name}: n_stations={cfg['n_stations']}, arrival_rate={cfg['arrival_rate']}")
    print(f"{'='*80}")
    
    results = {}
    
    # Phase 1: 训练所有策略
    print(f"\n{'#'*80}")
    print(f"# PHASE 1: TRAINING POLICIES")
    print(f"{'#'*80}")
    
    trained_models = {}
    training_stats_dict = {}  # 新格式：包含score, q_loss, mu_loss等
    envs = {}
    
    for exp_name, exp_config in EXPERIMENT_CONFIGS.items():
        mu, training_stats, env = train_policy(BASE_CONFIG, exp_config, save_dir)
        trained_models[exp_name] = mu
        training_stats_dict[exp_name] = training_stats
        envs[exp_name] = env
    
    # Phase 2: 测试运行时间
    print(f"\n{'#'*80}")
    print(f"# PHASE 2: RUNTIME TESTING")
    print(f"{'#'*80}")
    
    timing_results = {}
    timing_stats = {}
    
    for exp_name, exp_config in EXPERIMENT_CONFIGS.items():
        mu = trained_models[exp_name]
        
        # 创建新的环境用于测试
        test_env = EVChargingEnv(
            n_stations=exp_config['n_stations'],
            p_max=exp_config['p_max'],
            arrival_rate=exp_config['arrival_rate']
        )
        
        timing_result, stats = test_runtime(
            mu, test_env, BASE_CONFIG, exp_config, 
            num_runs=BASE_CONFIG['num_timing_runs']
        )
        
        timing_results[exp_name] = timing_result
        timing_stats[exp_name] = stats
    
    # 保存结果
    results = {
        'timing_stats': timing_stats,
        'training_stats': training_stats_dict,  # 新格式
        'experiment_configs': EXPERIMENT_CONFIGS,
        'base_config': BASE_CONFIG
    }
    
    return results, timing_results, training_stats_dict


def save_results(results, timing_results, save_dir):
    """保存实验结果"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. 保存运行时间统计到CSV
    timing_csv_path = os.path.join(save_dir, f'runtime_timing_{timestamp}.csv')
    with open(timing_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Scale', 'n_stations', 'Component', 'Mean (ms)', 'Std (ms)', 'Min (ms)', 'Max (ms)'])
        
        for exp_name, stats in results['timing_stats'].items():
            n_stations = EXPERIMENT_CONFIGS[exp_name]['n_stations']
            for component in ['policy_network', 'hungarian', 'qp', 'total']:
                writer.writerow([
                    exp_name,
                    n_stations,
                    component,
                    f"{stats[component]['mean']:.6f}",
                    f"{stats[component]['std']:.6f}",
                    f"{stats[component]['min']:.6f}",
                    f"{stats[component]['max']:.6f}"
                ])
    
    print(f"Timing results saved to: {timing_csv_path}")
    
    # 2. 保存汇总表格
    summary_csv_path = os.path.join(save_dir, f'runtime_summary_{timestamp}.csv')
    with open(summary_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['n_stations', 'Policy Network (ms)', 'Hungarian (ms)', 'QP (ms)', 'Total (ms)'])
        
        for exp_name in ['small', 'medium', 'large']:
            stats = results['timing_stats'][exp_name]
            n_stations = EXPERIMENT_CONFIGS[exp_name]['n_stations']
            writer.writerow([
                n_stations,
                f"{stats['policy_network']['mean']:.4f}",
                f"{stats['hungarian']['mean']:.4f}",
                f"{stats['qp']['mean']:.4f}",
                f"{stats['total']['mean']:.4f}"
            ])
    
    print(f"Summary saved to: {summary_csv_path}")
    
    # 3. 保存详细结果到JSON
    json_path = os.path.join(save_dir, f'experiment_results_{timestamp}.json')
    
    # 转换numpy数组为列表
    json_results = {
        'timing_stats': results['timing_stats'],
        'experiment_configs': results['experiment_configs'],
        'base_config': {k: v for k, v in results['base_config'].items() if not callable(v)},
        'timestamp': timestamp
    }
    
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"Detailed results saved to: {json_path}")
    
    return timing_csv_path, summary_csv_path, json_path


def plot_results(results, save_dir):
    """绘制实验结果图表"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 准备数据
    scales = ['small', 'medium', 'large']
    n_stations_list = [EXPERIMENT_CONFIGS[s]['n_stations'] for s in scales]
    
    policy_times = [results['timing_stats'][s]['policy_network']['mean'] for s in scales]
    hungarian_times = [results['timing_stats'][s]['hungarian']['mean'] for s in scales]
    qp_times = [results['timing_stats'][s]['qp']['mean'] for s in scales]
    total_times = [results['timing_stats'][s]['total']['mean'] for s in scales]
    
    policy_stds = [results['timing_stats'][s]['policy_network']['std'] for s in scales]
    hungarian_stds = [results['timing_stats'][s]['hungarian']['std'] for s in scales]
    qp_stds = [results['timing_stats'][s]['qp']['std'] for s in scales]
    total_stds = [results['timing_stats'][s]['total']['std'] for s in scales]
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 各组件运行时间对比（柱状图）
    ax1 = axes[0, 0]
    x = np.arange(len(n_stations_list))
    width = 0.2
    
    bars1 = ax1.bar(x - 1.5*width, policy_times, width, yerr=policy_stds, label='Policy Network', capsize=3)
    bars2 = ax1.bar(x - 0.5*width, hungarian_times, width, yerr=hungarian_stds, label='Hungarian', capsize=3)
    bars3 = ax1.bar(x + 0.5*width, qp_times, width, yerr=qp_stds, label='QP', capsize=3)
    bars4 = ax1.bar(x + 1.5*width, total_times, width, yerr=total_stds, label='Total', capsize=3)
    
    ax1.set_xlabel('Number of Stations')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('Runtime Components Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(n_stations_list)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 总时间随规模变化（折线图）
    ax2 = axes[0, 1]
    ax2.errorbar(n_stations_list, total_times, yerr=total_stds, marker='o', linewidth=2, capsize=5, label='Total Decision Time')
    ax2.set_xlabel('Number of Stations')
    ax2.set_ylabel('Time (ms)')
    ax2.set_title('Total Decision Time vs. Scale')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. 各组件时间随规模变化（折线图）
    ax3 = axes[1, 0]
    ax3.errorbar(n_stations_list, policy_times, yerr=policy_stds, marker='o', linewidth=2, capsize=3, label='Policy Network')
    ax3.errorbar(n_stations_list, hungarian_times, yerr=hungarian_stds, marker='s', linewidth=2, capsize=3, label='Hungarian')
    ax3.errorbar(n_stations_list, qp_times, yerr=qp_stds, marker='^', linewidth=2, capsize=3, label='QP')
    ax3.set_xlabel('Number of Stations')
    ax3.set_ylabel('Time (ms)')
    ax3.set_title('Component Runtime vs. Scale')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 堆叠柱状图显示时间组成
    ax4 = axes[1, 1]
    ax4.bar(n_stations_list, policy_times, label='Policy Network', color='C0')
    ax4.bar(n_stations_list, hungarian_times, bottom=policy_times, label='Hungarian', color='C1')
    bottom_for_qp = [p + h for p, h in zip(policy_times, hungarian_times)]
    ax4.bar(n_stations_list, qp_times, bottom=bottom_for_qp, label='QP', color='C2')
    ax4.set_xlabel('Number of Stations')
    ax4.set_ylabel('Time (ms)')
    ax4.set_title('Runtime Composition')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # 保存图表
    plot_path = os.path.join(save_dir, f'runtime_scaling_plot_{timestamp}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")
    
    plt.show()
    
    return plot_path


def plot_training_curves(training_stats_dict, save_dir):
    """
    绘制详细的训练曲线
    
    包含:
    1. Episode Score (奖励)
    2. Q-loss (Critic损失)
    3. Mu-loss (Actor损失)
    4. Policy Output Statistics (策略输出统计)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 创建2x2的子图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = ['C0', 'C1', 'C2']
    
    # ========== 1. Episode Score ==========
    ax1 = axes[0, 0]
    for idx, (exp_name, stats) in enumerate(training_stats_dict.items()):
        n_stations = EXPERIMENT_CONFIGS[exp_name]['n_stations']
        scores = stats['score_record']
        episodes = range(len(scores))
        
        # 原始曲线（透明）
        ax1.plot(episodes, scores, alpha=0.2, color=colors[idx])
        
        # 移动平均
        window = 20
        if len(scores) >= window:
            moving_avg = pd.Series(scores).rolling(window=window).mean()
            ax1.plot(episodes, moving_avg, linewidth=2, color=colors[idx], 
                    label=f'n={n_stations} (avg={np.mean(scores[-20:]):.1f})')
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Episode Score')
    ax1.set_title('Training Score Curves')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # ========== 2. Q-loss (Critic Loss) ==========
    ax2 = axes[0, 1]
    for idx, (exp_name, stats) in enumerate(training_stats_dict.items()):
        n_stations = EXPERIMENT_CONFIGS[exp_name]['n_stations']
        q_losses = stats['q_loss_record']
        episodes = range(len(q_losses))
        
        # 过滤掉0值（训练开始前）
        valid_idx = [i for i, loss in enumerate(q_losses) if loss > 0]
        valid_losses = [q_losses[i] for i in valid_idx]
        
        if valid_losses:
            ax2.plot(valid_idx, valid_losses, alpha=0.2, color=colors[idx])
            
            window = 20
            if len(valid_losses) >= window:
                moving_avg = pd.Series(valid_losses).rolling(window=window).mean()
                ax2.plot(valid_idx, moving_avg.values, linewidth=2, color=colors[idx],
                        label=f'n={n_stations}')
    
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Q-Loss (Critic)')
    ax2.set_title('Critic Loss Curves')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')  # 使用对数刻度更易观察
    
    # ========== 3. Mu-loss (Actor Loss) ==========
    ax3 = axes[1, 0]
    for idx, (exp_name, stats) in enumerate(training_stats_dict.items()):
        n_stations = EXPERIMENT_CONFIGS[exp_name]['n_stations']
        mu_losses = stats['mu_loss_record']
        episodes = range(len(mu_losses))
        
        # 过滤掉0值
        valid_idx = [i for i, loss in enumerate(mu_losses) if loss != 0]
        valid_losses = [mu_losses[i] for i in valid_idx]
        
        if valid_losses:
            ax3.plot(valid_idx, valid_losses, alpha=0.2, color=colors[idx])
            
            window = 20
            if len(valid_losses) >= window:
                moving_avg = pd.Series(valid_losses).rolling(window=window).mean()
                ax3.plot(valid_idx, moving_avg.values, linewidth=2, color=colors[idx],
                        label=f'n={n_stations}')
    
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Mu-Loss (Actor)')
    ax3.set_title('Actor Loss Curves (Policy Gradient)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # ========== 4. Policy Output (Feature Weights) ==========
    ax4 = axes[1, 1]
    for idx, (exp_name, stats) in enumerate(training_stats_dict.items()):
        n_stations = EXPERIMENT_CONFIGS[exp_name]['n_stations']
        action_stats = stats['action_stats_record']
        
        if action_stats:
            episodes = range(len(action_stats))
            # 兼容旧数据格式
            if 'wait_weight_mean' in action_stats[0]:
                wait_weights = [s['wait_weight_mean'] for s in action_stats]
                energy_weights = [s['energy_weight_mean'] for s in action_stats]
            else:
                wait_weights = [s.get('station_pref_mean', 0.5) for s in action_stats]
                energy_weights = [s.get('vehicle_pref_mean', 0.5) for s in action_stats]
            power_prefs = [s['power_pref_mean'] for s in action_stats]
            
            # Wait weight (实线)
            if len(wait_weights) >= 20:
                moving_avg = pd.Series(wait_weights).rolling(window=20).mean()
                ax4.plot(episodes, moving_avg, linewidth=2, color=colors[idx],
                        label=f'n={n_stations} wait_w', linestyle='-')
            
            # Energy weight (虚线)
            if len(energy_weights) >= 20:
                moving_avg = pd.Series(energy_weights).rolling(window=20).mean()
                ax4.plot(episodes, moving_avg, linewidth=2, color=colors[idx],
                        label=f'n={n_stations} energy_w', linestyle='--', alpha=0.7)
            
            # Power preference (点线)
            if len(power_prefs) >= 20:
                moving_avg = pd.Series(power_prefs).rolling(window=20).mean()
                ax4.plot(episodes, moving_avg, linewidth=2, color=colors[idx],
                        label=f'n={n_stations} power', linestyle=':', alpha=0.7)
    
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Feature Weights (0-1)')
    ax4.set_title('Policy Network Output: Feature Weights Evolution')
    ax4.legend(loc='upper right', fontsize=7, ncol=2)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1])
    
    plt.tight_layout()
    
    plot_path = os.path.join(save_dir, f'training_curves_{timestamp}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to: {plot_path}")
    
    plt.show()
    
    return plot_path


def print_final_summary(results):
    """打印最终实验总结"""
    print(f"\n{'='*80}")
    print(f"FINAL EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    
    print(f"\nRuntime Results (in milliseconds):")
    print(f"{'Scale':<15} {'n_stations':<12} {'Policy Net':<15} {'Hungarian':<15} {'QP':<15} {'Total':<15}")
    print(f"{'-'*87}")
    
    for exp_name in ['small', 'medium', 'large']:
        stats = results['timing_stats'][exp_name]
        n_stations = EXPERIMENT_CONFIGS[exp_name]['n_stations']
        print(f"{exp_name:<15} {n_stations:<12} "
              f"{stats['policy_network']['mean']:.4f}±{stats['policy_network']['std']:.4f}  "
              f"{stats['hungarian']['mean']:.4f}±{stats['hungarian']['std']:.4f}  "
              f"{stats['qp']['mean']:.4f}±{stats['qp']['std']:.4f}  "
              f"{stats['total']['mean']:.4f}±{stats['total']['std']:.4f}")
    
    print(f"\n{'='*80}")
    print(f"Scaling Analysis:")
    
    # 计算扩展性
    small_total = results['timing_stats']['small']['total']['mean']
    medium_total = results['timing_stats']['medium']['total']['mean']
    large_total = results['timing_stats']['large']['total']['mean']
    
    print(f"  Small (5) -> Medium (50):  {medium_total/small_total:.2f}x slowdown")
    print(f"  Small (5) -> Large (100):  {large_total/small_total:.2f}x slowdown")
    print(f"  Medium (50) -> Large (100): {large_total/medium_total:.2f}x slowdown")
    
    print(f"\n{'='*80}")


def main():
    """主函数"""
    # 创建保存目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(os.path.dirname(__file__), f'runtime_scaling_exp_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Results will be saved to: {save_dir}")
    
    # 运行实验
    results, timing_results, training_stats_dict = run_all_experiments(save_dir)
    
    # 保存结果
    save_results(results, timing_results, save_dir)
    
    # 绘制图表
    plot_results(results, save_dir)
    plot_training_curves(training_stats_dict, save_dir)
    
    # 打印最终总结
    print_final_summary(results)
    
    print(f"\n{'='*80}")
    print(f"Experiment completed!")
    print(f"All results saved to: {save_dir}")
    print(f"{'='*80}")
    
    return results


if __name__ == "__main__":
    main()
