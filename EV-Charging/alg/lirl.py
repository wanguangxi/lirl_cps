import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import math
from scipy.optimize import linear_sum_assignment
import sys
import os
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import pandas as pd
from datetime import datetime, timedelta
sys.path.append(os.path.join(os.path.dirname(__file__), '../env'))
from ev import EVChargingEnv

# =======================
# HYPERPARAMETERS CONFIG
# =======================
CONFIG = {
    # Learning parameters
    'lr_mu': 0.0005,
    'lr_q': 0.001,
    'gamma': 0.98,
    'batch_size': 128,
    'buffer_limit': 1000000,
    'tau': 0.005,  # for target network soft update
    
    # Environment parameters
    'n_stations': 5,  # 充电桩数量
    'p_max': 150.0,    # 最大功率
    'arrival_rate': 0.75,  # 车辆到达率
    'num_of_episodes': 200,
    
    # Network architecture
    'hidden_dim1': 128,
    'hidden_dim2': 64,
    'critic_hidden': 32,
    
    # Training parameters
    'memory_threshold': 500,
    'training_iterations': 20,
    'noise_params': {'theta': 0.1, 'dt': 0.05, 'sigma': 0.1},
    
    # Multi-run training parameters
    'enable_multi_run': True,  # Enable multi-run training by default
    # 'seeds': [3047,294,714],  # Multiple random seeds for training
    'seeds': [3047,294,714,1092,1386,2856,42,114514,2025,1993],  # Multiple random seeds for training
    # 'seeds': [3047,294],  # Multiple random seeds for training

    'num_runs': 10,  # Number of training runs (usually equals len(seeds))
    
    # Testing parameters
    'max_test_steps': 288,  # 一天的最大步数
    
    # Output parameters
    'print_interval': 10,
    'enable_gantt_plots': False,  # Set to True to enable real-time plotting
    'plot_training_curve': True,
    'save_models': True,
}


class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=CONFIG['buffer_limit'])

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
        
        # 先转换为numpy数组，再转换为tensor，提高效率
        s_lst_= torch.FloatTensor(np.array(s_lst))
        a_lst_= torch.tensor(np.array(a_lst), dtype=torch.float)
        r_lst_= torch.tensor(np.array(r_lst), dtype=torch.float)
        s_prime_lst_ = torch.tensor(np.array(s_prime_lst), dtype=torch.float)
        done_mask_lst_ = torch.tensor(np.array(done_mask_lst), dtype=torch.float)

        return s_lst_,a_lst_,r_lst_,s_prime_lst_,done_mask_lst_
    
    def size(self):
        return len(self.buffer)
    

class MuNet(nn.Module):
    def __init__(self, state_size, action_size):
        super(MuNet, self).__init__()
        self.fc1 = nn.Linear(state_size, CONFIG['hidden_dim1'])
        self.fc2 = nn.Linear(CONFIG['hidden_dim1'], CONFIG['hidden_dim2'])
        self.fc_mu = nn.Linear(CONFIG['hidden_dim2'], action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.fc_mu(x)
        # 对不同的动作分量使用不同的激活函数
        # station_id 和 vehicle_id 使用 sigmoid (0-1)
        # power 已经在环境中限制在 50-150 范围内
        if len(mu.shape) == 1:
            mu = mu.unsqueeze(0)
        mu[:, :2] = torch.sigmoid(mu[:, :2])  # station_id, vehicle_id
        mu[:, 2:] = torch.sigmoid(mu[:, 2:])  # power (将映射到0-1，后续转换到50-150)
        return mu


class QNet(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNet, self).__init__()
        self.fc_s = nn.Linear(state_size, CONFIG['hidden_dim2'])
        self.fc_a = nn.Linear(action_size, CONFIG['hidden_dim2'])
        self.fc_q = nn.Linear(CONFIG['hidden_dim2'] * 2, CONFIG['critic_hidden'])
        self.fc_out = nn.Linear(CONFIG['critic_hidden'], 1)

    def forward(self, x, a):
        h1 = F.relu(self.fc_s(x))
        h2 = F.relu(self.fc_a(a))
        cat = torch.cat([h1, h2], dim=1)
        q = F.relu(self.fc_q(cat))
        q = self.fc_out(q)
        return q


class OrnsteinUhlenbeckNoise:
    def __init__(self, mu):
        self.theta = CONFIG['noise_params']['theta']
        self.dt = CONFIG['noise_params']['dt']
        self.sigma = CONFIG['noise_params']['sigma']
        self.mu = mu
        self.x_prev = np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x


def action_projection_ev(env, a):
    """
    将神经网络输出投影到有效的充电站调度动作
    env: EVChargingEnv环境对象
    a: 神经网络输出的动作向量 [station_prob, vehicle_prob, power_prob]
    返回: {'station_id': int, 'vehicle_id': int, 'power': np.array([float])}
    """
    a_ = a.detach().cpu().numpy()
    
    # 确保 a_ 是一维数组
    if len(a_.shape) > 1:
        a_ = a_.squeeze()
    
    # 检查动作维度
    if len(a_) < 3:
        # print(f"Warning: action dimension is {len(a_)}, expected 3. Using defaults.")
        # 扩展到3维
        if len(a_) == 1:
            a_ = np.array([a_[0], 0.5, 0.5])
        elif len(a_) == 2:
            a_ = np.array([a_[0], a_[1], 0.5])
    
    # 创建当前环境状态的快照，避免状态变化导致的不一致
    current_station_status = env.station_status.copy()
    current_vehicles_snapshot = []
    
    # 1. 获取可用的充电桩 (station_status == 1 表示可用)
    valid_stations = []
    for i in range(env.n_stations):
        if current_station_status[i] == 1:  # 充电桩可用
            valid_stations.append(i)
    
    # 2. 获取有效的车辆 (存在且未充电且未充满) - 使用快照避免状态变化
    valid_vehicles = []
    for i in range(env.max_vehicles):
        if env.vehicles[i] is not None:  # 车辆存在
            vehicle = env.vehicles[i]
            # 车辆未在充电且未充满
            if not vehicle['charging'] and not vehicle['fully_charged']:
                valid_vehicles.append(i)
                # 保存车辆快照用于后续验证
                current_vehicles_snapshot.append({
                    'id': i,
                    'charging': vehicle['charging'],
                    'fully_charged': vehicle['fully_charged'],
                    'exists': True
                })
    
    # 3. 如果没有可用充电桩或有效车辆，返回安全的默认动作
    if len(valid_stations) == 0:
        # print("Warning: No available charging stations")
        # 选择一个存在的车辆ID，即使动作会失败
        safe_vehicle_id = -1  # 默认为无效ID
        for i in range(env.max_vehicles):
            if env.vehicles[i] is not None:
                safe_vehicle_id = i
                break
        
        # 如果没有找到任何存在的车辆，使用0但这会导致约束违反
        if safe_vehicle_id == -1:
            safe_vehicle_id = 0
            
        return {
            'station_id': 0,  # 将被约束检查拒绝
            'vehicle_id': safe_vehicle_id,
            'power': np.array([100.0], dtype=np.float32)
        }
    
    if len(valid_vehicles) == 0:
        # print("Warning: No valid vehicles for charging")
        # 选择一个存在的车辆ID，即使动作会失败
        safe_vehicle_id = -1  # 默认为无效ID
        for i in range(env.max_vehicles):
            if env.vehicles[i] is not None:
                safe_vehicle_id = i
                break
        
        # 如果没有找到任何存在的车辆，使用0但这会导致约束违反
        if safe_vehicle_id == -1:
            safe_vehicle_id = 0
            
        return {
            'station_id': valid_stations[0] if valid_stations else 0,
            'vehicle_id': safe_vehicle_id,
            'power': np.array([100.0], dtype=np.float32)
        }
    
    # 4. 使用匈牙利算法进行最优分配
    cost_matrix = np.zeros((len(valid_vehicles), len(valid_stations)))
    
    for i, vehicle_id in enumerate(valid_vehicles):
        # 双重检查车辆是否仍然存在
        if env.vehicles[vehicle_id] is None:
            # print(f"Warning: Vehicle {vehicle_id} disappeared during action projection")
            continue
            
        vehicle = env.vehicles[vehicle_id]
        wait_time = vehicle['wait_time']
        energy_needed = vehicle['energy_required'] - vehicle['energy_charged']
        
        for j, station_id in enumerate(valid_stations):
            # 基于等待时间和能量需求构建成本
            wait_cost = -wait_time * 10  # 等待时间权重
            energy_cost = -energy_needed * 0.1
            
            # 结合网络输出的偏好
            station_preference = abs(a_[0] - (station_id / max(1, env.n_stations - 1)))
            vehicle_preference = abs(a_[1] - (vehicle_id / max(1, env.max_vehicles - 1)))
            preference_cost = (station_preference + vehicle_preference) * 5
            
            cost_matrix[i, j] = wait_cost + energy_cost + preference_cost
    
    # 5. 求解最优分配
    try:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        if len(row_ind) > 0:
            vehicle_id = valid_vehicles[row_ind[0]]
            station_id = valid_stations[col_ind[0]]
        else:
            vehicle_id = valid_vehicles[0]
            station_id = valid_stations[0]
    except Exception as e:
        # print(f"Warning: Hungarian algorithm failed: {e}")
        vehicle_id = valid_vehicles[0] if valid_vehicles else 0
        station_id = valid_stations[0] if valid_stations else 0
    
    # 6. 执行前最终安全检查 - 防止竞态条件
    final_valid = True
    
    # 检查充电桩状态（可能在自动推进中发生变化）
    if station_id >= env.n_stations or env.station_status[station_id] != 1:
        # print(f"Warning: Selected station {station_id} is no longer available")
        # 寻找替代充电桩
        for alt_station in range(env.n_stations):
            if env.station_status[alt_station] == 1:
                station_id = alt_station
                # print(f"Using alternative station {station_id}")
                break
        else:
            final_valid = False
    
    # 检查车辆状态（关键：防止车辆在执行前消失）
    if (vehicle_id >= env.max_vehicles or 
        env.vehicles[vehicle_id] is None or 
        env.vehicles[vehicle_id]['charging'] or 
        env.vehicles[vehicle_id]['fully_charged']):
        
        # print(f"Warning: Selected vehicle {vehicle_id} is no longer valid")
        # 寻找替代车辆
        alternative_found = False
        for alt_vehicle in range(env.max_vehicles):
            if (env.vehicles[alt_vehicle] is not None and
                not env.vehicles[alt_vehicle]['charging'] and
                not env.vehicles[alt_vehicle]['fully_charged']):
                vehicle_id = alt_vehicle
                # print(f"Using alternative vehicle {vehicle_id}")
                alternative_found = True
                break
        
        if not alternative_found:
            # print("No alternative valid vehicle found")
            # 最后的安全检查：选择任何存在的车辆，即使会导致约束违反
            found_existing_vehicle = False
            for i in range(env.max_vehicles):
                if env.vehicles[i] is not None:
                    vehicle_id = i
                    found_existing_vehicle = True
                    break
            
            # 如果真的没有任何车辆存在，这是一个异常情况
            if not found_existing_vehicle:
                # 这种情况下无论选择什么都会违反约束
                vehicle_id = 0  # 保持原值，让约束检查处理
                
            final_valid = False
    
    # 7. 将功率从[0,1]映射到[50,150]
    power_normalized = np.clip(a_[2] if len(a_) > 2 else 0.5, 0, 1)
    power = 50.0 + power_normalized * 100.0  # [50, 150]
    
    # 确保功率在环境限制范围内
    power = np.clip(power, 50.0, min(150.0, env.p_max))
    power = 150.00
    
    # 8. 执行前的最终车辆存在性验证
    if vehicle_id < env.max_vehicles and env.vehicles[vehicle_id] is not None:
        # 车辆存在，可以使用
        pass
    else:
        # 车辆不存在，寻找任何存在的车辆
        found_existing = False
        for i in range(env.max_vehicles):
            if env.vehicles[i] is not None:
                vehicle_id = i
                found_existing = True
                break
        
        # 如果没有找到任何存在的车辆，保持原值让约束检查处理
        if not found_existing:
            vehicle_id = 0  # 这会导致约束违反，但至少不会崩溃
    
    action = {
        'station_id': station_id,
        'vehicle_id': vehicle_id,
        'power': np.array([power], dtype=np.float32)
    }
    
    # 9. 最终动作验证日志 - 只保留约束违反相关信息
    # if not final_valid:
    #     print(f"Warning: Returning potentially invalid action due to state changes: {action}")
    #     print(f"Current available stations: {sum(1 for i in range(env.n_stations) if env.station_status[i] == 1)}")
    #     print(f"Current valid vehicles: {sum(1 for i in range(env.max_vehicles) if env.vehicles[i] is not None and not env.vehicles[i]['charging'] and not env.vehicles[i]['fully_charged'])}")
    
    return action


def train(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer):
    s, a, r, s_prime, done_mask = memory.sample(CONFIG['batch_size'])
    
    # 处理动作维度
    if len(a.shape) == 1:
        a = a.unsqueeze(1)
    
    target = torch.unsqueeze(r, dim=1) + CONFIG['gamma'] * q_target(s_prime, mu_target(s_prime)).mul(done_mask) 
    q_loss = F.smooth_l1_loss(q(s, a), target.detach())
    q_optimizer.zero_grad()
    q_loss.backward()
    q_optimizer.step()
    
    mu_loss = -q(s, mu(s)).mean()  # That's all for the policy loss.
    mu_optimizer.zero_grad()
    mu_loss.backward()
    mu_optimizer.step()


def soft_update(net, net_target):
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * (1.0 - CONFIG['tau']) + param.data * CONFIG['tau'])


def main(config=None):
    """Main training function"""
    if config is None:
        config = CONFIG
    
    # Environment setup
    env = EVChargingEnv(
        n_stations=config['n_stations'],
        p_max=config['p_max'],
        arrival_rate=config['arrival_rate']
    )
    state_size = env.observation_space.shape[0]
    action_size = 3  # station_id选择, vehicle_id选择, power大小
    
    # Network initialization
    q, q_target = QNet(state_size, action_size), QNet(state_size, action_size)
    q_target.load_state_dict(q.state_dict())
    mu, mu_target = MuNet(state_size, action_size), MuNet(state_size, action_size)
    mu_target.load_state_dict(mu.state_dict())

    # Optimizer setup
    mu_optimizer = optim.Adam(mu.parameters(), lr=config['lr_mu'])
    q_optimizer = optim.Adam(q.parameters(), lr=config['lr_q'])
    
    # Training components
    ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(action_size))
    memory = ReplayBuffer()
    
    # Training variables
    action_restore = []
    score_record = []
    
    # Episode statistics tracking
    episode_stats = {
        'fully_charged_vehicles': [],  # 每个episode充满的车辆数
        'cumulative_revenue': [],      # 每个episode的累计收益
        'energy_delivered': [],        # 每个episode的能量输送
        'average_power': [],           # 每个episode的平均功率
        'station_utilization': [],     # 每个episode的充电桩利用率
        'total_arrivals': [],          # 每个episode进站的车辆总数
        'vehicles_charged': [],        # 每个episode充满电离开的车辆数
        'vehicles_left_uncharged': [], # 每个episode未充满就离开的车辆数
        'charging_success_rate': []    # 每个episode的充电成功率
    }
    
    # Constraint violation tracking
    constraint_violations = {
        'total_violations': 0,
        'episode_violations': [],
        'violation_rate': [],
        'violation_types': {},  # 统计不同类型的违反
        'violation_details': []  # 保存详细违反信息
    }

    print(f"Starting DDPG-LIRL training for EV Charging Station:")
    print(f"Stations: {config['n_stations']}, Max Power: {config['p_max']}kW")
    print(f"Episodes: {config['num_of_episodes']}")

    for n_epi in range(config['num_of_episodes']):
        s = env.reset()
        done = False
        action_eps = []
        step = 0
        episode_reward = 0
        episode_violations = 0
        
        while not done:
            # 获取动作
            a = mu(torch.from_numpy(s).float())
            # 添加噪声用于探索
            noise = torch.from_numpy(ou_noise()).float()
            a = a + noise * max(0.1, 1.0 - n_epi / 500)  # 噪声随训练递减
            a = torch.clamp(a, 0, 1)
            a = a.squeeze() if len(a.shape) > 1 else a
            
            # 投影到有效动作
            action = action_projection_ev(env, a)
            
            # 执行动作
            s_prime, r, done, info = env.step(action)
            
            # 根据环境返回的约束违反信息进行统计，而不是通过奖励判断
            violation_info = info.get('constraint_violation', None)
            if violation_info and violation_info['has_violation']:
                episode_violations += 1
                constraint_violations['total_violations'] += 1
                
                violation_type = violation_info['violation_type']
                # 统计不同类型的违反
                if violation_type not in constraint_violations['violation_types']:
                    constraint_violations['violation_types'][violation_type] = 0
                constraint_violations['violation_types'][violation_type] += 1
                
                # 保存详细信息用于分析
                constraint_violations['violation_details'].append({
                    'episode': n_epi,
                    'step': step,
                    'violation_type': violation_type,
                    'violation_details': violation_info['violation_details'],
                    'attempted_action': violation_info['attempted_action'],
                    'reward': r  # 同时记录奖励值
                })
                
                print(f"Episode {n_epi}, Step {step+1} - {violation_type}: {violation_info['violation_details']}")
            
            # 存储经验
            a_store = a.detach().numpy()
            memory.put((s, a_store, r, s_prime, done))
            
            s = s_prime
            step += 1
            episode_reward += r
            action_eps.append(a_store)
            
        action_restore.append(action_eps)    
        score_record.append(episode_reward)
        
        # 统计本episode的车辆进站、充满和未充满离开情况
        fully_charged_count = info.get('episode_charged_count', 0)  # 使用环境提供的统计
        total_arrivals = info.get('episode_arrivals', 1)  # 使用环境提供的统计
        cumulative_revenue = 0.0
        
        # 根据充电记录计算收益（加入寿命损伤成本扣减）
        if hasattr(env, 'charging_records'):
            for record in env.charging_records:
                if (record['start_step'] >= 0 and 
                    record['end_step'] <= env.current_step):
                    # 基础收益 = 能量 * 基础电价 * 时段倍数
                    hour = (record['start_step'] * 5 // 60) % 24
                    price_multiplier = env._get_price_multiplier(hour)
                    revenue = record['energy'] * env.base_price * price_multiplier
                    
                    # 寿命损伤成本（为负值），优先使用 record['damage_delta']；兼容其它字段名
                    damage_delta = record.get('damage_delta')
                    if damage_delta is None:
                        # 备用字段名（若环境以后改名，可在此扩展）
                        damage_delta = record.get('lifetime_damage_delta')
                    
                    lifetime_penalty = 0.0
                    if damage_delta is not None:
                        lifetime_penalty = -damage_delta * 5 # 成本（负数）
                    
                    cumulative_revenue += (revenue + lifetime_penalty)
            
            # 兜底：若逐条记录未提供 damage_delta，但环境维护了总寿命损伤，则统一扣减
            # 仅当循环中未累计任何 lifetime_penalty 时执行（避免双扣）
            if not any(('damage_delta' in r or 'lifetime_damage_delta' in r) for r in env.charging_records):
                if hasattr(env, 'total_lifetime_damage'):
                    cumulative_revenue -= getattr(env, 'total_lifetime_damage', 0.0) * 100
        # 如果没有充电记录，使用备用计算方法
        if cumulative_revenue == 0:
            # 用总收益作为累计收益的近似（收益 = 总能量 - 总成本）
            cumulative_revenue = max(0, info['total_energy'] * env.base_price - info['total_cost'])
        
        # 计算episode车辆统计
        current_vehicles_count = info['num_vehicles']  # 当前仍在场的车辆数
        episode_charged_vehicles = fully_charged_count
        episode_uncharged_left = max(0, total_arrivals - episode_charged_vehicles - current_vehicles_count)
        
        # 计算充电成功率
        charging_success_rate = (episode_charged_vehicles / total_arrivals * 100) if total_arrivals > 0 else 0
        
        # 记录episode统计信息
        episode_stats['fully_charged_vehicles'].append(fully_charged_count)
        episode_stats['cumulative_revenue'].append(cumulative_revenue)
        episode_stats['energy_delivered'].append(info['total_energy'])
        episode_stats['total_arrivals'].append(total_arrivals)
        episode_stats['vehicles_charged'].append(episode_charged_vehicles)
        episode_stats['vehicles_left_uncharged'].append(episode_uncharged_left)
        episode_stats['charging_success_rate'].append(charging_success_rate)
        
        # 计算平均功率
        if hasattr(env, 'charging_records') and len(env.charging_records) > 0:
            avg_power = np.mean([record['power'] for record in env.charging_records])
        else:
            avg_power = 0.0
        episode_stats['average_power'].append(avg_power)
        
        # 计算充电桩利用率（简化版）
        total_possible_time = env.n_stations * env.current_step
        actual_usage_time = sum(1 for i in range(env.n_stations) if env.station_status[i] == 0) * env.current_step
        utilization_rate = (actual_usage_time / total_possible_time * 100) if total_possible_time > 0 else 0
        episode_stats['station_utilization'].append(utilization_rate)
        
        # 记录本轮约束违反次数
        constraint_violations['episode_violations'].append(episode_violations)
        
        # 计算约束违反率（当前episode的违反率）
        violation_rate = episode_violations / step if step > 0 else 0
        constraint_violations['violation_rate'].append(violation_rate)
        
        # Training update
        if memory.size() > config['memory_threshold']:   
            for i in range(config['training_iterations']):
                train(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer)
                soft_update(mu, mu_target)
                soft_update(q, q_target)
        
        if n_epi % config['print_interval'] == 0 and n_epi != 0:
            avg_score = np.mean(score_record[-config['print_interval']:])
            avg_violations = np.mean(constraint_violations['episode_violations'][-config['print_interval']:])
            avg_violation_rate = np.mean(constraint_violations['violation_rate'][-config['print_interval']:]) * 100
            
            # 新增统计信息
            avg_charged_vehicles = np.mean(episode_stats['fully_charged_vehicles'][-config['print_interval']:])
            avg_revenue = np.mean(episode_stats['cumulative_revenue'][-config['print_interval']:])
            avg_energy = np.mean(episode_stats['energy_delivered'][-config['print_interval']:])
            avg_total_arrivals = np.mean(episode_stats['total_arrivals'][-config['print_interval']:])
            avg_uncharged_left = np.mean(episode_stats['vehicles_left_uncharged'][-config['print_interval']:])
            avg_success_rate = np.mean(episode_stats['charging_success_rate'][-config['print_interval']:])
            
            print(f"Episode {n_epi}: Average Score = {avg_score:.4f}, "
                  f"Vehicles: {info['num_vehicles']}, Energy: {info['total_energy']:.2f}kWh")
            print(f"  Performance - Charged Vehicles: {avg_charged_vehicles:.1f}, "
                  f"Revenue: {avg_revenue:.2f}, Avg Energy: {avg_energy:.2f}kWh")
            print(f"  Vehicle Flow - Total Arrivals: {avg_total_arrivals:.1f}, "
                  f"Charged: {avg_charged_vehicles:.1f}, Left Uncharged: {avg_uncharged_left:.1f}, "
                  f"Success Rate: {avg_success_rate:.1f}%")
            print(f"  Constraint Violations - Total: {constraint_violations['total_violations']}, "
                  f"Episode: {episode_violations}, Avg: {avg_violations:.1f}, Rate: {avg_violation_rate:.1f}%")
    
    # 打印最终约束违反统计
    total_steps = sum(len(actions) for actions in action_restore)
    final_violation_rate = (constraint_violations['total_violations'] / total_steps * 100) if total_steps > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"Constraint Violation Statistics:")
    print(f"{'='*60}")
    print(f"Total constraint violations: {constraint_violations['total_violations']}")
    print(f"Total steps taken: {total_steps}")
    print(f"Overall violation rate: {final_violation_rate:.2f}%")
    print(f"Average violations per episode: {np.mean(constraint_violations['episode_violations']):.2f}")
    print(f"Max violations in single episode: {np.max(constraint_violations['episode_violations'])}")
    print(f"Episodes with zero violations: {sum(1 for v in constraint_violations['episode_violations'] if v == 0)}")
    
    # 显示不同类型约束违反的统计
    if constraint_violations['violation_types']:
        print(f"\nViolation Types Breakdown:")
        for violation_type, count in constraint_violations['violation_types'].items():
            percentage = (count / constraint_violations['total_violations']) * 100
            print(f"  {violation_type}: {count} ({percentage:.1f}%)")
    
    # 显示episode统计总结
    if episode_stats['fully_charged_vehicles']:
        print(f"\nEpisode Performance Statistics:")
        print(f"  Total vehicles charged: {sum(episode_stats['fully_charged_vehicles'])}")
        print(f"  Average vehicles per episode: {np.mean(episode_stats['fully_charged_vehicles']):.2f}")
        print(f"  Total cumulative revenue: {sum(episode_stats['cumulative_revenue']):.2f}")
        print(f"  Average revenue per episode: {np.mean(episode_stats['cumulative_revenue']):.2f}")
        print(f"  Average energy per episode: {np.mean(episode_stats['energy_delivered']):.2f} kWh")
        print(f"  Average station utilization: {np.mean(episode_stats['station_utilization']):.1f}%")
        
        # 新增车辆流动统计
        print(f"\nVehicle Flow Statistics:")
        print(f"  Total vehicle arrivals: {sum(episode_stats['total_arrivals'])}")
        print(f"  Average arrivals per episode: {np.mean(episode_stats['total_arrivals']):.2f}")
        print(f"  Total vehicles charged: {sum(episode_stats['vehicles_charged'])}")
        print(f"  Total vehicles left uncharged: {sum(episode_stats['vehicles_left_uncharged'])}")
        print(f"  Overall charging success rate: {np.mean(episode_stats['charging_success_rate']):.1f}%")
        
        # 分析最好和最差的episode
        best_success_episode = np.argmax(episode_stats['charging_success_rate'])
        worst_success_episode = np.argmin(episode_stats['charging_success_rate'])
        print(f"  Best episode (Episode {best_success_episode}): "
              f"{episode_stats['total_arrivals'][best_success_episode]} arrivals, "
              f"{episode_stats['vehicles_charged'][best_success_episode]} charged, "
              f"{episode_stats['charging_success_rate'][best_success_episode]:.1f}% success rate")
        print(f"  Worst episode (Episode {worst_success_episode}): "
              f"{episode_stats['total_arrivals'][worst_success_episode]} arrivals, "
              f"{episode_stats['vehicles_charged'][worst_success_episode]} charged, "
              f"{episode_stats['charging_success_rate'][worst_success_episode]:.1f}% success rate")
    
    print(f"{'='*60}")
    
    return score_record, action_restore, [mu, mu_target, q, q_target], constraint_violations, episode_stats


def plot_charging_gantt(env, save_path=None):
    """绘制充电站调度甘特图"""
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # 时间轴设置
    start_time = datetime(2024, 1, 1, 0, 0)
    time_labels = []
    for i in range(0, 289, 12):  # 每小时标记一次
        time = start_time + timedelta(minutes=i*5)
        time_labels.append(time.strftime("%H:%M"))
    
    # 颜色映射
    colors = plt.cm.tab20(np.linspace(0, 1, env.max_vehicles))
    
    # 记录充电历史
    charging_history = []
    
    # 遍历所有充电桩
    for station_id in range(env.n_stations):
        y_pos = station_id
        
        # 绘制充电桩基线
        ax.axhline(y=y_pos, color='gray', linestyle='-', alpha=0.3)
        
    # 收集充电记录（这需要在环境中添加历史记录功能）
    if hasattr(env, 'charging_records'):
        for record in env.charging_records:
            station_id = record['station_id']
            vehicle_id = record['vehicle_id']
            start_step = record['start_step']
            end_step = record['end_step']
            power = record['power']
            energy = record['energy']
            
            # 绘制充电块
            rect = Rectangle((start_step, station_id - 0.4), 
                           end_step - start_step, 0.8,
                           facecolor=colors[vehicle_id % len(colors)],
                           edgecolor='black',
                           alpha=0.8)
            ax.add_patch(rect)
            
            # 添加标签
            mid_point = (start_step + end_step) / 2
            ax.text(mid_point, station_id, 
                   f'V{vehicle_id}\n{power:.0f}kW\n{energy:.1f}kWh',
                   ha='center', va='center', fontsize=8)
    
    # 设置坐标轴
    ax.set_xlim(0, 288)
    ax.set_ylim(-0.5, env.n_stations - 0.5)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Charging Station ID', fontsize=12)
    ax.set_title('EV Charging Station Schedule Gantt Chart', fontsize=16)
    
    # 设置时间标签
    x_ticks = list(range(0, 289, 12))
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(time_labels, rotation=45)
    
    # 设置充电桩标签
    ax.set_yticks(range(env.n_stations))
    ax.set_yticklabels([f'Station {i}' for i in range(env.n_stations)])
    
    # 添加网格
    ax.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gantt chart saved to: {save_path}")
    
    plt.show()
    return fig


def analyze_energy_consumption(env, save_path=None):
    """分析能耗情况"""
    # 创建分析图表
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 时段能耗分布
    hourly_energy = np.zeros(24)
    hourly_cost = np.zeros(24)
    
    if hasattr(env, 'charging_records'):
        for record in env.charging_records:
            hour = (record['start_step'] * 5 // 60) % 24
            duration_hours = (record['end_step'] - record['start_step']) * 5 / 60
            hourly_energy[hour] += record['energy']
            hourly_cost[hour] += record['cost']
    
    hours = list(range(24))
    ax1.bar(hours, hourly_energy, color='skyblue', alpha=0.7, label='Energy (kWh)')
    ax1_twin = ax1.twinx()
    ax1_twin.plot(hours, hourly_cost, color='red', marker='o', label='Cost')
    
    ax1.set_xlabel('Hour of Day')
    ax1.set_ylabel('Energy (kWh)', color='blue')
    ax1_twin.set_ylabel('Cost', color='red')
    ax1.set_title('Hourly Energy Consumption and Cost')
    ax1.grid(True, alpha=0.3)
    
    # 标记峰谷时段
    for start, end in env.peak_hours:
        ax1.axvspan(start, end, alpha=0.2, color='red', label='Peak Hours')
    ax1.axvspan(23, 24, alpha=0.2, color='green')
    ax1.axvspan(0, 7, alpha=0.2, color='green', label='Valley Hours')
    
    # 2. 充电桩利用率
    station_utilization = np.zeros(env.n_stations)
    station_energy = np.zeros(env.n_stations)
    
    if hasattr(env, 'charging_records'):
        for record in env.charging_records:
            station_id = record['station_id']
            duration = record['end_step'] - record['start_step']
            station_utilization[station_id] += duration
            station_energy[station_id] += record['energy']
    
    station_utilization = (station_utilization / 288) * 100  # 转换为百分比
    
    x = range(env.n_stations)
    ax2.bar(x, station_utilization, color='lightgreen', alpha=0.7)
    ax2.set_xlabel('Station ID')
    ax2.set_ylabel('Utilization Rate (%)')
    ax2.set_title('Charging Station Utilization')
    ax2.set_xticks(x)
    ax2.grid(True, alpha=0.3)
    
    # 添加能量标签
    for i, (util, energy) in enumerate(zip(station_utilization, station_energy)):
        ax2.text(i, util + 1, f'{energy:.1f}kWh', ha='center', va='bottom', fontsize=8)
    
    # 3. 功率分布直方图
    power_distribution = []
    if hasattr(env, 'charging_records'):
        power_distribution = [record['power'] for record in env.charging_records]
    
    if power_distribution:
        ax3.hist(power_distribution, bins=20, color='orange', alpha=0.7, edgecolor='black')
        ax3.axvline(np.mean(power_distribution), color='red', linestyle='dashed', 
                   linewidth=2, label=f'Mean: {np.mean(power_distribution):.1f}kW')
        ax3.set_xlabel('Charging Power (kW)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Charging Power Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. 累计能量和成本曲线
    cumulative_energy = []
    cumulative_cost = []
    current_energy = 0
    current_cost = 0
    
    for step in range(288):
        # 更新累计值
        current_energy = env.total_energy_delivered if step == env.current_step else current_energy
        current_cost = env.total_cost if step == env.current_step else current_cost
        cumulative_energy.append(current_energy)
        cumulative_cost.append(current_cost)
    
    time_hours = [i * 5 / 60 for i in range(288)]
    ax4.plot(time_hours, cumulative_energy, 'b-', label='Energy', linewidth=2)
    ax4_twin = ax4.twinx()
    ax4_twin.plot(time_hours, cumulative_cost, 'r-', label='Cost', linewidth=2)
    
    ax4.set_xlabel('Time (hours)')
    ax4.set_ylabel('Cumulative Energy (kWh)', color='blue')
    ax4_twin.set_ylabel('Cumulative Cost', color='red')
    ax4.set_title('Cumulative Energy and Cost Over Time')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Energy analysis saved to: {save_path}")
    
    plt.show()
    
    # 返回统计数据
    stats = {
        'total_energy': env.total_energy_delivered,
        'total_cost': env.total_cost,
        'average_power': np.mean(power_distribution) if power_distribution else 0,
        'peak_hour_energy': sum(hourly_energy[h] for start, end in env.peak_hours for h in range(start, end)),
        'valley_hour_energy': sum(hourly_energy[h] for h in list(range(23, 24)) + list(range(0, 7))),
        'station_utilization': station_utilization.tolist(),
        'lifetime_damage': env.total_lifetime_damage
    }
    
    return stats


def save_multi_run_vehicle_flow_statistics(all_episode_stats, all_constraint_violations, config, save_dir):
    """保存多次运行的车辆流动统计信息"""
    import csv
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. 保存每次运行的汇总统计
    runs_summary_csv_path = os.path.join(save_dir, f'multi_run_vehicle_flow_summary_{timestamp}.csv')
    with open(runs_summary_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # 写入表头
        writer.writerow([
            'Run', 'Seed', 'Total_Arrivals', 'Total_Charged', 'Total_Left_Uncharged', 
            'Success_Rate_%', 'Avg_Revenue', 'Avg_Energy_kWh', 'Avg_Power_kW', 
            'Avg_Station_Utilization_%', 'Total_Violations', 'Violation_Rate_%'
        ])
        
        # 写入每次运行的数据
        for i, (episode_stats, constraint_violations) in enumerate(zip(all_episode_stats, all_constraint_violations)):
            total_arrivals = sum(episode_stats['total_arrivals'])
            total_charged = sum(episode_stats['vehicles_charged'])
            total_left_uncharged = sum(episode_stats['vehicles_left_uncharged'])
            success_rate = np.mean(episode_stats['charging_success_rate'])
            avg_revenue = np.mean(episode_stats['cumulative_revenue'])
            avg_energy = np.mean(episode_stats['energy_delivered'])
            avg_power = np.mean(episode_stats['average_power'])
            avg_utilization = np.mean(episode_stats['station_utilization'])
            total_violations = constraint_violations['total_violations']
            violation_rate = np.mean(constraint_violations['violation_rate']) * 100
            
            writer.writerow([
                i + 1,  # Run编号
                config['seeds'][i],  # 种子
                total_arrivals,
                total_charged,
                total_left_uncharged,
                round(success_rate, 2),
                round(avg_revenue, 2),
                round(avg_energy, 2),
                round(avg_power, 2),
                round(avg_utilization, 2),
                total_violations,
                round(violation_rate, 2)
            ])
    
    # 2. 保存整体统计
    overall_summary_csv_path = os.path.join(save_dir, f'multi_run_overall_summary_{timestamp}.csv')
    with open(overall_summary_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # 计算所有运行的整体统计
        all_total_arrivals = [sum(stats['total_arrivals']) for stats in all_episode_stats]
        all_total_charged = [sum(stats['vehicles_charged']) for stats in all_episode_stats]
        all_total_left_uncharged = [sum(stats['vehicles_left_uncharged']) for stats in all_episode_stats]
        all_success_rates = [np.mean(stats['charging_success_rate']) for stats in all_episode_stats]
        all_avg_revenues = [np.mean(stats['cumulative_revenue']) for stats in all_episode_stats]
        all_avg_energies = [np.mean(stats['energy_delivered']) for stats in all_episode_stats]
        all_avg_powers = [np.mean(stats['average_power']) for stats in all_episode_stats]
        all_avg_utilizations = [np.mean(stats['station_utilization']) for stats in all_episode_stats]
        all_total_violations = [cv['total_violations'] for cv in all_constraint_violations]
        all_violation_rates = [np.mean(cv['violation_rate']) * 100 for cv in all_constraint_violations]
        
        # 写入整体统计
        writer.writerow(['Metric', 'Mean', 'Std', 'Min', 'Max', 'Unit'])
        
        writer.writerow(['Total Arrivals per Run', 
                        round(np.mean(all_total_arrivals), 2),
                        round(np.std(all_total_arrivals), 2),
                        min(all_total_arrivals),
                        max(all_total_arrivals),
                        'vehicles'])
        
        writer.writerow(['Total Charged per Run', 
                        round(np.mean(all_total_charged), 2),
                        round(np.std(all_total_charged), 2),
                        min(all_total_charged),
                        max(all_total_charged),
                        'vehicles'])
        
        writer.writerow(['Total Left Uncharged per Run', 
                        round(np.mean(all_total_left_uncharged), 2),
                        round(np.std(all_total_left_uncharged), 2),
                        min(all_total_left_uncharged),
                        max(all_total_left_uncharged),
                        'vehicles'])
        
        writer.writerow(['Success Rate per Run', 
                        round(np.mean(all_success_rates), 2),
                        round(np.std(all_success_rates), 2),
                        round(min(all_success_rates), 2),
                        round(max(all_success_rates), 2),
                        '%'])
        
        writer.writerow(['Average Revenue per Run', 
                        round(np.mean(all_avg_revenues), 2),
                        round(np.std(all_avg_revenues), 2),
                        round(min(all_avg_revenues), 2),
                        round(max(all_avg_revenues), 2),
                        'currency'])
        
        writer.writerow(['Average Energy per Run', 
                        round(np.mean(all_avg_energies), 2),
                        round(np.std(all_avg_energies), 2),
                        round(min(all_avg_energies), 2),
                        round(max(all_avg_energies), 2),
                        'kWh'])
        
        writer.writerow(['Average Power per Run', 
                        round(np.mean(all_avg_powers), 2),
                        round(np.std(all_avg_powers), 2),
                        round(min(all_avg_powers), 2),
                        round(max(all_avg_powers), 2),
                        'kW'])
        
        writer.writerow(['Average Station Utilization per Run', 
                        round(np.mean(all_avg_utilizations), 2),
                        round(np.std(all_avg_utilizations), 2),
                        round(min(all_avg_utilizations), 2),
                        round(max(all_avg_utilizations), 2),
                        '%'])
        
        writer.writerow(['Total Violations per Run', 
                        round(np.mean(all_total_violations), 2),
                        round(np.std(all_total_violations), 2),
                        min(all_total_violations),
                        max(all_total_violations),
                        'violations'])
        
        writer.writerow(['Violation Rate per Run', 
                        round(np.mean(all_violation_rates), 2),
                        round(np.std(all_violation_rates), 2),
                        round(min(all_violation_rates), 2),
                        round(max(all_violation_rates), 2),
                        '%'])
    
    # 3. 保存详细的逐episode数据（合并所有运行）
    detailed_csv_path = os.path.join(save_dir, f'multi_run_all_episodes_{timestamp}.csv')
    with open(detailed_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # 写入表头
        writer.writerow([
            'Run', 'Seed', 'Episode', 'Total_Arrivals', 'Vehicles_Charged', 
            'Vehicles_Left_Uncharged', 'Charging_Success_Rate_%', 'Fully_Charged_Vehicles', 
            'Cumulative_Revenue', 'Energy_Delivered_kWh', 'Average_Power_kW', 
            'Station_Utilization_%', 'Constraint_Violations', 'Violation_Rate_%'
        ])
        
        # 写入每次运行的每个episode数据
        for run_idx, (episode_stats, constraint_violations) in enumerate(zip(all_episode_stats, all_constraint_violations)):
            for episode in range(len(episode_stats['total_arrivals'])):
                writer.writerow([
                    run_idx + 1,  # Run编号
                    config['seeds'][run_idx],  # 种子
                    episode,  # Episode编号
                    episode_stats['total_arrivals'][episode],
                    episode_stats['vehicles_charged'][episode],
                    episode_stats['vehicles_left_uncharged'][episode],
                    round(episode_stats['charging_success_rate'][episode], 2),
                    episode_stats['fully_charged_vehicles'][episode],
                    round(episode_stats['cumulative_revenue'][episode], 2),
                    round(episode_stats['energy_delivered'][episode], 2),
                    round(episode_stats['average_power'][episode], 2),
                    round(episode_stats['station_utilization'][episode], 2),
                    constraint_violations['episode_violations'][episode],
                    round(constraint_violations['violation_rate'][episode] * 100, 2)
                ])
    
    print(f"Multi-run vehicle flow statistics saved to:")
    print(f"  Runs summary: {runs_summary_csv_path}")
    print(f"  Overall summary: {overall_summary_csv_path}")
    print(f"  All episodes details: {detailed_csv_path}")
    
    return runs_summary_csv_path, overall_summary_csv_path, detailed_csv_path


def save_vehicle_flow_statistics(episode_stats, constraint_violations, config, save_dir):
    """保存车辆流动统计信息到CSV文件"""
    import csv
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. 保存逐episode的详细统计
    episode_csv_path = os.path.join(save_dir, f'vehicle_flow_episodes_{timestamp}.csv')
    with open(episode_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # 写入表头
        writer.writerow([
            'Episode', 'Total_Arrivals', 'Vehicles_Charged', 'Vehicles_Left_Uncharged', 
            'Charging_Success_Rate_%', 'Fully_Charged_Vehicles', 'Cumulative_Revenue',
            'Energy_Delivered_kWh', 'Average_Power_kW', 'Station_Utilization_%',
            'Constraint_Violations', 'Violation_Rate_%'
        ])
        
        # 写入每个episode的数据
        for i in range(len(episode_stats['total_arrivals'])):
            writer.writerow([
                i,  # Episode编号
                episode_stats['total_arrivals'][i],
                episode_stats['vehicles_charged'][i],
                episode_stats['vehicles_left_uncharged'][i],
                round(episode_stats['charging_success_rate'][i], 2),
                episode_stats['fully_charged_vehicles'][i],
                round(episode_stats['cumulative_revenue'][i], 2),
                round(episode_stats['energy_delivered'][i], 2),
                round(episode_stats['average_power'][i], 2),
                round(episode_stats['station_utilization'][i], 2),
                constraint_violations['episode_violations'][i],
                round(constraint_violations['violation_rate'][i] * 100, 2)
            ])
    
    # 2. 保存汇总统计
    summary_csv_path = os.path.join(save_dir, f'vehicle_flow_summary_{timestamp}.csv')
    with open(summary_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # 写入汇总统计
        writer.writerow(['Metric', 'Value', 'Unit'])
        
        # 车辆流动统计
        writer.writerow(['Total Vehicle Arrivals', sum(episode_stats['total_arrivals']), 'vehicles'])
        writer.writerow(['Average Arrivals per Episode', round(np.mean(episode_stats['total_arrivals']), 2), 'vehicles'])
        writer.writerow(['Total Vehicles Charged', sum(episode_stats['vehicles_charged']), 'vehicles'])
        writer.writerow(['Average Charged per Episode', round(np.mean(episode_stats['vehicles_charged']), 2), 'vehicles'])
        writer.writerow(['Total Vehicles Left Uncharged', sum(episode_stats['vehicles_left_uncharged']), 'vehicles'])
        writer.writerow(['Average Left Uncharged per Episode', round(np.mean(episode_stats['vehicles_left_uncharged']), 2), 'vehicles'])
        writer.writerow(['Overall Charging Success Rate', round(np.mean(episode_stats['charging_success_rate']), 2), '%'])
        
        # 性能统计
        writer.writerow(['Total Cumulative Revenue', round(sum(episode_stats['cumulative_revenue']), 2), 'currency'])
        writer.writerow(['Average Revenue per Episode', round(np.mean(episode_stats['cumulative_revenue']), 2), 'currency'])
        writer.writerow(['Average Energy per Episode', round(np.mean(episode_stats['energy_delivered']), 2), 'kWh'])
        writer.writerow(['Average Power per Episode', round(np.mean(episode_stats['average_power']), 2), 'kW'])
        writer.writerow(['Average Station Utilization', round(np.mean(episode_stats['station_utilization']), 2), '%'])
        
        # 约束违反统计
        writer.writerow(['Total Constraint Violations', constraint_violations['total_violations'], 'violations'])
        writer.writerow(['Average Violations per Episode', round(np.mean(constraint_violations['episode_violations']), 2), 'violations'])
        writer.writerow(['Overall Violation Rate', round(np.mean(constraint_violations['violation_rate']) * 100, 2), '%'])
        
        # 最佳和最差episode
        if len(episode_stats['charging_success_rate']) > 0:
            best_episode = np.argmax(episode_stats['charging_success_rate'])
            worst_episode = np.argmin(episode_stats['charging_success_rate'])
            
            writer.writerow(['Best Episode Number', best_episode, 'episode'])
            writer.writerow(['Best Episode Success Rate', round(episode_stats['charging_success_rate'][best_episode], 2), '%'])
            writer.writerow(['Best Episode Arrivals', episode_stats['total_arrivals'][best_episode], 'vehicles'])
            writer.writerow(['Best Episode Charged', episode_stats['vehicles_charged'][best_episode], 'vehicles'])
            
            writer.writerow(['Worst Episode Number', worst_episode, 'episode'])
            writer.writerow(['Worst Episode Success Rate', round(episode_stats['charging_success_rate'][worst_episode], 2), '%'])
            writer.writerow(['Worst Episode Arrivals', episode_stats['total_arrivals'][worst_episode], 'vehicles'])
            writer.writerow(['Worst Episode Charged', episode_stats['vehicles_charged'][worst_episode], 'vehicles'])
    
    # 3. 保存训练配置信息
    config_csv_path = os.path.join(save_dir, f'training_config_{timestamp}.csv')
    with open(config_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Parameter', 'Value'])
        
        for key, value in config.items():
            if not callable(value):  # 排除函数类型
                writer.writerow([key, str(value)])
    
    print(f"Vehicle flow statistics saved to:")
    print(f"  Episodes details: {episode_csv_path}")
    print(f"  Summary statistics: {summary_csv_path}")
    print(f"  Training config: {config_csv_path}")
    
    return episode_csv_path, summary_csv_path, config_csv_path


def save_scheduling_data(env, episode_data, save_dir):
    """保存调度数据到文件"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存环境数据
    env_data = {
        'n_stations': int(env.n_stations),
        'max_vehicles': int(env.max_vehicles),
        'arrival_rate': float(env.arrival_rate),
        'total_energy': float(env.total_energy_delivered),
        'total_cost': float(env.total_cost),
        'total_lifetime_damage': float(env.total_lifetime_damage),
        'current_step': int(env.current_step)
    }
    
    # 保存充电记录
    charging_records = []
    if hasattr(env, 'charging_records'):
        # Convert numpy types to Python native types
        for record in env.charging_records:
            charging_records.append({
                'vehicle_id': int(record['vehicle_id']),
                'station_id': int(record['station_id']),
                'start_step': int(record['start_step']),
                'end_step': int(record['end_step']),
                'power': float(record['power']),
                'energy': float(record['energy']),
                'cost': float(record['cost']),
                'wait_time': int(record['wait_time'])
            })
    
    # 保存到JSON文件
    import json
    data_to_save = {
        'timestamp': timestamp,
        'environment': env_data,
        'episode_data': {
            'total_reward': float(episode_data['total_reward']),
            'total_steps': int(episode_data['total_steps']),
            'average_reward': float(episode_data['average_reward'])
        },
        'charging_records': charging_records
    }
    
    json_path = os.path.join(save_dir, f'scheduling_data_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump(data_to_save, f, indent=2)
    
    # 保存到CSV文件（充电记录）
    if charging_records:
        df = pd.DataFrame(charging_records)
        csv_path = os.path.join(save_dir, f'charging_records_{timestamp}.csv')
        df.to_csv(csv_path, index=False)
        print(f"Charging records saved to: {csv_path}")
    
    print(f"Scheduling data saved to: {json_path}")
    return json_path


def visualize_training_results(score_records, save_dir=None):
    """可视化训练结果"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. 训练曲线
    episodes = range(len(score_records))
    ax1.plot(episodes, score_records, 'b-', alpha=0.6, label='Episode Reward')
    
    # 添加移动平均
    window = min(20, len(score_records) // 10)
    if window > 1:
        moving_avg = pd.Series(score_records).rolling(window=window).mean()
        ax1.plot(episodes, moving_avg, 'r-', linewidth=2, label=f'{window}-Episode Moving Average')
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Training Progress')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 奖励分布
    ax2.hist(score_records, bins=30, color='green', alpha=0.7, edgecolor='black')
    ax2.axvline(np.mean(score_records), color='red', linestyle='dashed', 
               linewidth=2, label=f'Mean: {np.mean(score_records):.2f}')
    ax2.axvline(np.median(score_records), color='blue', linestyle='dashed', 
               linewidth=2, label=f'Median: {np.median(score_records):.2f}')
    ax2.set_xlabel('Total Reward')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Reward Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        save_path = os.path.join(save_dir, 'training_results.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training results visualization saved to: {save_path}")
    
    plt.show()


def test_and_visualize(config=None, model_path=None):
    """Test trained model and visualize charging process"""
    if config is None:
        config = CONFIG
        
    print("\n=== Starting DDPG-LIRL Testing for EV Charging ===")
    
    # Create environment
    env = EVChargingEnv(
        n_stations=config['n_stations'],
        p_max=config['p_max'],
        arrival_rate=config['arrival_rate']
    )
    state_size = env.observation_space.shape[0]
    action_size = 3
    
    # Load trained model
    mu = MuNet(state_size, action_size)
    if model_path and os.path.exists(model_path):
        try:
            mu.load_state_dict(torch.load(model_path))
            mu.eval()
            print(f"Successfully loaded model: {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using randomly initialized model")
    else:
        print("Warning: Model path not provided or file not found, using random initialization")
    
    # Reset environment
    s = env.reset()
    done = False
    step = 0
    total_reward = 0
    
    print(f"\nStarting EV charging scheduling - Stations: {config['n_stations']}")
    print("-" * 50)
    
    # Execute scheduling process
    while not done and step < config['max_test_steps']:
        # Use trained policy network to select action
        with torch.no_grad():
            a = mu(torch.from_numpy(s).float())
            a = torch.clamp(a, 0, 1)
            # 确保输出是正确的维度
            if len(a.shape) > 1 and a.shape[0] == 1:
                a = a.squeeze(0)
            
        # Use action projection method
        action = action_projection_ev(env, a)
        
        # 屏蔽step-by-step输出
        # print(f"Step {step+1}: Station{action['station_id']}, Vehicle{action['vehicle_id']}, Power{action['power'][0]:.1f}kW")
        
        # Execute action
        s_prime, reward, done, info = env.step(action)
        
        # 根据环境返回的约束违反信息显示，而不是通过奖励判断
        violation_info = info.get('constraint_violation', None)
        if violation_info and violation_info['has_violation']:
            print(f"Step {step+1}: CONSTRAINT VIOLATION - {violation_info['violation_type']}: {violation_info['violation_details']}")
        
        # print(f"  Reward: {reward:.4f}, Vehicles: {info['num_vehicles']}, Total Energy: {info['total_energy']:.2f}kWh")
        total_reward += reward
        s = s_prime
        step += 1
        
        if done:
            print(f"\nSimulation completed! Total steps: {step}")
            break
    
    # Print final results
    print(f"\n=== Charging Station Results Summary ===")
    print(f"Total steps: {step}")
    print(f"Total reward: {total_reward:.4f}")
    print(f"Average reward per step: {total_reward/step:.4f}")
    print(f"Total energy delivered: {info['total_energy']:.2f} kWh")
    print(f"Total cost: {info['total_cost']:.2f}")
    print(f"Total lifetime damage: {info['total_lifetime_damage']:.4f}")
    
    # 生成可视化
    if config.get('enable_visualization', True):
        # 创建保存目录
        vis_dir = f"visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(vis_dir, exist_ok=True)
        
        # 绘制甘特图
        gantt_path = os.path.join(vis_dir, 'charging_gantt.png')
        plot_charging_gantt(env, gantt_path)
        
        # 分析能耗
        energy_path = os.path.join(vis_dir, 'energy_analysis.png')
        energy_stats = analyze_energy_consumption(env, energy_path)
        
        # 保存数据
        episode_data = {
            'total_reward': total_reward,
            'total_steps': step,
            'average_reward': total_reward/step
        }
        save_scheduling_data(env, episode_data, vis_dir)
        
        # 打印能耗统计
        print(f"\n=== Energy Consumption Analysis ===")
        print(f"Peak hour energy: {energy_stats['peak_hour_energy']:.2f} kWh")
        print(f"Valley hour energy: {energy_stats['valley_hour_energy']:.2f} kWh")
        print(f"Average charging power: {energy_stats['average_power']:.2f} kW")
        print(f"Average station utilization: {np.mean(energy_stats['station_utilization']):.1f}%")
        
        print(f"\nAll visualizations saved to: {vis_dir}")
    
    return total_reward, step


def multi_run_training(config=None):
    """Execute multiple training runs with different seeds"""
    if config is None:
        config = CONFIG
    
    all_score_records = []
    all_action_restores = []
    all_models = []
    all_constraint_violations = []
    all_episode_stats = []
    
    print(f"\n{'='*80}")
    print(f"Starting Multi-Run DDPG-LIRL Training")
    print(f"Seeds: {config['seeds']}")
    print(f"Total runs: {len(config['seeds'])}")
    print(f"{'='*80}")
    
    for run_idx, seed in enumerate(config['seeds']):
        print(f"\n{'='*60}")
        print(f"Run {run_idx + 1}/{len(config['seeds'])} - Seed: {seed}")
        print(f"{'='*60}")
        
        # Set random seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        # Run training
        score_record, action_restore, models, constraint_violations, episode_stats = main(config)
        
        # Store results
        all_score_records.append(score_record)
        all_action_restores.append(action_restore)
        all_models.append(models)
        all_constraint_violations.append(constraint_violations)
        all_episode_stats.append(episode_stats)
        
        print(f"Run {run_idx + 1} completed - Final Score: {score_record[-1]:.4f}")
        print(f"  Total violations: {constraint_violations['total_violations']}")
        
    print(f"\n{'='*60}")
    print(f"All {len(config['seeds'])} runs completed!")
    print(f"{'='*60}")
    
    # Print multi-run constraint violation summary
    print(f"\n{'='*60}")
    print(f"Multi-Run Constraint Violation Summary:")
    print(f"{'='*60}")
    
    total_violations_all_runs = sum(cv['total_violations'] for cv in all_constraint_violations)
    total_steps_all_runs = sum(sum(len(actions) for actions in action_restore) 
                              for action_restore in all_action_restores)
    
    print(f"Total violations across all runs: {total_violations_all_runs}")
    print(f"Total steps across all runs: {total_steps_all_runs}")
    print(f"Overall violation rate: {(total_violations_all_runs / total_steps_all_runs * 100):.2f}%")
    
    # Per-run statistics
    for i, cv in enumerate(all_constraint_violations):
        run_steps = sum(len(actions) for actions in all_action_restores[i])
        run_rate = (cv['total_violations'] / run_steps * 100) if run_steps > 0 else 0
        print(f"Run {i+1}: {cv['total_violations']} violations, {run_rate:.2f}% rate")
    
    print(f"{'='*60}")
    
    return all_score_records, all_action_restores, all_models, all_constraint_violations, all_episode_stats


def evaluate_multi_run_results(all_score_records, config=None):
    """Evaluate and analyze results from multiple runs"""
    if config is None:
        config = CONFIG
    
    print(f"\n{'='*60}")
    print(f"Multi-Run Training Results Analysis")
    print(f"{'='*60}")
    
    # Calculate statistics
    final_scores = [scores[-1] for scores in all_score_records]
    mean_scores = [np.mean(scores[-20:]) if len(scores) >= 20 else np.mean(scores) for scores in all_score_records]
    
    print(f"Final Episode Scores:")
    for i, (seed, score) in enumerate(zip(config['seeds'], final_scores)):
        print(f"  Run {i+1} (Seed {seed}): {score:.4f}")
    
    print(f"\nLast 20 Episodes Average Scores:")
    for i, (seed, score) in enumerate(zip(config['seeds'], mean_scores)):
        print(f"  Run {i+1} (Seed {seed}): {score:.4f}")
    
    print(f"\nOverall Statistics:")
    print(f"  Mean Final Score: {np.mean(final_scores):.4f} ± {np.std(final_scores):.4f}")
    print(f"  Best Final Score: {np.max(final_scores):.4f}")
    print(f"  Worst Final Score: {np.min(final_scores):.4f}")
    print(f"  Mean of Last 20 Episodes: {np.mean(mean_scores):.4f} ± {np.std(mean_scores):.4f}")
    
    return {
        'final_scores': final_scores,
        'mean_scores': mean_scores,
        'overall_mean': np.mean(final_scores),
        'overall_std': np.std(final_scores),
        'best_score': np.max(final_scores),
        'worst_score': np.min(final_scores)
    }


def save_results(score_records, action_restores, models_restore, config):
    """Save training results and models"""
    if not config['save_models']:
        return None, None
        
    # Create save directory with timestamp
    alg_name = "ddpg_lirl_pi"
    now_str = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"{alg_name}_{now_str}"
    os.makedirs(save_dir, exist_ok=True)

    # Save training data
    np.save(os.path.join(save_dir, f"{alg_name}_scores_{now_str}.npy"), score_records)
    np.save(os.path.join(save_dir, f"{alg_name}_actions_{now_str}.npy"), action_restores)

    # Save models
    model_paths = []
    for idx, models in enumerate(models_restore):
        mu, mu_target, q, q_target = models
        mu_path = os.path.join(save_dir, f"{alg_name}_mu_{idx}_{now_str}.pth")
        torch.save(mu.state_dict(), mu_path)
        torch.save(mu_target.state_dict(), os.path.join(save_dir, f"{alg_name}_mu_target_{idx}_{now_str}.pth"))
        torch.save(q.state_dict(), os.path.join(save_dir, f"{alg_name}_q_{idx}_{now_str}.pth"))
        torch.save(q_target.state_dict(), os.path.join(save_dir, f"{alg_name}_q_target_{idx}_{now_str}.pth"))
        model_paths.append(mu_path)
    
    print(f"Results saved to directory: {save_dir}")
    return save_dir, model_paths[0] if model_paths else None


def plot_training_curve(score_records, save_dir=None):
    """Plot training curve"""
    x = range(len(score_records))
    y = score_records
    plt.figure(figsize=(10, 6))
    plt.plot(x, y)
    plt.title('DDPG-LIRL Training Score over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.grid(True)
    
    # Save the plot if save_dir is provided
    if save_dir:
        save_path = os.path.join(save_dir, 'training_curve.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curve saved to: {save_path}")
    
    plt.show()


def save_multi_run_results(all_score_records, all_action_restores, all_models, config):
    """Save results from multiple training runs"""
    if not config['save_models']:
        return None, None
        
    # Create save directory with timestamp
    alg_name = "ddpg_lirl_pi"
    now_str = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"{alg_name}_multi_run_{now_str}"
    os.makedirs(save_dir, exist_ok=True)

    # Save training data for all runs - handle irregular shapes
    np.save(os.path.join(save_dir, f"{alg_name}_all_scores_{now_str}.npy"), all_score_records, allow_pickle=True)
    
    # Save action restores with pickle to handle irregular shapes
    import pickle
    with open(os.path.join(save_dir, f"{alg_name}_all_actions_{now_str}.pkl"), 'wb') as f:
        pickle.dump(all_action_restores, f)

    # Save models from all runs
    model_paths = []
    if hasattr(all_models, '__len__') and len(all_models) > 0:
        for run_idx, models in enumerate(all_models):
            mu, mu_target, q, q_target = models
            run_save_dir = os.path.join(save_dir, f"run_{run_idx+1}_seed_{config['seeds'][run_idx]}")
            os.makedirs(run_save_dir, exist_ok=True)
            
            mu_path = os.path.join(run_save_dir, f"{alg_name}_mu_{now_str}.pth")
            torch.save(mu.state_dict(), mu_path)
            torch.save(mu_target.state_dict(), os.path.join(run_save_dir, f"{alg_name}_mu_target_{now_str}.pth"))
            torch.save(q.state_dict(), os.path.join(run_save_dir, f"{alg_name}_q_{now_str}.pth"))
            torch.save(q_target.state_dict(), os.path.join(run_save_dir, f"{alg_name}_q_target_{now_str}.pth"))
            model_paths.append(mu_path)
    
    # Save configuration
    config_path = os.path.join(save_dir, f"config_{now_str}.json")
    import json
    with open(config_path, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        json_config = {}
        for key, value in config.items():
            if isinstance(value, np.ndarray):
                json_config[key] = value.tolist()
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.integer):
                json_config[key] = [int(v) for v in value]
            else:
                json_config[key] = value
        json.dump(json_config, f, indent=2)
    
    print(f"Multi-run results saved to directory: {save_dir}")
    return save_dir, model_paths


def plot_multi_run_training_curves(all_score_records, config=None, save_dir=None):
    """Plot training curves for multiple runs"""
    if config is None:
        config = CONFIG
    
    plt.figure(figsize=(12, 8))
    
    # Plot individual runs
    for i, scores in enumerate(all_score_records):
        x = range(len(scores))
        plt.plot(x, scores, alpha=0.6, label=f'Run {i+1} (Seed {config["seeds"][i]})')
    
    # Plot mean curve
    min_length = min(len(scores) for scores in all_score_records)
    mean_scores = np.mean([scores[:min_length] for scores in all_score_records], axis=0)
    std_scores = np.std([scores[:min_length] for scores in all_score_records], axis=0)
    
    x = range(min_length)
    plt.plot(x, mean_scores, 'k-', linewidth=2, label='Mean')
    plt.fill_between(x, mean_scores - std_scores, mean_scores + std_scores, alpha=0.2, color='black')
    
    plt.title('DDPG-LIRL Multi-Run Training Curves')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save the plot if save_dir is provided
    if save_dir:
        save_path = os.path.join(save_dir, 'multi_run_training_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Multi-run training curves saved to: {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # Parse command line arguments or use default configuration
    import argparse
    
    parser = argparse.ArgumentParser(description='DDPG-LIRL Policy for EV Charging Station')
    parser.add_argument('--stations', type=int, default=CONFIG['n_stations'], help='Number of charging stations')
    parser.add_argument('--power', type=float, default=CONFIG['p_max'], help='Maximum power per station')
    parser.add_argument('--arrival-rate', type=float, default=CONFIG['arrival_rate'], help='Vehicle arrival rate')
    parser.add_argument('--episodes', type=int, default=CONFIG['num_of_episodes'], help='Number of episodes')
    parser.add_argument('--test-only', action='store_true', help='Run test only (skip training)')
    parser.add_argument('--model-path', type=str, default=None, help='Path to saved model for testing')
    parser.add_argument('--multi-run', action='store_true', default=CONFIG['enable_multi_run'], 
                       help='Run multiple training sessions with different seeds')
    parser.add_argument('--single-run', action='store_true', help='Force single run training (override config)')
    parser.add_argument('--seeds', nargs='+', type=int, default=CONFIG['seeds'], help='Random seeds for multi-run training')
    
    args = parser.parse_args()
    
    # Update CONFIG with command line arguments
    config = CONFIG.copy()
    config.update({
        'n_stations': args.stations,
        'p_max': args.power,
        'arrival_rate': args.arrival_rate,
        'num_of_episodes': args.episodes,
        'seeds': args.seeds,
        'enable_multi_run': args.multi_run and not args.single_run  # Allow override with --single-run
    })
    
    print(f"\n{'='*60}")
    print(f"DDPG-LIRL Policy for EV Charging Station")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  Stations: {config['n_stations']}")
    print(f"  Max Power: {config['p_max']} kW")
    print(f"  Arrival Rate: {config['arrival_rate']}")
    print(f"  Episodes: {config['num_of_episodes']}")
    print(f"  Seeds: {config['seeds']}")
    print(f"  Test only: {args.test_only}")
    print(f"  Multi-run mode: {config['enable_multi_run']}")
    if args.model_path:
        print(f"  Model path: {args.model_path}")
    print(f"{'='*60}")
    
    if args.test_only:
        # Test only mode
        test_and_visualize(config, args.model_path)
    elif config['enable_multi_run']:
        # Multi-run training mode (based on config or command line)
        all_score_records, all_action_restores, all_models, all_constraint_violations, all_episode_stats = multi_run_training(config)
        
        # Evaluate results
        stats = evaluate_multi_run_results(all_score_records, config)
        
        # Save multi-run results if enabled
        if config['save_models']:
            save_dir, model_paths = save_multi_run_results(all_score_records, all_action_restores, all_models, config)
            # Save vehicle flow statistics for multi-run
            save_multi_run_vehicle_flow_statistics(all_episode_stats, all_constraint_violations, config, save_dir)
        
        # Plot multi-run training curves
        if config['plot_training_curve']:
            plot_multi_run_training_curves(all_score_records, config, save_dir if config['save_models'] else None)
        
        # Test with the best performing model
        if config['save_models'] and model_paths:
            best_run_idx = np.argmax([scores[-1] for scores in all_score_records])
            best_model_path = model_paths[best_run_idx]
            print(f"\n{'='*40}")
            print(f"Testing with best model (Run {best_run_idx+1})...")
            print(f"{'='*40}")
            test_and_visualize(config, best_model_path)
    else:
        # Single run training mode
        # Set random seed
        if config['seeds']:
            seed = config['seeds'][0]
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            print(f"Using random seed: {seed}")
        
        score_record, action_restore, models, constraint_violations, episode_stats = main(config)
        
        # Save results if enabled
        save_dir, model_path = save_results([score_record], [action_restore], [models], config)
        
        # Save vehicle flow statistics
        if config['save_models'] and save_dir:
            save_vehicle_flow_statistics(episode_stats, constraint_violations, config, save_dir)
        
        # Print constraint violation summary for single run
        print(f"\n{'='*60}")
        print(f"Single Run Constraint Violation Summary:")
        print(f"{'='*60}")
        total_steps = sum(len(actions) for actions in [action_restore])
        final_violation_rate = (constraint_violations['total_violations'] / total_steps * 100) if total_steps > 0 else 0
        print(f"Total constraint violations: {constraint_violations['total_violations']}")
        print(f"Total steps taken: {total_steps}")
        print(f"Overall violation rate: {final_violation_rate:.2f}%")
        print(f"{'='*60}")
        
        # Plot training curve
        if config['plot_training_curve']:
            plot_training_curve(score_record, save_dir if config['save_models'] else None)
        
        # Test with trained model
        if model_path:
            print(f"\n{'='*40}")
            print(f"Testing with trained model...")
            print(f"{'='*40}")
            test_and_visualize(config, model_path)