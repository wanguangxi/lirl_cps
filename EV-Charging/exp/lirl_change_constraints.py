"""
LIRL Change Constraints Experiment
===================================
实验目的：测试LIRL算法在约束变化时的适应能力

实验设置：
- 阶段一：加载已训练模型，原始约束（p_max=150kW）测试
- 阶段二：加载已训练模型，新约束（p_max=100kW）测试（不重新训练）
- 阶段三：在新约束下重新训练策略网络，然后测试

测试指标：
- Average Revenue: 日收入（不含充电桩折旧成本）
- Average Output Energy: 每日总输出能量 (kWh)
- Average Charging Power: 日平均充电功率 (kW)
- Average Station Utilization: 充电桩平均利用率 (%)
- Violation Rate: 调度决策违反操作约束的比例 (%)
- Success Rate: 到达车辆成功开始充电的概率 (%)
"""

import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import linear_sum_assignment
from datetime import datetime
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
# 预训练模型路径
PRETRAINED_MODEL_PATH = '/home/one/Project/LIRL/LIRL-CPS-main/EV-Charging/exp/runtime_scaling_exp_20251218_130354/medium_scale_50_stations'

# 环境配置
ENV_CONFIG = {
    'n_stations': 50,
    'p_max_original': 150.0,  # 原始最大功率约束
    'p_max_new': 100.0,       # 新的最大功率约束
    'arrival_rate': 0.8,
}

# 测试配置
TEST_CONFIG = {
    'num_test_runs': 10,      # 测试次数
    'max_steps_per_episode': 288,  # 每个episode的最大步数（1天=288个5分钟）
}

# 训练配置（用于阶段三重新训练）
TRAIN_CONFIG = {
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
    'seed': 2025,
    'num_of_episodes': 300,  # 重新训练的episode数
    'print_interval': 20,
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
# ACTION PROJECTION
# =======================
def action_projection(env, a, p_max_constraint=None):
    """
    动作投影函数
    
    Args:
        env: 环境实例
        a: 策略网络输出
        p_max_constraint: 功率上限约束，如果为None则使用env.p_max
    
    Returns:
        action: 投影后的动作
        info: 投影信息（用于统计违反约束情况）
    """
    info = {
        'power_violation': False,  # 功率是否违反约束
        'original_power': 0,
        'constrained_power': 0,
    }
    
    # 处理网络输出
    a_ = a.detach().cpu().numpy()
    if len(a_.shape) > 1:
        a_ = a_.squeeze()
    
    if len(a_) < 3:
        if len(a_) == 1:
            a_ = np.array([a_[0], 0.5, 0.5])
        elif len(a_) == 2:
            a_ = np.array([a_[0], a_[1], 0.5])
    
    # 网络输出解析
    wait_weight = a_[0]
    energy_weight = a_[1]
    power_pref = a_[2]
    
    # 获取可用充电桩列表
    valid_stations = [i for i in range(env.n_stations) if env.station_status[i] == 1]
    
    # 获取有效车辆列表
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
        return action, info
    
    # 匈牙利算法 - 基于策略权重的最优分配
    n_vehicles = len(valid_vehicles)
    n_stations = len(valid_stations)
    cost_matrix = np.zeros((n_vehicles, n_stations))
    
    # 预计算所有车辆的特征
    wait_times = []
    energy_needs = []
    for vehicle_id in valid_vehicles:
        vehicle = env.vehicles[vehicle_id]
        if vehicle is not None:
            wait_times.append(vehicle['wait_time'])
            energy_needs.append(vehicle['energy_required'] - vehicle['energy_charged'])
    
    max_wait_time = max(wait_times) if wait_times else 1
    max_energy_need = max(energy_needs) if energy_needs else 1
    max_wait_time = max(max_wait_time, 1)
    max_energy_need = max(max_energy_need, 1)
    
    for i, vehicle_id in enumerate(valid_vehicles):
        vehicle = env.vehicles[vehicle_id]
        if vehicle is None:
            continue
        
        wait_time = vehicle['wait_time']
        energy_needed = vehicle['energy_required'] - vehicle['energy_charged']
        
        wait_score = wait_time / max_wait_time
        energy_score = energy_needed / max_energy_need
        
        for j, station_id in enumerate(valid_stations):
            weighted_cost = -(wait_weight * wait_score * 100.0 + 
                             energy_weight * energy_score * 100.0)
            
            urgency_bonus = 0.0
            if wait_time >= env.max_wait_time - 1:
                urgency_bonus = -200.0
            elif wait_time >= env.max_wait_time - 3:
                urgency_bonus = -100.0
            
            cost_matrix[i, j] = weighted_cost + urgency_bonus
    
    # 求解最优分配
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
    
    # QP求解器 - 约束功率
    # 原始目标功率 (映射到50-100 kW)
    target_power = 50.0 + power_pref * 50.0
    info['original_power'] = target_power
    
    # 确定功率上限
    if p_max_constraint is not None:
        effective_p_max = p_max_constraint
    else:
        effective_p_max = env.p_max
    
    # 检查是否违反约束
    if target_power > effective_p_max:
        info['power_violation'] = True
    
    # 投影到可行域
    if HAS_CVXPY:
        try:
            p = cp.Variable(1)
            objective = cp.Minimize(cp.square(p - target_power))
            constraints = [p >= 50.0, p <= effective_p_max]
            prob = cp.Problem(objective, constraints)
            prob.solve(solver=cp.OSQP, verbose=False)
            power = float(p.value[0]) if p.value is not None else min(target_power, effective_p_max)
        except:
            power = np.clip(target_power, 50.0, effective_p_max)
    else:
        power = np.clip(target_power, 50.0, effective_p_max)
    
    info['constrained_power'] = power
    
    # 安全检查
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
    
    action = {
        'station_id': station_id,
        'vehicle_id': vehicle_id,
        'power': np.array([power], dtype=np.float32)
    }
    
    return action, info


# =======================
# METRICS CALCULATION
# =======================
class MetricsCollector:
    """收集和计算实验指标（简化版：只计算Revenue和Violation Rate）"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.total_revenue = 0.0
        self.total_violations = 0
        self.total_decisions = 0
    
    def update(self, env, action, reward, info, projection_info):
        """每步更新指标"""
        # 收入（奖励）
        self.total_revenue += reward
        
        # 违反约束
        if projection_info.get('power_violation', False):
            self.total_violations += 1
        self.total_decisions += 1
    
    def get_metrics(self):
        """计算最终指标"""
        violation_rate = (self.total_violations / self.total_decisions * 100 
                          if self.total_decisions > 0 else 0)
        
        return {
            'Average Revenue': self.total_revenue,
            'Violation Rate (%)': violation_rate,
        }


# =======================
# TESTING FUNCTIONS
# =======================
def load_model(model_path, state_size, action_size, config):
    """加载预训练模型"""
    mu = MuNet(state_size, action_size, config['hidden_dim1'], config['hidden_dim2'])
    mu_path = os.path.join(model_path, 'mu.pth')
    
    if os.path.exists(mu_path):
        mu.load_state_dict(torch.load(mu_path, map_location='cpu'))
        print(f"Model loaded from: {mu_path}")
    else:
        raise FileNotFoundError(f"Model not found: {mu_path}")
    
    return mu


def run_single_test(mu, env, p_max_constraint=None, max_steps=288):
    """运行单次测试"""
    mu.eval()
    collector = MetricsCollector()
    
    s = env.reset()
    done = False
    step = 0
    
    while not done and step < max_steps:
        with torch.no_grad():
            a = mu(torch.from_numpy(s).float())
            a = torch.clamp(a, 0, 1)
            if len(a.shape) > 1:
                a = a.squeeze(0)
        
        action, projection_info = action_projection(env, a, p_max_constraint)
        s_prime, reward, done, info = env.step(action)
        
        collector.update(env, action, reward, info, projection_info)
        
        s = s_prime
        step += 1
    
    return collector.get_metrics()


def run_test_phase(mu, env_config, p_max_constraint, num_runs, phase_name):
    """运行测试阶段"""
    print(f"\n{'='*60}")
    print(f"{phase_name}")
    print(f"Power constraint: {p_max_constraint} kW")
    print(f"Number of test runs: {num_runs}")
    print(f"{'='*60}")
    
    all_metrics = []
    
    for run in range(num_runs):
        # 创建新环境
        env = EVChargingEnv(
            n_stations=env_config['n_stations'],
            p_max=env_config['p_max_original'],  # 环境本身用原始约束
            arrival_rate=env_config['arrival_rate']
        )
        
        metrics = run_single_test(mu, env, p_max_constraint, 
                                  TEST_CONFIG['max_steps_per_episode'])
        all_metrics.append(metrics)
        print(f"  Run {run+1}/{num_runs}: Revenue={metrics['Average Revenue']:.2f}, "
              f"Violation={metrics['Violation Rate (%)']:.2f}%")
    
    # 计算平均值和标准差
    avg_metrics = {}
    std_metrics = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics]
        avg_metrics[key] = np.mean(values)
        std_metrics[key] = np.std(values)
    
    # 打印结果
    print(f"\n{phase_name} - Results (Mean ± Std):")
    print(f"-" * 50)
    for key in avg_metrics.keys():
        print(f"  {key}: {avg_metrics[key]:.4f} ± {std_metrics[key]:.4f}")
    
    return avg_metrics, std_metrics, all_metrics


# =======================
# TRAINING FUNCTIONS
# =======================
def train_step(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer, config):
    """执行一次训练更新"""
    s, a, r, s_prime, done_mask = memory.sample(config['batch_size'])
    
    if len(a.shape) == 1:
        a = a.unsqueeze(1)
    
    target = torch.unsqueeze(r, dim=1) + config['gamma'] * q_target(s_prime, mu_target(s_prime)).mul(done_mask)
    q_loss = F.smooth_l1_loss(q(s, a), target.detach())
    q_optimizer.zero_grad()
    q_loss.backward()
    q_optimizer.step()
    
    mu_loss = -q(s, mu(s)).mean()
    mu_optimizer.zero_grad()
    mu_loss.backward()
    mu_optimizer.step()
    
    return q_loss.item(), mu_loss.item()


def soft_update(net, net_target, tau):
    """软更新目标网络"""
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)


def retrain_with_new_constraint(env_config, p_max_constraint, config, save_dir):
    """在新约束下重新训练策略网络"""
    print(f"\n{'='*60}")
    print(f"PHASE 3: RETRAINING WITH NEW CONSTRAINT")
    print(f"New power constraint: {p_max_constraint} kW")
    print(f"{'='*60}")
    
    # 设置随机种子
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    
    # 创建环境
    env = EVChargingEnv(
        n_stations=env_config['n_stations'],
        p_max=env_config['p_max_original'],
        arrival_rate=env_config['arrival_rate']
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
    
    score_record = []
    
    for n_epi in range(config['num_of_episodes']):
        s = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            a = mu(torch.from_numpy(s).float())
            noise_scale = max(0.1, 1.0 - n_epi / (config['num_of_episodes'] * 0.8))
            noise = torch.from_numpy(ou_noise()).float()
            a = a + noise * noise_scale
            a = torch.clamp(a, 0, 1)
            a = a.squeeze() if len(a.shape) > 1 else a
            
            # 使用新约束进行动作投影
            action, _ = action_projection(env, a, p_max_constraint)
            s_prime, r, done, info = env.step(action)
            
            a_store = a.detach().numpy()
            memory.put((s, a_store, r, s_prime, done))
            
            s = s_prime
            episode_reward += r
        
        score_record.append(episode_reward)
        
        if memory.size() > config['memory_threshold']:
            for _ in range(config['training_iterations']):
                train_step(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer, config)
                soft_update(mu, mu_target, config['tau'])
                soft_update(q, q_target, config['tau'])
        
        if n_epi % config['print_interval'] == 0 and n_epi != 0:
            avg_score = np.mean(score_record[-config['print_interval']:])
            print(f"  Episode {n_epi}: Average Score = {avg_score:.4f}")
    
    # 保存重新训练的模型
    model_save_dir = os.path.join(save_dir, 'retrained_model')
    os.makedirs(model_save_dir, exist_ok=True)
    torch.save(mu.state_dict(), os.path.join(model_save_dir, 'mu.pth'))
    torch.save(q.state_dict(), os.path.join(model_save_dir, 'q.pth'))
    
    print(f"Retrained model saved to: {model_save_dir}")
    print(f"Final average score (last 20 episodes): {np.mean(score_record[-20:]):.4f}")
    
    return mu, score_record


# =======================
# RESULTS VISUALIZATION
# =======================
def plot_comparison(phase1_metrics, phase2_metrics, phase3_metrics, save_dir):
    """绘制三个阶段的对比图（简化版：只显示Revenue和Violation Rate）"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 准备数据
    phases = ['Phase 1\n(Original)', 'Phase 2\n(New Constraint)', 'Phase 3\n(Retrained)']
    metrics_list = [phase1_metrics, phase2_metrics, phase3_metrics]
    
    # 只绘制两个指标
    metric_keys = [
        'Average Revenue',
        'Violation Rate (%)'
    ]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    colors = ['#2ecc71', '#e74c3c', '#3498db']  # 绿、红、蓝
    
    for idx, key in enumerate(metric_keys):
        ax = axes[idx]
        values = [m[key] for m in metrics_list]
        
        bars = ax.bar(phases, values, color=colors)
        ax.set_ylabel(key.split('(')[0].strip())
        ax.set_title(key, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)
    
    plt.suptitle('Constraint Change Experiment Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plot_path = os.path.join(save_dir, f'constraint_comparison_{timestamp}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {plot_path}")
    
    plt.show()
    return plot_path


def save_results_to_csv(phase1_metrics, phase2_metrics, phase3_metrics, 
                        phase1_std, phase2_std, phase3_std, save_dir):
    """保存结果到CSV"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(save_dir, f'constraint_results_{timestamp}.csv')
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # 写入表头
        writer.writerow(['Metric', 
                        'Phase 1 (Mean)', 'Phase 1 (Std)',
                        'Phase 2 (Mean)', 'Phase 2 (Std)',
                        'Phase 3 (Mean)', 'Phase 3 (Std)'])
        
        # 写入数据
        for key in phase1_metrics.keys():
            writer.writerow([
                key,
                f"{phase1_metrics[key]:.4f}", f"{phase1_std[key]:.4f}",
                f"{phase2_metrics[key]:.4f}", f"{phase2_std[key]:.4f}",
                f"{phase3_metrics[key]:.4f}", f"{phase3_std[key]:.4f}"
            ])
    
    print(f"Results saved to: {csv_path}")
    return csv_path


def print_summary_table(phase1_metrics, phase2_metrics, phase3_metrics,
                        phase1_std, phase2_std, phase3_std):
    """打印汇总表格（带Mean±Std格式）"""
    print(f"\n{'='*95}")
    print(f"EXPERIMENT SUMMARY (Mean ± Std)")
    print(f"{'='*95}")
    
    print(f"\n{'Metric':<25} {'Phase 1 (p=150)':<22} {'Phase 2 (p=100)':<22} {'Phase 3 (Retrain)':<22}")
    print(f"{'-'*95}")
    
    for key in phase1_metrics.keys():
        p1_str = f"{phase1_metrics[key]:.4f}±{phase1_std[key]:.4f}"
        p2_str = f"{phase2_metrics[key]:.4f}±{phase2_std[key]:.4f}"
        p3_str = f"{phase3_metrics[key]:.4f}±{phase3_std[key]:.4f}"
        print(f"{key:<25} {p1_str:<22} {p2_str:<22} {p3_str:<22}")
    
    print(f"\n{'='*95}")
    print("Phase 1: Original constraint (p_max=150kW), pre-trained model")
    print("Phase 2: New constraint (p_max=100kW), pre-trained model (no retraining)")
    print("Phase 3: New constraint (p_max=100kW), retrained model")
    print(f"{'='*95}")


# =======================
# MAIN FUNCTION
# =======================
def main():
    """主函数"""
    # 创建保存目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(os.path.dirname(__file__), f'change_constraints_exp_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"{'='*80}")
    print(f"LIRL Change Constraints Experiment")
    print(f"{'='*80}")
    print(f"Results will be saved to: {save_dir}")
    print(f"Pre-trained model path: {PRETRAINED_MODEL_PATH}")
    
    # 创建临时环境获取状态空间大小
    temp_env = EVChargingEnv(
        n_stations=ENV_CONFIG['n_stations'],
        p_max=ENV_CONFIG['p_max_original'],
        arrival_rate=ENV_CONFIG['arrival_rate']
    )
    state_size = temp_env.observation_space.shape[0]
    action_size = 3
    
    # 加载预训练模型
    mu_pretrained = load_model(PRETRAINED_MODEL_PATH, state_size, action_size, TRAIN_CONFIG)
    
    # ========== 阶段一：原始约束测试 ==========
    phase1_avg, phase1_std, phase1_all = run_test_phase(
        mu_pretrained, ENV_CONFIG, 
        p_max_constraint=ENV_CONFIG['p_max_original'],
        num_runs=TEST_CONFIG['num_test_runs'],
        phase_name="PHASE 1: Original Constraint (p_max=150kW)"
    )
    
    # ========== 阶段二：新约束测试（不重新训练）==========
    phase2_avg, phase2_std, phase2_all = run_test_phase(
        mu_pretrained, ENV_CONFIG,
        p_max_constraint=ENV_CONFIG['p_max_new'],
        num_runs=TEST_CONFIG['num_test_runs'],
        phase_name="PHASE 2: New Constraint (p_max=100kW) - No Retraining"
    )
    
    # ========== 阶段三：新约束下重新训练 ==========
    mu_retrained, training_scores = retrain_with_new_constraint(
        ENV_CONFIG, 
        p_max_constraint=ENV_CONFIG['p_max_new'],
        config=TRAIN_CONFIG,
        save_dir=save_dir
    )
    
    # 测试重新训练的模型
    phase3_avg, phase3_std, phase3_all = run_test_phase(
        mu_retrained, ENV_CONFIG,
        p_max_constraint=ENV_CONFIG['p_max_new'],
        num_runs=TEST_CONFIG['num_test_runs'],
        phase_name="PHASE 3: New Constraint (p_max=100kW) - After Retraining"
    )
    
    # ========== 结果汇总和可视化 ==========
    print_summary_table(phase1_avg, phase2_avg, phase3_avg,
                        phase1_std, phase2_std, phase3_std)
    
    # 保存结果
    save_results_to_csv(phase1_avg, phase2_avg, phase3_avg,
                        phase1_std, phase2_std, phase3_std, save_dir)
    
    # 绘制对比图
    plot_comparison(phase1_avg, phase2_avg, phase3_avg, save_dir)
    
    # 保存训练曲线
    if training_scores:
        plt.figure(figsize=(10, 6))
        plt.plot(training_scores, alpha=0.3, color='blue')
        window = 20
        if len(training_scores) >= window:
            moving_avg = pd.Series(training_scores).rolling(window=window).mean()
            plt.plot(moving_avg, linewidth=2, color='blue', label='Moving Average')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.title('Phase 3: Retraining Score Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        curve_path = os.path.join(save_dir, f'retraining_curve_{timestamp}.png')
        plt.savefig(curve_path, dpi=300, bbox_inches='tight')
        print(f"Training curve saved to: {curve_path}")
        plt.show()
    
    # 保存完整结果到JSON
    results = {
        'phase1': {'avg': phase1_avg, 'std': phase1_std},
        'phase2': {'avg': phase2_avg, 'std': phase2_std},
        'phase3': {'avg': phase3_avg, 'std': phase3_std},
        'config': {
            'env_config': ENV_CONFIG,
            'test_config': TEST_CONFIG,
            'train_config': {k: v for k, v in TRAIN_CONFIG.items() if not callable(v)}
        },
        'timestamp': timestamp
    }
    
    json_path = os.path.join(save_dir, f'experiment_results_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Full results saved to: {json_path}")
    
    print(f"\n{'='*80}")
    print(f"Experiment completed!")
    print(f"All results saved to: {save_dir}")
    print(f"{'='*80}")
    
    return results


if __name__ == "__main__":
    main()
