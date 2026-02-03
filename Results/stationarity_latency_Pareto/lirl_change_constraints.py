#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LIRL Constraint Change Experiment
=================================
实验目的：测试LIRL算法在约束变化情况下的适应能力

实验分三个阶段：
1) 阶段一：加载已训练模型，正常决策测试
2) 阶段二：增加约束（机器人0-5禁用），测试约束违反率
3) 阶段三：在新约束下重新训练模型，收敛后测试
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
# GPU设备配置
# =======================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def print_gpu_info():
    """打印GPU信息"""
    if torch.cuda.is_available():
        print(f"GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("GPU not available, using CPU")

# =======================
# 实验配置
# =======================
CONFIG = {
    # Environment parameters (与medium scale一致)
    'num_of_jobs': 100,
    'num_of_robots': 50,
    'alpha': 0.5,
    'beta': 0.5,
    
    # Constraint parameters
    'disabled_robots': [0, 1, 2, 3, 4, 5],  # 禁用的机器人ID列表
    
    # Network architecture
    'hidden_dim1': 128,
    'hidden_dim2': 64,
    'critic_hidden': 32,
    
    # Training parameters (阶段三) - 微调预训练模型
    'lr_mu': 0.00005,     # 极小学习率，保护预训练模型
    'lr_q': 0.0001,       # 极小Critic学习率
    'gamma': 0.98,
    'batch_size': 128,
    'buffer_limit': 1000000,
    'tau': 0.001,         # 极慢目标网络更新
    'memory_threshold': 500,
    'training_iterations': 5,   # 减少训练迭代
    'num_of_episodes': 200,     # 微调不需要太多episode
    'noise_params': {'theta': 0.1, 'dt': 0.05, 'sigma': 0.1},
    'noise_scale': 0.1,         # 极小噪声，保护策略
    'train_from_scratch': False, # 微调预训练模型
    'violation_penalty': 0.0,   # 不添加额外惩罚，让投影处理约束
    
    # Test parameters
    'max_test_steps': 1000,
    'num_test_episodes': 10,  # 每个阶段测试的episode数
    
    # Pretrained model path
    'pretrained_model_path': '/home/one/Project/LIRL/LIRL-CPS-main/lirl_runtime_scaling_20251217_182102/medium',
    
    # Output
    'print_interval': 20,
    'save_results': True,
}


# =======================
# 支持禁用机器人的环境包装类
# =======================
class ConstrainedEnv:
    """包装原始环境，支持禁用机器人约束
    
    禁用的机器人不参与时间线计算，确保调度能够正常进行
    """
    def __init__(self, num_of_jobs, num_of_robots, alpha, beta, disabled_robots=None):
        self.base_env = ENV.Env(num_of_jobs, num_of_robots, alpha, beta)
        self.disabled_robots = set(disabled_robots) if disabled_robots else set()
        self.num_of_jobs = num_of_jobs
        self.num_of_robots = num_of_robots
        
        # 代理常用属性
        self._sync_from_base()
    
    def _sync_from_base(self):
        """从基础环境同步属性"""
        self.task_set = self.base_env.task_set
        self.task_state = self.base_env.task_state
        self.robot_state = self.base_env.robot_state
        self.state = self.base_env.state
        self.action = self.base_env.action
        self.robot_timeline = self.base_env.robot_timeline
        self.current_time = self.base_env.current_time
        self.future_time = self.base_env.future_time
        self.done = self.base_env.done
        self.reward = self.base_env.reward
        self.robot_task_history = self.base_env.robot_task_history
        self.task_prcoessing_time_state = self.base_env.task_prcoessing_time_state
    
    def _get_active_timelines(self):
        """获取非禁用机器人的时间线列表"""
        return [self.base_env.robot_timeline[i] 
                for i in range(self.num_of_robots) 
                if i not in self.disabled_robots]
    
    def _update_times_excluding_disabled(self):
        """更新current_time和future_time，排除禁用机器人"""
        active = self._get_active_timelines()
        if active:
            self.base_env.current_time = np.min(active)
            self.base_env.future_time = np.max(active)
    
    def _update_robot_availability(self):
        """更新机器人可用状态，禁用机器人始终不可用"""
        for robot_id in range(self.num_of_robots):
            if robot_id in self.disabled_robots:
                self.base_env.robot_state[robot_id] = 0
            elif self.base_env.robot_timeline[robot_id] <= self.base_env.current_time:
                self.base_env.robot_state[robot_id] = 1
            else:
                self.base_env.robot_state[robot_id] = 0
    
    def _rebuild_state(self):
        """重建状态向量"""
        self.base_env.state = np.concatenate((
            self.base_env.task_state,
            self.base_env.robot_state,
            self.base_env.task_prcoessing_time_state,
            self.base_env.last_action_state
        ))
        self._sync_from_base()
    
    def _update_disabled_robot_timelines(self):
        """将禁用机器人的时间线设为所有机器人时间线中的最大值"""
        # 获取非禁用机器人的最大时间线
        active_timelines = self._get_active_timelines()
        if active_timelines:
            max_timeline = np.max(active_timelines)
        else:
            max_timeline = 0.0
        
        # 设置禁用机器人的时间线为最大值
        for robot_id in self.disabled_robots:
            if robot_id < self.num_of_robots:
                self.base_env.robot_timeline[robot_id] = max_timeline
    
    def reset(self):
        """重置环境"""
        self.base_env.reset()
        
        # 设置禁用机器人：不可用 + 时间线为最大时间线
        for robot_id in self.disabled_robots:
            if robot_id < self.num_of_robots:
                self.base_env.robot_state[robot_id] = 0
        
        # 初始时所有时间线都是0，禁用机器人也设为0
        self._update_disabled_robot_timelines()
        self._rebuild_state()
        return self.state
    
    def step(self, action):
        """执行动作"""
        robot_id = round(action[1])
        if robot_id >= self.num_of_robots:
            robot_id = self.num_of_robots - 1
        
        # 选择禁用机器人时返回惩罚
        if robot_id in self.disabled_robots:
            self.base_env.reward = -1
            self.base_env.last_action_state[0] = 0
            self.base_env.last_action_state[1] = 0
            self._rebuild_state()
            return self.state, -1, self.done
        
        # 执行基础环境的step
        state, reward, done = self.base_env.step(action)
        
        # 更新禁用机器人时间线为当前最大时间线
        self._update_disabled_robot_timelines()
        # 修正时间计算
        self._update_times_excluding_disabled()
        self._update_robot_availability()
        self._rebuild_state()
        
        return self.state, self.reward, self.done
    
    def calculate_robot_idle_times(self, reference_time=None):
        """计算空闲时间（代理到基础环境）"""
        return self.base_env.calculate_robot_idle_times(reference_time)


# =======================
# 神经网络定义
# =======================
class MuNet(nn.Module):
    """Actor网络"""
    def __init__(self, state_size, action_size, config=None):
        super(MuNet, self).__init__()
        if config is None:
            config = CONFIG
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
    """Critic网络"""
    def __init__(self, state_size, action_size, config=None):
        super(QNet, self).__init__()
        if config is None:
            config = CONFIG
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


class ReplayBuffer:
    """经验回放缓冲区"""
    def __init__(self, buffer_limit=None, device=None):
        if buffer_limit is None:
            buffer_limit = CONFIG['buffer_limit']
        self.buffer = collections.deque(maxlen=buffer_limit)
        self.device = device if device else DEVICE
    
    def put(self, transition):
        self.buffer.append(transition)
        
    def sample(self, n):
        actual_n = min(n, len(self.buffer))
        mini_batch = random.sample(self.buffer, actual_n)
        
        s_arr = np.array([t[0] for t in mini_batch], dtype=np.float32)
        a_arr = np.array([t[1] for t in mini_batch], dtype=np.float32)
        r_arr = np.array([t[2] for t in mini_batch], dtype=np.float32)
        s_prime_arr = np.array([t[3] for t in mini_batch], dtype=np.float32)
        done_arr = np.array([[0.0 if t[4] else 1.0] for t in mini_batch], dtype=np.float32)
        
        return (torch.from_numpy(s_arr).to(self.device),
                torch.from_numpy(a_arr).to(self.device),
                torch.from_numpy(r_arr).to(self.device),
                torch.from_numpy(s_prime_arr).to(self.device),
                torch.from_numpy(done_arr).to(self.device))
    
    def size(self):
        return len(self.buffer)


class OrnsteinUhlenbeckNoise:
    def __init__(self, mu, config=None):
        if config is None:
            config = CONFIG
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
# 约束感知的动作投影器
# =======================
class ConstraintAwareActionProjector:
    """约束感知的动作投影器 - 支持机器人禁用约束"""
    
    def __init__(self, config=None, disabled_robots=None):
        self.config = config if config else CONFIG
        self.disabled_robots = set(disabled_robots) if disabled_robots else set()
        
        # 统计约束违反
        self.constraint_violations = 0  # 机器人约束违反次数
        self.job_violations = 0  # 选择已完成作业的次数
        self.total_actions = 0
    
    def reset_statistics(self):
        """重置统计数据"""
        self.constraint_violations = 0
        self.job_violations = 0
        self.total_actions = 0
    
    def get_valid_jobs_and_robots(self, env):
        """获取有效的作业和机器人（考虑禁用约束）"""
        # 获取环境中可用的机器人
        if hasattr(env, 'robot_state') and isinstance(env.robot_state, np.ndarray):
            env_available = np.where(env.robot_state == 1)[0].tolist()
        else:
            env_available = [i for i, state in enumerate(env.robot_state) if state == 1]
        
        # 排除禁用的机器人
        valid_robots = [r for r in env_available if r not in self.disabled_robots]
        
        # 获取未完成的作业
        valid_jobs = []
        if hasattr(env, 'task_state') and hasattr(env, 'num_of_jobs'):
            task_state = env.task_state
            num_jobs = env.num_of_jobs
            for job_id in range(num_jobs):
                start_idx = job_id * 5
                end_idx = start_idx + 5
                if end_idx <= len(task_state):
                    if not np.all(task_state[start_idx:end_idx] == 1):
                        valid_jobs.append(job_id)
        else:
            for job_id in range(len(env.task_set)):
                job = env.task_set[job_id]
                finished = all(task.state for task in job)
                if not finished:
                    valid_jobs.append(job_id)
        
        return valid_jobs, valid_robots
    
    def project(self, env, a, check_violation=True):
        """
        执行动作投影（考虑机器人禁用约束）
        
        投影过程：
        1. 检查网络输出是否尝试选择禁用的机器人（记录约束违反）
        2. 从有效机器人集合中选择（排除禁用机器人和环境中不可用的机器人）
        3. 考虑任务-机器人兼容性（available_modules）
        
        Args:
            env: 环境对象
            a: 动作向量 (torch.Tensor)
            check_violation: 是否检查约束违反
        
        Returns:
            [job_id, robot_id, param], was_violated
        """
        if isinstance(a, torch.Tensor):
            a_ = a.detach().cpu().numpy()
        else:
            a_ = np.asarray(a, dtype=np.float32)
        
        self.total_actions += 1
        was_violated = False
        
        # 获取有效作业和机器人（已排除禁用的机器人和已完成的作业）
        valid_jobs, valid_robots = self.get_valid_jobs_and_robots(env)
        
        num_jobs = len(env.task_set)
        num_robots = len(env.robot_state)
        
        # 边界情况：所有作业已完成或无可用机器人
        if len(valid_jobs) == 0 or len(valid_robots) == 0:
            # 找一个任意有效的作业和机器人（用于返回，但环境会处理done状态）
            # 不应该发送已完成的作业，返回-1表示无效动作
            return [-1, -1, 0.0], False
        
        job_preference = a_[0]
        robot_preference = a_[1]
        
        # ====== 作业选择（从有效作业中选择，确保有未完成的操作）======
        target_job = int(job_preference * num_jobs)
        
        # 从有效作业中找到有未完成操作的作业
        job_id = None
        current_op_idx = None
        
        # 按偏好排序有效作业
        sorted_valid_jobs = sorted(valid_jobs, key=lambda x: abs(x - target_job))
        
        for candidate_job in sorted_valid_jobs:
            operations = env.task_set[candidate_job]
            # 查找该作业的第一个未完成操作
            for op_idx, task in enumerate(operations):
                if not task.state:
                    job_id = candidate_job
                    current_op_idx = op_idx
                    break
            if job_id is not None:
                break
        
        # 如果没有找到有效的作业（理论上不应该发生）
        if job_id is None:
            return [-1, -1, 0.0], False
        
        # 获取当前待执行的任务
        operations = env.task_set[job_id]
        current_task = operations[current_op_idx]
        
        # ====== 机器人选择（考虑约束和兼容性）======
        # 获取任务支持的机器人列表
        task_compatible_robots = set(getattr(current_task, 'available_modules', range(num_robots)))
        
        # 从有效机器人中筛选出同时满足：
        # 1. 环境可用（robot_state == 1）
        # 2. 未被禁用（不在disabled_robots中）- 已在valid_robots中处理
        # 3. 任务兼容（在available_modules中）
        compatible_valid_robots = [r for r in valid_robots if r in task_compatible_robots]
        
        # 如果没有同时满足约束和兼容性的机器人，退而求其次使用有效机器人
        if len(compatible_valid_robots) == 0:
            compatible_valid_robots = valid_robots
        
        # 从兼容的有效机器人中选择最接近偏好的
        target_robot = int(robot_preference * num_robots)
        robot_id = min(compatible_valid_robots, key=lambda x: abs(x - target_robot))
        
        # ====== 连续参数 ======
        param = float(np.clip(a_[2] if len(a_) > 2 else 0.0, 0.0, 1.0))
        
        # ====== 最终动作的约束违反检测 ======
        if check_violation:
            # 检查最终选择的机器人是否在禁用列表中
            if robot_id in self.disabled_robots:
                was_violated = True
                self.constraint_violations += 1
            # 检查最终选择的作业是否已完成
            if job_id not in valid_jobs:
                self.job_violations += 1
        
        return [job_id, robot_id, param], was_violated
    
    def get_violation_rate(self):
        """获取机器人约束违反率"""
        if self.total_actions == 0:
            return 0.0
        return self.constraint_violations / self.total_actions
    
    def get_job_violation_rate(self):
        """获取作业违反率（选择已完成作业的比例）"""
        if self.total_actions == 0:
            return 0.0
        return self.job_violations / self.total_actions
    
    def get_statistics(self):
        """获取所有统计数据"""
        return {
            'total_actions': self.total_actions,
            'robot_constraint_violations': self.constraint_violations,
            'robot_violation_rate': self.get_violation_rate(),
            'job_violations': self.job_violations,
            'job_violation_rate': self.get_job_violation_rate()
        }


# =======================
# 能耗计算函数
# =======================
def calculate_total_energy(env):
    """计算调度方案的总能耗"""
    total_energy = 0.0
    
    # 计算所有已完成任务的能耗
    for job_id in range(env.num_of_jobs):
        operations = env.task_set[job_id]
        for task in operations:
            if task.state:  # 任务已完成
                # 使用能量模型计算
                try:
                    import energy_model as EM
                    energy = EM.energy_dynamic(task.target_position, task.mass, task.processing_time)
                    total_energy += energy
                except:
                    # 如果能量模型不可用，使用简化估算
                    total_energy += task.processing_time * 10  # 假设每时间单位10能量
    
    # 加上空闲能耗
    idle_stats = env.calculate_robot_idle_times(env.future_time)
    idle_energy = idle_stats['summary']['total_idle_time'] * 5.0  # 空闲能耗率5
    total_energy += idle_energy
    
    return total_energy


# =======================
# 阶段一：加载模型并测试
# =======================
def phase1_test_pretrained_model(config):
    """阶段一：加载已训练模型，正常决策测试"""
    print("\n" + "="*80)
    print("阶段一：加载已训练模型进行正常决策测试")
    print("="*80)
    
    device = DEVICE
    
    # 创建环境
    env = ENV.Env(config['num_of_jobs'], config['num_of_robots'], 
                  config['alpha'], config['beta'])
    state_size = len(env.state)
    action_size = len(env.action)
    
    print(f"环境配置: Jobs={config['num_of_jobs']}, Robots={config['num_of_robots']}")
    print(f"状态维度: {state_size}, 动作维度: {action_size}")
    
    # 加载模型
    mu = MuNet(state_size, action_size, config).to(device)
    model_path = os.path.join(config['pretrained_model_path'], 'mu.pth')
    
    if os.path.exists(model_path):
        mu.load_state_dict(torch.load(model_path, map_location=device))
        mu.eval()
        print(f"成功加载模型: {model_path}")
    else:
        print(f"错误: 模型文件不存在 {model_path}")
        return None
    
    # 创建动作投影器（无约束）
    action_projector = ConstraintAwareActionProjector(config, disabled_robots=None)
    
    # 测试多个episode
    results = []
    
    for ep in range(config['num_test_episodes']):
        s = env.reset()
        done = False
        step = 0
        total_reward = 0
        
        s_buffer = torch.zeros(state_size, dtype=torch.float32, device=device)
        
        while not done and step < config['max_test_steps']:
            with torch.no_grad():
                s_buffer.copy_(torch.from_numpy(s.astype(np.float32)))
                a = mu(s_buffer.unsqueeze(0)).squeeze(0)
                a = torch.clamp(a, 0, 1)
            
            action, _ = action_projector.project(env, a, check_violation=False)
            
            # 检查是否有效动作
            if action[0] == -1:
                break
            
            s_prime, reward, done = env.step(action)
            
            total_reward += reward
            s = s_prime
            step += 1
        
        # 计算结果
        makespan = env.future_time
        total_energy = calculate_total_energy(env)
        
        results.append({
            'episode': ep + 1,
            'makespan': makespan,
            'total_energy': total_energy,
            'total_reward': total_reward,
            'steps': step
        })
        
        print(f"  Episode {ep+1}: Makespan={makespan:.2f}, Energy={total_energy:.2f}, Steps={step}")
    
    # 汇总结果
    avg_makespan = np.mean([r['makespan'] for r in results])
    avg_energy = np.mean([r['total_energy'] for r in results])
    
    print(f"\n阶段一结果汇总:")
    print(f"  平均完工时间 (Makespan): {avg_makespan:.2f}")
    print(f"  平均总能耗 (Energy): {avg_energy:.2f}")
    
    return {
        'phase': 1,
        'description': '正常决策测试（无约束）',
        'avg_makespan': avg_makespan,
        'avg_energy': avg_energy,
        'details': results
    }


# =======================
# 阶段二：带约束测试
# =======================
def phase2_test_with_constraints(config):
    """阶段二：增加约束（机器人0-5禁用），测试约束违反率"""
    print("\n" + "="*80)
    print("阶段二：增加约束测试（机器人0-5禁用）")
    print("="*80)
    
    device = DEVICE
    disabled_robots = config['disabled_robots']
    print(f"禁用机器人: {disabled_robots}")
    
    # 创建带约束的环境（禁用机器人不参与时间线计算）
    env = ConstrainedEnv(config['num_of_jobs'], config['num_of_robots'], 
                         config['alpha'], config['beta'],
                         disabled_robots=disabled_robots)
    state_size = len(env.state)
    action_size = len(env.action)
    
    # 加载模型
    mu = MuNet(state_size, action_size, config).to(device)
    model_path = os.path.join(config['pretrained_model_path'], 'mu.pth')
    
    if os.path.exists(model_path):
        mu.load_state_dict(torch.load(model_path, map_location=device))
        mu.eval()
        print(f"成功加载模型: {model_path}")
    else:
        print(f"错误: 模型文件不存在 {model_path}")
        return None
    
    # 创建约束感知的动作投影器
    action_projector = ConstraintAwareActionProjector(config, disabled_robots=disabled_robots)
    
    # 测试多个episode
    results = []
    
    for ep in range(config['num_test_episodes']):
        env.reset()
        action_projector.reset_statistics()
        
        s = env.state
        done = False
        step = 0
        total_reward = 0
        violation_count = 0
        
        s_buffer = torch.zeros(state_size, dtype=torch.float32, device=device)
        
        while not done and step < config['max_test_steps']:
            with torch.no_grad():
                s_buffer.copy_(torch.from_numpy(s.astype(np.float32)))
                a = mu(s_buffer.unsqueeze(0)).squeeze(0)
                a = torch.clamp(a, 0, 1)
            
            action, was_violated = action_projector.project(env, a, check_violation=True)
            
            # 检查是否有效动作（-1表示无有效作业/机器人）
            if action[0] == -1:
                # 所有作业已完成或无可用机器人
                break
            
            s_prime, reward, done = env.step(action)
            
            # 统计实际执行时的约束违反（环境返回负奖励表示违反）
            if reward == -1:
                violation_count += 1
            
            total_reward += reward
            s = s_prime
            step += 1
        
        # 计算结果
        makespan = env.future_time
        total_energy = calculate_total_energy(env)
        # 基于实际执行的约束违反率
        violation_rate = violation_count / step if step > 0 else 0.0
        
        results.append({
            'episode': ep + 1,
            'makespan': makespan,
            'total_energy': total_energy,
            'total_reward': total_reward,
            'steps': step,
            'violations': violation_count,
            'violation_rate': violation_rate
        })
        
        print(f"  Episode {ep+1}: Makespan={makespan:.2f}, Energy={total_energy:.2f}, "
              f"Violations={violation_count}, ViolationRate={violation_rate*100:.1f}%")
    
    # 汇总结果
    avg_makespan = np.mean([r['makespan'] for r in results])
    avg_energy = np.mean([r['total_energy'] for r in results])
    avg_violation_rate = np.mean([r['violation_rate'] for r in results])
    total_violations = sum([r['violations'] for r in results])
    
    print(f"\n阶段二结果汇总:")
    print(f"  平均完工时间 (Makespan): {avg_makespan:.2f}")
    print(f"  平均总能耗 (Energy): {avg_energy:.2f}")
    print(f"  平均约束违反率: {avg_violation_rate*100:.2f}%")
    print(f"  总约束违反次数: {total_violations}")
    
    return {
        'phase': 2,
        'description': f'约束测试（机器人{disabled_robots}禁用）',
        'disabled_robots': disabled_robots,
        'avg_makespan': avg_makespan,
        'avg_energy': avg_energy,
        'avg_violation_rate': avg_violation_rate,
        'total_violations': total_violations,
        'details': results
    }


# =======================
# 阶段三：重新训练并测试
# =======================
def train_step(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer, config, device):
    """单步训练"""
    s, a, r, s_prime, done_mask = memory.sample(config['batch_size'])
    
    with torch.no_grad():
        next_action = mu_target(s_prime)
        target_q = q_target(s_prime, next_action)
        target = r.unsqueeze(1) + config['gamma'] * target_q * done_mask
    
    current_q = q(s, a)
    q_loss = F.smooth_l1_loss(current_q, target)
    
    q_optimizer.zero_grad()
    q_loss.backward()
    torch.nn.utils.clip_grad_norm_(q.parameters(), 1.0)
    q_optimizer.step()
    
    mu_loss = -q(s, mu(s)).mean()
    
    mu_optimizer.zero_grad()
    mu_loss.backward()
    torch.nn.utils.clip_grad_norm_(mu.parameters(), 1.0)
    mu_optimizer.step()
    
    return q_loss.item(), mu_loss.item()


def soft_update(net, net_target, tau):
    """软更新目标网络"""
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)


def phase3_retrain_and_test(config):
    """阶段三：在新约束下重新训练模型，收敛后测试"""
    print("\n" + "="*80)
    print("阶段三：在新约束下重新训练模型")
    print("="*80)
    
    device = DEVICE
    disabled_robots = config['disabled_robots']
    print(f"禁用机器人: {disabled_robots}")
    print(f"训练episodes: {config['num_of_episodes']}")
    
    # 创建带约束的环境（禁用机器人不参与时间线计算）
    env = ConstrainedEnv(config['num_of_jobs'], config['num_of_robots'], 
                         config['alpha'], config['beta'],
                         disabled_robots=disabled_robots)
    state_size = len(env.state)
    action_size = len(env.action)
    
    # 初始化网络（从预训练模型加载）
    mu = MuNet(state_size, action_size, config).to(device)
    mu_target = MuNet(state_size, action_size, config).to(device)
    q = QNet(state_size, action_size, config).to(device)
    q_target = QNet(state_size, action_size, config).to(device)
    
    # 根据配置决定是否加载预训练模型
    if not config.get('train_from_scratch', False):
        model_path = os.path.join(config['pretrained_model_path'], 'mu.pth')
        q_path = os.path.join(config['pretrained_model_path'], 'q.pth')
        
        if os.path.exists(model_path):
            mu.load_state_dict(torch.load(model_path, map_location=device))
            mu_target.load_state_dict(mu.state_dict())
            print(f"从预训练模型初始化Actor: {model_path}")
        
        if os.path.exists(q_path):
            q.load_state_dict(torch.load(q_path, map_location=device))
            q_target.load_state_dict(q.state_dict())
            print(f"从预训练模型初始化Critic: {q_path}")
    else:
        print("从头训练模型（不加载预训练权重）")
        mu_target.load_state_dict(mu.state_dict())
        q_target.load_state_dict(q.state_dict())
    
    # 优化器
    mu_optimizer = optim.Adam(mu.parameters(), lr=config['lr_mu'])
    q_optimizer = optim.Adam(q.parameters(), lr=config['lr_q'])
    
    # 训练组件
    memory = ReplayBuffer(config['buffer_limit'], device=device)
    ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(action_size), config=config)
    action_projector = ConstraintAwareActionProjector(config, disabled_robots=disabled_robots)
    
    # 预分配GPU张量
    s_buffer = torch.zeros(state_size, dtype=torch.float32, device=device)
    
    # 训练记录
    score_record = []
    best_avg_score = float('-inf')
    
    print(f"\n开始训练...")
    training_start = time.time()
    
    for n_epi in range(config['num_of_episodes']):
        s = env.reset()
        done = False
        episode_reward = 0.0
        step = 0
        
        while not done and step < config['max_test_steps']:
            with torch.no_grad():
                s_buffer.copy_(torch.from_numpy(s.astype(np.float32)))
                a = mu(s_buffer.unsqueeze(0)).squeeze(0)
                
                # 添加探索噪声（使用配置的噪声比例）
                noise = ou_noise()
                noise_scale = config.get('noise_scale', 0.2)
                a_np = a.cpu().numpy() + noise * noise_scale
                a_np = np.clip(a_np, 0, 1)
            
            # 动作投影（约束感知）
            a_tensor = torch.from_numpy(a_np.astype(np.float32)).to(device)
            action, was_violated = action_projector.project(env, a_tensor, check_violation=True)
            
            # 检查是否有效动作
            if action[0] == -1:
                break
            
            s_prime, r, done = env.step(action)
            
            # 如果网络输出被投影修正，可选添加惩罚
            violation_penalty = config.get('violation_penalty', 0.0)
            if was_violated and violation_penalty > 0:
                r = r - violation_penalty
            
            # 存储网络输出动作
            memory.put((s, a_np, r, s_prime, done))
            s = s_prime
            episode_reward += r
            step += 1
        
        score_record.append(episode_reward)
        
        # 训练更新
        if memory.size() > config['memory_threshold']:
            for _ in range(config['training_iterations']):
                train_step(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer, config, device)
                soft_update(mu, mu_target, config['tau'])
                soft_update(q, q_target, config['tau'])
        
        # 打印进度
        if n_epi % config['print_interval'] == 0 and n_epi != 0:
            avg_score = np.mean(score_record[-config['print_interval']:])
            elapsed = time.time() - training_start
            print(f"  Episode {n_epi}: Avg Score = {avg_score:.4f}, Time = {elapsed:.1f}s")
            
            if avg_score > best_avg_score:
                best_avg_score = avg_score
    
    training_time = time.time() - training_start
    print(f"\n训练完成! 用时: {training_time:.2f}s")
    print(f"最佳平均分数: {best_avg_score:.4f}")
    
    # 测试重新训练后的模型
    print(f"\n测试重新训练后的模型...")
    mu.eval()
    
    results = []
    action_projector.reset_statistics()
    
    for ep in range(config['num_test_episodes']):
        env.reset()
        action_projector.reset_statistics()
        
        s = env.state
        done = False
        step = 0
        total_reward = 0
        violation_count = 0
        
        while not done and step < config['max_test_steps']:
            with torch.no_grad():
                s_buffer.copy_(torch.from_numpy(s.astype(np.float32)))
                a = mu(s_buffer.unsqueeze(0)).squeeze(0)
                a = torch.clamp(a, 0, 1)
            
            action, was_violated = action_projector.project(env, a, check_violation=True)
            
            # 检查是否有效动作
            if action[0] == -1:
                break
            
            s_prime, reward, done = env.step(action)
            
            # 统计实际执行时的约束违反（环境返回负奖励表示违反）
            if reward == -1:
                violation_count += 1
            
            total_reward += reward
            s = s_prime
            step += 1
        
        makespan = env.future_time
        total_energy = calculate_total_energy(env)
        # 基于实际执行的约束违反率
        violation_rate = violation_count / step if step > 0 else 0.0
        
        results.append({
            'episode': ep + 1,
            'makespan': makespan,
            'total_energy': total_energy,
            'total_reward': total_reward,
            'steps': step,
            'violations': violation_count,
            'violation_rate': violation_rate
        })
        
        print(f"  Episode {ep+1}: Makespan={makespan:.2f}, Energy={total_energy:.2f}, "
              f"Violations={violation_count}, ViolationRate={violation_rate*100:.1f}%")
    
    # 汇总结果
    avg_makespan = np.mean([r['makespan'] for r in results])
    avg_energy = np.mean([r['total_energy'] for r in results])
    avg_violation_rate = np.mean([r['violation_rate'] for r in results])
    
    print(f"\n阶段三结果汇总:")
    print(f"  平均完工时间 (Makespan): {avg_makespan:.2f}")
    print(f"  平均总能耗 (Energy): {avg_energy:.2f}")
    print(f"  平均约束违反率: {avg_violation_rate*100:.2f}%")
    
    return {
        'phase': 3,
        'description': f'重新训练后测试（机器人{disabled_robots}禁用）',
        'disabled_robots': disabled_robots,
        'training_episodes': config['num_of_episodes'],
        'training_time': training_time,
        'best_avg_score': best_avg_score,
        'avg_makespan': avg_makespan,
        'avg_energy': avg_energy,
        'avg_violation_rate': avg_violation_rate,
        'score_record': score_record,
        'details': results
    }


# =======================
# 结果可视化
# =======================
def visualize_results(phase1_result, phase2_result, phase3_result):
    """可视化三个阶段的结果对比"""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    phases = ['Phase 1\n(No Constraint)', 'Phase 2\n(With Constraint)', 'Phase 3\n(Retrained)']
    
    # 完工时间对比
    makespans = [phase1_result['avg_makespan'], 
                 phase2_result['avg_makespan'], 
                 phase3_result['avg_makespan']]
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    ax1 = axes[0]
    bars1 = ax1.bar(phases, makespans, color=colors)
    ax1.set_ylabel('Makespan')
    ax1.set_title('Average Makespan Comparison')
    ax1.bar_label(bars1, fmt='%.2f')
    ax1.grid(axis='y', alpha=0.3)
    
    # 能耗对比
    energies = [phase1_result['avg_energy'], 
                phase2_result['avg_energy'], 
                phase3_result['avg_energy']]
    
    ax2 = axes[1]
    bars2 = ax2.bar(phases, energies, color=colors)
    ax2.set_ylabel('Total Energy')
    ax2.set_title('Average Energy Consumption')
    ax2.bar_label(bars2, fmt='%.2f')
    ax2.grid(axis='y', alpha=0.3)
    
    # 约束违反率对比
    violations = [0, 
                  phase2_result['avg_violation_rate'] * 100, 
                  phase3_result['avg_violation_rate'] * 100]
    
    ax3 = axes[2]
    bars3 = ax3.bar(phases, violations, color=colors)
    ax3.set_ylabel('Violation Rate (%)')
    ax3.set_title('Constraint Violation Rate')
    ax3.bar_label(bars3, fmt='%.1f%%')
    ax3.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"constraint_change_results_{timestamp}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n结果图片已保存: {save_path}")
    
    plt.show()
    
    return save_path


def save_experiment_results(phase1_result, phase2_result, phase3_result, config):
    """保存实验结果到JSON文件"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results = {
        'timestamp': timestamp,
        'config': {
            'num_of_jobs': config['num_of_jobs'],
            'num_of_robots': config['num_of_robots'],
            'disabled_robots': config['disabled_robots'],
            'training_episodes': config['num_of_episodes']
        },
        'phase1': {
            'description': phase1_result['description'],
            'avg_makespan': phase1_result['avg_makespan'],
            'avg_energy': phase1_result['avg_energy']
        },
        'phase2': {
            'description': phase2_result['description'],
            'avg_makespan': phase2_result['avg_makespan'],
            'avg_energy': phase2_result['avg_energy'],
            'avg_violation_rate': phase2_result['avg_violation_rate'],
            'total_violations': phase2_result['total_violations']
        },
        'phase3': {
            'description': phase3_result['description'],
            'training_time': phase3_result['training_time'],
            'best_avg_score': phase3_result['best_avg_score'],
            'avg_makespan': phase3_result['avg_makespan'],
            'avg_energy': phase3_result['avg_energy'],
            'avg_violation_rate': phase3_result['avg_violation_rate']
        },
        'comparison': {
            'makespan_improvement_phase2_to_phase3': 
                (phase2_result['avg_makespan'] - phase3_result['avg_makespan']) / phase2_result['avg_makespan'] * 100,
            'energy_improvement_phase2_to_phase3':
                (phase2_result['avg_energy'] - phase3_result['avg_energy']) / phase2_result['avg_energy'] * 100,
            'violation_reduction':
                phase2_result['avg_violation_rate'] - phase3_result['avg_violation_rate']
        }
    }
    
    save_path = f"constraint_change_results_{timestamp}.json"
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"实验结果已保存: {save_path}")
    return save_path


# =======================
# 主函数
# =======================
def main():
    """主函数 - 执行三阶段实验"""
    print("="*80)
    print("LIRL约束变化实验")
    print("="*80)
    print(f"实验配置:")
    print(f"  Jobs: {CONFIG['num_of_jobs']}")
    print(f"  Robots: {CONFIG['num_of_robots']}")
    print(f"  禁用机器人: {CONFIG['disabled_robots']}")
    print(f"  预训练模型: {CONFIG['pretrained_model_path']}")
    print("="*80)
    
    print_gpu_info()
    
    # 阶段一：正常测试
    phase1_result = phase1_test_pretrained_model(CONFIG)
    
    # 阶段二：带约束测试
    phase2_result = phase2_test_with_constraints(CONFIG)
    
    # 阶段三：重新训练并测试
    phase3_result = phase3_retrain_and_test(CONFIG)
    
    # 结果对比
    print("\n" + "="*80)
    print("实验结果对比")
    print("="*80)
    print(f"{'指标':<20} {'阶段一(无约束)':<20} {'阶段二(有约束)':<20} {'阶段三(重训练)':<20}")
    print("-"*80)
    print(f"{'平均完工时间':<20} {phase1_result['avg_makespan']:<20.2f} {phase2_result['avg_makespan']:<20.2f} {phase3_result['avg_makespan']:<20.2f}")
    print(f"{'平均总能耗':<20} {phase1_result['avg_energy']:<20.2f} {phase2_result['avg_energy']:<20.2f} {phase3_result['avg_energy']:<20.2f}")
    print(f"{'约束违反率(%)':<20} {'N/A':<20} {phase2_result['avg_violation_rate']*100:<20.2f} {phase3_result['avg_violation_rate']*100:<20.2f}")
    print("="*80)
    
    # 计算改进
    if phase2_result['avg_makespan'] > 0:
        makespan_improve = (phase2_result['avg_makespan'] - phase3_result['avg_makespan']) / phase2_result['avg_makespan'] * 100
        print(f"\n重训练后完工时间改进: {makespan_improve:.2f}%")
    
    if phase2_result['avg_energy'] > 0:
        energy_improve = (phase2_result['avg_energy'] - phase3_result['avg_energy']) / phase2_result['avg_energy'] * 100
        print(f"重训练后能耗改进: {energy_improve:.2f}%")
    
    violation_reduce = (phase2_result['avg_violation_rate'] - phase3_result['avg_violation_rate']) * 100
    print(f"约束违反率降低: {violation_reduce:.2f}%")
    
    # 可视化结果
    try:
        visualize_results(phase1_result, phase2_result, phase3_result)
    except Exception as e:
        print(f"可视化失败: {e}")
    
    # 保存结果
    if CONFIG['save_results']:
        save_experiment_results(phase1_result, phase2_result, phase3_result, CONFIG)
    
    print("\n实验完成!")
    
    return phase1_result, phase2_result, phase3_result


if __name__ == "__main__":
    results = main()
