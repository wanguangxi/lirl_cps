#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LIRL Constraint Change Experiment
=================================
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

sys.path.append(os.path.join(os.path.dirname(__file__), '../env'))
try:
    import importlib
    ENV = importlib.import_module('env')
    if not hasattr(ENV, 'Env'):
        ENV = importlib.import_module('env.env')
except Exception:
    from env import env as ENV

# =======================
# GPU Device Configuration
# =======================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def print_gpu_info():
    """Print GPU information"""
    if torch.cuda.is_available():
        print(f"GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("GPU not available, using CPU")

# =======================
# Experiment Configuration
# =======================
CONFIG = {
    # Environment parameters (consistent with medium scale)
    'num_of_jobs': 100,
    'num_of_robots': 50,
    'alpha': 0.5,
    'beta': 0.5,
    
    # Constraint parameters
    'disabled_robots': [0, 1, 2, 3, 4, 5],
    
    # Network architecture
    'hidden_dim1': 128,
    'hidden_dim2': 64,
    'critic_hidden': 32,
    
    # Training parameters (Phase 3) - fine-tune pretrained model
    'lr_mu': 0.00005,
    'lr_q': 0.0001,
    'gamma': 0.98,
    'batch_size': 128,
    'buffer_limit': 1000000,
    'tau': 0.001,
    'memory_threshold': 500,
    'training_iterations': 5,
    'num_of_episodes': 200,
    'noise_params': {'theta': 0.1, 'dt': 0.05, 'sigma': 0.1},
    'noise_scale': 0.1,
    'train_from_scratch': False,
    'violation_penalty': 0.0,
    
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
# Environment wrapper class supporting disabled robots
# =======================
class ConstrainedEnv:
    """Wrapper for original environment, supporting disabled robot constraints
    
    Disabled robots do not participate in timeline calculation, ensuring scheduling can proceed normally
    """
    def __init__(self, num_of_jobs, num_of_robots, alpha, beta, disabled_robots=None):
        self.base_env = ENV.Env(num_of_jobs, num_of_robots, alpha, beta)
        self.disabled_robots = set(disabled_robots) if disabled_robots else set()
        self.num_of_jobs = num_of_jobs
        self.num_of_robots = num_of_robots
        
        self._sync_from_base()
    
    def _sync_from_base(self):
        """Sync attributes from base environment"""
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
        """Get timeline list of non-disabled robots"""
        return [self.base_env.robot_timeline[i] 
                for i in range(self.num_of_robots) 
                if i not in self.disabled_robots]
    
    def _update_times_excluding_disabled(self):
        """Update current_time and future_time, excluding disabled robots"""
        active = self._get_active_timelines()
        if active:
            self.base_env.current_time = np.min(active)
            self.base_env.future_time = np.max(active)
    
    def _update_robot_availability(self):
        """Update robot availability status, disabled robots are always unavailable"""
        for robot_id in range(self.num_of_robots):
            if robot_id in self.disabled_robots:
                self.base_env.robot_state[robot_id] = 0
            elif self.base_env.robot_timeline[robot_id] <= self.base_env.current_time:
                self.base_env.robot_state[robot_id] = 1
            else:
                self.base_env.robot_state[robot_id] = 0
    
    def _rebuild_state(self):
        """Rebuild state vector"""
        self.base_env.state = np.concatenate((
            self.base_env.task_state,
            self.base_env.robot_state,
            self.base_env.task_prcoessing_time_state,
            self.base_env.last_action_state
        ))
        self._sync_from_base()
    
    def _update_disabled_robot_timelines(self):
        """Set disabled robot timelines to the maximum of all robot timelines"""
        active_timelines = self._get_active_timelines()
        if active_timelines:
            max_timeline = np.max(active_timelines)
        else:
            max_timeline = 0.0
        
        for robot_id in self.disabled_robots:
            if robot_id < self.num_of_robots:
                self.base_env.robot_timeline[robot_id] = max_timeline
    
    def reset(self):
        """Reset environment"""
        self.base_env.reset()
        
        for robot_id in self.disabled_robots:
            if robot_id < self.num_of_robots:
                self.base_env.robot_state[robot_id] = 0
        
        self._update_disabled_robot_timelines()
        self._rebuild_state()
        return self.state
    
    def step(self, action):
        """Execute action"""
        robot_id = round(action[1])
        if robot_id >= self.num_of_robots:
            robot_id = self.num_of_robots - 1
        
        if robot_id in self.disabled_robots:
            self.base_env.reward = -1
            self.base_env.last_action_state[0] = 0
            self.base_env.last_action_state[1] = 0
            self._rebuild_state()
            return self.state, -1, self.done
        
        state, reward, done = self.base_env.step(action)
        
        self._update_disabled_robot_timelines()
        self._update_times_excluding_disabled()
        self._update_robot_availability()
        self._rebuild_state()
        
        return self.state, self.reward, self.done
    
    def calculate_robot_idle_times(self, reference_time=None):
        """Calculate idle time (proxy to base environment)"""
        return self.base_env.calculate_robot_idle_times(reference_time)


# =======================
# Neural Network Definitions
# =======================
class MuNet(nn.Module):
    """Actor network"""
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
    """Critic network"""
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
    """Experience replay buffer"""
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
# Constraint-aware action projector
# =======================
class ConstraintAwareActionProjector:
    """Constraint-aware action projector - supports robot disable constraints"""
    
    def __init__(self, config=None, disabled_robots=None):
        self.config = config if config else CONFIG
        self.disabled_robots = set(disabled_robots) if disabled_robots else set()
        
        self.constraint_violations = 0
        self.job_violations = 0
        self.total_actions = 0
    
    def reset_statistics(self):
        """Reset statistics"""
        self.constraint_violations = 0
        self.job_violations = 0
        self.total_actions = 0
    
    def get_valid_jobs_and_robots(self, env):
        """Get valid jobs and robots (considering disable constraints)"""
        if hasattr(env, 'robot_state') and isinstance(env.robot_state, np.ndarray):
            env_available = np.where(env.robot_state == 1)[0].tolist()
        else:
            env_available = [i for i, state in enumerate(env.robot_state) if state == 1]
        
        valid_robots = [r for r in env_available if r not in self.disabled_robots]
        
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
        Execute action projection (considering robot disable constraints)
        
        Projection process:
        1. Check if network output attempts to select disabled robot (record constraint violation)
        2. Select from valid robot set (exclude disabled robots and unavailable robots in environment)
        3. Consider task-robot compatibility (available_modules)
        
        Args:
            env: Environment object
            a: Action vector (torch.Tensor)
            check_violation: Whether to check constraint violation
        
        Returns:
            [job_id, robot_id, param], was_violated
        """
        if isinstance(a, torch.Tensor):
            a_ = a.detach().cpu().numpy()
        else:
            a_ = np.asarray(a, dtype=np.float32)
        
        self.total_actions += 1
        was_violated = False
        
        valid_jobs, valid_robots = self.get_valid_jobs_and_robots(env)
        
        num_jobs = len(env.task_set)
        num_robots = len(env.robot_state)
        
        if len(valid_jobs) == 0 or len(valid_robots) == 0:
            return [-1, -1, 0.0], False
        
        job_preference = a_[0]
        robot_preference = a_[1]
        
        target_job = int(job_preference * num_jobs)
        
        job_id = None
        current_op_idx = None
        
        sorted_valid_jobs = sorted(valid_jobs, key=lambda x: abs(x - target_job))
        
        for candidate_job in sorted_valid_jobs:
            operations = env.task_set[candidate_job]
            for op_idx, task in enumerate(operations):
                if not task.state:
                    job_id = candidate_job
                    current_op_idx = op_idx
                    break
            if job_id is not None:
                break
        
        if job_id is None:
            return [-1, -1, 0.0], False
        
        operations = env.task_set[job_id]
        current_task = operations[current_op_idx]
        
        task_compatible_robots = set(getattr(current_task, 'available_modules', range(num_robots)))
        
        compatible_valid_robots = [r for r in valid_robots if r in task_compatible_robots]
        
        if len(compatible_valid_robots) == 0:
            compatible_valid_robots = valid_robots
        
        target_robot = int(robot_preference * num_robots)
        robot_id = min(compatible_valid_robots, key=lambda x: abs(x - target_robot))
        
        param = float(np.clip(a_[2] if len(a_) > 2 else 0.0, 0.0, 1.0))
        
        if check_violation:
            if robot_id in self.disabled_robots:
                was_violated = True
                self.constraint_violations += 1
            if job_id not in valid_jobs:
                self.job_violations += 1
        
        return [job_id, robot_id, param], was_violated
    
    def get_violation_rate(self):
        """Get robot constraint violation rate"""
        if self.total_actions == 0:
            return 0.0
        return self.constraint_violations / self.total_actions
    
    def get_job_violation_rate(self):
        """Get job violation rate (proportion of selecting completed jobs)"""
        if self.total_actions == 0:
            return 0.0
        return self.job_violations / self.total_actions
    
    def get_statistics(self):
        """Get all statistics"""
        return {
            'total_actions': self.total_actions,
            'robot_constraint_violations': self.constraint_violations,
            'robot_violation_rate': self.get_violation_rate(),
            'job_violations': self.job_violations,
            'job_violation_rate': self.get_job_violation_rate()
        }


# =======================
# Energy calculation function
# =======================
def calculate_total_energy(env):
    """Calculate total energy consumption of scheduling solution"""
    total_energy = 0.0
    
    for job_id in range(env.num_of_jobs):
        operations = env.task_set[job_id]
        for task in operations:
            if task.state:
                try:
                    import energy_model as EM
                    energy = EM.energy_dynamic(task.target_position, task.mass, task.processing_time)
                    total_energy += energy
                except:
                    total_energy += task.processing_time * 10
    
    idle_stats = env.calculate_robot_idle_times(env.future_time)
    idle_energy = idle_stats['summary']['total_idle_time'] * 5.0
    total_energy += idle_energy
    
    return total_energy


# =======================
# Phase 1: Load model and test
# =======================
def phase1_test_pretrained_model(config):
    """Phase 1: Load trained model, normal decision testing"""
    print("\n" + "="*80)
    print("Phase 1: Load trained model for normal decision testing")
    print("="*80)
    
    device = DEVICE
    
    env = ENV.Env(config['num_of_jobs'], config['num_of_robots'], 
                  config['alpha'], config['beta'])
    state_size = len(env.state)
    action_size = len(env.action)
    
    print(f"Environment config: Jobs={config['num_of_jobs']}, Robots={config['num_of_robots']}")
    print(f"State dimension: {state_size}, Action dimension: {action_size}")
    
    mu = MuNet(state_size, action_size, config).to(device)
    model_path = os.path.join(config['pretrained_model_path'], 'mu.pth')
    
    if os.path.exists(model_path):
        mu.load_state_dict(torch.load(model_path, map_location=device))
        mu.eval()
        print(f"Successfully loaded model: {model_path}")
    else:
        print(f"Error: Model file does not exist {model_path}")
        return None
    
    action_projector = ConstraintAwareActionProjector(config, disabled_robots=None)
    
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
    
    avg_makespan = np.mean([r['makespan'] for r in results])
    avg_energy = np.mean([r['total_energy'] for r in results])
    
    print(f"\nPhase 1 Results Summary:")
    print(f"  Average Makespan: {avg_makespan:.2f}")
    print(f"  Average Total Energy: {avg_energy:.2f}")
    
    return {
        'phase': 1,
        'description': 'Normal decision testing (no constraints)',
        'avg_makespan': avg_makespan,
        'avg_energy': avg_energy,
        'details': results
    }


# =======================
# Phase 2: Test with constraints
# =======================
def phase2_test_with_constraints(config):
    """Phase 2: Add constraints (robots 0-5 disabled), test constraint violation rate"""
    print("\n" + "="*80)
    print("Phase 2: Constraint testing (robots 0-5 disabled)")
    print("="*80)
    
    device = DEVICE
    disabled_robots = config['disabled_robots']
    print(f"Disabled robots: {disabled_robots}")
    
    env = ConstrainedEnv(config['num_of_jobs'], config['num_of_robots'], 
                         config['alpha'], config['beta'],
                         disabled_robots=disabled_robots)
    state_size = len(env.state)
    action_size = len(env.action)
    
    mu = MuNet(state_size, action_size, config).to(device)
    model_path = os.path.join(config['pretrained_model_path'], 'mu.pth')
    
    if os.path.exists(model_path):
        mu.load_state_dict(torch.load(model_path, map_location=device))
        mu.eval()
        print(f"Successfully loaded model: {model_path}")
    else:
        print(f"Error: Model file does not exist {model_path}")
        return None
    
    action_projector = ConstraintAwareActionProjector(config, disabled_robots=disabled_robots)
    
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
            
            if action[0] == -1:
                break
            
            s_prime, reward, done = env.step(action)
            
            if reward == -1:
                violation_count += 1
            
            total_reward += reward
            s = s_prime
            step += 1
        
        makespan = env.future_time
        total_energy = calculate_total_energy(env)
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
    
    avg_makespan = np.mean([r['makespan'] for r in results])
    avg_energy = np.mean([r['total_energy'] for r in results])
    avg_violation_rate = np.mean([r['violation_rate'] for r in results])
    total_violations = sum([r['violations'] for r in results])
    
    print(f"\nPhase 2 Results Summary:")
    print(f"  Average Makespan: {avg_makespan:.2f}")
    print(f"  Average Total Energy: {avg_energy:.2f}")
    print(f"  Average Constraint Violation Rate: {avg_violation_rate*100:.2f}%")
    print(f"  Total Constraint Violations: {total_violations}")
    
    return {
        'phase': 2,
        'description': f'Constraint testing (robots {disabled_robots} disabled)',
        'disabled_robots': disabled_robots,
        'avg_makespan': avg_makespan,
        'avg_energy': avg_energy,
        'avg_violation_rate': avg_violation_rate,
        'total_violations': total_violations,
        'details': results
    }


# =======================
# Phase 3: Retrain and test
# =======================
def train_step(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer, config, device):
    """Single training step"""
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
    """Soft update target network"""
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)


def phase3_retrain_and_test(config):
    """Phase 3: Retrain model under new constraints, test after convergence"""
    print("\n" + "="*80)
    print("Phase 3: Retrain model under new constraints")
    print("="*80)
    
    device = DEVICE
    disabled_robots = config['disabled_robots']
    print(f"Disabled robots: {disabled_robots}")
    print(f"Training episodes: {config['num_of_episodes']}")
    
    env = ConstrainedEnv(config['num_of_jobs'], config['num_of_robots'], 
                         config['alpha'], config['beta'],
                         disabled_robots=disabled_robots)
    state_size = len(env.state)
    action_size = len(env.action)
    
    mu = MuNet(state_size, action_size, config).to(device)
    mu_target = MuNet(state_size, action_size, config).to(device)
    q = QNet(state_size, action_size, config).to(device)
    q_target = QNet(state_size, action_size, config).to(device)
    
    if not config.get('train_from_scratch', False):
        model_path = os.path.join(config['pretrained_model_path'], 'mu.pth')
        q_path = os.path.join(config['pretrained_model_path'], 'q.pth')
        
        if os.path.exists(model_path):
            mu.load_state_dict(torch.load(model_path, map_location=device))
            mu_target.load_state_dict(mu.state_dict())
            print(f"Initialize Actor from pretrained model: {model_path}")
        
        if os.path.exists(q_path):
            q.load_state_dict(torch.load(q_path, map_location=device))
            q_target.load_state_dict(q.state_dict())
            print(f"Initialize Critic from pretrained model: {q_path}")
    else:
        print("Training from scratch (not loading pretrained weights)")
        mu_target.load_state_dict(mu.state_dict())
        q_target.load_state_dict(q.state_dict())
    
    mu_optimizer = optim.Adam(mu.parameters(), lr=config['lr_mu'])
    q_optimizer = optim.Adam(q.parameters(), lr=config['lr_q'])
    
    memory = ReplayBuffer(config['buffer_limit'], device=device)
    ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(action_size), config=config)
    action_projector = ConstraintAwareActionProjector(config, disabled_robots=disabled_robots)
    
    s_buffer = torch.zeros(state_size, dtype=torch.float32, device=device)
    
    score_record = []
    best_avg_score = float('-inf')
    
    print(f"\nStarting training...")
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
                
                noise = ou_noise()
                noise_scale = config.get('noise_scale', 0.2)
                a_np = a.cpu().numpy() + noise * noise_scale
                a_np = np.clip(a_np, 0, 1)
            
            a_tensor = torch.from_numpy(a_np.astype(np.float32)).to(device)
            action, was_violated = action_projector.project(env, a_tensor, check_violation=True)
            
            if action[0] == -1:
                break
            
            s_prime, r, done = env.step(action)
            
            violation_penalty = config.get('violation_penalty', 0.0)
            if was_violated and violation_penalty > 0:
                r = r - violation_penalty
            
            memory.put((s, a_np, r, s_prime, done))
            s = s_prime
            episode_reward += r
            step += 1
        
        score_record.append(episode_reward)
        
        if memory.size() > config['memory_threshold']:
            for _ in range(config['training_iterations']):
                train_step(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer, config, device)
                soft_update(mu, mu_target, config['tau'])
                soft_update(q, q_target, config['tau'])
        
        if n_epi % config['print_interval'] == 0 and n_epi != 0:
            avg_score = np.mean(score_record[-config['print_interval']:])
            elapsed = time.time() - training_start
            print(f"  Episode {n_epi}: Avg Score = {avg_score:.4f}, Time = {elapsed:.1f}s")
            
            if avg_score > best_avg_score:
                best_avg_score = avg_score
    
    training_time = time.time() - training_start
    print(f"\nTraining completed! Time: {training_time:.2f}s")
    print(f"Best average score: {best_avg_score:.4f}")
    
    print(f"\nTesting retrained model...")
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
            
            if action[0] == -1:
                break
            
            s_prime, reward, done = env.step(action)
            
            if reward == -1:
                violation_count += 1
            
            total_reward += reward
            s = s_prime
            step += 1
        
        makespan = env.future_time
        total_energy = calculate_total_energy(env)
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
    
    avg_makespan = np.mean([r['makespan'] for r in results])
    avg_energy = np.mean([r['total_energy'] for r in results])
    avg_violation_rate = np.mean([r['violation_rate'] for r in results])
    
    print(f"\nPhase 3 Results Summary:")
    print(f"  Average Makespan: {avg_makespan:.2f}")
    print(f"  Average Total Energy: {avg_energy:.2f}")
    print(f"  Average Constraint Violation Rate: {avg_violation_rate*100:.2f}%")
    
    return {
        'phase': 3,
        'description': f'Retrained testing (robots {disabled_robots} disabled)',
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
# Result visualization
# =======================
def visualize_results(phase1_result, phase2_result, phase3_result):
    """Visualize comparison of results from three phases"""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    phases = ['Phase 1\n(No Constraint)', 'Phase 2\n(With Constraint)', 'Phase 3\n(Retrained)']
    
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
    
    energies = [phase1_result['avg_energy'], 
                phase2_result['avg_energy'], 
                phase3_result['avg_energy']]
    
    ax2 = axes[1]
    bars2 = ax2.bar(phases, energies, color=colors)
    ax2.set_ylabel('Total Energy')
    ax2.set_title('Average Energy Consumption')
    ax2.bar_label(bars2, fmt='%.2f')
    ax2.grid(axis='y', alpha=0.3)
    
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
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"constraint_change_results_{timestamp}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nResult image saved: {save_path}")
    
    plt.show()
    
    return save_path


def save_experiment_results(phase1_result, phase2_result, phase3_result, config):
    """Save experiment results to JSON file"""
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
    
    print(f"Experiment results saved: {save_path}")
    return save_path


# =======================
# Main Function
# =======================
def main():
    """Main function - execute three-phase experiment"""
    print("="*80)
    print("LIRL Constraint Change Experiment")
    print("="*80)
    print(f"Experiment Configuration:")
    print(f"  Jobs: {CONFIG['num_of_jobs']}")
    print(f"  Robots: {CONFIG['num_of_robots']}")
    print(f"  Disabled robots: {CONFIG['disabled_robots']}")
    print(f"  Pretrained model: {CONFIG['pretrained_model_path']}")
    print("="*80)
    
    print_gpu_info()
    
    phase1_result = phase1_test_pretrained_model(CONFIG)
    
    phase2_result = phase2_test_with_constraints(CONFIG)
    
    phase3_result = phase3_retrain_and_test(CONFIG)
    
    print("\n" + "="*80)
    print("Experiment Results Comparison")
    print("="*80)
    print(f"{'Metric':<20} {'Phase 1 (No Constraint)':<25} {'Phase 2 (With Constraint)':<25} {'Phase 3 (Retrained)':<25}")
    print("-"*95)
    print(f"{'Average Makespan':<20} {phase1_result['avg_makespan']:<25.2f} {phase2_result['avg_makespan']:<25.2f} {phase3_result['avg_makespan']:<25.2f}")
    print(f"{'Average Total Energy':<20} {phase1_result['avg_energy']:<25.2f} {phase2_result['avg_energy']:<25.2f} {phase3_result['avg_energy']:<25.2f}")
    print(f"{'Violation Rate (%)':<20} {'N/A':<25} {phase2_result['avg_violation_rate']*100:<25.2f} {phase3_result['avg_violation_rate']*100:<25.2f}")
    print("="*95)
    
    if phase2_result['avg_makespan'] > 0:
        makespan_improve = (phase2_result['avg_makespan'] - phase3_result['avg_makespan']) / phase2_result['avg_makespan'] * 100
        print(f"\nMakespan improvement after retraining: {makespan_improve:.2f}%")
    
    if phase2_result['avg_energy'] > 0:
        energy_improve = (phase2_result['avg_energy'] - phase3_result['avg_energy']) / phase2_result['avg_energy'] * 100
        print(f"Energy improvement after retraining: {energy_improve:.2f}%")
    
    violation_reduce = (phase2_result['avg_violation_rate'] - phase3_result['avg_violation_rate']) * 100
    print(f"Constraint violation rate reduction: {violation_reduce:.2f}%")
    
    try:
        visualize_results(phase1_result, phase2_result, phase3_result)
    except Exception as e:
        print(f"Visualization failed: {e}")
    
    if CONFIG['save_results']:
        save_experiment_results(phase1_result, phase2_result, phase3_result, CONFIG)
    
    print("\nExperiment completed!")
    
    return phase1_result, phase2_result, phase3_result


if __name__ == "__main__":
    results = main()
