#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LIRL Runtime Scaling Experiment
================================
Purpose: Test LIRL algorithm runtime performance at different scales

Experiment settings:
1) num_of_jobs=100, num_of_robots=5    (small scale)
2) num_of_jobs=100, num_of_robots=50   (medium scale)
3) num_of_jobs=1000, num_of_robots=500 (large scale)

Metrics:
- Policy network forward inference time
- Hungarian algorithm execution time
- QP solving time
- Total decision time
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

# Add environment path
sys.path.append(os.path.join(os.path.dirname(__file__), '../env'))
try:
    import importlib
    ENV = importlib.import_module('env')
    if not hasattr(ENV, 'Env'):
        ENV = importlib.import_module('env.env')
except Exception:
    from env import env as ENV


# =======================
# Experiment Configuration
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

# Base training configuration
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
    
    # Large-scale optimization parameters
    'use_fast_action_selection': True,
    'max_steps_per_episode': None,
}

# Scale-specific optimization configuration
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
        'training_iterations': 20,
        'max_steps_per_episode': None,
        'use_fast_action_selection': False,
        'max_hungarian_size': 1000,
    }
}

# Runtime test configuration
TIMING_CONFIG = {
    'num_timing_episodes': 5,
    'warmup_steps': 100,
}


# =======================
# GPU Device Configuration
# =======================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_device():
    """Get current device"""
    return DEVICE

def warmup_gpu():
    """
    Global GPU warmup function
    Call before running any experiments to ensure:
    1. CUDA kernels are compiled (JIT)
    2. cuDNN algorithms are auto-tuned
    3. GPU memory pool is initialized
    4. GPU frequency is boosted to maximum
    """
    if not torch.cuda.is_available():
        print("GPU not available, skipping warmup")
        return
    
    print("Warming up GPU...")
    device = DEVICE
    
    dummy_sizes = [128, 256, 512, 1024, 2048]
    for size in dummy_sizes:
        dummy = torch.randn(size, size, device=device)
        for _ in range(10):
            _ = torch.mm(dummy, dummy)
    
    for in_features in [1000, 2000, 5000, 10000]:
        layer = torch.nn.Linear(in_features, 128).to(device)
        dummy_input = torch.randn(1, in_features, device=device)
        for _ in range(20):
            _ = layer(dummy_input)
    
    torch.cuda.synchronize()
    
    torch.cuda.empty_cache()
    
    print(f"GPU warmup completed (Device: {torch.cuda.get_device_name(0)})")

# =======================
# Neural Network Definitions
# =======================
class ReplayBuffer():
    """Experience replay buffer - optimized GPU transfer"""
    def __init__(self, buffer_limit, device=None):
        self.buffer = collections.deque(maxlen=buffer_limit)
        self.device = device if device else DEVICE
    
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
# Action Projection Function (with timing)
# =======================
class ActionProjectionWithTiming:
    """Action projection class with timing functionality"""
    
    def __init__(self):
        self.reset_timings()
    
    def reset_timings(self):
        """Reset timers"""
        self.hungarian_times = []
        self.qp_times = []
        self.total_projection_times = []
    
    def get_valid_jobs_and_robots(self, env):
        """Get valid jobs and robots - optimized version"""
        valid_robots = np.where(env.robot_state == 1)[0].tolist()
        
        valid_jobs = []
        task_state = env.task_state
        num_jobs = env.num_of_jobs
        
        for job_id in range(num_jobs):
            start_idx = job_id * 5
            end_idx = start_idx + 5
            if not np.all(task_state[start_idx:end_idx] == 1):
                valid_jobs.append(job_id)
        
        return valid_jobs, valid_robots
    
    def fast_action_selection(self, env, a_):
        """
        Fast action selection - skip Hungarian algorithm, use greedy strategy
        Suitable for large-scale problems, trade optimality for speed
        """
        valid_jobs, valid_robots = self.get_valid_jobs_and_robots(env)
        
        if len(valid_jobs) == 0 or len(valid_robots) == 0:
            return [0, 0, a_[2] if len(a_) > 2 else 0.0]
        
        job_preference = a_[0]
        robot_preference = a_[1]
        
        num_jobs = len(env.task_set)
        num_robots = len(env.robot_state)
        
        target_job = int(job_preference * num_jobs)
        target_robot = int(robot_preference * num_robots)
        
        job_id = min(valid_jobs, key=lambda x: abs(x - target_job))
        robot_id = min(valid_robots, key=lambda x: abs(x - target_robot))
        
        param = float(np.clip(a_[2] if len(a_) > 2 else 0.0, 0.0, 1.0))
        
        return [job_id, robot_id, param]
    
    def build_cost_matrix(self, env, valid_jobs, valid_robots, a_):
        """Build cost matrix - optimized version using vectorized operations"""
        job_preference = a_[0]
        robot_preference = a_[1]
        
        n_jobs = len(valid_jobs)
        n_robots = len(valid_robots)
        
        if n_jobs * n_robots > 10000:
            job_indices = np.array(valid_jobs)
            robot_indices = np.array(valid_robots)
            
            job_costs = np.abs(job_preference - job_indices / len(env.task_set))
            robot_costs = np.abs(robot_preference - robot_indices / len(env.robot_state))
            
            cost_matrix = job_costs[:, np.newaxis] + robot_costs[np.newaxis, :]
            return cost_matrix
        
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
        """Solve optimal assignment using Hungarian algorithm (with timing)"""
        start_time = time.perf_counter()
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        hungarian_time = time.perf_counter() - start_time
        self.hungarian_times.append(hungarian_time)
        return row_ind, col_ind, hungarian_time
    
    def solve_qp(self, v, A, b):
        """Solve quadratic programming problem (with timing)"""
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
        Execute action projection (with full timing)
        
        Args:
            env: Environment object
            a: Action vector
            record_timing: Whether to record timing
            max_hungarian_size: Maximum size limit for Hungarian algorithm (for accelerating large-scale problems)
        
        Returns: [job_id, robot_id, param], timing_info
        """
        total_start = time.perf_counter()
        
        a_ = a.detach().cpu().numpy()
        
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
        
        if len(valid_jobs) > max_hungarian_size or len(valid_robots) > max_hungarian_size:
            job_scores = [abs(job_preference - (j / len(env.task_set))) for j in valid_jobs]
            robot_scores = [abs(robot_preference - (r / len(env.robot_state))) for r in valid_robots]
            
            k_jobs = min(max_hungarian_size, len(valid_jobs))
            k_robots = min(max_hungarian_size, len(valid_robots))
            
            top_job_indices = np.argsort(job_scores)[:k_jobs]
            top_robot_indices = np.argsort(robot_scores)[:k_robots]
            
            selected_jobs = [valid_jobs[i] for i in top_job_indices]
            selected_robots = [valid_robots[i] for i in top_robot_indices]
        else:
            selected_jobs = valid_jobs
            selected_robots = valid_robots
        
        cost_matrix = self.build_cost_matrix(env, selected_jobs, selected_robots, a_)
        
        row_ind, col_ind, hungarian_time = self.solve_hungarian(cost_matrix)
        
        if len(row_ind) > 0:
            job_id = selected_jobs[row_ind[0]]
            robot_id = selected_robots[col_ind[0]]
        else:
            job_id = selected_jobs[0]
            robot_id = selected_robots[0]
        
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
        """Get timing statistics"""
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
# Training Function
# =======================
def train_step(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer, config, device='cpu', scaler=None):
    s, a, r, s_prime, done_mask = memory.sample(config['batch_size'])
    
    use_amp = scaler is not None and device.type == 'cuda'
    
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
        
        mu_loss = -q(s, mu(s)).mean()
        
        mu_optimizer.zero_grad()
        mu_loss.backward()
        mu_optimizer.step()


def soft_update(net, net_target, tau):
    """Soft update target network"""
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)


def train_policy(config, experiment_name):
    """
    Train policy network
    
    Args:
        config: Training configuration
        experiment_name: Experiment name ('small', 'medium', 'large')
    
    Returns:
        Trained model and training records
    """
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
    
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    
    device = DEVICE
    print(f"Using device: {device}")
    
    scaler = None
    if device.type == 'cuda':
        torch.cuda.manual_seed(config['seed'])
        torch.backends.cudnn.benchmark = True
        scaler = torch.amp.GradScaler('cuda')
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"Mixed precision training: Enabled")
    
    env = ENV.Env(config['num_of_jobs'], config['num_of_robots'], config['alpha'], config['beta'])
    state_size = len(env.state)
    action_size = len(env.action)
    
    print(f"State size: {state_size}, Action size: {action_size}")
    
    q = QNet(state_size, action_size, config).to(device)
    q_target = QNet(state_size, action_size, config).to(device)
    q_target.load_state_dict(q.state_dict())
    
    mu = MuNet(state_size, action_size, config).to(device)
    mu_target = MuNet(state_size, action_size, config).to(device)
    mu_target.load_state_dict(mu.state_dict())
    
    mu_optimizer = optim.Adam(mu.parameters(), lr=config['lr_mu'])
    q_optimizer = optim.Adam(q.parameters(), lr=config['lr_q'])
    
    memory = ReplayBuffer(config['buffer_limit'], device=device)
    action_projector = ActionProjectionWithTiming()
    
    use_fast_action = config.get('use_fast_action_selection', False)
    max_steps = config.get('max_steps_per_episode', None)
    training_iterations = config.get('training_iterations', 20)
    
    batch_size = config['batch_size']
    if device.type == 'cuda':
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        if gpu_memory_gb >= 20:
            batch_size = min(batch_size * 8, 2048)
        elif gpu_memory_gb >= 10:
            batch_size = min(batch_size * 4, 1024)
        else:
            batch_size = min(batch_size * 2, 512)
        
        memory_threshold = max(config['memory_threshold'], batch_size + 100)
        config = {**config, 'batch_size': batch_size, 'memory_threshold': memory_threshold}
        print(f"GPU batch size: {batch_size} (GPU memory: {gpu_memory_gb:.1f} GB)")
        print(f"Memory threshold: {memory_threshold}")
    
    score_record = []
    
    max_hungarian_size = config.get('max_hungarian_size', 100)
    print(f"Max Hungarian size: {max_hungarian_size}")
    
    s_buffer = torch.zeros(state_size, dtype=torch.float32, device=device)
    
    training_start_time = time.time()
    
    for n_epi in range(config['num_of_episodes']):
        s = env.reset()
        done = False
        episode_reward = 0.0
        step_count = 0
        
        while not done:
            with torch.no_grad():
                s_buffer.copy_(torch.from_numpy(s.astype(np.float32)))
                a = mu(s_buffer.unsqueeze(0)).squeeze(0)
                a = torch.clamp(a, 0, 1)
                a_np = a.cpu().numpy()
            
            if use_fast_action:
                action = action_projector.fast_action_selection(env, a_np)
            else:
                action, _ = action_projector.project(env, a, record_timing=False, max_hungarian_size=max_hungarian_size)
            
            s_prime, r, done = env.step(action)
            
            memory.put((s, a_np, r, s_prime, done))
            s = s_prime
            episode_reward += r
            step_count += 1
            
            if max_steps is not None and step_count >= max_steps:
                break
        
        score_record.append(episode_reward)
        
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
# Runtime Testing Function
# =======================
def test_runtime(config, mu, experiment_name):
    """
    Test policy runtime
    
    Args:
        config: Configuration
        mu: Trained policy network
        experiment_name: Experiment name
    
    Returns:
        Runtime statistics
    """
    print(f"\n{'='*60}")
    print(f"Testing Runtime for: {EXPERIMENT_CONFIGS[experiment_name]['name']}")
    print(f"Jobs: {config['num_of_jobs']}, Robots: {config['num_of_robots']}")
    print(f"{'='*60}")
    
    mu.eval()
    
    device = next(mu.parameters()).device
    print(f"Model device: {device}")
    
    env = ENV.Env(config['num_of_jobs'], config['num_of_robots'], config['alpha'], config['beta'])
    
    action_projector = ActionProjectionWithTiming()
    
    network_forward_times = []
    total_decision_times = []
    
    print(f"Warming up ({TIMING_CONFIG['warmup_steps']} steps)...")
    s = env.reset()
    for _ in range(TIMING_CONFIG['warmup_steps']):
        with torch.no_grad():
            s_tensor = torch.from_numpy(s.astype(np.float32)).to(device)
            a = mu(s_tensor)
            a = torch.clamp(a, 0, 1)
            a_np = a.cpu().numpy()
        action = action_projector.fast_action_selection(env, a_np)
        s_prime, _, done = env.step(action)
        if done:
            s = env.reset()
        else:
            s = s_prime
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    action_projector.reset_timings()
    
    print(f"Running timing tests ({TIMING_CONFIG['num_timing_episodes']} episodes)...")
    
    for episode in range(TIMING_CONFIG['num_timing_episodes']):
        s = env.reset()
        done = False
        step = 0
        
        while not done:
            decision_start = time.perf_counter()
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            network_start = time.perf_counter()
            with torch.no_grad():
                s_tensor = torch.from_numpy(s.astype(np.float32)).to(device)
                a = mu(s_tensor)
                a = torch.clamp(a, 0, 1)
                a_np = a.cpu().numpy()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            network_time = time.perf_counter() - network_start
            network_forward_times.append(network_time)
            
            max_hungarian_size = config.get('max_hungarian_size', 100)
            action, timing_info = action_projector.project(env, torch.from_numpy(a_np), record_timing=True, max_hungarian_size=max_hungarian_size)
            
            total_decision_time = time.perf_counter() - decision_start
            total_decision_times.append(total_decision_time)
            
            s_prime, _, done = env.step(action)
            s = s_prime
            step += 1
        
        print(f"  Episode {episode + 1}: {step} steps")
    
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
    print(f"\n{'='*70}")
    print(f"Runtime Results: {timing_results['config_name']}")
    print(f"Jobs: {timing_results['num_of_jobs']}, Robots: {timing_results['num_of_robots']}")
    print(f"{'='*70}")
    
    print(f"\n1. Policy Network Forward Inference Time:")
    net = timing_results['network_forward']
    print(f"   Mean: {net['mean']*1000:.4f} ms")
    print(f"   Std: {net['std']*1000:.4f} ms")
    print(f"   Min: {net['min']*1000:.4f} ms")
    print(f"   Max: {net['max']*1000:.4f} ms")
    
    print(f"\n2. Hungarian Algorithm Execution Time:")
    hung = timing_results['hungarian']
    if hung:
        print(f"   Mean: {hung['mean']*1000:.4f} ms")
        print(f"   Std: {hung['std']*1000:.4f} ms")
        print(f"   Min: {hung['min']*1000:.4f} ms")
        print(f"   Max: {hung['max']*1000:.4f} ms")
    
    print(f"\n3. QP Solving Time:")
    qp = timing_results['qp']
    if qp:
        print(f"   Mean: {qp['mean']*1000:.4f} ms")
        print(f"   Std: {qp['std']*1000:.4f} ms")
        print(f"   Min: {qp['min']*1000:.4f} ms")
        print(f"   Max: {qp['max']*1000:.4f} ms")
    
    print(f"\n4. Total Decision Time:")
    total = timing_results['total_decision']
    print(f"   Mean: {total['mean']*1000:.4f} ms")
    print(f"   Std: {total['std']*1000:.4f} ms")
    print(f"   Min: {total['min']*1000:.4f} ms")
    print(f"   Max: {total['max']*1000:.4f} ms")
    
    print(f"\n5. Time Percentage Analysis:")
    total_mean = total['mean']
    if total_mean > 0:
        net_pct = (net['mean'] / total_mean) * 100
        hung_pct = (hung['mean'] / total_mean) * 100 if hung else 0
        qp_pct = (qp['mean'] / total_mean) * 100 if qp else 0
        other_pct = 100 - net_pct - hung_pct - qp_pct
        
        print(f"   Network Forward: {net_pct:.2f}%")
        print(f"   Hungarian Algorithm: {hung_pct:.2f}%")
        print(f"   QP Solving: {qp_pct:.2f}%")
        print(f"   Other: {other_pct:.2f}%")


def save_results(all_results, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    results_json = {
        'timestamp': datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        'experiments': {}
    }
    
    for exp_name, result in all_results.items():
        model_dir = os.path.join(save_dir, exp_name)
        os.makedirs(model_dir, exist_ok=True)
        
        torch.save(result['training']['mu'].state_dict(), 
                   os.path.join(model_dir, 'mu.pth'))
        torch.save(result['training']['q'].state_dict(), 
                   os.path.join(model_dir, 'q.pth'))
        
        np.save(os.path.join(model_dir, 'score_record.npy'), 
                result['training']['score_record'])
        
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
    
    with open(os.path.join(save_dir, 'results_summary.json'), 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\nResults saved to: {save_dir}")
    return save_dir


def plot_comparison(all_results, save_dir):
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
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    x = np.arange(len(labels))
    width = 0.6
    
    ax1 = axes[0, 0]
    bars1 = ax1.bar(x, total_times, width, color=['#3498db', '#e74c3c', '#2ecc71'])
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('Total Decision Time Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Small\n(100×5)', 'Medium\n(100×50)', 'Large\n(1000×500)'])
    ax1.bar_label(bars1, fmt='%.2f')
    ax1.grid(axis='y', alpha=0.3)
    
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
    
    ax3 = axes[1, 0]
    bars3 = ax3.bar(x, hungarian_times, width, color=['#3498db', '#e74c3c', '#2ecc71'])
    ax3.set_ylabel('Time (ms)')
    ax3.set_title('Hungarian Algorithm Time')
    ax3.set_xticks(x)
    ax3.set_xticklabels(['Small\n(100×5)', 'Medium\n(100×50)', 'Large\n(1000×500)'])
    ax3.bar_label(bars3, fmt='%.2f')
    ax3.grid(axis='y', alpha=0.3)
    
    ax4 = axes[1, 1]
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    for i, name in enumerate(exp_names):
        scores = all_results[name]['training']['score_record']
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
    print("\n" + "="*100)
    print("Runtime Results (in milliseconds):")
    print(f"{'Scale':<15} {'Jobs×Robots':<12} {'Policy Net':<16} {'Hungarian':<16} {'QP':<16} {'Total':<16}")
    print("-"*100)
    
    for name in ['small', 'medium', 'large']:
        if name in all_results:
            timing = all_results[name]['timing']
            config_str = f"{timing['num_of_jobs']}×{timing['num_of_robots']}"
            
            net_mean = timing['network_forward']['mean'] * 1000
            net_std = timing['network_forward']['std'] * 1000
            
            hung_mean = timing['hungarian']['mean'] * 1000 if timing['hungarian'] else 0
            hung_std = timing['hungarian']['std'] * 1000 if timing['hungarian'] else 0
            
            qp_mean = timing['qp']['mean'] * 1000 if timing['qp'] else 0
            qp_std = timing['qp']['std'] * 1000 if timing['qp'] else 0
            
            total_mean = timing['total_decision']['mean'] * 1000
            total_std = timing['total_decision']['std'] * 1000
            
            net_str = f"{net_mean:.4f}±{net_std:.4f}"
            hung_str = f"{hung_mean:.4f}±{hung_std:.4f}"
            qp_str = f"{qp_mean:.4f}±{qp_std:.4f}"
            total_str = f"{total_mean:.4f}±{total_std:.4f}"
            
            print(f"{name:<15} {config_str:<12} {net_str:<16} {hung_str:<16} {qp_str:<16} {total_str:<16}")
    
    print("="*100)
    
    if 'small' in all_results and 'large' in all_results:
        small_total = all_results['small']['timing']['total_decision']['mean']
        large_total = all_results['large']['timing']['total_decision']['mean']
        scale_factor = large_total / small_total if small_total > 0 else 0
        
        print(f"\nScaling Factor (Large/Small): {scale_factor:.2f}x")
        print(f"Problem Size Increase: {(1000*100)/(100*5):.0f}x")


# =======================
# Main Function
# =======================
def main():
    print("="*80)
    print("LIRL Runtime Scaling Experiment")
    print("="*80)
    print(f"Experiment Configurations:")
    for name, cfg in EXPERIMENT_CONFIGS.items():
        print(f"  {name}: {cfg['name']}")
    print("="*80)
    
    warmup_gpu()
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_folder = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(exp_folder, f"lirl_runtime_scaling_{timestamp}")
    
    all_results = {}
    
    for exp_name in ['small', 'medium', 'large']:
        print(f"\n\n{'#'*80}")
        print(f"# Experiment: {EXPERIMENT_CONFIGS[exp_name]['name']}")
        print(f"{'#'*80}")
        
        config = BASE_CONFIG.copy()
        config.update(EXPERIMENT_CONFIGS[exp_name])
        if exp_name in SCALE_SPECIFIC_CONFIG:
            config.update(SCALE_SPECIFIC_CONFIG[exp_name])
        
        training_result = train_policy(config, exp_name)
        
        timing_result = test_runtime(config, training_result['mu'], exp_name)
        
        print_timing_results(timing_result)
        
        all_results[exp_name] = {
            'training': training_result,
            'timing': timing_result,
            'config': config
        }
    
    print_summary_table(all_results)
    
    saved_dir = save_results(all_results, save_dir)
    
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

