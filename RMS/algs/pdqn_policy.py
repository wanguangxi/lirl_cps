import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List
import os
import sys
import random
import datetime
import matplotlib.pyplot as plt
from collections import deque
sys.path.append(os.path.join(os.path.dirname(__file__), '../env'))
import env as ENV

# =======================
# HYPERPARAMETERS CONFIG
# =======================
CONFIG = {
    # Learning parameters - PDQN specific
    'actor_lr': 1e-4,  # Learning rate for actor (parameter network)
    'critic_lr': 1e-3,  # Learning rate for critic (Q-network)
    'gamma': 0.99,  # Discount factor
    'tau': 0.001,  # Soft update parameter for target networks
    'batch_size': 128,  # Batch size for training
    'buffer_limit': 10000,  # Experience replay buffer size
    'epsilon_start': 1.0,  # Initial exploration rate
    'epsilon_end': 0.01,  # Final exploration rate
    'epsilon_decay': 0.995,  # Epsilon decay rate
    'update_interval': 4,  # Update networks every N steps
    
    # Environment parameters
    'num_jobs': 100,
    'num_machines': 5,
    'max_operations': 5,
    'alpha': 0.5,
    'beta': 0.5,
    'num_episodes': 1000,
    'steps_per_episode': 550,
    
    # Training parameters
    'warmup_steps': 1000,  # Steps before training starts
    'eval_interval': 50,
    'save_interval': 100,
    'target_update_freq': 10,
    'max_steps_per_episode': 500,  # Maximum steps per episode (hard limit)
    
    # Multi-run training parameters
    'enable_multi_run': True,
    'seeds': [3047, 294, 714, 1092, 1386, 2856, 42, 114514, 2025, 1993],
    'num_runs': 10,
    
    # Testing parameters
    'max_test_steps': 550,
    
    # Output parameters
    'print_interval': 10,
    'enable_gantt_plots': False,
    'plot_training_curve': True,
    'save_models': True,
    'save_dir': 'checkpoints',
    
    # Performance optimization
    'use_optimized_action_selection': True,
    'action_selection_batch_size': 100,  # Max actions to evaluate in batch
    'update_sub_batch_size': 10,  # Sub-batch size for next Q-value computation
    'enable_action_caching': False,  # Experimental: cache Q-values for repeated states
}


class QNetwork(nn.Module):
    """Q-network for evaluating state-action values"""
    
    def __init__(self, state_dim: int, num_jobs: int, num_machines: int, 
                 param_dim: int, hidden_dim: int = 256):
        super().__init__()
        # Input: state + discrete actions (job, machine) + continuous parameters
        input_dim = state_dim + 2 + param_dim  # 2 for job and machine indices
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, state: torch.Tensor, job_idx: torch.Tensor, 
                machine_idx: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: State tensor [batch, state_dim]
            job_idx: Job indices [batch]
            machine_idx: Machine indices [batch]
            params: Continuous parameters [batch, param_dim]
        Returns:
            Q-values [batch, 1]
        """
        # Reshape indices to [batch, 1]
        job_idx = job_idx.view(-1, 1).float()
        machine_idx = machine_idx.view(-1, 1).float()
        
        # Concatenate all inputs
        x = torch.cat([state, job_idx, machine_idx, params], dim=-1)
        return self.network(x)


class ParameterNetwork(nn.Module):
    """Actor network for generating continuous parameters given discrete actions"""
    
    def __init__(self, state_dim: int, num_jobs: int, num_machines: int,
                 param_dim: int, hidden_dim: int = 256):
        super().__init__()
        # Input: state + discrete actions (one-hot encoded)
        input_dim = state_dim + num_jobs + num_machines
        
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.param_dim = param_dim
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, param_dim)
        )
        
    def forward(self, state: torch.Tensor, job_idx: torch.Tensor, 
                machine_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: State tensor [batch, state_dim]
            job_idx: Job indices [batch]
            machine_idx: Machine indices [batch]
        Returns:
            Parameters [batch, param_dim] (normalized to [0, 1])
        """
        batch_size = state.shape[0]
        
        # Ensure indices are 1D tensors
        if job_idx.dim() > 1:
            job_idx = job_idx.squeeze(-1)
        if machine_idx.dim() > 1:
            machine_idx = machine_idx.squeeze(-1)
        
        # Create one-hot encodings
        job_one_hot = F.one_hot(job_idx.long(), num_classes=self.num_jobs).float()
        machine_one_hot = F.one_hot(machine_idx.long(), num_classes=self.num_machines).float()
        
        # Concatenate inputs
        x = torch.cat([state, job_one_hot, machine_one_hot], dim=-1)
        
        # Generate parameters and apply sigmoid to bound them to [0, 1]
        params = torch.sigmoid(self.network(x))
        return params


class PDQNAgent:
    """Parameterized Deep Q-Network agent for flexible job shop scheduling"""
    
    def __init__(self, state_dim: int, num_jobs: int, num_machines: int,
                 max_operations: int, actor_lr: float = 1e-4, 
                 critic_lr: float = 1e-3, gamma: float = 0.99,
                 tau: float = 0.001, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.param_dim = max_operations
        self.gamma = gamma
        self.tau = tau
        
        # Initialize networks
        self.q_network = QNetwork(state_dim, num_jobs, num_machines, 
                                 self.param_dim).to(self.device)
        self.q_target = QNetwork(state_dim, num_jobs, num_machines, 
                                self.param_dim).to(self.device)
        self.param_network = ParameterNetwork(state_dim, num_jobs, num_machines,
                                            self.param_dim).to(self.device)
        self.param_target = ParameterNetwork(state_dim, num_jobs, num_machines,
                                           self.param_dim).to(self.device)
        
        # Initialize target networks
        self.q_target.load_state_dict(self.q_network.state_dict())
        self.param_target.load_state_dict(self.param_network.state_dict())
        
        # Initialize optimizers
        self.q_optimizer = torch.optim.Adam(self.q_network.parameters(), lr=critic_lr)
        self.param_optimizer = torch.optim.Adam(self.param_network.parameters(), lr=actor_lr)
        
        # Exploration parameters
        self.epsilon = CONFIG['epsilon_start']
        self.epsilon_end = CONFIG['epsilon_end']
        self.epsilon_decay = CONFIG['epsilon_decay']
        
        # Add cache for action evaluation
        self.action_cache = {}
        self.cache_size = 1000
    
    def select_action(self, state: torch.Tensor, job_mask: torch.Tensor,
                     machine_mask: torch.Tensor, operation_bounds: torch.Tensor,
                     deterministic: bool = False) -> Dict[str, torch.Tensor]:
        """Select action using epsilon-greedy for discrete actions and deterministic for parameters"""
        with torch.no_grad():
            batch_size = state.shape[0]
            
            if not deterministic and random.random() < self.epsilon:
                # Random exploration for discrete actions
                valid_jobs = torch.where(job_mask[0] > 0)[0]
                valid_machines = torch.where(machine_mask[0] > 0)[0]
                
                if len(valid_jobs) > 0 and len(valid_machines) > 0:
                    job_idx = valid_jobs[torch.randint(len(valid_jobs), (1,))]
                    machine_idx = valid_machines[torch.randint(len(valid_machines), (1,))]
                else:
                    # Fallback if no valid actions
                    job_idx = torch.zeros(1, dtype=torch.long, device=self.device)
                    machine_idx = torch.zeros(1, dtype=torch.long, device=self.device)
            else:
                # Greedy action selection
                best_q = -float('inf')
                best_job = 0
                best_machine = 0
                
                # Evaluate Q-values for all valid action combinations
                for job in range(self.num_jobs):
                    if job_mask[0, job] == 0:
                        continue
                    for machine in range(self.num_machines):
                        if machine_mask[0, machine] == 0:
                            continue
                        
                        job_tensor = torch.tensor([job], device=self.device)
                        machine_tensor = torch.tensor([machine], device=self.device)
                        
                        # Get parameters for this discrete action
                        params = self.param_network(state, job_tensor, machine_tensor)
                        
                        # Scale parameters to bounds
                        lower_bounds = operation_bounds[0, :, 0]
                        upper_bounds = operation_bounds[0, :, 1]
                        scaled_params = params * (upper_bounds - lower_bounds) + lower_bounds
                        
                        # Evaluate Q-value
                        q_value = self.q_network(state, job_tensor, machine_tensor, scaled_params)
                        
                        if q_value.item() > best_q:
                            best_q = q_value.item()
                            best_job = job
                            best_machine = machine
                
                job_idx = torch.tensor([best_job], device=self.device)
                machine_idx = torch.tensor([best_machine], device=self.device)
            
            # Get continuous parameters for selected discrete actions
            params = self.param_network(state, job_idx, machine_idx)
            
            # Scale parameters to bounds
            lower_bounds = operation_bounds[0, :, 0]
            upper_bounds = operation_bounds[0, :, 1]
            continuous_param = params * (upper_bounds - lower_bounds) + lower_bounds
            
            # Add noise to parameters for exploration (if not deterministic)
            if not deterministic:
                noise = torch.randn_like(continuous_param) * 0.1
                continuous_param = torch.clamp(continuous_param + noise, lower_bounds, upper_bounds)
            
            return {
                'job_idx': job_idx,
                'machine_idx': machine_idx,
                'continuous_param': continuous_param,
                'params_normalized': params  # Keep normalized params for training
            }
    
    def select_action_optimized(self, state: torch.Tensor, job_mask: torch.Tensor,
                                machine_mask: torch.Tensor, operation_bounds: torch.Tensor,
                                deterministic: bool = False) -> Dict[str, torch.Tensor]:
        """Optimized action selection using batch computation"""
        with torch.no_grad():
            if not deterministic and random.random() < self.epsilon:
                # Random exploration for discrete actions
                valid_jobs = torch.where(job_mask[0] > 0)[0]
                valid_machines = torch.where(machine_mask[0] > 0)[0]
                
                if len(valid_jobs) > 0 and len(valid_machines) > 0:
                    job_idx = valid_jobs[torch.randint(len(valid_jobs), (1,))]
                    machine_idx = valid_machines[torch.randint(len(valid_machines), (1,))]
                else:
                    job_idx = torch.zeros(1, dtype=torch.long, device=self.device)
                    machine_idx = torch.zeros(1, dtype=torch.long, device=self.device)
            else:
                # Batch evaluate all valid actions
                valid_jobs = torch.where(job_mask[0] > 0)[0]
                valid_machines = torch.where(machine_mask[0] > 0)[0]
                
                if len(valid_jobs) == 0 or len(valid_machines) == 0:
                    job_idx = torch.zeros(1, dtype=torch.long, device=self.device)
                    machine_idx = torch.zeros(1, dtype=torch.long, device=self.device)
                else:
                    # Create all valid action combinations
                    job_indices = valid_jobs.repeat_interleave(len(valid_machines))
                    machine_indices = valid_machines.repeat(len(valid_jobs))
                    
                    # Batch compute parameters for all actions
                    batch_states = state.repeat(len(job_indices), 1)
                    batch_params = self.param_network(batch_states, job_indices, machine_indices)
                    
                    # Scale parameters
                    lower_bounds = operation_bounds[0, :, 0]
                    upper_bounds = operation_bounds[0, :, 1]
                    scaled_params = batch_params * (upper_bounds - lower_bounds) + lower_bounds
                    
                    # Batch evaluate Q-values
                    q_values = self.q_network(batch_states, job_indices, machine_indices, scaled_params)
                    
                    # Find best action
                    best_idx = q_values.squeeze().argmax()
                    job_idx = job_indices[best_idx].unsqueeze(0)
                    machine_idx = machine_indices[best_idx].unsqueeze(0)
            
            # Get continuous parameters for selected discrete actions
            params = self.param_network(state, job_idx, machine_idx)
            
            # Scale parameters to bounds
            lower_bounds = operation_bounds[0, :, 0]
            upper_bounds = operation_bounds[0, :, 1]
            continuous_param = params * (upper_bounds - lower_bounds) + lower_bounds
            
            # Add noise to parameters for exploration (if not deterministic)
            if not deterministic:
                noise = torch.randn_like(continuous_param) * 0.1
                continuous_param = torch.clamp(continuous_param + noise, lower_bounds, upper_bounds)
            
            return {
                'job_idx': job_idx,
                'machine_idx': machine_idx,
                'continuous_param': continuous_param,
                'params_normalized': params
            }
    
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Update Q-network and parameter network"""
        states = batch['states']
        job_indices = batch['actions']['job_idx']
        machine_indices = batch['actions']['machine_idx']
        params = batch['actions']['params_normalized']
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones']
        operation_bounds = batch['operation_bounds']
        
        # Scale parameters for Q-network
        lower_bounds = operation_bounds[:, :, 0]
        upper_bounds = operation_bounds[:, :, 1]
        scaled_params = params * (upper_bounds - lower_bounds) + lower_bounds
        
        # Update Q-network
        current_q = self.q_network(states, job_indices, machine_indices, scaled_params)
        
        with torch.no_grad():
            # Find best actions for next states
            next_job_masks = batch['next_job_masks']
            next_machine_masks = batch['next_machine_masks']
            
            best_next_q = torch.zeros(states.shape[0], device=self.device)
            
            for i in range(states.shape[0]):
                if dones[i]:
                    continue
                    
                best_q = -float('inf')
                
                # Search over valid actions
                for job in range(self.num_jobs):
                    if next_job_masks[i, job] == 0:
                        continue
                    for machine in range(self.num_machines):
                        if next_machine_masks[i, machine] == 0:
                            continue
                        
                        job_tensor = torch.tensor([job], device=self.device)
                        machine_tensor = torch.tensor([machine], device=self.device)
                        
                        # Get parameters from target network
                        next_params = self.param_target(next_states[i:i+1], job_tensor, machine_tensor)
                        
                        # Scale parameters
                        next_lower = operation_bounds[i, :, 0]
                        next_upper = operation_bounds[i, :, 1]
                        next_scaled_params = next_params * (next_upper - next_lower) + next_lower
                        
                        # Evaluate Q-value with target network
                        q_value = self.q_target(next_states[i:i+1], job_tensor, 
                                              machine_tensor, next_scaled_params)
                        
                        if q_value.item() > best_q:
                            best_q = q_value.item()
                
                if best_q > -float('inf'):
                    best_next_q[i] = best_q
            
            # Ensure rewards and dones have correct shape
            rewards = rewards.squeeze()
            dones = dones.squeeze()
            
            target_q = rewards + self.gamma * best_next_q * (1 - dones)
        
        # Ensure both tensors have the same shape for loss calculation
        current_q = current_q.squeeze()
        q_loss = F.mse_loss(current_q, target_q)
        
        self.q_optimizer.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.q_optimizer.step()
        
        # Update parameter network
        predicted_params = self.param_network(states, job_indices, machine_indices)
        predicted_scaled = predicted_params * (upper_bounds - lower_bounds) + lower_bounds
        
        # Actor loss: negative Q-value (we want to maximize Q)
        actor_loss = -self.q_network(states, job_indices, machine_indices, predicted_scaled).mean()
        
        self.param_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.param_network.parameters(), max_norm=1.0)
        self.param_optimizer.step()
        
        # Soft update target networks
        self._soft_update(self.q_network, self.q_target)
        self._soft_update(self.param_network, self.param_target)
        
        # Update epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return {
            'q_loss': q_loss.item(),
            'actor_loss': actor_loss.item(),
            'epsilon': self.epsilon
        }
    
    def update_optimized(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Optimized update using vectorized operations"""
        states = batch['states']
        job_indices = batch['actions']['job_idx']
        machine_indices = batch['actions']['machine_idx']
        params = batch['actions']['params_normalized']
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones']
        operation_bounds = batch['operation_bounds']
        
        # Scale parameters for Q-network
        lower_bounds = operation_bounds[:, :, 0]
        upper_bounds = operation_bounds[:, :, 1]
        scaled_params = params * (upper_bounds - lower_bounds) + lower_bounds
        
        # Update Q-network
        current_q = self.q_network(states, job_indices, machine_indices, scaled_params)
        
        with torch.no_grad():
            # Vectorized computation of next Q-values
            next_job_masks = batch['next_job_masks']
            next_machine_masks = batch['next_machine_masks']
            
            # Pre-compute valid actions for each sample
            batch_size = states.shape[0]
            max_actions = self.num_jobs * self.num_machines
            
            # Initialize tensor to store best Q-values
            best_next_q = torch.zeros(batch_size, device=self.device)
            
            # Process in smaller sub-batches to avoid memory issues
            sub_batch_size = 10
            for i in range(0, batch_size, sub_batch_size):
                end_idx = min(i + sub_batch_size, batch_size)
                sub_batch_indices = list(range(i, end_idx))
                
                for idx in sub_batch_indices:
                    if dones[idx]:
                        continue
                    
                    valid_jobs = torch.where(next_job_masks[idx] > 0)[0]
                    valid_machines = torch.where(next_machine_masks[idx] > 0)[0]
                    
                    if len(valid_jobs) > 0 and len(valid_machines) > 0:
                        # Create action combinations
                        job_indices_next = valid_jobs.repeat_interleave(len(valid_machines))
                        machine_indices_next = valid_machines.repeat(len(valid_jobs))
                        
                        # Batch compute parameters
                        state_batch = next_states[idx:idx+1].repeat(len(job_indices_next), 1)
                        params_next = self.param_target(state_batch, job_indices_next, machine_indices_next)
                        
                        # Scale parameters
                        next_lower = operation_bounds[idx, :, 0]
                        next_upper = operation_bounds[idx, :, 1]
                        scaled_params_next = params_next * (next_upper - next_lower) + next_lower
                        
                        # Batch evaluate Q-values
                        q_values_next = self.q_target(state_batch, job_indices_next, 
                                                    machine_indices_next, scaled_params_next)
                        
                        best_next_q[idx] = q_values_next.max().item()
            
            # Ensure rewards and dones have correct shape
            rewards = rewards.squeeze()
            dones = dones.squeeze()
            
            target_q = rewards + self.gamma * best_next_q * (1 - dones)
        
        # Ensure both tensors have the same shape for loss calculation
        current_q = current_q.squeeze()
        q_loss = F.mse_loss(current_q, target_q)
        
        self.q_optimizer.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.q_optimizer.step()
        
        # Update parameter network
        predicted_params = self.param_network(states, job_indices, machine_indices)
        predicted_scaled = predicted_params * (upper_bounds - lower_bounds) + lower_bounds
        
        # Actor loss: negative Q-value (we want to maximize Q)
        actor_loss = -self.q_network(states, job_indices, machine_indices, predicted_scaled).mean()
        
        self.param_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.param_network.parameters(), max_norm=1.0)
        self.param_optimizer.step()
        
        # Soft update target networks
        self._soft_update(self.q_network, self.q_target)
        self._soft_update(self.param_network, self.param_target)
        
        # Update epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return {
            'q_loss': q_loss.item(),
            'actor_loss': actor_loss.item(),
            'epsilon': self.epsilon
        }
    
    def _soft_update(self, source: nn.Module, target: nn.Module):
        """Soft update target network parameters"""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1.0 - self.tau) * target_param.data)
    
    def save(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'q_target_state_dict': self.q_target.state_dict(),
            'param_network_state_dict': self.param_network.state_dict(),
            'param_target_state_dict': self.param_target.state_dict(),
            'q_optimizer_state_dict': self.q_optimizer.state_dict(),
            'param_optimizer_state_dict': self.param_optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
    
    def load(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.q_target.load_state_dict(checkpoint['q_target_state_dict'])
        self.param_network.load_state_dict(checkpoint['param_network_state_dict'])
        self.param_target.load_state_dict(checkpoint['param_target_state_dict'])
        self.q_optimizer.load_state_dict(checkpoint['q_optimizer_state_dict'])
        self.param_optimizer.load_state_dict(checkpoint['param_optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', CONFIG['epsilon_end'])


class ReplayBuffer:
    """Experience replay buffer for PDQN"""
    
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, experience: Dict[str, torch.Tensor]):
        """Add experience to buffer"""
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a batch of experiences"""
        experiences = random.sample(self.buffer, batch_size)
        
        # Organize batch data
        batch = {
            'states': torch.stack([e['states'] for e in experiences]),
            'actions': {
                'job_idx': torch.stack([e['actions']['job_idx'] for e in experiences]),
                'machine_idx': torch.stack([e['actions']['machine_idx'] for e in experiences]),
                'continuous_param': torch.stack([e['actions']['continuous_param'] for e in experiences]),
                'params_normalized': torch.stack([e['actions']['params_normalized'] for e in experiences])
            },
            'rewards': torch.stack([e['rewards'] for e in experiences]),
            'next_states': torch.stack([e['next_states'] for e in experiences]),
            'dones': torch.stack([e['dones'] for e in experiences]),
            'job_masks': torch.stack([e['job_masks'] for e in experiences]),
            'machine_masks': torch.stack([e['machine_masks'] for e in experiences]),
            'operation_bounds': torch.stack([e['operation_bounds'] for e in experiences]),
            'next_job_masks': torch.stack([e['next_job_masks'] for e in experiences]),
            'next_machine_masks': torch.stack([e['next_machine_masks'] for e in experiences])
        }
        
        return batch
    
    def __len__(self):
        return len(self.buffer)


# Reuse utility functions from hppo_policy.py
def get_job_mask(env):
    """Generate a job mask based on the environment state."""
    mask = np.zeros(env.num_of_jobs, dtype=np.float32)
    
    for job_id in range(len(env.task_set)):
        job = env.task_set[job_id]
        job_finished = True
        for op in range(len(job)):
            if not job[op].state:
                job_finished = False
                break
        
        if not job_finished:
            mask[job_id] = 1.0
    
    return mask


def get_machine_mask(env):
    """Generate a machine mask based on the environment state."""
    mask = np.zeros(env.num_of_robots, dtype=np.float32)
    
    for robot_id in range(len(env.robot_state)):
        if env.robot_state[robot_id] == 1:
            mask[robot_id] = 1.0
    
    return mask


def get_operation_bounds(env):
    """Generate bounds for continuous parameters based on the environment state."""
    operation_bounds = np.array([
        [4.0, 7.2],
        [2.0, 14.2],
        [2.5, 16.5],
        [2.1, 18.0],
        [2.4, 16.8]
    ])
    
    if CONFIG['max_operations'] > len(operation_bounds):
        default_bounds = np.tile(np.array([0.0, 1.0]), (CONFIG['max_operations'] - len(operation_bounds), 1))
        operation_bounds = np.vstack([operation_bounds, default_bounds])
    
    if CONFIG['max_operations'] < len(operation_bounds):
        operation_bounds = operation_bounds[:CONFIG['max_operations']]
    
    return operation_bounds


def collect_experience_pdqn(env, agent: PDQNAgent, num_steps: int, 
                           buffer: ReplayBuffer, deterministic: bool = False,
                           max_episode_steps: int = None):
    """Collect experience for PDQN training with episode step limit"""
    state = env.reset()
    episode_reward = 0
    step_counter = 0
    
    # Use config max steps if not specified
    if max_episode_steps is None:
        max_episode_steps = CONFIG['max_steps_per_episode']
    
    # Add timing information
    import time
    start_time = time.time()
    
    for step in range(num_steps):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
        
        job_mask = torch.FloatTensor(get_job_mask(env)).unsqueeze(0).to(agent.device)
        machine_mask = torch.FloatTensor(get_machine_mask(env)).unsqueeze(0).to(agent.device)
        operation_bounds = torch.FloatTensor(get_operation_bounds(env)).unsqueeze(0).to(agent.device)
        
        # Time action selection
        action_start = time.time()
        action_dict = agent.select_action(
            state_tensor, job_mask, machine_mask, operation_bounds, deterministic
        )
        action_time = time.time() - action_start
        
        job_id = action_dict['job_idx'].item()
        machine_id = action_dict['machine_idx'].item()
        param_value = float(action_dict['continuous_param'].squeeze(0).cpu().numpy()[0])
        
        action = [job_id, machine_id, param_value]
        
        step_result = env.step(action)
        step_counter += 1
        
        # Log detailed step information periodically
        if step_counter % 100 == 0:
            valid_jobs = torch.sum(job_mask).item()
            valid_machines = torch.sum(machine_mask).item()
            print(f"  Step {step_counter}: Valid jobs={valid_jobs:.0f}, Valid machines={valid_machines:.0f}, "
                  f"Action=({job_id}, {machine_id}, {param_value:.2f}), Action time={action_time*1000:.1f}ms")
        
        if len(step_result) == 3:
            next_state, reward, done = step_result
        else:
            next_state, reward, done, _ = step_result
        
        episode_reward += reward
        
        # Check if maximum steps reached
        max_steps_reached = step_counter >= max_episode_steps
        
        # Get next state masks
        next_job_mask = torch.FloatTensor(get_job_mask(env)).to(agent.device)
        next_machine_mask = torch.FloatTensor(get_machine_mask(env)).to(agent.device)
        
        # Store experience with modified done flag
        effective_done = done or max_steps_reached
        
        experience = {
            'states': state_tensor.squeeze(0),
            'actions': {
                'job_idx': action_dict['job_idx'].squeeze(0),
                'machine_idx': action_dict['machine_idx'].squeeze(0),
                'continuous_param': action_dict['continuous_param'].squeeze(0),
                'params_normalized': action_dict['params_normalized'].squeeze(0)
            },
            'rewards': torch.FloatTensor([reward]).to(agent.device),
            'next_states': torch.FloatTensor(next_state).to(agent.device),
            'dones': torch.FloatTensor([effective_done]).to(agent.device),
            'job_masks': job_mask.squeeze(0),
            'machine_masks': machine_mask.squeeze(0),
            'operation_bounds': operation_bounds.squeeze(0),
            'next_job_masks': next_job_mask,
            'next_machine_masks': next_machine_mask
        }
        buffer.push(experience)
        
        # End episode if done or max steps reached
        if done or max_steps_reached:
            elapsed_time = time.time() - start_time
            termination_reason = "completed" if done else "max steps reached"
            print(f"  Episode {termination_reason}: Steps={step_counter}, Reward={episode_reward:.2f}, "
                  f"Time={elapsed_time:.2f}s, Avg time/step={elapsed_time/step_counter*1000:.1f}ms")
            return episode_reward, step_counter, max_steps_reached
        
        state = next_state
    
    # Reached num_steps limit (for data collection)
    elapsed_time = time.time() - start_time
    print(f"  Collection steps reached: Steps={step_counter}, Reward={episode_reward:.2f}, "
          f"Time={elapsed_time:.2f}s, Avg time/step={elapsed_time/step_counter*1000:.1f}ms")
    return episode_reward, step_counter, False


def train_single_run_pdqn(config=None):
    """Execute a single training run with PDQN"""
    if config is None:
        config = CONFIG
    
    import time
    
    # Environment setup
    env_kwargs = {
        'num_of_jobs': config['num_jobs'],
        'num_of_robots': config['num_machines'],
        'alpha': config['alpha'],
        'beta': config['beta']
    }
    
    env = ENV.Env(**env_kwargs)
    print(f"Successfully created environment with parameters: {env_kwargs}")
    
    # Get environment dimensions
    state_dim = len(env.state)
    num_jobs = env_kwargs['num_of_jobs']
    num_machines = env_kwargs['num_of_robots']
    max_operations = config['max_operations']
    
    # Create agent
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agent = PDQNAgent(
        state_dim=state_dim,
        num_jobs=num_jobs,
        num_machines=num_machines,
        max_operations=max_operations,
        actor_lr=config['actor_lr'],
        critic_lr=config['critic_lr'],
        gamma=config['gamma'],
        tau=config['tau'],
        device=device
    )
    
    # Initialize replay buffer
    buffer = ReplayBuffer(capacity=config['buffer_limit'])
    
    # Training metrics
    episode_rewards = []
    episode_steps = []
    episode_truncated = []  # Track if episode was truncated by max steps
    evaluation_rewards = []
    
    print(f"\nStarting PDQN training:")
    print(f"  State dimension: {state_dim}")
    print(f"  Jobs: {num_jobs}, Machines: {num_machines}")
    print(f"  Alpha: {config['alpha']}, Beta: {config['beta']}")
    print(f"  Episodes: {config['num_episodes']}")
    print(f"  Max steps per episode: {config['max_steps_per_episode']}")
    print(f"  Device: {device}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Buffer limit: {config['buffer_limit']}")
    print(f"  Update interval: {config['update_interval']}")
    print(f"  Warmup steps: {config['warmup_steps']}")
    print(f"  Epsilon: {config['epsilon_start']} -> {config['epsilon_end']} (decay={config['epsilon_decay']})")
    print(f"  Learning rates: Actor={config['actor_lr']}, Critic={config['critic_lr']}")
    print(f"  Optimization mode: {'Optimized' if config['use_optimized_action_selection'] else 'Standard'}")
    print()
    
    best_reward = -float('inf')
    total_steps = 0
    training_start_time = time.time()
    
    for episode in range(config['num_episodes']):
        episode_start_time = time.time()
        
        print(f"\nEpisode {episode + 1}/{config['num_episodes']} starting...")
        
        # Collect experience with max step limit
        episode_reward, steps, truncated = collect_experience_pdqn(
            env, agent, config['steps_per_episode'], buffer, 
            deterministic=False, max_episode_steps=config['max_steps_per_episode']
        )
        
        episode_rewards.append(episode_reward)
        episode_steps.append(steps)
        episode_truncated.append(truncated)
        total_steps += steps
        
        # Update best reward
        if episode_reward > best_reward:
            best_reward = episode_reward
            print(f"  *** New best reward: {best_reward:.2f} ***")
        
        # Train after warmup
        update_losses = []
        if total_steps > config['warmup_steps'] and len(buffer) >= config['batch_size']:
            # Perform multiple updates per episode
            # num_updates = min(steps // config['update_interval'], 10)
            num_updates = 1  # Only one update per episode for faster training
            print(f"  Training: {num_updates} updates (buffer size: {len(buffer)})")
            
            update_start_time = time.time()
            for update_idx in range(num_updates):
                batch = buffer.sample(config['batch_size'])
                losses = agent.update(batch)
                update_losses.append(losses)
                
                if (update_idx + 1) % 5 == 0:
                    avg_q_loss = np.mean([l['q_loss'] for l in update_losses[-5:]])
                    avg_actor_loss = np.mean([l['actor_loss'] for l in update_losses[-5:]])
                    print(f"    Update {update_idx + 1}/{num_updates}: "
                          f"Q-loss={avg_q_loss:.4f}, Actor-loss={avg_actor_loss:.4f}")
            
            update_time = time.time() - update_start_time
            if num_updates > 0:
                print(f"  Training completed in {update_time:.2f}s ({update_time/num_updates:.3f}s per update)")
        else:
            remaining_warmup = max(0, config['warmup_steps'] - total_steps)
            if remaining_warmup > 0:
                print(f"  Warmup phase: {remaining_warmup} steps remaining")
            else:
                print(f"  Skipping training: Buffer size {len(buffer)} < {config['batch_size']}")
        
        episode_time = time.time() - episode_start_time
        
        # Print progress
        if episode % config['print_interval'] == 0 or episode == 0:
            window_size = min(20, len(episode_rewards))
            if len(episode_rewards) >= window_size:
                moving_avg = np.mean(episode_rewards[-window_size:])
                moving_std = np.std(episode_rewards[-window_size:])
                avg_steps = np.mean(episode_steps[-window_size:])
                truncation_rate = np.mean(episode_truncated[-window_size:]) * 100
                
                # Calculate training statistics
                total_elapsed = time.time() - training_start_time
                eps_per_hour = (episode + 1) / (total_elapsed / 3600)
                eta_hours = (config['num_episodes'] - episode - 1) / eps_per_hour if eps_per_hour > 0 else 0
                
                print(f"\n{'='*80}")
                print(f"Episode {episode + 1}/{config['num_episodes']} Summary:")
                print(f"  Current: R={episode_reward:.2f}, Steps={steps}, Time={episode_time:.2f}s")
                print(f"  Moving Avg (last {window_size}): R={moving_avg:.2f}±{moving_std:.2f}, Steps={avg_steps:.1f}")
                print(f"  Truncation rate: {truncation_rate:.1f}%")
                print(f"  Best: R={best_reward:.2f}")
                print(f"  Total steps: {total_steps}, Buffer: {len(buffer)}/{config['buffer_limit']}")
                print(f"  Epsilon: {agent.epsilon:.3f}")
                if update_losses:
                    avg_q_loss = np.mean([l['q_loss'] for l in update_losses])
                    avg_actor_loss = np.mean([l['actor_loss'] for l in update_losses])
                    print(f"  Avg losses: Q={avg_q_loss:.4f}, Actor={avg_actor_loss:.4f}")
                print(f"  Training speed: {eps_per_hour:.1f} eps/hour, ETA: {eta_hours:.1f} hours")
                print(f"{'='*80}")
            else:
                print(f"\nEpisode {episode + 1}/{config['num_episodes']}: "
                      f"R={episode_reward:.2f}, Steps={steps}, Time={episode_time:.2f}s, ε={agent.epsilon:.3f}")
        
        # Evaluate
        if (episode + 1) % config['eval_interval'] == 0:
            print(f"\n{'='*60}")
            print(f"Evaluation at Episode {episode + 1}:")
            print(f"{'='*60}")
            
            eval_start_time = time.time()
            eval_rewards = []
            eval_steps = []
            eval_truncated = []
            
            for eval_idx in range(5):
                print(f"  Eval episode {eval_idx + 1}/5...", end='', flush=True)
                eval_reward, eval_step, eval_trunc = collect_experience_pdqn(
                    env, agent, config['steps_per_episode'], 
                    ReplayBuffer(1), deterministic=True,
                    max_episode_steps=config['max_steps_per_episode']
                )
                eval_rewards.append(eval_reward)
                eval_steps.append(eval_step)
                eval_truncated.append(eval_trunc)
                status = " (truncated)" if eval_trunc else " (completed)"
                print(f" R={eval_reward:.2f}, Steps={eval_step}{status}")
            
            eval_mean = np.mean(eval_rewards)
            eval_std = np.std(eval_rewards)
            eval_steps_mean = np.mean(eval_steps)
            eval_truncation_rate = np.mean(eval_truncated) * 100
            eval_time = time.time() - eval_start_time
            
            evaluation_rewards.append((episode + 1, eval_mean, eval_std))
            print(f"\nEvaluation Results:")
            print(f"  Reward: {eval_mean:.4f} ± {eval_std:.4f}")
            print(f"  Steps: {eval_steps_mean:.1f}")
            print(f"  Truncation rate: {eval_truncation_rate:.1f}%")
            print(f"  Time: {eval_time:.2f}s")
            print(f"{'='*60}\n")
        
        # Save checkpoint
        if config['save_models'] and (episode + 1) % config['save_interval'] == 0:
            checkpoint_path = os.path.join(config['save_dir'], f'pdqn_checkpoint_ep{episode+1}.pt')
            os.makedirs(config['save_dir'], exist_ok=True)
            agent.save(checkpoint_path)
            print(f"  Checkpoint saved to: {checkpoint_path}")
    
    total_training_time = time.time() - training_start_time
    
    print(f"\n{'='*80}")
    print(f"Training Complete!")
    print(f"{'='*80}")
    print(f"Total training time: {total_training_time/3600:.2f} hours")
    print(f"Final 100 episodes average: {np.mean(episode_rewards[-100:]):.4f}")
    print(f"Best episode reward: {best_reward:.4f}")
    print(f"Final epsilon: {agent.epsilon:.4f}")
    print(f"Total steps: {total_steps}")
    print(f"Average truncation rate: {np.mean(episode_truncated) * 100:.1f}%")
    print(f"{'='*80}")
    
    return episode_rewards, [], agent


def multi_run_training_pdqn(config=None):
    """Execute multiple training runs with different seeds for PDQN"""
    if config is None:
        config = CONFIG
    
    all_score_records = []
    all_models = []
    
    print(f"\n{'='*80}")
    print(f"Starting Multi-Run PDQN Training")
    print(f"Seeds: {config['seeds'][:config['num_runs']]}")
    print(f"Total runs: {config['num_runs']}")
    print(f"{'='*80}")
    
    for run_idx, seed in enumerate(config['seeds'][:config['num_runs']]):
        print(f"\n{'='*60}")
        print(f"Run {run_idx + 1}/{config['num_runs']} - Seed: {seed}")
        print(f"{'='*60}")
        
        # Set random seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        # Run training
        score_record, _, model = train_single_run_pdqn(config)
        
        # Store results
        all_score_records.append(score_record)
        all_models.append(model)
        
        print(f"Run {run_idx + 1} completed - Final Score: {np.mean(score_record[-10:]):.4f}")
    
    print(f"\n{'='*60}")
    print(f"All {config['num_runs']} runs completed!")
    print(f"{'='*60}")
    
    return all_score_records, [], all_models


# Reuse visualization and evaluation functions
def evaluate_agent(env, agent: PDQNAgent, num_episodes: int = 10, max_steps: int = None):
    """Evaluate agent performance with step limit"""
    if max_steps is None:
        max_steps = CONFIG['max_steps_per_episode']
        
    total_rewards = []
    step_counts = []
    truncated_episodes = 0
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        step_counter = 0
        
        while not done and step_counter < max_steps:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            job_mask = torch.FloatTensor(get_job_mask(env)).unsqueeze(0).to(agent.device)
            machine_mask = torch.FloatTensor(get_machine_mask(env)).unsqueeze(0).to(agent.device)
            operation_bounds = torch.FloatTensor(get_operation_bounds(env)).unsqueeze(0).to(agent.device)
            
            action_dict = agent.select_action(
                state_tensor, job_mask, machine_mask, operation_bounds, deterministic=True
            )
            
            job_id = action_dict['job_idx'].item()
            machine_id = action_dict['machine_idx'].item()
            param_value = float(action_dict['continuous_param'].squeeze(0).cpu().numpy()[0])
            action = [job_id, machine_id, param_value]
            
            step_result = env.step(action)
            step_counter += 1
            
            if len(step_result) == 3:
                state, reward, done = step_result
            else:
                state, reward, done, _ = step_result
                
            episode_reward += reward
        
        if step_counter >= max_steps and not done:
            truncated_episodes += 1
            status = " (truncated)"
        else:
            status = " (completed)"
            
        total_rewards.append(episode_reward)
        step_counts.append(step_counter)
        print(f"Evaluation episode {episode+1} finished in {step_counter} steps with reward {episode_reward:.4f}{status}")
        
    print(f"\nEvaluation Summary:")
    print(f"  Average reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"  Average steps: {np.mean(step_counts):.2f} ± {np.std(step_counts):.2f}")
    print(f"  Truncated episodes: {truncated_episodes}/{num_episodes} ({truncated_episodes/num_episodes*100:.1f}%)")
    
    return np.mean(total_rewards), np.std(total_rewards)


def test_and_visualize(config=None, model_path=None):
    """Test trained model and visualize scheduling process"""
    if config is None:
        config = CONFIG
        
    print("\n=== Starting PDQN Testing and Visualization ===")
    
    env_kwargs = {
        'num_of_jobs': config['num_jobs'],
        'num_of_robots': config['num_machines'],
        'alpha': config['alpha'],
        'beta': config['beta']
    }
    
    env = ENV.Env(**env_kwargs)
    
    state_dim = len(env.state)
    num_jobs = env_kwargs['num_of_jobs']
    num_machines = env_kwargs['num_of_robots']
    max_operations = config['max_operations']
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agent = PDQNAgent(
        state_dim=state_dim,
        num_jobs=num_jobs,
        num_machines=num_machines,
        max_operations=max_operations,
        actor_lr=config['actor_lr'],
        critic_lr=config['critic_lr'],
        gamma=config['gamma'],
        tau=config['tau'],
        device=device
    )
    
    if model_path:
        agent.load(model_path)
        print(f"Model loaded from: {model_path}")
    else:
        print("No model path provided, using untrained agent.")
    
    state = env.reset()
    done = False
    total_reward = 0
    step_counter = 0
    max_test_steps = config.get('max_test_steps', config['max_steps_per_episode'])
    
    print(f"Testing with max steps: {max_test_steps}")
    
    while not done and step_counter < max_test_steps:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
        job_mask = torch.FloatTensor(get_job_mask(env)).unsqueeze(0).to(agent.device)
        machine_mask = torch.FloatTensor(get_machine_mask(env)).unsqueeze(0).to(agent.device)
        operation_bounds = torch.FloatTensor(get_operation_bounds(env)).unsqueeze(0).to(agent.device)
        
        with torch.no_grad():
            action_dict = agent.select_action(
                state_tensor, job_mask, machine_mask, operation_bounds, deterministic=True
            )
        
        job_id = action_dict['job_idx'].item()
        machine_id = action_dict['machine_idx'].item()
        param_value = float(action_dict['continuous_param'].squeeze(0).cpu().numpy()[0])
        action = [job_id, machine_id, param_value]
        
        step_result = env.step(action)
        step_counter += 1
        
        if len(step_result) == 3:
            state, reward, done = step_result
        else:
            state, reward, done, _ = step_result
            
        total_reward += reward
        
        if config['enable_gantt_plots']:
            env.render_gantt()
    
    if step_counter >= max_test_steps and not done:
        print(f"Episode truncated at {max_test_steps} steps")
    else:
        print(f"Episode completed in {step_counter} steps")
    
    print(f"Total Reward: {total_reward}")
    plt.show()


# Reuse plotting functions from hppo_policy.py
def plot_multi_run_training_curves(all_score_records, config=None):
    """Plot training curves for multiple runs"""
    if config is None:
        config = CONFIG
    
    plt.figure(figsize=(12, 8))
    
    for i, scores in enumerate(all_score_records):
        x = range(len(scores))
        plt.plot(x, scores, alpha=0.6, label=f'Run {i+1} (Seed {config["seeds"][i]})')
    
    if len(all_score_records) > 1:
        min_length = min(len(scores) for scores in all_score_records)
        mean_scores = np.mean([scores[:min_length] for scores in all_score_records], axis=0)
        std_scores = np.std([scores[:min_length] for scores in all_score_records], axis=0)
        
        x = range(min_length)
        plt.plot(x, mean_scores, 'k-', linewidth=2, label='Mean')
        plt.fill_between(x, mean_scores - std_scores, mean_scores + std_scores, alpha=0.2, color='black')
    
    plt.title('PDQN Multi-Run Training Curves')
    plt.xlabel('Training Episode')
    plt.ylabel('Episode Reward')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plt.text(0.02, 0.98, f'Total Episodes: {config["num_episodes"]}', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.show()


def save_multi_run_results(all_score_records, all_action_restores, all_models, config):
    """Save results from multiple training runs"""
    if not config['save_models']:
        return None, None
    
    import json
    alg_name = "pdqn"
    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"{alg_name}_multi_run_{now_str}"
    os.makedirs(save_dir, exist_ok=True)

    np.save(os.path.join(save_dir, f"{alg_name}_all_scores_{now_str}.npy"), all_score_records)
    
    model_paths = []
    for run_idx, model in enumerate(all_models):
        run_save_dir = os.path.join(save_dir, f"run_{run_idx+1}_seed_{config['seeds'][run_idx]}")
        os.makedirs(run_save_dir, exist_ok=True)
        
        model_path = os.path.join(run_save_dir, f"{alg_name}_model_{now_str}.pt")
        model.save(model_path)
        model_paths.append(model_path)
    
    config_path = os.path.join(save_dir, f"config_{now_str}.json")
    with open(config_path, 'w') as f:
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


def main():
    """Main function for PDQN policy"""
    import argparse
    
    parser = argparse.ArgumentParser(description='PDQN Policy for Flexible Job Shop Scheduling')
    parser.add_argument('--jobs', type=int, default=CONFIG['num_jobs'], help='Number of jobs')
    parser.add_argument('--machines', type=int, default=CONFIG['num_machines'], help='Number of machines')
    parser.add_argument('--alpha', type=float, default=CONFIG['alpha'], help='Alpha parameter')
    parser.add_argument('--beta', type=float, default=CONFIG['beta'], help='Beta parameter')
    parser.add_argument('--episodes', type=int, default=CONFIG['num_episodes'], help='Number of episodes')
    parser.add_argument('--test-only', action='store_true', help='Run test only (skip training)')
    parser.add_argument('--model-path', type=str, default=None, help='Path to saved model for testing')
    parser.add_argument('--multi-run', action='store_true', default=CONFIG['enable_multi_run'], 
                       help='Run multiple training sessions with different seeds')
    parser.add_argument('--single-run', action='store_true', help='Force single run training (override config)')
    parser.add_argument('--seeds', nargs='+', type=int, default=CONFIG['seeds'], help='Random seeds for multi-run training')
    
    args = parser.parse_args()
    
    config = CONFIG.copy()
    config.update({
        'num_jobs': args.jobs,
        'num_machines': args.machines,
        'alpha': args.alpha,
        'beta': args.beta,
        'num_episodes': args.episodes,
        'seeds': args.seeds,
        'enable_multi_run': args.multi_run and not args.single_run
    })
    
    print(f"\n{'='*60}")
    print(f"PDQN Policy for Flexible Job Shop Scheduling")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  Jobs: {config['num_jobs']}")
    print(f"  Machines: {config['num_machines']}")
    print(f"  Alpha: {config['alpha']}")
    print(f"  Beta: {config['beta']}")
    print(f"  Episodes: {config['num_episodes']}")
    print(f"  Multi-run mode: {config['enable_multi_run']}")
    print(f"{'='*60}")
    
    if args.test_only:
        test_and_visualize(config, args.model_path)
    elif config['enable_multi_run']:
        all_score_records, all_action_restores, all_models = multi_run_training_pdqn(config)
        
        if config['save_models']:
            save_dir, model_paths = save_multi_run_results(all_score_records, all_action_restores, all_models, config)
        
        if config['plot_training_curve']:
            plot_multi_run_training_curves(all_score_records, config)
        
        if config['save_models'] and model_paths:
            last_n = 20
            mean_scores = [np.mean(scores[-last_n:]) if len(scores) >= last_n else np.mean(scores) 
                          for scores in all_score_records]
            best_run_idx = np.argmax(mean_scores)
            best_model_path = model_paths[best_run_idx]
            
            print(f"\nTesting with best model (Run {best_run_idx+1})...")
            test_and_visualize(config, best_model_path)
    else:
        if config['seeds']:
            seed = config['seeds'][0]
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            print(f"Using random seed: {seed}")
        
        score_record, action_restore, model = train_single_run_pdqn(config)
        
        if config['save_models']:
            save_dir, model_path = save_multi_run_results([score_record], [action_restore], [model], config)
        
        if config['plot_training_curve']:
            plt.figure(figsize=(10, 6))
            plt.plot(score_record)
            plt.title('PDQN Training Curve')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.grid(True)
            plt.show()
        
        if config['save_models'] and model_path:
            print(f"\nTesting with trained model...")
            test_and_visualize(config, model_path[0])


if __name__ == '__main__':
    main()
