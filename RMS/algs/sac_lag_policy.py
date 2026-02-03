import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical, Normal
from typing import Dict, Tuple, Optional, List, Union
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
    # Learning parameters - SAC-Lag specific
    'actor_lr': 3e-4,  # Actor learning rate
    'critic_lr': 3e-4,  # Critic learning rate
    'lambda_lr': 1e-3,  # Lagrange multiplier learning rate
    'gamma': 0.99,  # Discount factor
    'tau': 0.005,  # Soft update parameter
    'alpha': 0.2,  # Entropy temperature
    'alpha_lr': 3e-4,  # Temperature learning rate
    'batch_size': 256,  # Batch size for training
    'buffer_limit': 100000,  # Experience buffer size
    'gradient_steps': 1,  # Gradient steps per environment step
    
    # Constraint parameters
    'constraint_threshold': 0.0,  # Threshold for constraint violations
    'lambda_init': 5.0,  # Higher initial Lagrange multiplier values
    'lambda_min': 0.0,  # Minimum Lagrange multiplier
    'lambda_max': 200.0,  # Higher maximum Lagrange multiplier
    'use_hard_constraints': True,  # Use hard constraints for actions
    
    # Environment parameters
    'num_jobs': 10,
    'num_machines': 5,
    'max_operations': 5,
    'alpha_env': 0.5,
    'beta_env': 0.5,
    'num_episodes': 1000,
    'steps_per_episode': 55,
    
    # Training parameters
    'start_steps': 1000,  # Random action steps before training
    'update_after': 1000,  # Start updates after this many steps
    'update_every': 50,  # Update frequency
    'eval_interval': 50,
    'save_interval': 100,
    
    # Multi-run training parameters
    'enable_multi_run': False,
    'seeds': [3047, 294, 714, 1092, 1386, 2856, 42, 114514, 2025, 1993],
    'num_runs': 10,
    
    # Testing parameters
    'max_test_steps': 100,
    
    # Output parameters
    'print_interval': 10,
    'enable_gantt_plots': False,
    'plot_training_curve': True,
    'save_models': True,
    'save_dir': 'checkpoints',
}


class SACLagActor(nn.Module):
    """Actor network for SAC-Lag with hierarchical action space"""
    
    def __init__(self, state_dim: int, num_jobs: int, num_machines: int, 
                 max_operations: int, hidden_dim: int = 256):
        super().__init__()
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.max_operations = max_operations
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Job selection head (discrete)
        self.job_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_jobs)
        )
        
        # Machine selection head (discrete, conditioned on job)
        self.machine_head = nn.Sequential(
            nn.Linear(hidden_dim + num_jobs, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_machines)
        )
        
        # Continuous parameter heads (mean and log_std)
        self.param_mean_head = nn.Sequential(
            nn.Linear(hidden_dim + num_jobs + num_machines, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, max_operations)
        )
        
        self.param_log_std_head = nn.Sequential(
            nn.Linear(hidden_dim + num_jobs + num_machines, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, max_operations)
        )
        
        # Log std bounds
        self.log_std_min = -20
        self.log_std_max = 2
        
    def forward(self, state: torch.Tensor, job_mask: Optional[torch.Tensor] = None,
                machine_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass with soft masking through large negative logits"""
        features = self.feature_extractor(state)
        
        # Job selection with soft masking
        job_logits = self.job_head(features)
        if job_mask is not None:
            # Apply soft mask by adding large negative values to invalid actions
            mask_value = -1e8
            job_logits = job_logits + (1 - job_mask) * mask_value
        
        # Get job probabilities for conditioning
        job_probs = F.softmax(job_logits, dim=-1)
        
        # Machine selection conditioned on job with soft masking
        machine_input = torch.cat([features, job_probs], dim=-1)
        machine_logits = self.machine_head(machine_input)
        if machine_mask is not None:
            mask_value = -1e8
            machine_logits = machine_logits + (1 - machine_mask) * mask_value
        
        # Continuous parameters conditioned on discrete actions
        machine_probs = F.softmax(machine_logits, dim=-1)
        param_input = torch.cat([features, job_probs, machine_probs], dim=-1)
        
        param_mean = self.param_mean_head(param_input)
        param_log_std = self.param_log_std_head(param_input)
        param_log_std = torch.clamp(param_log_std, self.log_std_min, self.log_std_max)
        
        return {
            'job_logits': job_logits,
            'machine_logits': machine_logits,
            'param_mean': param_mean,
            'param_log_std': param_log_std
        }
    
    def sample(self, state: torch.Tensor, job_mask: Optional[torch.Tensor] = None,
               machine_mask: Optional[torch.Tensor] = None,
               operation_bounds: Optional[torch.Tensor] = None) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Sample action and compute log probability"""
        outputs = self.forward(state, job_mask, machine_mask)
        
        # Sample discrete actions
        job_dist = Categorical(logits=outputs['job_logits'])
        job_action = job_dist.sample()
        job_log_prob = job_dist.log_prob(job_action)
        
        machine_dist = Categorical(logits=outputs['machine_logits'])
        machine_action = machine_dist.sample()
        machine_log_prob = machine_dist.log_prob(machine_action)
        
        # Sample continuous actions
        param_std = torch.exp(outputs['param_log_std'])
        param_dist = Normal(outputs['param_mean'], param_std)
        param_action_raw = param_dist.rsample()  # Reparameterization trick
        param_log_prob = param_dist.log_prob(param_action_raw).sum(dim=-1)
        
        # Apply tanh squashing and adjust log probability
        param_action = torch.tanh(param_action_raw)
        param_log_prob -= torch.log(1 - param_action.pow(2) + 1e-6).sum(dim=-1)
        
        # Scale to operation bounds if provided
        if operation_bounds is not None:
            lower_bounds = operation_bounds[..., 0]
            upper_bounds = operation_bounds[..., 1]
            param_action_scaled = 0.5 * (param_action + 1) * (upper_bounds - lower_bounds) + lower_bounds
        else:
            param_action_scaled = param_action
        
        # Total log probability
        log_prob = job_log_prob + machine_log_prob + param_log_prob
        
        actions = {
            'job_idx': job_action,
            'machine_idx': machine_action,
            'continuous_param': param_action_scaled,
            'continuous_param_raw': param_action_raw
        }
        
        return actions, log_prob


class SACLagCritic(nn.Module):
    """Critic network for SAC-Lag"""
    
    def __init__(self, state_dim: int, num_jobs: int, num_machines: int,
                 max_operations: int, hidden_dim: int = 256):
        super().__init__()
        
        # Q1 network
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + num_jobs + num_machines + max_operations, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Q2 network (for double Q-learning)
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + num_jobs + num_machines + max_operations, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        
    def forward(self, state: torch.Tensor, actions: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for both Q networks"""
        # One-hot encode discrete actions
        job_one_hot = F.one_hot(actions['job_idx'].long(), num_classes=self.num_jobs).float()
        machine_one_hot = F.one_hot(actions['machine_idx'].long(), num_classes=self.num_machines).float()
        
        # Concatenate state and actions
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        if len(job_one_hot.shape) == 1:
            job_one_hot = job_one_hot.unsqueeze(0)
        if len(machine_one_hot.shape) == 1:
            machine_one_hot = machine_one_hot.unsqueeze(0)
        if len(actions['continuous_param'].shape) == 1:
            continuous_param = actions['continuous_param'].unsqueeze(0)
        else:
            continuous_param = actions['continuous_param']
            
        x = torch.cat([state, job_one_hot, machine_one_hot, continuous_param], dim=-1)
        
        q1 = self.q1(x)
        q2 = self.q2(x)
        
        return q1, q2


class ConstraintCritic(nn.Module):
    """Constraint critic network for SAC-Lag"""
    
    def __init__(self, state_dim: int, num_jobs: int, num_machines: int,
                 max_operations: int, num_constraints: int = 3, hidden_dim: int = 256):
        super().__init__()
        
        self.num_constraints = num_constraints
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        
        # Constraint cost networks (one for each constraint)
        self.constraint_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim + num_jobs + num_machines + max_operations, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            ) for _ in range(num_constraints)
        ])
        
    def forward(self, state: torch.Tensor, actions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass for constraint critics"""
        # One-hot encode discrete actions
        job_one_hot = F.one_hot(actions['job_idx'].long(), num_classes=self.num_jobs).float()
        machine_one_hot = F.one_hot(actions['machine_idx'].long(), num_classes=self.num_machines).float()
        
        # Handle dimension issues
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        if len(job_one_hot.shape) == 1:
            job_one_hot = job_one_hot.unsqueeze(0)
        if len(machine_one_hot.shape) == 1:
            machine_one_hot = machine_one_hot.unsqueeze(0)
        if len(actions['continuous_param'].shape) == 1:
            continuous_param = actions['continuous_param'].unsqueeze(0)
        else:
            continuous_param = actions['continuous_param']
            
        x = torch.cat([state, job_one_hot, machine_one_hot, continuous_param], dim=-1)
        
        # Compute constraint costs
        constraint_costs = []
        for net in self.constraint_nets:
            cost = net(x)
            constraint_costs.append(cost)
        
        return torch.cat(constraint_costs, dim=-1)


class SACLagAgent:
    """SAC-Lagrangian agent for constrained job shop scheduling"""
    
    def __init__(self, state_dim: int, num_jobs: int, num_machines: int,
                 max_operations: int, num_constraints: int = 3, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.max_operations = max_operations
        self.num_constraints = num_constraints
        
        # Initialize networks
        self.actor = SACLagActor(state_dim, num_jobs, num_machines, max_operations).to(self.device)
        self.critic = SACLagCritic(state_dim, num_jobs, num_machines, max_operations).to(self.device)
        self.critic_target = SACLagCritic(state_dim, num_jobs, num_machines, max_operations).to(self.device)
        self.constraint_critic = ConstraintCritic(
            state_dim, num_jobs, num_machines, max_operations, num_constraints
        ).to(self.device)
        
        # Initialize target network
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Initialize optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=CONFIG['actor_lr'])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=CONFIG['critic_lr'])
        self.constraint_critic_optimizer = torch.optim.Adam(
            self.constraint_critic.parameters(), lr=CONFIG['critic_lr']
        )
        
        # Initialize temperature
        self.log_alpha = torch.tensor(np.log(CONFIG['alpha']), requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=CONFIG['alpha_lr'])
        self.target_entropy = -(num_jobs + num_machines + max_operations)  # Heuristic
        
        # Initialize Lagrange multipliers with higher initial values - create as leaf tensors
        init_lambda_values = [CONFIG['lambda_init']] * num_constraints
        self.log_lambdas = torch.tensor(
            np.log(init_lambda_values), 
            dtype=torch.float32,
            device=self.device, 
            requires_grad=True
        )
        self.lambdas = torch.exp(self.log_lambdas.detach())  # Detach to avoid graph issues
        self.lambda_optimizer = torch.optim.Adam([self.log_lambdas], lr=CONFIG['lambda_lr'])
        
        # Constraint thresholds (stricter for discrete constraints)
        self.constraint_thresholds = torch.tensor([0.0, 0.0, CONFIG['constraint_threshold']]).to(self.device)
        
        # Add constraint violation penalty weight
        self.constraint_penalty_weight = 10.0
        
    def select_action(self, state: torch.Tensor, job_mask: torch.Tensor,
                     machine_mask: torch.Tensor, operation_bounds: torch.Tensor,
                     deterministic: bool = False) -> Dict[str, torch.Tensor]:
        """Select action with soft constraint consideration"""
        with torch.no_grad():
            if deterministic:
                # Get mean action for evaluation with masks
                outputs = self.actor(state, job_mask, machine_mask)
                
                # Discrete actions: argmax (already influenced by soft masking)
                job_idx = outputs['job_logits'].argmax(dim=-1)
                machine_idx = outputs['machine_logits'].argmax(dim=-1)
                
                # Continuous actions: mean
                continuous_param = torch.tanh(outputs['param_mean'])
                
                # Scale to bounds
                lower_bounds = operation_bounds[..., 0]
                upper_bounds = operation_bounds[..., 1]
                continuous_param = 0.5 * (continuous_param + 1) * (upper_bounds - lower_bounds) + lower_bounds
                
                return {
                    'job_idx': job_idx,
                    'machine_idx': machine_idx,
                    'continuous_param': continuous_param
                }
            else:
                # Sample action with masks
                actions, _ = self.actor.sample(state, job_mask, machine_mask, operation_bounds)
                return actions
    
    # Use cases for three constraint types
    def compute_constraint_violations(self, states, actions, job_masks, machine_masks, operation_bounds):
        """
        1. Job completion constraint - avoid rescheduling finished jobs
        2. Machine availability constraint - avoid assigning new tasks to busy machines
        3. Operation parameter constraint - ensure processing parameters stay within physical bounds
        """
        batch_size = states.shape[0]
        violations = torch.zeros(batch_size, self.num_constraints).to(self.device)
        
        # Constraint 1: Completed jobs should not be selected (binary violation)
        job_selected = F.one_hot(actions['job_idx'].long(), num_classes=self.num_jobs).float()
        job_violation = (job_selected * (1 - job_masks)).sum(dim=-1)
        violations[:, 0] = job_violation * self.constraint_penalty_weight
        
        # Constraint 2: Busy machines should not be selected (binary violation)
        machine_selected = F.one_hot(actions['machine_idx'].long(), num_classes=self.num_machines).float()
        machine_violation = (machine_selected * (1 - machine_masks)).sum(dim=-1)
        violations[:, 1] = machine_violation * self.constraint_penalty_weight
        
        # Constraint 3: Continuous parameters should be within bounds
        lower_bounds = operation_bounds[..., 0]
        upper_bounds = operation_bounds[..., 1]
        param_violation = torch.relu(lower_bounds - actions['continuous_param']) + \
                         torch.relu(actions['continuous_param'] - upper_bounds)
        violations[:, 2] = param_violation.sum(dim=-1)
        # print("Operation-parameter constraint violations:", violations[:, 2].cpu().numpy())
        return violations
    
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Update networks using SAC-Lag algorithm with enhanced constraint handling"""
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones']
        job_masks = batch['job_masks']
        machine_masks = batch['machine_masks']
        next_job_masks = batch['next_job_masks']
        next_machine_masks = batch['next_machine_masks']
        operation_bounds = batch['operation_bounds']
        
        # Update constraint critic
        with torch.no_grad():
            constraint_violations = self.compute_constraint_violations(
                states, actions, job_masks, machine_masks, operation_bounds
            )
        
        constraint_costs = self.constraint_critic(states, actions)
        constraint_critic_loss = F.mse_loss(constraint_costs, constraint_violations)
        
        self.constraint_critic_optimizer.zero_grad()
        constraint_critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.constraint_critic.parameters(), 1.0)
        self.constraint_critic_optimizer.step()
        
        # Update critics with constraint-augmented rewards
        with torch.no_grad():
            # Add constraint penalty to rewards
            constraint_penalty = (self.lambdas * constraint_violations).sum(dim=-1, keepdim=True)
            augmented_rewards = rewards - constraint_penalty
            
            next_actions, next_log_prob = self.actor.sample(
                next_states, next_job_masks, next_machine_masks, operation_bounds
            )
            next_q1, next_q2 = self.critic_target(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2) - self.log_alpha.exp() * next_log_prob.unsqueeze(-1)
            target_q = augmented_rewards + CONFIG['gamma'] * (1 - dones) * next_q
        
        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # Update actor with stronger constraint guidance
        new_actions, log_prob = self.actor.sample(states, job_masks, machine_masks, operation_bounds)
        q1_new, q2_new = self.critic(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        # Compute expected constraint costs with current policy
        constraint_costs_new = self.constraint_critic(states, new_actions)
        
        # Actor loss with Lagrangian augmentation and direct constraint penalty
        actor_loss = (self.log_alpha.exp() * log_prob.unsqueeze(-1) - q_new).mean()
        
        # Add both Lagrangian term and direct penalty
        lagrangian_term = (self.lambdas * constraint_costs_new).sum(dim=-1).mean()
        direct_penalty = constraint_costs_new.sum(dim=-1).mean() * 0.1  # Additional direct penalty
        
        actor_loss += lagrangian_term + direct_penalty
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # Update temperature
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        # Update Lagrange multipliers with momentum
        with torch.no_grad():
            avg_constraint_violations = constraint_violations.mean(dim=0)
            # Use exponential moving average for smoother updates
            violation_ema = 0.95 * avg_constraint_violations + 0.05 * self.constraint_thresholds
        
        # Gradient ascent on Lagrange multipliers
        lambda_loss = -(self.log_lambdas * (violation_ema - self.constraint_thresholds).detach()).sum()
        
        self.lambda_optimizer.zero_grad()
        lambda_loss.backward()
        self.lambda_optimizer.step()
        
        # Update lambdas with increased bounds for discrete constraints
        with torch.no_grad():
            self.lambdas = torch.exp(self.log_lambdas)
            # Higher max for discrete constraints
            lambda_max_discrete = CONFIG['lambda_max'] * 2.0
            lambda_max_continuous = CONFIG['lambda_max']
            max_values = torch.tensor([lambda_max_discrete, lambda_max_discrete, lambda_max_continuous]).to(self.device)
            min_values = torch.tensor([CONFIG['lambda_min']] * self.num_constraints).to(self.device)
            self.lambdas = torch.maximum(torch.minimum(self.lambdas, max_values), min_values)
            self.log_lambdas.data = torch.log(self.lambdas + 1e-8)
        
        # Soft update target network
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(CONFIG['tau'] * param.data + (1 - CONFIG['tau']) * target_param.data)
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'constraint_critic_loss': constraint_critic_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'lambda_loss': lambda_loss.item(),
            'alpha': self.log_alpha.exp().item(),
            'lambdas': self.lambdas.detach().cpu().numpy(),
            'avg_constraint_violations': avg_constraint_violations.detach().cpu().numpy()
        }
    
    def save(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'constraint_critic_state_dict': self.constraint_critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'constraint_critic_optimizer_state_dict': self.constraint_critic_optimizer.state_dict(),
            'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict(),
            'lambda_optimizer_state_dict': self.lambda_optimizer.state_dict(),
            'log_alpha': self.log_alpha,
            'log_lambdas': self.log_lambdas
        }, path)
    
    def load(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.constraint_critic.load_state_dict(checkpoint['constraint_critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.constraint_critic_optimizer.load_state_dict(checkpoint['constraint_critic_optimizer_state_dict'])
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
        self.lambda_optimizer.load_state_dict(checkpoint['lambda_optimizer_state_dict'])
        self.log_alpha = checkpoint['log_alpha'].to(self.device)
        self.log_lambdas = checkpoint['log_lambdas'].to(self.device)
        self.lambdas = torch.exp(self.log_lambdas.detach())


class ReplayBuffer:
    """Experience replay buffer for SAC-Lag"""
    
    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
    def push(self, state: np.ndarray, action: Dict[str, Union[int, np.ndarray]], 
             reward: float, next_state: np.ndarray, done: bool,
             job_mask: np.ndarray, machine_mask: np.ndarray,
             next_job_mask: np.ndarray, next_machine_mask: np.ndarray,
             operation_bounds: np.ndarray):
        """Add experience to buffer"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        self.buffer[self.position] = (
            state, action, reward, next_state, done,
            job_mask, machine_mask, next_job_mask, next_machine_mask,
            operation_bounds
        )
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a batch of experiences"""
        batch = random.sample(self.buffer, batch_size)
        
        states, actions, rewards, next_states, dones, \
        job_masks, machine_masks, next_job_masks, next_machine_masks, \
        operation_bounds = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states))
        next_states = torch.FloatTensor(np.array(next_states))
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        
        # Handle actions
        job_indices = torch.LongTensor([a['job_idx'] for a in actions])
        machine_indices = torch.LongTensor([a['machine_idx'] for a in actions])
        continuous_params = torch.FloatTensor(np.array([a['continuous_param'] for a in actions]))
        
        actions_dict = {
            'job_idx': job_indices,
            'machine_idx': machine_indices,
            'continuous_param': continuous_params
        }
        
        # Masks and bounds
        job_masks = torch.FloatTensor(np.array(job_masks))
        machine_masks = torch.FloatTensor(np.array(machine_masks))
        next_job_masks = torch.FloatTensor(np.array(next_job_masks))
        next_machine_masks = torch.FloatTensor(np.array(next_machine_masks))
        operation_bounds = torch.FloatTensor(np.array(operation_bounds))
        
        return {
            'states': states,
            'actions': actions_dict,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones,
            'job_masks': job_masks,
            'machine_masks': machine_masks,
            'next_job_masks': next_job_masks,
            'next_machine_masks': next_machine_masks,
            'operation_bounds': operation_bounds
        }
        
    def __len__(self):
        return len(self.buffer)


# Utility functions (reuse from other policies)
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


def train_single_run_sac_lag(config=None):
    """Execute a single training run with SAC-Lag"""
    if config is None:
        config = CONFIG
    
    # Environment setup
    env_kwargs = {
        'num_of_jobs': config['num_jobs'],
        'num_of_robots': config['num_machines'],
        'alpha': config['alpha_env'],
        'beta': config['beta_env']
    }
    
    env = ENV.Env(**env_kwargs)
    print(f"Successfully created environment with parameters: {env_kwargs}")
    
    # Get environment dimensions
    state_dim = len(env.state)
    num_jobs = env_kwargs['num_of_jobs']
    num_machines = env_kwargs['num_of_robots']
    max_operations = config['max_operations']
    
    # Create agent and buffer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agent = SACLagAgent(
        state_dim=state_dim,
        num_jobs=num_jobs,
        num_machines=num_machines,
        max_operations=max_operations,
        num_constraints=3,
        device=device
    )
    
    buffer = ReplayBuffer(capacity=config['buffer_limit'])
    
    # Training metrics
    episode_rewards = []
    episode_steps = []
    evaluation_rewards = []
    constraint_violations_history = []
    
    print(f"Starting SAC-Lag training:")
    print(f"Jobs: {num_jobs}, Machines: {num_machines}")
    print(f"Alpha: {config['alpha_env']}, Beta: {config['beta_env']}, Episodes: {config['num_episodes']}")
    print(f"Device: {device}")
    print(f"Max steps per episode: {config['steps_per_episode']}")
    print(f"Constraints: Job completion, Machine availability, Operation bounds (No hard masking)")
    
    best_reward = -float('inf')
    total_steps = 0
    
    for episode in range(config['num_episodes']):
        state = env.reset()
        episode_reward = 0
        episode_constraint_violations = np.zeros(3)
        done = False
        step_counter = 0
        
        # Episode loop with max steps limit
        while not done and step_counter < config['steps_per_episode']:
            # Get masks and bounds
            job_mask = get_job_mask(env)
            machine_mask = get_machine_mask(env)
            operation_bounds = get_operation_bounds(env)
            
            # Convert to tensors
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            job_mask_tensor = torch.FloatTensor(job_mask).unsqueeze(0).to(device)
            machine_mask_tensor = torch.FloatTensor(machine_mask).unsqueeze(0).to(device)
            operation_bounds_tensor = torch.FloatTensor(operation_bounds).unsqueeze(0).to(device)
            
            # Select action
            if total_steps < config['start_steps']:
                # Random action for initial exploration (still use valid actions for better exploration)
                valid_jobs = np.where(job_mask > 0)[0]
                valid_machines = np.where(machine_mask > 0)[0]
                
                if len(valid_jobs) > 0 and len(valid_machines) > 0:
                    job_id = np.random.choice(valid_jobs)
                    machine_id = np.random.choice(valid_machines)
                else:
                    job_id = np.random.randint(0, num_jobs)
                    machine_id = np.random.randint(0, num_machines)
                
                # Generate random continuous parameters for all operations
                param_values = []
                for i in range(max_operations):
                    param_value = np.random.uniform(operation_bounds[i, 0], operation_bounds[i, 1])
                    param_values.append(param_value)
                continuous_param = np.array(param_values)
                
                action_dict = {
                    'job_idx': job_id,
                    'machine_idx': machine_id,
                    'continuous_param': continuous_param
                }
            else:
                with torch.no_grad():
                    action_dict = agent.select_action(
                        state_tensor, job_mask_tensor, machine_mask_tensor, 
                        operation_bounds_tensor, deterministic=False
                    )
                    # Convert to numpy for environment
                    action_dict = {
                        'job_idx': action_dict['job_idx'].cpu().item(),
                        'machine_idx': action_dict['machine_idx'].cpu().item(),
                        'continuous_param': action_dict['continuous_param'].cpu().numpy()[0]
                    }
            
            # Create action for environment (only use first continuous parameter)
            action = [action_dict['job_idx'], action_dict['machine_idx'], action_dict['continuous_param'][0]]
            
            # Step environment
            step_result = env.step(action)
            step_counter += 1
            total_steps += 1
            
            if len(step_result) == 3:
                next_state, reward, done = step_result
            else:
                next_state, reward, done, _ = step_result
                
            episode_reward += reward
            
            # Check constraint violations
            if job_mask[action_dict['job_idx']] == 0:
                episode_constraint_violations[0] += 1
            if machine_mask[action_dict['machine_idx']] == 0:
                episode_constraint_violations[1] += 1
            # Check bounds for all continuous parameters
            for i in range(max_operations):
                if action_dict['continuous_param'][i] < operation_bounds[i, 0] or \
                   action_dict['continuous_param'][i] > operation_bounds[i, 1]:
                    episode_constraint_violations[2] += 1
                    break
            
            # Force episode termination if max steps reached
            if step_counter >= config['steps_per_episode']:
                done = True
                if config['print_interval'] <= 10:  # Only print if verbose
                    print(f"  Episode {episode + 1} reached max steps limit ({config['steps_per_episode']})")
            
            # Get next masks
            next_job_mask = get_job_mask(env) if not done else job_mask
            next_machine_mask = get_machine_mask(env) if not done else machine_mask
            
            # Store experience
            buffer.push(
                state, action_dict, reward, next_state, done,
                job_mask, machine_mask, next_job_mask, next_machine_mask,
                operation_bounds
            )
            
            # Update networks
            if total_steps >= config['update_after'] and total_steps % config['update_every'] == 0:
                for _ in range(config['gradient_steps']):
                    if len(buffer) > config['batch_size']:
                        batch = buffer.sample(config['batch_size'])
                        # Move batch to device
                        for key in batch:
                            if isinstance(batch[key], dict):
                                for sub_key in batch[key]:
                                    batch[key][sub_key] = batch[key][sub_key].to(device)
                            else:
                                batch[key] = batch[key].to(device)
                        
                        losses = agent.update(batch)
            
            state = next_state
        
        episode_rewards.append(episode_reward)
        episode_steps.append(step_counter)
        constraint_violations_history.append(episode_constraint_violations)
        
        # Update best reward
        if episode_reward > best_reward:
            best_reward = episode_reward
        
        # Print progress
        if episode % config['print_interval'] == 0:
            window_size = min(20, len(episode_rewards))
            if len(episode_rewards) >= window_size:
                moving_avg = np.mean(episode_rewards[-window_size:])
                moving_std = np.std(episode_rewards[-window_size:])
                avg_violations = np.mean(constraint_violations_history[-window_size:], axis=0)
                avg_steps = np.mean(episode_steps[-window_size:])
                print(f"Episode {episode + 1}/{config['num_episodes']}: "
                      f"R={episode_reward:.2f}, Steps={step_counter}, "
                      f"MA={moving_avg:.2f}±{moving_std:.2f}, "
                      f"AvgSteps={avg_steps:.1f}, "
                      f"Best={best_reward:.2f}, "
                      f"Violations={avg_violations}")
            else:
                print(f"Episode {episode + 1}/{config['num_episodes']}: "
                      f"R={episode_reward:.2f}, Steps={step_counter}")
        
        # Evaluate
        if (episode + 1) % config['eval_interval'] == 0:
            eval_rewards = []
            eval_violations = []
            eval_steps = []
            
            for _ in range(5):
                eval_state = env.reset()
                eval_reward = 0
                eval_done = False
                eval_step_counter = 0
                eval_constraint_violations = np.zeros(3)
                
                # Evaluation loop with max steps limit
                while not eval_done and eval_step_counter < config['steps_per_episode']:
                    eval_job_mask = torch.FloatTensor(get_job_mask(env)).unsqueeze(0).to(device)
                    eval_machine_mask = torch.FloatTensor(get_machine_mask(env)).unsqueeze(0).to(device)
                    eval_bounds = torch.FloatTensor(get_operation_bounds(env)).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        eval_action = agent.select_action(
                            torch.FloatTensor(eval_state).unsqueeze(0).to(device),
                            eval_job_mask, eval_machine_mask, eval_bounds,
                            deterministic=True
                        )
                    
                    eval_action_list = [
                        eval_action['job_idx'].cpu().item(),
                        eval_action['machine_idx'].cpu().item(),
                        eval_action['continuous_param'].cpu().numpy()[0][0]  # Use first parameter
                    ]
                    
                    eval_step_result = env.step(eval_action_list)
                    eval_step_counter += 1
                    
                    if len(eval_step_result) == 3:
                        eval_state, eval_r, eval_done = eval_step_result
                    else:
                        eval_state, eval_r, eval_done, _ = eval_step_result
                    
                    eval_reward += eval_r
                    
                    # Force termination if max steps reached
                    if eval_step_counter >= config['steps_per_episode']:
                        eval_done = True
                
                eval_rewards.append(eval_reward)
                eval_violations.append(eval_constraint_violations)
                eval_steps.append(eval_step_counter)
            
            eval_mean = np.mean(eval_rewards)
            eval_std = np.std(eval_rewards)
            eval_violations_mean = np.mean(eval_violations, axis=0)
            eval_steps_mean = np.mean(eval_steps)
            evaluation_rewards.append((episode + 1, eval_mean, eval_std))
            print(f"\nEvaluation at Episode {episode + 1}: {eval_mean:.4f} ± {eval_std:.4f}")
            print(f"Average steps: {eval_steps_mean:.1f}")
            print(f"Constraint violations: {eval_violations_mean}\n")
            
            # Save best model
            if eval_mean > best_reward and config['save_models']:
                best_model_path = os.path.join(config['save_dir'], 'best_sac_lag_model.pt')
                os.makedirs(config['save_dir'], exist_ok=True)
                agent.save(best_model_path)
                print(f"Saved best model with eval reward: {eval_mean:.4f}")
    
    print(f"\nTraining Complete!")
    print(f"Final 100 episodes average: {np.mean(episode_rewards[-100:]):.4f}")
    print(f"Best episode reward: {best_reward:.4f}")
    print(f"Average steps in final 100 episodes: {np.mean(episode_steps[-100:]):.1f}")
    print(f"Final constraint violations: {np.mean(constraint_violations_history[-100:], axis=0)}")
    
    return episode_rewards, [], agent


def multi_run_training_sac_lag(config=None):
    """Execute multiple training runs with different seeds for SAC-Lag"""
    if config is None:
        config = CONFIG
    
    all_score_records = []
    all_models = []
    
    print(f"\n{'='*80}")
    print(f"Starting Multi-Run SAC-Lag Training")
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
        score_record, _, model = train_single_run_sac_lag(config)
        
        # Store results
        all_score_records.append(score_record)
        all_models.append(model)
        
        print(f"Run {run_idx + 1} completed - Final Score: {np.mean(score_record[-10:]):.4f}")
    
    print(f"\n{'='*60}")
    print(f"All {config['num_runs']} runs completed!")
    print(f"{'='*60}")
    
    return all_score_records, [], all_models


# Testing and visualization functions (similar to other policies)
def test_and_visualize(config=None, model_path=None):
    """Test trained model and visualize scheduling process"""
    if config is None:
        config = CONFIG
        
    print("\n=== Starting SAC-Lag Testing and Visualization ===")
    
    env_kwargs = {
        'num_of_jobs': config['num_jobs'],
        'num_of_robots': config['num_machines'],
        'alpha': config['alpha_env'],
        'beta': config['beta_env']
    }
    
    env = ENV.Env(**env_kwargs)
    
    state_dim = len(env.state)
    num_jobs = env_kwargs['num_of_jobs']
    num_machines = env_kwargs['num_of_robots']
    max_operations = config['max_operations']
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agent = SACLagAgent(
        state_dim=state_dim,
        num_jobs=num_jobs,
        num_machines=num_machines,
        max_operations=max_operations,
        num_constraints=3,
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
    constraint_violations = np.zeros(3)
    
    while not done and step_counter < config['max_test_steps']:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        job_mask = torch.FloatTensor(get_job_mask(env)).unsqueeze(0).to(device)
        machine_mask = torch.FloatTensor(get_machine_mask(env)).unsqueeze(0).to(device)
        operation_bounds = torch.FloatTensor(get_operation_bounds(env)).unsqueeze(0).to(device)
        
        with torch.no_grad():
            action_dict = agent.select_action(
                state_tensor, job_mask, machine_mask, operation_bounds, deterministic=True
            )
        
        job_id = action_dict['job_idx'].cpu().item()
        machine_id = action_dict['machine_idx'].cpu().item()
        param_value = action_dict['continuous_param'].cpu().numpy()[0][0]  # Use first parameter
        
        # Check constraints
        if job_mask[0, job_id] == 0:
            print(f"Warning: Selected completed job {job_id}")
            constraint_violations[0] += 1
        if machine_mask[0, machine_id] == 0:
            print(f"Warning: Selected busy machine {machine_id}")
            constraint_violations[1] += 1
        
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
    
    print(f"Total Reward: {total_reward}")
    print(f"Episode completed in {step_counter} steps")
    print(f"Constraint violations: {constraint_violations}")
    plt.show()


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
    
    plt.title('SAC-Lag Multi-Run Training Curves')
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
    alg_name = "sac_lag"
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
    """Main function for SAC-Lag policy"""
    import argparse
    
    parser = argparse.ArgumentParser(description='SAC-Lag Policy for Flexible Job Shop Scheduling with Constraints')
    parser.add_argument('--jobs', type=int, default=CONFIG['num_jobs'], help='Number of jobs')
    parser.add_argument('--machines', type=int, default=CONFIG['num_machines'], help='Number of machines')
    parser.add_argument('--alpha', type=float, default=CONFIG['alpha_env'], help='Alpha parameter')
    parser.add_argument('--beta', type=float, default=CONFIG['beta_env'], help='Beta parameter')
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
        'alpha_env': args.alpha,
        'beta_env': args.beta,
        'num_episodes': args.episodes,
        'seeds': args.seeds,
        'enable_multi_run': args.multi_run and not args.single_run
    })
    
    print(f"\n{'='*60}")
    print(f"SAC-Lag Policy for Flexible Job Shop Scheduling")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  Jobs: {config['num_jobs']}")
    print(f"  Machines: {config['num_machines']}")
    print(f"  Alpha: {config['alpha_env']}")
    print(f"  Beta: {config['beta_env']}")
    print(f"  Episodes: {config['num_episodes']}")
    print(f"  Multi-run mode: {config['enable_multi_run']}")
    print(f"  Constraints:")
    print(f"    1. Completed jobs cannot be selected")
    print(f"    2. Busy machines cannot be selected")
    print(f"    3. Continuous parameters respect operation bounds")
    print(f"{'='*60}")
    
    if args.test_only:
        test_and_visualize(config, args.model_path)
    elif config['enable_multi_run']:
        all_score_records, all_action_restores, all_models = multi_run_training_sac_lag(config)
        
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
        
        score_record, action_restore, model = train_single_run_sac_lag(config)
        
        if config['save_models']:
            save_dir, model_path = save_multi_run_results([score_record], [action_restore], [model], config)
        
        if config['plot_training_curve']:
            plt.figure(figsize=(10, 6))
            plt.plot(score_record)
            plt.title('SAC-Lag Training Curve')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.grid(True)
            plt.show()
        
        if config['save_models'] and model_path:
            print(f"\nTesting with trained model...")
            test_and_visualize(config, model_path[0])




if __name__ == '__main__':
    main()
