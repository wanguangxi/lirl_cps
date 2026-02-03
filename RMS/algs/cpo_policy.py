import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical, Normal
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
    # Learning parameters - CPO specific
    'lr': 0.005,  # Learning rate (reduced from 3e-3)
    'gamma': 0.99,  # Discount factor
    'lambda_gae': 0.95,  # GAE lambda
    'delta': 0.05,  # KL constraint threshold (increased from 0.01 for more exploration)
    'damping': 0.1,  # Damping for conjugate gradient
    'max_backtracks': 10,  # Maximum line search steps
    'batch_size': 64,  # Batch size for training (reduced from 256)
    'buffer_limit': 10000,  # Experience buffer size
    'cg_iterations': 10,  # Conjugate gradient iterations
    'value_epochs': 5,  # Value function update epochs (reduced from 10)
    
    # Environment parameters
    'num_jobs': 100,
    'num_machines': 5,
    'max_operations': 5,
    'alpha': 0.5,
    'beta': 0.5,
    'num_episodes': 1000,  # Increased from 1000
    'steps_per_episode': 550,
    
    # Training parameters
    'update_interval': 512,  # Steps between policy updates (reduced from 1024)
    'eval_interval': 50,
    'save_interval': 100,
    'normalize_advantages': True,  # Normalize advantages
    'clip_grad_norm': 1.0,  # Gradient clipping (increased from 0.5)
    'entropy_coef': 0.05,  # Entropy regularization (increased from 0.01)
    'max_steps_per_episode': 500,  # Maximum steps per episode (hard limit)
    
    # Constraint parameters
    'constraint_threshold': 0.05,  # lower threshold for stricter constraints
    'use_hard_constraints': True,
    'line_search_steps': 10,
    'accept_ratio': 0.1,
    'violation_penalty': 10.0,  # increase penalty strength
    'constraint_weights': [5.0, 5.0, 1.0],  # increase weights
    'constraint_decay_rate': 0.99,  # slower decay
    
    # Multi-run training parameters
    'enable_multi_run': True,  # Start with single run for debugging
    # 'seeds': [3047],
    'seeds': [3047, 294, 714, 1092, 1386, 2856, 42, 114514, 2025, 1993],
    'num_runs': 1,
    
    # Testing parameters
    'max_test_steps': 100,
    
    # Output parameters
    'print_interval': 10,
    'enable_gantt_plots': False,
    'plot_training_curve': True,
    'save_models': True,
    'save_dir': 'checkpoints',
    
    # Add early stopping parameter
    'enable_early_stopping': False,  # Disable early stopping
    'early_stopping_patience': 10,  # Number of evaluations without improvement before stopping
}


class CPOPolicyNetwork(nn.Module):
    """Policy network for CPO with hierarchical action space"""
    
    def __init__(self, state_dim: int, num_jobs: int, num_machines: int, 
                 max_operations: int, hidden_dim: int = 256):
        super().__init__()
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.max_operations = max_operations
        
        # Shared feature extractor with batch normalization
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),  # Add dropout for regularization
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)  # Add dropout for regularization
        )
        
        # Job selection head
        self.job_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_jobs)
        )
        
        # Machine selection head (conditioned on job)
        self.machine_head = nn.Sequential(
            nn.Linear(hidden_dim + num_jobs, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_machines)
        )
        
        # Continuous parameter heads (mean and std)
        self.param_mean_head = nn.Sequential(
            nn.Linear(hidden_dim + num_jobs + num_machines, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, max_operations),
            nn.Tanh()  # Bound mean to [-1, 1]
        )
        
        self.param_std_head = nn.Sequential(
            nn.Linear(hidden_dim + num_jobs + num_machines, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, max_operations),
            nn.Softplus()  # Ensure positive std
        )
        
        # Add action validity prediction heads
        self.job_validity_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_jobs),
            nn.Sigmoid()  # Output probability of validity
        )
        
        self.machine_validity_head = nn.Sequential(
            nn.Linear(hidden_dim + num_jobs, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_machines),
            nn.Sigmoid()  # Output probability of validity
        )
        
        # Initialize weights with smaller values
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights with Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)  # Increased gain from 0.01
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, state: torch.Tensor, job_mask: Optional[torch.Tensor] = None,
                machine_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass with validity prediction
        """
        features = self.feature_extractor(state)
        
        # Job selection with validity prediction
        job_logits = self.job_head(features)
        job_validity = self.job_validity_head(features)
        
        # Apply soft masking using validity predictions
        if job_mask is not None:
            # Combine learned validity with actual mask
            job_validity = job_validity * job_mask
        
        # Modify logits based on validity
        job_logits = job_logits + torch.log(job_validity + 1e-8)
        
        # Get job probabilities for conditioning
        job_probs = F.softmax(job_logits, dim=-1)
        
        # Machine selection conditioned on job with validity prediction
        machine_input = torch.cat([features, job_probs], dim=-1)
        machine_logits = self.machine_head(machine_input)
        machine_validity = self.machine_validity_head(machine_input)
        
        # Apply soft masking using validity predictions
        if machine_mask is not None:
            machine_validity = machine_validity * machine_mask
        
        # Modify logits based on validity
        machine_logits = machine_logits + torch.log(machine_validity + 1e-8)
        
        # Continuous parameters conditioned on discrete actions
        machine_probs = F.softmax(machine_logits, dim=-1)
        param_input = torch.cat([features, job_probs, machine_probs], dim=-1)
        
        param_mean = self.param_mean_head(param_input)
        param_log_std = self.param_std_head(param_input)
        param_std = torch.exp(torch.clamp(param_log_std, -10, 2))
        
        return {
            'job_logits': job_logits,
            'machine_logits': machine_logits,
            'param_mean': param_mean,
            'param_std': param_std,
            'job_validity': job_validity,
            'machine_validity': machine_validity
        }


class CPOValueNetwork(nn.Module):
    """Value network for CPO"""
    
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class CPOAgent:
    """Constrained Policy Optimization agent for job shop scheduling"""
    
    def __init__(self, state_dim: int, num_jobs: int, num_machines: int,
                 max_operations: int, lr: float = 3e-4, gamma: float = 0.99,
                 lambda_gae: float = 0.95, delta: float = 0.01,
                 damping: float = 0.1, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.max_operations = max_operations
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.delta = delta
        self.damping = damping
        
        # Initialize networks
        self.policy = CPOPolicyNetwork(state_dim, num_jobs, num_machines, 
                                      max_operations).to(self.device)
        self.value = CPOValueNetwork(state_dim).to(self.device)
        self.old_policy = CPOPolicyNetwork(state_dim, num_jobs, num_machines,
                                          max_operations).to(self.device)
        
        # Initialize old policy with current policy
        self.old_policy.load_state_dict(self.policy.state_dict())
        
        # Optimizer for value function only
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=lr)
        
        # Add entropy coefficient
        self.entropy_coef = CONFIG['entropy_coef']
        
        # Add constraint threshold
        self.constraint_threshold = CONFIG['constraint_threshold']
        
        # Add constraint violation penalty weight from config
        self.violation_penalty = CONFIG['violation_penalty']
        
        # Add learning rate scheduler without verbose parameter
        self.value_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.value_optimizer, mode='min', factor=0.5, patience=10
        )
        
        # Initialize update counter
        self.update_count = 0

    def compute_constraint_violations(self, states: torch.Tensor, actions: Dict[str, torch.Tensor],
                                    job_masks: torch.Tensor, machine_masks: torch.Tensor,
                                    operation_bounds: torch.Tensor) -> torch.Tensor:
        """Compute constraint violations for the given state-action pairs"""
        batch_size = states.shape[0]
        violations = torch.zeros(batch_size, 3).to(self.device)  # 3 types of constraints
        
        # Constraint 1: Job completion constraint (selecting completed jobs)
        job_indices = actions['job_idx'].long()
        job_violations = torch.zeros(batch_size).to(self.device)
        for i in range(batch_size):
            if job_masks[i, job_indices[i]] == 0:  # Job is completed
                job_violations[i] = 1.0
        violations[:, 0] = job_violations
        
        # Constraint 2: Machine availability constraint (selecting busy machines)
        machine_indices = actions['machine_idx'].long()
        machine_violations = torch.zeros(batch_size).to(self.device)
        for i in range(batch_size):
            if machine_masks[i, machine_indices[i]] == 0:  # Machine is busy
                machine_violations[i] = 1.0
        violations[:, 1] = machine_violations
        
        # Constraint 3: Operation bounds constraint
        continuous_params = actions['continuous_param']
        lower_bounds = operation_bounds[..., 0]
        upper_bounds = operation_bounds[..., 1]
        
        # Check if parameters are outside bounds
        below_lower = (continuous_params < lower_bounds).float()
        above_upper = (continuous_params > upper_bounds).float()
        bound_violations = (below_lower + above_upper).sum(dim=-1) > 0
        violations[:, 2] = bound_violations.float()
        
        return violations
    
    def compute_constraint_cost(self, violations: torch.Tensor) -> torch.Tensor:
        """Compute total constraint cost from violations with configurable penalties"""
        # Use weights from config
        constraint_weights = torch.tensor(CONFIG['constraint_weights']).to(self.device)
        weighted_violations = violations * constraint_weights
        return weighted_violations.sum(dim=-1)
        
    def select_action(self, state: torch.Tensor, job_mask: torch.Tensor,
                     machine_mask: torch.Tensor, operation_bounds: torch.Tensor,
                     deterministic: bool = False) -> Dict[str, torch.Tensor]:
        """Select action with constraints and validity checking"""
        with torch.no_grad():
            # Get policy outputs with validity predictions
            policy_output = self.policy(state, job_mask, machine_mask)
            
            # Add exploration noise to logits during training
            if not deterministic:
                noise_scale = 0.1
                job_noise = torch.randn_like(policy_output['job_logits']) * noise_scale
                machine_noise = torch.randn_like(policy_output['machine_logits']) * noise_scale
                policy_output['job_logits'] = policy_output['job_logits'] + job_noise
                policy_output['machine_logits'] = policy_output['machine_logits'] + machine_noise
            
            # Maximum attempts to sample valid action
            max_attempts = 10
            
            for attempt in range(max_attempts):
                # Sample or select job (discrete)
                if deterministic:
                    job_idx = policy_output['job_logits'].argmax(dim=-1)
                else:
                    job_dist = Categorical(logits=policy_output['job_logits'])
                    job_idx = job_dist.sample()
                
                # Check job validity
                if job_mask is not None and job_mask[0, job_idx] == 0:
                    # Invalid job selected, modify logits to prevent reselection
                    policy_output['job_logits'][0, job_idx] = -1e9  # set invalid action logit to a very small value
                    continue
                
                # Sample or select machine (discrete)
                if deterministic:
                    machine_idx = policy_output['machine_logits'].argmax(dim=-1)
                else:
                    machine_dist = Categorical(logits=policy_output['machine_logits'])
                    machine_idx = machine_dist.sample()
                
                # Check machine validity
                if machine_mask is not None and machine_mask[0, machine_idx] == 0:
                    # Invalid machine selected, modify logits to prevent reselection
                    policy_output['machine_logits'][0, machine_idx] = -1e9
                    continue
                
                # If we get here, we have valid discrete actions
                break
            
            # Sample continuous parameters with operation-specific bounds
            param_mean = policy_output['param_mean']
            param_std = policy_output['param_std']
            
            # Increase exploration for continuous parameters
            if not deterministic:
                param_std = param_std * 1.5  # Increase std for more exploration
            
            if deterministic:
                continuous_param = param_mean
            else:
                param_dist = Normal(param_mean, param_std)
                continuous_param = param_dist.sample()
            
            # Apply operation-specific bounds (constraint 3)
            lower_bounds = operation_bounds[..., 0]
            upper_bounds = operation_bounds[..., 1]
            
            # Scale from standard normal to bounded range
            continuous_param = torch.sigmoid(continuous_param)  # Map to [0, 1]
            continuous_param = continuous_param * (upper_bounds - lower_bounds) + lower_bounds
            
            # Ensure parameters are within bounds
            continuous_param = torch.clamp(continuous_param, lower_bounds, upper_bounds)
            
            # Compute value
            value = self.value(state)
            
            return {
                'job_idx': job_idx,
                'machine_idx': machine_idx,
                'continuous_param': continuous_param,
                'value': value,
                'job_logits': policy_output['job_logits'],
                'machine_logits': policy_output['machine_logits'],
                'param_mean': param_mean,
                'param_std': param_std,
                'job_validity': policy_output['job_validity'],
                'machine_validity': policy_output['machine_validity']
            }
    
    def compute_advantages(self, rewards: torch.Tensor, values: torch.Tensor,
                          next_values: torch.Tensor, dones: torch.Tensor,
                          constraint_costs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute GAE advantages and returns with constraint costs"""
        # Compute advantages for rewards
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values[t]
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = delta + self.gamma * self.lambda_gae * (1 - dones[t]) * last_advantage
            last_advantage = advantages[t]
        
        returns = advantages + values
        
        # Compute advantages for constraint costs (negative rewards)
        cost_advantages = torch.zeros_like(constraint_costs)
        last_cost_advantage = 0
        
        for t in reversed(range(len(constraint_costs))):
            if t == len(constraint_costs) - 1:
                next_cost = 0
            else:
                next_cost = constraint_costs[t + 1]
            
            cost_delta = constraint_costs[t] + self.gamma * next_cost * (1 - dones[t])
            cost_advantages[t] = cost_delta + self.gamma * self.lambda_gae * (1 - dones[t]) * last_cost_advantage
            last_cost_advantage = cost_advantages[t]
        
        return returns, advantages, cost_advantages
    
    def compute_kl_divergence(self, old_dist_info: Dict, new_dist_info: Dict) -> torch.Tensor:
        """Compute KL divergence between old and new policies"""
        # KL for discrete distributions (jobs and machines)
        old_job_probs = F.softmax(old_dist_info['job_logits'], dim=-1)
        new_job_probs = F.softmax(new_dist_info['job_logits'], dim=-1)
        kl_job = (old_job_probs * (old_job_probs.log() - new_job_probs.log())).sum(dim=-1)
        
        old_machine_probs = F.softmax(old_dist_info['machine_logits'], dim=-1)
        new_machine_probs = F.softmax(new_dist_info['machine_logits'], dim=-1)
        kl_machine = (old_machine_probs * (old_machine_probs.log() - new_machine_probs.log())).sum(dim=-1)
        
        # KL for continuous distributions (parameters)
        old_mean = old_dist_info['param_mean']
        old_std = old_dist_info['param_std']
        new_mean = new_dist_info['param_mean']
        new_std = new_dist_info['param_std']
        
        kl_param = (0.5 * (
            (old_std / new_std).pow(2) + 
            ((new_mean - old_mean) / new_std).pow(2) - 
            1 + 
            2 * (new_std.log() - old_std.log())
        )).sum(dim=-1)
        
        return (kl_job + kl_machine + kl_param).mean()
    
    def conjugate_gradient(self, b: torch.Tensor, nsteps: int = 10, residual_tol: float = 1e-10):
        """Conjugate gradient solver for CPO"""
        x = torch.zeros_like(b)
        r = b.clone()
        p = b.clone()
        rdotr = r.dot(r)
        
        for _ in range(nsteps):
            if rdotr < residual_tol:
                break
            
            # Compute Fisher-vector product
            Ap = self.fisher_vector_product(p)
            alpha = rdotr / p.dot(Ap)
            x += alpha * p
            r -= alpha * Ap
            new_rdotr = r.dot(r)
            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr
            
        return x
    
    def fisher_vector_product(self, v: torch.Tensor, damping: float = 0.1) -> torch.Tensor:
        """Compute Fisher-vector product for natural gradient"""
        # This is a simplified version - in practice, you'd compute the actual FVP
        return v + damping * v
    
    def compute_policy_entropy(self, policy_output: Dict[str, torch.Tensor], 
                             actions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute entropy of the policy for regularization"""
        # Entropy for discrete distributions
        job_probs = F.softmax(policy_output['job_logits'], dim=-1)
        job_entropy = -(job_probs * job_probs.log()).sum(dim=-1)
        
        machine_probs = F.softmax(policy_output['machine_logits'], dim=-1)
        machine_entropy = -(machine_probs * machine_probs.log()).sum(dim=-1)
        
        # Entropy for continuous distribution
        param_entropy = 0.5 * torch.log(2 * np.pi * np.e * policy_output['param_std'].pow(2)).sum(dim=-1)
        
        return (job_entropy + machine_entropy + param_entropy).mean()
    
    def line_search(self, states: torch.Tensor, actions: Dict[str, torch.Tensor],
                    advantages: torch.Tensor, cost_advantages: torch.Tensor,
                    job_masks: torch.Tensor, machine_masks: torch.Tensor, 
                    operation_bounds: torch.Tensor,
                    old_policy_output: Dict[str, torch.Tensor],
                    old_log_probs: Dict[str, torch.Tensor],
                    max_steps: int = 10) -> bool:
        """Perform line search with constraint satisfaction"""
        # Save initial policy parameters
        initial_params = [param.clone().detach() for param in self.policy.parameters()]
        
        # Clear any existing gradients
        self.policy.zero_grad()
        
        # Compute gradient direction for objective
        new_policy_output = self.policy(states, job_masks, machine_masks)
        
        # Compute policy loss with entropy bonus
        job_dist = Categorical(logits=new_policy_output['job_logits'])
        machine_dist = Categorical(logits=new_policy_output['machine_logits'])
        
        # Clamp std to avoid numerical issues
        param_std = torch.clamp(new_policy_output['param_std'], min=1e-6)
        param_dist = Normal(new_policy_output['param_mean'], param_std)
        
        new_job_log_prob = job_dist.log_prob(actions['job_idx'])
        new_machine_log_prob = machine_dist.log_prob(actions['machine_idx'])
        new_param_log_prob = param_dist.log_prob(actions['continuous_param']).sum(dim=-1)
        
        new_log_prob = new_job_log_prob + new_machine_log_prob + new_param_log_prob
        ratio = torch.exp(new_log_prob - old_log_probs['total'])
        
        # Clip ratio to avoid numerical issues
        ratio = torch.clamp(ratio, min=1e-8, max=1e8)
        
        # Add entropy bonus
        entropy = self.compute_policy_entropy(new_policy_output, actions)
        policy_loss = -(ratio * advantages).mean() - self.entropy_coef * entropy
        
        # Compute gradients
        policy_loss.backward(retain_graph=True)
        
        # Store objective gradients
        objective_grads = []
        for param in self.policy.parameters():
            if param.grad is not None:
                grad = param.grad.clone()
                objective_grads.append(grad)
            else:
                objective_grads.append(torch.zeros_like(param))
        
        # Clear gradients
        self.policy.zero_grad()
        
        # Compute gradient direction for constraints
        constraint_loss = (ratio * cost_advantages).mean()
        constraint_loss.backward()
        
        # Store constraint gradients
        constraint_grads = []
        for param in self.policy.parameters():
            if param.grad is not None:
                grad = param.grad.clone()
                constraint_grads.append(grad)
            else:
                constraint_grads.append(torch.zeros_like(param))
        
        # Clear gradients
        self.policy.zero_grad()
        
        # Compute natural gradient direction using conjugate gradient
        # This is a simplified version - full CPO would solve a constrained optimization problem
        step_direction = []
        for obj_grad, con_grad in zip(objective_grads, constraint_grads):
            # Project objective gradient to satisfy constraint
            direction = obj_grad - 0.1 * con_grad  # Simple projection
            direction = torch.clamp(direction, min=-CONFIG['clip_grad_norm'], max=CONFIG['clip_grad_norm'])
            step_direction.append(direction)
        
        # Line search with constraint checking
        step_sizes = [1.0, 0.5, 0.25, 0.125, 0.0625]  # Predefined step sizes
        
        for step_size in step_sizes:
            # Update parameters with current step size
            with torch.no_grad():
                for param, init_param, direction in zip(self.policy.parameters(), initial_params, step_direction):
                    param.data = init_param - step_size * direction
            
            try:
                # Check KL divergence
                with torch.no_grad():
                    new_policy_output = self.policy(states, job_masks, machine_masks)
                    
                    # Check for NaN in outputs
                    for key, value in new_policy_output.items():
                        if torch.isnan(value).any():
                            raise ValueError(f"NaN detected in {key}")
                    
                    kl = self.compute_kl_divergence(old_policy_output, new_policy_output)
                    
                    if torch.isnan(kl):
                        raise ValueError("NaN detected in KL divergence")
                    
                    # Check expected constraint cost
                    violations = self.compute_constraint_violations(
                        states, actions, job_masks, machine_masks, operation_bounds
                    )
                    expected_cost = self.compute_constraint_cost(violations).mean()
                
                # More lenient acceptance criteria
                if kl < self.delta * 1.5 and expected_cost <= self.constraint_threshold + 0.2:
                    return True
                    
            except (ValueError, RuntimeError) as e:
                # If we get NaN or other numerical issues, try next step size
                continue
        
        # Restore original parameters if line search failed
        with torch.no_grad():
            for param, init_param in zip(self.policy.parameters(), initial_params):
                param.data = init_param.data
        
        return False
    
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Update policy using CPO with enhanced constraint handling"""
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones']
        job_masks = batch['job_masks']
        machine_masks = batch['machine_masks']
        operation_bounds = batch['operation_bounds']
        old_log_probs = batch['old_log_probs']
        constraint_violations = batch['constraint_violations']
        
        # Ensure rewards and dones are 1D tensors
        rewards = rewards.squeeze()
        dones = dones.squeeze()
        
        # Apply penalty to rewards for constraint violations (with decay)
        violation_penalty = self.compute_constraint_cost(constraint_violations)
        penalty_scale = self.violation_penalty * (0.95 ** (self.update_count / 100))  # Decay penalty over time
        rewards = rewards - penalty_scale * violation_penalty
        
        # Normalize rewards for stability
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        # Compute constraint costs
        constraint_costs = self.compute_constraint_cost(constraint_violations)
        
        # Compute advantages
        with torch.no_grad():
            values = self.value(states).squeeze()
            next_values = self.value(next_states).squeeze()
            returns, advantages, cost_advantages = self.compute_advantages(
                rewards, values, next_values, dones, constraint_costs
            )
            
            # Normalize advantages if enabled
            if CONFIG['normalize_advantages']:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                cost_advantages = (cost_advantages - cost_advantages.mean()) / (cost_advantages.std() + 1e-8)
        
        # Update value function with early stopping
        best_value_loss = float('inf')
        patience = 3
        no_improve = 0
        
        for epoch in range(CONFIG['value_epochs']):
            value_pred = self.value(states).squeeze()
            value_loss = F.mse_loss(value_pred, returns)
            
            self.value_optimizer.zero_grad()
            value_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.value.parameters(), CONFIG['clip_grad_norm'])
            
            self.value_optimizer.step()
            
            # Early stopping
            if value_loss.item() < best_value_loss:
                best_value_loss = value_loss.item()
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break
        
        # Update learning rate scheduler with detached value loss
        self.value_scheduler.step(value_loss.detach())
        
        # Get old policy output
        with torch.no_grad():
            old_policy_output = self.old_policy(states, job_masks, machine_masks)
        
        # Clear any existing gradients
        self.policy.zero_grad()
        
        # Perform line search for policy update with constraints
        line_search_success = self.line_search(
            states, actions, advantages, cost_advantages, job_masks, machine_masks, 
            operation_bounds, old_policy_output, old_log_probs, 
            max_steps=CONFIG['line_search_steps']
        )
        
        # Clear gradients after line search
        self.policy.zero_grad()
        
        # Compute final KL and constraint cost for logging
        with torch.no_grad():
            new_policy_output = self.policy(states, job_masks, machine_masks)
            kl = self.compute_kl_divergence(old_policy_output, new_policy_output)
            avg_constraint_cost = constraint_costs.mean()
        
        # Update old policy only if line search succeeded
        if line_search_success:
            self.old_policy.load_state_dict(self.policy.state_dict())
        
        # Compute policy gradient for logging
        with torch.no_grad():
            job_dist = Categorical(logits=new_policy_output['job_logits'])
            machine_dist = Categorical(logits=new_policy_output['machine_logits'])
            
            param_std = torch.clamp(new_policy_output['param_std'], min=1e-6)
            param_dist = Normal(new_policy_output['param_mean'], param_std)
            
            new_job_log_prob = job_dist.log_prob(actions['job_idx'])
            new_machine_log_prob = machine_dist.log_prob(actions['machine_idx'])
            new_param_log_prob = param_dist.log_prob(actions['continuous_param']).sum(dim=-1)
            new_log_prob = new_job_log_prob + new_machine_log_prob + new_param_log_prob
            
            ratio = torch.exp(new_log_prob - old_log_probs['total'])
            ratio = torch.clamp(ratio, min=1e-8, max=1e8)
            policy_gradient = (ratio * advantages).mean()
        
        # Add validity loss to encourage learning valid actions
        with torch.no_grad():
            policy_output = self.policy(states, job_masks, machine_masks)
            
            # Compute validity loss
            job_validity_target = job_masks
            machine_validity_target = machine_masks
            
            job_validity_loss = F.binary_cross_entropy(
                policy_output['job_validity'], job_validity_target
            )
            machine_validity_loss = F.binary_cross_entropy(
                policy_output['machine_validity'], machine_validity_target
            )
            
            validity_loss = job_validity_loss + machine_validity_loss
        
        return {
            'value_loss': value_loss.item(),
            'policy_gradient': policy_gradient.item(),
            'kl_divergence': kl.item() if not torch.isnan(kl) else 0.0,
            'line_search_success': float(line_search_success),
            'avg_constraint_cost': avg_constraint_cost.item(),
            'validity_loss': validity_loss.item()
        }
    
    def save(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'value_state_dict': self.value.state_dict(),
            'old_policy_state_dict': self.old_policy.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict()
        }, path)
    
    def load(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.value.load_state_dict(checkpoint['value_state_dict'])
        self.old_policy.load_state_dict(checkpoint['old_policy_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])


class CPOBuffer:
    """Experience buffer for CPO"""
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
    def push(self, experience: Dict[str, torch.Tensor]):
        """Add experience to buffer"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.capacity
        
    def get_batch(self) -> Dict[str, torch.Tensor]:
        """Get all experiences as a batch"""
        batch = {}
        
        for key in self.buffer[0].keys():
            if isinstance(self.buffer[0][key], dict):
                batch[key] = {}
                for sub_key in self.buffer[0][key].keys():
                    batch[key][sub_key] = torch.stack([self.buffer[i][key][sub_key] for i in range(len(self.buffer))])
            else:
                batch[key] = torch.stack([self.buffer[i][key] for i in range(len(self.buffer))])
                
        return batch
    
    def clear(self):
        """Clear the buffer"""
        self.buffer.clear()
        self.position = 0
        
    def __len__(self):
        return len(self.buffer)


# Reuse utility functions
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


def collect_experience_cpo(env, agent: CPOAgent, num_steps: int, 
                          buffer: CPOBuffer, deterministic: bool = False,
                          max_episode_steps: int = None):
    """Collect experience for CPO training with enhanced constraint tracking"""
    state = env.reset()
    episode_reward = 0
    step_counter = 0
    total_violations = 0
    
    # Use config max steps if not specified
    if max_episode_steps is None:
        max_episode_steps = CONFIG['max_steps_per_episode']
    
    # Add reward shaping
    previous_makespan = float('inf')
    
    for step in range(num_steps):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
        
        # Get constraint masks
        job_mask = torch.FloatTensor(get_job_mask(env)).unsqueeze(0).to(agent.device)
        machine_mask = torch.FloatTensor(get_machine_mask(env)).unsqueeze(0).to(agent.device)
        operation_bounds = torch.FloatTensor(get_operation_bounds(env)).unsqueeze(0).to(agent.device)
        
        # Select action with validity checking
        action_dict = agent.select_action(
            state_tensor, job_mask, machine_mask, operation_bounds, deterministic
        )
        
        job_id = action_dict['job_idx'].item()
        machine_id = action_dict['machine_idx'].item()
        param_value = float(action_dict['continuous_param'].squeeze(0).cpu().numpy()[0])
        
        # Double-check action validity before stepping
        if job_mask[0, job_id] == 0 or machine_mask[0, machine_id] == 0:
            # Force select a valid action
            valid_jobs = torch.where(job_mask[0] > 0)[0]
            valid_machines = torch.where(machine_mask[0] > 0)[0]
            
            if len(valid_jobs) > 0 and len(valid_machines) > 0:
                job_id = valid_jobs[0].item()
                machine_id = valid_machines[0].item()
                action_dict['job_idx'] = torch.tensor([job_id], device=agent.device)
                action_dict['machine_idx'] = torch.tensor([machine_id], device=agent.device)
            else:
                # No valid actions available, episode should end
                print(f"Warning: No valid actions available at step {step_counter}")
                break
        
        action = [job_id, machine_id, param_value]
        
        # Compute constraint violations for this action
        with torch.no_grad():
            violations = agent.compute_constraint_violations(
                state_tensor, 
                {
                    'job_idx': action_dict['job_idx'],
                    'machine_idx': action_dict['machine_idx'],
                    'continuous_param': action_dict['continuous_param']
                },
                job_mask, machine_mask, operation_bounds
            )
            total_violations += violations.sum().item()
        
        # Step environment
        step_result = env.step(action)
        step_counter += 1
        
        if len(step_result) == 3:
            next_state, reward, done = step_result
        else:
            next_state, reward, done, _ = step_result
        
        # Add reward shaping based on makespan improvement
        current_makespan = env.current_time
        if current_makespan < previous_makespan:
            reward += 0.1  # Small bonus for improving makespan
        previous_makespan = current_makespan
        
        # Add small step penalty to encourage efficiency
        reward -= 0.01
        
        episode_reward += reward
        
        # Check if maximum steps reached
        max_steps_reached = step_counter >= max_episode_steps
        
        # Calculate log probabilities for CPO update
        with torch.no_grad():
            job_dist = Categorical(logits=action_dict['job_logits'])
            machine_dist = Categorical(logits=action_dict['machine_logits'])
            param_dist = Normal(action_dict['param_mean'], action_dict['param_std'])
            
            job_log_prob = job_dist.log_prob(action_dict['job_idx'])
            machine_log_prob = machine_dist.log_prob(action_dict['machine_idx'])
            param_log_prob = param_dist.log_prob(action_dict['continuous_param']).sum(dim=-1)
            
            total_log_prob = job_log_prob + machine_log_prob + param_log_prob
        
        # Store experience with modified done flag
        effective_done = done or max_steps_reached
        
        experience = {
            'states': state_tensor.squeeze(0),
            'actions': {
                'job_idx': action_dict['job_idx'].squeeze(0),
                'machine_idx': action_dict['machine_idx'].squeeze(0),
                'continuous_param': action_dict['continuous_param'].squeeze(0)
            },
            'rewards': torch.FloatTensor([reward]).squeeze().to(agent.device),
            'next_states': torch.FloatTensor(next_state).to(agent.device),
            'dones': torch.FloatTensor([effective_done]).squeeze().to(agent.device),
            'job_masks': job_mask.squeeze(0),
            'machine_masks': machine_mask.squeeze(0),
            'operation_bounds': operation_bounds.squeeze(0),
            'old_log_probs': {
                'job': job_log_prob.squeeze(0),
                'machine': machine_log_prob.squeeze(0),
                'param': param_log_prob.squeeze(0),
                'total': total_log_prob.squeeze(0)
            },
            'constraint_violations': violations.squeeze(0)
        }
        buffer.push(experience)
        
        # End episode if done or max steps reached
        if done or max_steps_reached:
            return episode_reward, step_counter, total_violations, max_steps_reached
        
        state = next_state
    
    # Reached num_steps limit (for data collection)
    return episode_reward, step_counter, total_violations, False


def train_single_run_cpo(config=None):
    """Execute a single training run with CPO"""
    if config is None:
        config = CONFIG
    
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
    agent = CPOAgent(
        state_dim=state_dim,
        num_jobs=num_jobs,
        num_machines=num_machines,
        max_operations=max_operations,
        lr=config['lr'],
        gamma=config['gamma'],
        lambda_gae=config['lambda_gae'],
        delta=config['delta'],
        damping=config['damping'],
        device=device
    )
    
    # Initialize buffer
    buffer = CPOBuffer(capacity=config['buffer_limit'])
    
    # Training metrics
    episode_rewards = []
    episode_steps = []
    episode_violations = []
    episode_truncated = []  # Track if episode was truncated by max steps
    evaluation_rewards = []
    kl_divergences = []
    constraint_costs = []
    
    print(f"Starting CPO training:")
    print(f"Jobs: {num_jobs}, Machines: {num_machines}")
    print(f"Alpha: {config['alpha']}, Beta: {config['beta']}, Episodes: {config['num_episodes']}")
    print(f"Max steps per episode: {config['max_steps_per_episode']}")
    print(f"Device: {device}")
    print(f"Using enhanced CPO with adaptive penalties")
    
    best_reward = -float('inf')
    total_steps = 0
    
    # Add early stopping
    no_improvement_count = 0
    best_eval_reward = -float('inf')
    
    for episode in range(config['num_episodes']):
        # Collect experience with max step limit
        episode_reward, steps, violations, truncated = collect_experience_cpo(
            env, agent, config['steps_per_episode'], buffer, 
            deterministic=False, max_episode_steps=config['max_steps_per_episode']
        )
        
        episode_rewards.append(episode_reward)
        episode_steps.append(steps)
        episode_violations.append(violations)
        episode_truncated.append(truncated)
        total_steps += steps
        
        # Update best reward
        if episode_reward > best_reward:
            best_reward = episode_reward
        
        # Update policy when buffer has enough samples
        if len(buffer) >= config['update_interval']:
            batch = buffer.get_batch()
            losses = agent.update(batch)
            kl_divergences.append(losses['kl_divergence'])
            constraint_costs.append(losses['avg_constraint_cost'])
            buffer.clear()
            
            # Log update info
            if episode % config['print_interval'] == 0:
                print(f"  Update: KL={losses['kl_divergence']:.4f}, "
                      f"VLoss={losses['value_loss']:.4f}, "
                      f"PGrad={losses['policy_gradient']:.4f}, "
                      f"ConstraintCost={losses['avg_constraint_cost']:.4f}, "
                      f"LineSearch={'Success' if losses['line_search_success'] else 'Failed'}")
        
        # Print progress
        if episode % config['print_interval'] == 0:
            window_size = min(20, len(episode_rewards))
            if len(episode_rewards) >= window_size:
                moving_avg = np.mean(episode_rewards[-window_size:])
                moving_std = np.std(episode_rewards[-window_size:])
                avg_steps = np.mean(episode_steps[-window_size:])
                avg_violations = np.mean(episode_violations[-window_size:])
                truncation_rate = np.mean(episode_truncated[-window_size:]) * 100
                
                status = " (truncated)" if truncated else " (completed)"
                print(f"Episode {episode + 1}/{config['num_episodes']}: "
                      f"R={episode_reward:.2f}, Steps={steps}{status}, "
                      f"Violations={violations}, "
                      f"MA={moving_avg:.2f}±{moving_std:.2f}, "
                      f"AvgViol={avg_violations:.2f}, "
                      f"TruncRate={truncation_rate:.1f}%, "
                      f"Best={best_reward:.2f}")
            else:
                status = " (truncated)" if truncated else " (completed)"
                print(f"Episode {episode + 1}/{config['num_episodes']}: "
                      f"R={episode_reward:.2f}, Steps={steps}{status}, "
                      f"Violations={violations}")

        # Evaluate
        if (episode + 1) % config['eval_interval'] == 0:
            eval_rewards = []
            eval_steps = []
            eval_truncated = []
            
            print(f"\nEvaluation at Episode {episode + 1}:")
            for eval_idx in range(5):
                eval_buffer = CPOBuffer(1)
                eval_reward, eval_step, _, eval_trunc = collect_experience_cpo(
                    env, agent, config['steps_per_episode'], 
                    eval_buffer, deterministic=True,
                    max_episode_steps=config['max_steps_per_episode']
                )
                eval_rewards.append(eval_reward)
                eval_steps.append(eval_step)
                eval_truncated.append(eval_trunc)
                
                status = " (truncated)" if eval_trunc else " (completed)"
                print(f"  Eval {eval_idx+1}: R={eval_reward:.2f}, Steps={eval_step}{status}")
            
            eval_mean = np.mean(eval_rewards)
            eval_std = np.std(eval_rewards)
            eval_truncation_rate = np.mean(eval_truncated) * 100
            evaluation_rewards.append((episode + 1, eval_mean, eval_std))
            
            print(f"Evaluation Summary: {eval_mean:.4f} ± {eval_std:.4f}, "
                  f"Truncation rate: {eval_truncation_rate:.1f}%")
            
            # Early stopping check (only if enabled)
            if config.get('enable_early_stopping', True):  # Default to True for backward compatibility
                if eval_mean > best_eval_reward:
                    best_eval_reward = eval_mean
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                    
                if no_improvement_count >= 20:  # Increased from 10 to 20
                    print(f"Early stopping triggered at episode {episode + 1}")
                    break
            
            # Save best model
            if eval_mean > best_reward and config['save_models']:
                best_model_path = os.path.join(config['save_dir'], 'best_cpo_model.pt')
                os.makedirs(config['save_dir'], exist_ok=True)
                agent.save(best_model_path)
                print(f"Saved best model with eval reward: {eval_mean:.4f}\n")
        
        # Save checkpoint
        if config['save_models'] and (episode + 1) % config['save_interval'] == 0:
            checkpoint_path = os.path.join(config['save_dir'], f'cpo_checkpoint_ep{episode+1}.pt')
            os.makedirs(config['save_dir'], exist_ok=True)
            agent.save(checkpoint_path)
            print(f"Checkpoint saved to: {checkpoint_path}")
    
    print(f"\nTraining Complete!")
    print(f"Final 100 episodes average: {np.mean(episode_rewards[-100:]):.4f}")
    print(f"Best episode reward: {best_reward:.4f}")
    print(f"Average truncation rate: {np.mean(episode_truncated) * 100:.1f}%")
    
    return episode_rewards, [], agent


def multi_run_training_cpo(config=None):
    """Execute multiple training runs with different seeds for CPO"""
    if config is None:
        config = CONFIG
    
    all_score_records = []
    all_models = []
    
    print(f"\n{'='*80}")
    print(f"Starting Multi-Run CPO Training")
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
        score_record, _, model = train_single_run_cpo(config)
        
        # Store results
        all_score_records.append(score_record)
        all_models.append(model)
        
        print(f"Run {run_idx + 1} completed - Final Score: {np.mean(score_record[-10:]):.4f}")
    
    print(f"\n{'='*60}")
    print(f"All {config['num_runs']} runs completed!")
    print(f"{'='*60}")
    
    return all_score_records, [], all_models


# Reuse evaluation and visualization functions with minor modifications
def evaluate_agent(env, agent: CPOAgent, num_episodes: int = 10, max_steps: int = None):
    """Evaluate agent performance with step limit"""
    if max_steps is None:
        max_steps = CONFIG['max_steps_per_episode']
        
    total_rewards = []
    step_counts = []
    constraint_violations = []
    truncated_episodes = 0
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        step_counter = 0
        violations = 0
        
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
            
            # Check for constraint violations
            if job_mask[0, job_id] == 0:
                violations += 1  # Selected completed job
            if machine_mask[0, machine_id] == 0:
                violations += 1  # Selected busy machine
            
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
        constraint_violations.append(violations)
        print(f"Evaluation episode {episode+1} finished in {step_counter} steps "
              f"with reward {episode_reward:.4f} and {violations} constraint violations{status}")
    
    print(f"\nEvaluation Summary:")
    print(f"  Average reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"  Average steps: {np.mean(step_counts):.2f} ± {np.std(step_counts):.2f}")
    print(f"  Average violations: {np.mean(constraint_violations):.2f} ± {np.std(constraint_violations):.2f}")
    print(f"  Truncated episodes: {truncated_episodes}/{num_episodes} ({truncated_episodes/num_episodes*100:.1f}%)")
    
    return np.mean(total_rewards), np.std(total_rewards)


def test_and_visualize(config=None, model_path=None):
    """Test trained model and visualize scheduling process"""
    if config is None:
        config = CONFIG
        
    print("\n=== Starting CPO Testing and Visualization ===")
    
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
    agent = CPOAgent(
        state_dim=state_dim,
        num_jobs=num_jobs,
        num_machines=num_machines,
        max_operations=max_operations,
        lr=config['lr'],
        gamma=config['gamma'],
        lambda_gae=config['lambda_gae'],
        delta=config['delta'],
        damping=config['damping'],
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
    constraint_violations = 0
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
        
        # Check constraints
        if job_mask[0, job_id] == 0:
            print(f"Warning: Selected completed job {job_id}")
            constraint_violations += 1
        if machine_mask[0, machine_id] == 0:
            print(f"Warning: Selected busy machine {machine_id}")
            constraint_violations += 1
        
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
    print(f"Constraint violations: {constraint_violations}")
    plt.show()


# Reuse plotting and saving functions
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
    
    plt.title('CPO Multi-Run Training Curves')
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
    alg_name = "cpo"
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
    """Main function for CPO policy"""
    import argparse
    
    parser = argparse.ArgumentParser(description='CPO Policy for Flexible Job Shop Scheduling with Constraints')
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
    print(f"CPO Policy for Flexible Job Shop Scheduling")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  Jobs: {config['num_jobs']}")
    print(f"  Machines: {config['num_machines']}")
    print(f"  Alpha: {config['alpha']}")
    print(f"  Beta: {config['beta']}")
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
        all_score_records, all_action_restores, all_models = multi_run_training_cpo(config)
        
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
        
        score_record, action_restore, model = train_single_run_cpo(config)
        
        if config['save_models']:
            save_dir, model_path = save_multi_run_results([score_record], [action_restore], [model], config)
        
        if config['plot_training_curve']:
            plt.figure(figsize=(10, 6))
            plt.plot(score_record)
            plt.title('CPO Training Curve')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.grid(True)
            plt.show()
        
        if config['save_models'] and model_path:
            print(f"\nTesting with trained model...")
            test_and_visualize(config, model_path[0])


if __name__ == '__main__':
    main()
