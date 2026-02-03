import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical, Normal
from typing import Dict, Tuple, Optional
import os
import sys
import random
import datetime
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__), '../env'))
import env as ENV

# =======================
# HYPERPARAMETERS CONFIG
# =======================
CONFIG = {
    # Learning parameters - redesigned for more stable configuration
    'lr': 0.035,  # Standard learning rate 0.05, 0.045
    'gamma': 0.99,  # Standard discount factor
    'eps_clip': 0.2,  # Standard PPO clip
    'value_coef': 0.5,  # Standard value coefficient
    'entropy_coef': 0.01,  # Standard entropy coefficient
    'batch_size': 32,  # Standard batch size
    'buffer_limit': 10000,  # Standard buffer
    'num_epochs': 10,  # Standard PPO epochs
    'gae_lambda': 0.95,  # Standard GAE
    
    # Environment parameters
    'num_jobs': 5,
    'num_machines': 5,
    'max_operations': 5,
    'alpha': 0.5,
    'beta': 0.5,
    'num_episodes': 5,
    'steps_per_episode': 30,  # Increased max steps
    
    # Training parameters
    'update_interval': 5,  # Update every episode
    'eval_interval': 50,
    'save_interval': 100,
    'memory_threshold': 32,  # Lower threshold to start training
    'target_update_freq': 10,  # Add target network update frequency
    
    # Reward shaping parameters - simplified reward design
    'use_reward_shaping': False,  # Disable complex reward shaping
    'reward_normalization': False,  # Add reward normalization
    'return_normalization': True,  # Add return normalization
    
    # Multi-run training parameters
    'enable_multi_run': True,
    # 'seeds': [3047],
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

class HighLevelPolicy(nn.Module):
    """High-level policy network for discrete action selection (job and machine)"""
    
    def __init__(self, state_dim: int, num_jobs: int, num_machines: int, hidden_dim: int = 256):
        super().__init__()
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Job selection head
        self.job_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_jobs)
        )
        
        # Machine selection head
        self.machine_head = nn.Sequential(
            nn.Linear(hidden_dim + num_jobs, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_machines)
        )
        
    def forward(self, state: torch.Tensor, job_mask: Optional[torch.Tensor] = None, 
                machine_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            state: Current state tensor
            job_mask: Binary mask for valid jobs (1 for valid, 0 for invalid)
            machine_mask: Binary mask for valid machines
        Returns:
            job_logits: Logits for job selection
            machine_logits: Logits for machine selection
        """
        features = self.feature_extractor(state)
        
        # Job selection with masking
        job_logits = self.job_head(features)
        if job_mask is not None:
            job_logits = job_logits.masked_fill(~job_mask.bool(), -1e9)
        
        # Machine selection conditioned on job selection
        job_probs = F.softmax(job_logits, dim=-1)
        machine_input = torch.cat([features, job_probs], dim=-1)
        machine_logits = self.machine_head(machine_input)
        if machine_mask is not None:
            machine_logits = machine_logits.masked_fill(~machine_mask.bool(), -1e9)
        
        return job_logits, machine_logits


class LowLevelPolicy(nn.Module):
    """Low-level policy network for continuous parameter selection"""
    
    def __init__(self, state_dim: int, num_jobs: int, num_machines: int, 
                 max_operations: int, hidden_dim: int = 256):
        super().__init__()
        self.max_operations = max_operations
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        
        # Input includes state + selected job and machine (one-hot)
        input_dim = state_dim + num_jobs + num_machines
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Output mean and log_std for each possible operation
        self.mean_head = nn.Linear(hidden_dim, max_operations)
        self.log_std_head = nn.Linear(hidden_dim, max_operations)
        
    def forward(self, state: torch.Tensor, job_idx: torch.Tensor, machine_idx: torch.Tensor,
                operation_bounds: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            state: Current state tensor
            job_idx: Selected job index
            machine_idx: Selected machine index
            operation_bounds: Bounds for continuous parameters [batch, max_operations, 2]
        Returns:
            mean: Mean of continuous parameter distribution
            std: Standard deviation of continuous parameter distribution
        """
        batch_size = state.shape[0]
        
        # Create one-hot encodings
        job_one_hot = F.one_hot(job_idx, num_classes=self.num_jobs)
        machine_one_hot = F.one_hot(machine_idx, num_classes=self.num_machines)
        
        # Concatenate inputs
        x = torch.cat([state, job_one_hot.float(), machine_one_hot.float()], dim=-1)
        
        features = self.network(x)
        raw_mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        std = torch.exp(torch.clamp(log_std, -10, 2))
        
        # Scale mean to bounds
        lower_bounds = operation_bounds[..., 0]
        upper_bounds = operation_bounds[..., 1]
        mean = torch.sigmoid(raw_mean) * (upper_bounds - lower_bounds) + lower_bounds
        
        return mean, std


class ValueNetwork(nn.Module):
    """Shared value network for advantage estimation"""
    
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


class HPPOAgent:
    """Hybrid PPO agent for flexible job shop scheduling"""
    
    def __init__(self, state_dim: int, num_jobs: int, num_machines: int, 
                 max_operations: int, lr: float = 3e-4, gamma: float = 0.99,
                 eps_clip: float = 0.2, value_coef: float = 0.5, 
                 entropy_coef: float = 0.01, device: str = 'cuda',
                 gae_lambda: float = 0.95):  # add GAE lambda
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.gae_lambda = gae_lambda  # add GAE lambda
        
        # Initialize networks
        self.high_policy = HighLevelPolicy(state_dim, num_jobs, num_machines).to(self.device)
        self.low_policy = LowLevelPolicy(state_dim, num_jobs, num_machines, max_operations).to(self.device)
        self.value_net = ValueNetwork(state_dim).to(self.device)
        
    # Initialize weights to reduce initial output variance
        self._init_weights()
        
        # Optimizers with weight decay
        self.optimizer = torch.optim.Adam([
            {'params': self.high_policy.parameters()},
            {'params': self.low_policy.parameters()},
            {'params': self.value_net.parameters()}
        ], lr=lr, eps=1e-5, weight_decay=1e-4)  # add weight decay
        
    # Use cosine annealing learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )
    
    def _init_weights(self):
        """Initialize network weights with smaller values"""
        for module in [self.high_policy, self.low_policy, self.value_net]:
            for param in module.parameters():
                if len(param.shape) >= 2:
                    nn.init.orthogonal_(param, gain=0.01)
                else:
                    nn.init.zeros_(param)
    
    def compute_gae(self, rewards, values, next_values, dones):
        """Compute Generalized Advantage Estimation"""
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values[t]
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_advantage
            last_advantage = advantages[t]
        
        returns = advantages + values
        return returns, advantages
    
    def select_action(self, state: torch.Tensor, job_mask: torch.Tensor, 
                     machine_mask: torch.Tensor, operation_bounds: torch.Tensor,
                     deterministic: bool = False) -> Dict[str, torch.Tensor]:
        """Select hierarchical action"""
        with torch.no_grad():
            # High-level discrete actions
            job_logits, machine_logits = self.high_policy(state, job_mask, machine_mask)
            
            if deterministic:
                job_idx = job_logits.argmax(dim=-1)
                machine_idx = machine_logits.argmax(dim=-1)
            else:
                job_dist = Categorical(logits=job_logits)
                machine_dist = Categorical(logits=machine_logits)
                job_idx = job_dist.sample()
                machine_idx = machine_dist.sample()
            
            # Low-level continuous parameters
            mean, std = self.low_policy(state, job_idx, machine_idx, operation_bounds)
            
            if deterministic:
                continuous_param = mean
            else:
                param_dist = Normal(mean, std)
                continuous_param = param_dist.sample()
                continuous_param = torch.clamp(continuous_param, 
                                             operation_bounds[..., 0], 
                                             operation_bounds[..., 1])
            
            # Value estimation
            value = self.value_net(state)
            
            return {
                'job_idx': job_idx,
                'machine_idx': machine_idx,
                'continuous_param': continuous_param,
                'value': value,
                'job_logits': job_logits,
                'machine_logits': machine_logits,
                'param_mean': mean,
                'param_std': std
            }
    
    def compute_loss(self, states: torch.Tensor, actions: Dict[str, torch.Tensor],
                    rewards: torch.Tensor, next_states: torch.Tensor, 
                    dones: torch.Tensor, job_masks: torch.Tensor,
                    machine_masks: torch.Tensor, operation_bounds: torch.Tensor,
                    old_log_probs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute PPO loss for hierarchical policy"""
        
        # Compute advantages with GAE
        values = self.value_net(states).squeeze()
        next_values = self.value_net(next_states).squeeze()
        
        # Ensure rewards and dones are 1D tensors
        rewards = rewards.squeeze()
        dones = dones.squeeze()
        
    # Use GAE to calculate returns and advantages
        returns, advantages = self.compute_gae(rewards, values, next_values, dones)
        
    # Normalize advantages
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # High-level policy loss
        job_logits, machine_logits = self.high_policy(states, job_masks, machine_masks)
        job_dist = Categorical(logits=job_logits)
        machine_dist = Categorical(logits=machine_logits)
        
        job_log_probs = job_dist.log_prob(actions['job_idx'])
        machine_log_probs = machine_dist.log_prob(actions['machine_idx'])
        high_log_probs = job_log_probs + machine_log_probs
        
        ratio_high = torch.exp(high_log_probs - old_log_probs['high'])
        surr1_high = ratio_high * advantages
        surr2_high = torch.clamp(ratio_high, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
        policy_loss_high = -torch.min(surr1_high, surr2_high).mean()
        
        # Low-level policy loss
        mean, std = self.low_policy(states, actions['job_idx'], actions['machine_idx'], operation_bounds)
        param_dist = Normal(mean, std)
        low_log_probs = param_dist.log_prob(actions['continuous_param']).sum(dim=-1)
        
        ratio_low = torch.exp(low_log_probs - old_log_probs['low'])
        surr1_low = ratio_low * advantages
        surr2_low = torch.clamp(ratio_low, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
        policy_loss_low = -torch.min(surr1_low, surr2_low).mean()
        
        # Value loss
        values_clipped = values + torch.clamp(
            self.value_net(states).squeeze() - values,
            -self.eps_clip,
            self.eps_clip
        )
        value_loss_unclipped = F.mse_loss(values, returns.detach())
        value_loss_clipped = F.mse_loss(values_clipped, returns.detach())
        value_loss = torch.max(value_loss_unclipped, value_loss_clipped)
        
        # Entropy bonus
        entropy_high = job_dist.entropy() + machine_dist.entropy()
        entropy_low = param_dist.entropy().sum(dim=-1)
        entropy_loss = -(entropy_high.mean() + entropy_low.mean()) * self.entropy_coef
        
        # Total loss
        total_loss = policy_loss_high + policy_loss_low + self.value_coef * value_loss + entropy_loss
        
        return {
            'total_loss': total_loss,
            'policy_loss_high': policy_loss_high,
            'policy_loss_low': policy_loss_low,
            'value_loss': value_loss,
            'entropy_loss': entropy_loss
        }
    
    def update(self, batch: Dict[str, torch.Tensor], epochs: int = 10) -> Dict[str, float]:
        """Update policy using PPO"""
        losses = []
        
        for _ in range(epochs):
            loss_dict = self.compute_loss(
                states=batch['states'],
                actions=batch['actions'],
                rewards=batch['rewards'],
                next_states=batch['next_states'],
                dones=batch['dones'],
                job_masks=batch['job_masks'],
                machine_masks=batch['machine_masks'],
                operation_bounds=batch['operation_bounds'],
                old_log_probs=batch['old_log_probs']
            )
            
            self.optimizer.zero_grad()
            loss_dict['total_loss'].backward()
            
        # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(
                list(self.high_policy.parameters()) + 
                list(self.low_policy.parameters()) + 
                list(self.value_net.parameters()), 
                max_norm=0.5
            )
            self.optimizer.step()
            
            losses.append({k: v.item() for k, v in loss_dict.items()})
        
    # Update learning rate
        self.scheduler.step()
        
        # Average losses over epochs
        avg_losses = {k: np.mean([l[k] for l in losses]) for k in losses[0].keys()}
        return avg_losses
    
    def save(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'high_policy_state_dict': self.high_policy.state_dict(),
            'low_policy_state_dict': self.low_policy.state_dict(),
            'value_net_state_dict': self.value_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
    
    def load(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.high_policy.load_state_dict(checkpoint['high_policy_state_dict'])
        self.low_policy.load_state_dict(checkpoint['low_policy_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


class ExperienceBuffer:
    """Buffer for storing and sampling experience tuples"""
    
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
        
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a batch of experiences"""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = {}
        
        for key in self.buffer[0].keys():
            if isinstance(self.buffer[0][key], dict):
                batch[key] = {}
                for sub_key in self.buffer[0][key].keys():
                    batch[key][sub_key] = torch.stack([self.buffer[i][key][sub_key] for i in indices])
            else:
                batch[key] = torch.stack([self.buffer[i][key] for i in indices])
                
        return batch
    
    def clear(self):
        """Clear the buffer"""
        self.buffer.clear()
        self.position = 0
        
    def __len__(self):
        return len(self.buffer)


def get_job_mask(env):
    """Generate a job mask based on the environment state.
    Returns a binary mask where 1 indicates a valid job and 0 an invalid job."""
    mask = np.zeros(env.num_of_jobs, dtype=np.float32)
    
    # Assuming task_set contains all jobs, and each job has operations
    for job_id in range(len(env.task_set)):
        job = env.task_set[job_id]
    # A job is valid if not all of its operations are completed
        job_finished = True
        for op in range(len(job)):
            if not job[op].state:  # Assuming task.state indicates completion
                job_finished = False
                break
        
    # If the job is not finished, mark it as valid
        if not job_finished:
            mask[job_id] = 1.0
    
    return mask

def get_machine_mask(env):
    """Generate a machine mask based on the environment state.
    Returns a binary mask where 1 indicates a valid machine and 0 an invalid machine."""
    mask = np.zeros(env.num_of_robots, dtype=np.float32)
    
    # Assuming robot_state is a list of robot states, where 1 means available
    for robot_id in range(len(env.robot_state)):
        if env.robot_state[robot_id] == 1:  # Robot is available
            mask[robot_id] = 1.0
    
    return mask

def get_operation_bounds(env):
    """Generate bounds for continuous parameters based on the environment state.
    Returns a tensor of shape [max_operations, 2] with [min, max] for each operation."""
    # Define specific operation time bounds
    operation_bounds = np.array([
        [4.0, 7.2],    # Operation 0 time range
        [2.0, 14.2],   # Operation 1 time range
        [2.5, 16.5],   # Operation 2 time range
        [2.1, 18.0],   # Operation 3 time range
        [2.4, 16.8]    # Operation 4 time range
    ])
    
    # If we need more operations than defined, fill with defaults
    if CONFIG['max_operations'] > len(operation_bounds):
        default_bounds = np.tile(np.array([0.0, 1.0]), (CONFIG['max_operations'] - len(operation_bounds), 1))
        operation_bounds = np.vstack([operation_bounds, default_bounds])
    
    # If we need fewer operations, truncate
    if CONFIG['max_operations'] < len(operation_bounds):
        operation_bounds = operation_bounds[:CONFIG['max_operations']]
    
    return operation_bounds

def collect_experience(env, agent: HPPOAgent, num_steps: int, 
                      buffer: ExperienceBuffer, deterministic: bool = False,
                      config: dict = None, reward_normalizer=None):
    """Collect experience with simplified reward structure"""
    if config is None:
        config = CONFIG
        
    state = env.reset()
    episode_reward = 0
    step_counter = 0
    
    for step in range(num_steps):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
        
        job_mask = torch.FloatTensor(get_job_mask(env)).unsqueeze(0).to(agent.device)
        machine_mask = torch.FloatTensor(get_machine_mask(env)).unsqueeze(0).to(agent.device)
        operation_bounds = torch.FloatTensor(get_operation_bounds(env)).unsqueeze(0).to(agent.device)
        
        action_dict = agent.select_action(
            state_tensor, job_mask, machine_mask, operation_bounds, deterministic
        )
        
        job_id = action_dict['job_idx'].item()
        machine_id = action_dict['machine_idx'].item()
        param_value = float(action_dict['continuous_param'].squeeze(0).cpu().numpy()[0])
        
        action = [job_id, machine_id, param_value]
        
        step_result = env.step(action)
        step_counter += 1
        
        if len(step_result) == 3:
            next_state, reward, done = step_result
            info = {}
        else:
            next_state, reward, done, info = step_result
        
    # Simplified reward processing - use only raw reward or normalization
        if config.get('reward_normalization', False) and reward_normalizer is not None:
            shaped_reward = reward_normalizer.normalize(reward)
        else:
            shaped_reward = reward  # simple scaling
        
    episode_reward += reward  # Record raw reward
        
        # Calculate log probabilities for PPO
        with torch.no_grad():
            job_dist = Categorical(logits=action_dict['job_logits'])
            machine_dist = Categorical(logits=action_dict['machine_logits'])
            high_log_prob = (job_dist.log_prob(action_dict['job_idx']) + 
                           machine_dist.log_prob(action_dict['machine_idx']))
            
            param_dist = Normal(action_dict['param_mean'], action_dict['param_std'])
            low_log_prob = param_dist.log_prob(action_dict['continuous_param']).sum(dim=-1)
        
        experience = {
            'states': state_tensor.squeeze(0),
            'actions': {
                'job_idx': action_dict['job_idx'].squeeze(0),
                'machine_idx': action_dict['machine_idx'].squeeze(0),
                'continuous_param': action_dict['continuous_param'].squeeze(0)
            },
            'rewards': torch.FloatTensor([shaped_reward]).to(agent.device),
            'next_states': torch.FloatTensor(next_state).to(agent.device),
            'dones': torch.FloatTensor([done]).to(agent.device),
            'job_masks': job_mask.squeeze(0),
            'machine_masks': machine_mask.squeeze(0),
            'operation_bounds': operation_bounds.squeeze(0),
            'old_log_probs': {
                'high': high_log_prob.squeeze(0),
                'low': low_log_prob.squeeze(0)
            }
        }
        buffer.push(experience)
        
        if done:
            # Episode finished, return immediately
            return episode_reward, step_counter
        
        state = next_state
    
    # If max steps reached but episode not finished, return current accumulated reward
    return episode_reward, step_counter


def evaluate_agent(env, agent: HPPOAgent, num_episodes: int = 10):
    """Evaluate agent performance"""
    total_rewards = []
    step_counts = []  # Track steps for each episode
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
    step_counter = 0  # Initialize step counter
        
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            job_mask = torch.FloatTensor(get_job_mask(env)).unsqueeze(0).to(agent.device)
            machine_mask = torch.FloatTensor(get_machine_mask(env)).unsqueeze(0).to(agent.device)
            operation_bounds = torch.FloatTensor(get_operation_bounds(env)).unsqueeze(0).to(agent.device)
            
            action_dict = agent.select_action(
                state_tensor, job_mask, machine_mask, operation_bounds, deterministic=True
            )
            
            # Format action as a list as expected by the environment
            job_id = action_dict['job_idx'].item()
            machine_id = action_dict['machine_idx'].item()
            param_value = float(action_dict['continuous_param'].squeeze(0).cpu().numpy()[0])
            action = [job_id, machine_id, param_value]
            
            # Handle the case where env.step returns 3 values instead of 4
            step_result = env.step(action)
            step_counter += 1  # Increment step counter
            
            if len(step_result) == 3:
                state, reward, done = step_result
            else:
                state, reward, done, _ = step_result
                
            episode_reward += reward
            
        total_rewards.append(episode_reward)
        step_counts.append(step_counter)
        print(f"Evaluation episode {episode+1} completed in {step_counter} steps with reward {episode_reward:.4f}")
        
    print(f"Average steps to completion: {np.mean(step_counts):.2f} ± {np.std(step_counts):.2f}")
    return np.mean(total_rewards), np.std(total_rewards)

def test_and_visualize(config=None, model_path=None):
    """Test trained model and visualize scheduling process"""
    if config is None:
        config = CONFIG
        
    print("\n=== Starting H-PPO Testing and Visualization ===")
    
    # Create environment with correct parameter names
    env_kwargs = {
        'num_of_jobs': config['num_jobs'],
        'num_of_robots': config['num_machines'],
        'alpha': config['alpha'],
        'beta': config['beta']
    }
    
    try:
        # Create environment with correct parameters
        env = ENV.Env(
            num_of_jobs=env_kwargs['num_of_jobs'],
            num_of_robots=env_kwargs['num_of_robots'],
            alpha=env_kwargs['alpha'],
            beta=env_kwargs['beta']
        )
        print(f"Successfully created environment with parameters: {env_kwargs}")
    except Exception as e:
        print(f"Error creating environment: {e}")
        print("Available attributes in ENV module:", [attr for attr in dir(ENV) if not attr.startswith('_')])
        
        # Try to print the signature of the Env constructor for debugging
        import inspect
        try:
            print("Env constructor signature:", inspect.signature(ENV.Env.__init__))
        except Exception as sig_error:
            print("Could not get constructor signature:", sig_error)
            
        raise
    
    # Get environment dimensions
    state_dim = len(env.state)
    num_jobs = env_kwargs['num_of_jobs']
    num_machines = env_kwargs['num_of_robots']
    max_operations = config['max_operations']
    
    # Create agent
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agent = HPPOAgent(
        state_dim=state_dim,
        num_jobs=num_jobs,
        num_machines=num_machines,
        max_operations=max_operations,
        lr=config['lr'],
        gamma=config['gamma'],
        eps_clip=config['eps_clip'],
        value_coef=config['value_coef'],
        entropy_coef=config['entropy_coef'],
        device=device
    )
    
    # Load model
    if model_path:
        agent.load(model_path)
        print(f"Model loaded from: {model_path}")
    else:
        print("No model path provided, using untrained agent.")
    
    # Testing loop
    state = env.reset()
    done = False
    total_reward = 0
    step_counter = 0  # Initialize step counter
    
    while not done:
        # Prepare state
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
        job_mask = torch.FloatTensor(get_job_mask(env)).unsqueeze(0).to(agent.device)
        machine_mask = torch.FloatTensor(get_machine_mask(env)).unsqueeze(0).to(agent.device)
        operation_bounds = torch.FloatTensor(get_operation_bounds(env)).unsqueeze(0).to(agent.device)
        
        # Select action
        with torch.no_grad():
            action_dict = agent.select_action(
                state_tensor, job_mask, machine_mask, operation_bounds, deterministic=True
            )
        
        # Format action as a list as expected by the environment
        job_id = action_dict['job_idx'].item()
        machine_id = action_dict['machine_idx'].item()
        param_value = float(action_dict['continuous_param'].squeeze(0).cpu().numpy()[0])
        action = [job_id, machine_id, param_value]
        
        # Handle the case where env.step returns 3 values instead of 4
        step_result = env.step(action)
        step_counter += 1  # Increment step counter
        
        if len(step_result) == 3:
            state, reward, done = step_result
        else:
            state, reward, done, _ = step_result
            
        total_reward += reward
        
        # Visualization (Gantt chart or other)
        if config['enable_gantt_plots']:
            env.render_gantt()
    
    print(f"Total Reward: {total_reward}")
    print(f"Episode completed in {step_counter} steps")
    plt.show()


def multi_run_training(config=None):
    """Execute multiple training runs with different seeds"""
    if config is None:
        config = CONFIG
    
    all_score_records = []
    all_action_restores = []
    all_models = []
    
    print(f"\n{'='*80}")
    print(f"Starting Multi-Run H-PPO Training")
    print(f"Seeds: {config['seeds'][:config['num_runs']]}")
    print(f"Total runs: {config['num_runs']}")
    print(f"{'='*80}")
    
    for run_idx, seed in enumerate(config['seeds'][:config['num_runs']]):
    print(f"\n{'='*60}")
    print(f"Run {run_idx + 1}/{config['num_runs']} - Seed: {seed}")
    print(f"{'='*60}")
        
        # Set random seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        # Run training
        score_record, action_restore, model = train_single_run(config)
        
        # Store results
        all_score_records.append(score_record)
        all_action_restores.append(action_restore)
        all_models.append(model)
        
        print(f"Run {run_idx + 1} completed - Final Score: {np.mean(score_record[-10:]):.4f}")
        
    print(f"\n{'='*60}")
    print(f"All {config['num_runs']} runs completed!")
    print(f"{'='*60}")
    
    return all_score_records, all_action_restores, all_models

def train_single_run(config=None):
    """Execute a single training run with debugging info"""
    if config is None:
        config = CONFIG
    
    # Environment setup with correct parameter names
    env_kwargs = {
        'num_of_jobs': config['num_jobs'],
        'num_of_robots': config['num_machines'],
        'alpha': config['alpha'],
        'beta': config['beta']
    }
    
    try:
        # Create environment with correct parameters
        # Use the exact parameter names from the Env class
        env = ENV.Env(
            num_of_jobs=env_kwargs['num_of_jobs'],
            num_of_robots=env_kwargs['num_of_robots'],
            alpha=env_kwargs['alpha'],
            beta=env_kwargs['beta']
        )
        print(f"Successfully created environment with parameters: {env_kwargs}")
    except Exception as e:
        print(f"Error creating environment: {e}")
        print("Available attributes in ENV module:", [attr for attr in dir(ENV) if not attr.startswith('_')])
        
        # Try to print the signature of the Env constructor for debugging
        import inspect
        try:
            print("Env constructor signature:", inspect.signature(ENV.Env.__init__))
        except Exception as sig_error:
            print("Could not get constructor signature:", sig_error)
            
        raise
    
    # Get environment dimensions
    state_dim = len(env.state)
    num_jobs = env_kwargs['num_of_jobs']
    num_machines = env_kwargs['num_of_robots']
    max_operations = config['max_operations']
    
    # Create agent
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agent = HPPOAgent(
        state_dim=state_dim,
        num_jobs=num_jobs,
        num_machines=num_machines,
        max_operations=max_operations,
        lr=config['lr'],
        gamma=config['gamma'],
        eps_clip=config['eps_clip'],
        value_coef=config['value_coef'],
        entropy_coef=config['entropy_coef'],
        device=device,
        gae_lambda=config.get('gae_lambda', 0.95)
    )
    
    # Initialize experience buffer and normalizers
    buffer = ExperienceBuffer(capacity=config['buffer_limit'])
    reward_normalizer = RewardNormalizer() if config.get('reward_normalization', False) else None
    
    # Training metrics
    episode_rewards = []
    episode_steps = []
    evaluation_rewards = []
    action_restore = []
    
    # Add debug info collection
    loss_history = {
        'policy_high': [],
        'policy_low': [],
        'value': [],
        'entropy': []
    }
    
    print(f"Starting H-PPO training:")
    print(f"Jobs: {num_jobs}, Machines: {num_machines}")
    print(f"Alpha: {config['alpha']}, Beta: {config['beta']}, Episodes: {config['num_episodes']}")
    print(f"Device: {device}")
    
    best_reward = -float('inf')
    no_improvement_count = 0
    update_counter = 0
    
    for episode in range(config['num_episodes']):
        # Collect experience for ONE episode
        episode_reward, steps = collect_experience(
            env, agent, config['steps_per_episode'], buffer, 
            deterministic=False,
            config=config,
            reward_normalizer=reward_normalizer
        )
        
        episode_rewards.append(episode_reward)
        episode_steps.append(steps)
        
        # Update reward normalizer
        if reward_normalizer is not None:
            reward_normalizer.update(episode_reward)
        
        # Check for improvement
        if episode_reward > best_reward:
            best_reward = episode_reward
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        
        # Print training progress more frequently for debugging
        if episode % 5 == 0 or episode == config['num_episodes'] - 1:
            window_size = min(20, len(episode_rewards))
            if len(episode_rewards) >= window_size:
                moving_avg = np.mean(episode_rewards[-window_size:])
                moving_std = np.std(episode_rewards[-window_size:])
                avg_steps = np.mean(episode_steps[-window_size:])
                print(f"Episode {episode + 1}/{config['num_episodes']}: "
                      f"R={episode_reward:.2f}, Steps={steps}, "
                      f"MA={moving_avg:.2f}±{moving_std:.2f}, "
                      f"Best={best_reward:.2f}, BufferSize={len(buffer)}")
            else:
                print(f"Episode {episode + 1}/{config['num_episodes']}: "
                      f"R={episode_reward:.2f}, Steps={steps}, BufferSize={len(buffer)}")
        
    # Update policy every episode if buffer has enough samples
        if len(buffer) >= config['memory_threshold']:
            batch_size = min(config['batch_size'], len(buffer))
            batch = buffer.sample(batch_size)
            losses = agent.update(batch, epochs=config['num_epochs'])
            
            # Record losses for debugging
            loss_history['policy_high'].append(losses['policy_loss_high'])
            loss_history['policy_low'].append(losses['policy_loss_low'])
            loss_history['value'].append(losses['value_loss'])
            loss_history['entropy'].append(losses['entropy_loss'])
            
            update_counter += 1
            if update_counter % 10 == 0:
                # Print recent loss trends
                recent_losses = {
                    k: np.mean(v[-10:]) if len(v) >= 10 else np.mean(v) 
                    for k, v in loss_history.items()
                }
                print(f"  Update {update_counter} - Avg Losses: "
                      f"PH={recent_losses['policy_high']:.4f}, "
                      f"PL={recent_losses['policy_low']:.4f}, "
                      f"V={recent_losses['value']:.4f}, "
                      f"E={recent_losses['entropy']:.4f}")
        
        # Clear buffer periodically to avoid stale data
        if episode % 100 == 0 and episode > 0:
            buffer.clear()
            print(f"  Buffer cleared at episode {episode}")
        
        # Evaluate
        if (episode + 1) % config['eval_interval'] == 0:
            eval_mean, eval_std = evaluate_agent(env, agent, num_episodes=5)
            evaluation_rewards.append((episode + 1, eval_mean, eval_std))
            print(f"\n{'='*50}")
            print(f"Evaluation at Episode {episode + 1}: {eval_mean:.4f} ± {eval_std:.4f}")
            print(f"Training performance: {np.mean(episode_rewards[-50:]):.4f}")
            print(f"{'='*50}\n")
    
    # Print final statistics
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Final 100 episodes average: {np.mean(episode_rewards[-100:]):.4f}")
    print(f"Best episode reward: {best_reward:.4f}")
    print(f"Final loss values:")
    for k, v in loss_history.items():
        if len(v) > 0:
            print(f"  {k}: {np.mean(v[-50:]):.4f}")
    print(f"{'='*60}")
    
    return episode_rewards, action_restore, agent

class RewardNormalizer:
    """Running normalization for rewards"""
    def __init__(self, shape=(), epsilon=1e-8):
        self.epsilon = epsilon
        self.shape = shape
        self.mean = np.zeros(shape)
        self.var = np.ones(shape)
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x)
        batch_var = np.var(x)
        batch_count = len(x) if isinstance(x, (list, np.ndarray)) else 1
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x):
        return (x - self.mean) / np.sqrt(self.var + self.epsilon)

def evaluate_multi_run_results(all_score_records, config=None):
    """Evaluate and analyze results from multiple runs"""
    if config is None:
        config = CONFIG
    
    print(f"\n{'='*60}")
    print(f"Multi-Run Training Results Analysis")
    print(f"{'='*60}")
    
    # Calculate statistics from last N episodes
    n_last = 20
    final_scores = [np.mean(scores[-n_last:]) if len(scores) >= n_last else np.mean(scores) 
                   for scores in all_score_records]
    
    print(f"Last {n_last} Episodes Average Scores:")
    for i, (seed, score) in enumerate(zip(config['seeds'][:len(all_score_records)], final_scores)):
        print(f"  Run {i+1} (Seed {seed}): {score:.4f}")
    
    print(f"\nOverall Statistics:")
    print(f"  Mean Score: {np.mean(final_scores):.4f} ± {np.std(final_scores):.4f}")
    print(f"  Best Score: {np.max(final_scores):.4f}")
    print(f"  Worst Score: {np.min(final_scores):.4f}")
    
    return {
        'final_scores': final_scores,
        'overall_mean': np.mean(final_scores),
        'overall_std': np.std(final_scores),
        'best_score': np.max(final_scores),
        'worst_score': np.min(final_scores)
    }

def save_multi_run_results(all_score_records, all_action_restores, all_models, config):
    """Save results from multiple training runs"""
    if not config['save_models']:
        return None, None
    
    # Create save directory with timestamp
    import json
    alg_name = "hppo"
    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"{alg_name}_multi_run_{now_str}"
    os.makedirs(save_dir, exist_ok=True)

    # Save training data for all runs
    np.save(os.path.join(save_dir, f"{alg_name}_all_scores_{now_str}.npy"), all_score_records)
    np.save(os.path.join(save_dir, f"{alg_name}_all_actions_{now_str}.npy"), all_action_restores)

    # Save models from all runs
    model_paths = []
    for run_idx, model in enumerate(all_models):
        run_save_dir = os.path.join(save_dir, f"run_{run_idx+1}_seed_{config['seeds'][run_idx]}")
        os.makedirs(run_save_dir, exist_ok=True)
        
        model_path = os.path.join(run_save_dir, f"{alg_name}_model_{now_str}.pt")
        model.save(model_path)
        model_paths.append(model_path)
    
    # Save configuration
    config_path = os.path.join(save_dir, f"config_{now_str}.json")
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

def save_results(all_score_records, all_action_restores, all_models, config):
    """Save single run results - wrapper for compatibility"""
    return save_multi_run_results(all_score_records, all_action_restores, all_models, config)

def plot_multi_run_training_curves(all_score_records, config=None):
    """Plot training curves for multiple runs"""
    if config is None:
        config = CONFIG
    
    plt.figure(figsize=(12, 8))
    
    # Plot individual runs
    for i, scores in enumerate(all_score_records):
    # Ensure x-axis matches training episode count
        x = range(len(scores))
        plt.plot(x, scores, alpha=0.6, label=f'Run {i+1} (Seed {config["seeds"][i]})')
    
    # Plot mean curve if we have multiple runs
    if len(all_score_records) > 1:
        # Find minimum length across all runs
        min_length = min(len(scores) for scores in all_score_records)
        mean_scores = np.mean([scores[:min_length] for scores in all_score_records], axis=0)
        std_scores = np.std([scores[:min_length] for scores in all_score_records], axis=0)
        
        x = range(min_length)
        plt.plot(x, mean_scores, 'k-', linewidth=2, label='Mean')
        plt.fill_between(x, mean_scores - std_scores, mean_scores + std_scores, alpha=0.2, color='black')
    
    plt.title('H-PPO Multi-Run Training Curves')
    plt.xlabel('Training Episode')
    plt.ylabel('Episode Reward')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Add extra info
    plt.text(0.02, 0.98, f'Total Episodes: {config["num_episodes"]}', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.show()

def plot_training_curve(score_record):
    """Plot single training curve"""
    plt.figure(figsize=(10, 6))
    plt.plot(score_record)
    plt.title('H-PPO Training Curve')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.show()

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='H-PPO Policy for Flexible Job Shop Scheduling')
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
    
    # Update CONFIG with command line arguments
    config = CONFIG.copy()
    config.update({
        'num_jobs': args.jobs,
        'num_machines': args.machines,
        'alpha': args.alpha,
        'beta': args.beta,
        'num_episodes': args.episodes,
        'seeds': args.seeds,
        'enable_multi_run': args.multi_run and not args.single_run  # Allow override with --single-run
    })
    
    print(f"\n{'='*60}")
    print(f"H-PPO Policy for Flexible Job Shop Scheduling")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  Jobs: {config['num_jobs']}")
    print(f"  Machines: {config['num_machines']}")
    print(f"  Alpha: {config['alpha']}")
    print(f"  Beta: {config['beta']}")
    print(f"  Episodes: {config['num_episodes']}")
    print(f"  Seeds: {config['seeds'][:5]}..." if len(config['seeds']) > 5 else f"  Seeds: {config['seeds']}")
    print(f"  Test only: {args.test_only}")
    print(f"  Multi-run enabled (config): {CONFIG['enable_multi_run']}")
    print(f"  Multi-run mode (final): {config['enable_multi_run']}")
    if args.model_path:
        print(f"  Model path: {args.model_path}")
    print(f"{'='*60}")
    
    if args.test_only:
        # Test only mode
        test_and_visualize(config, args.model_path)
    elif config['enable_multi_run']:
        # Multi-run training mode
        all_score_records, all_action_restores, all_models = multi_run_training(config)
        
        # Evaluate results
        stats = evaluate_multi_run_results(all_score_records, config)
        
        # Save multi-run results if enabled
        if config['save_models']:
            save_dir, model_paths = save_multi_run_results(all_score_records, all_action_restores, all_models, config)
        
        # Plot multi-run training curves
        if config['plot_training_curve']:
            plot_multi_run_training_curves(all_score_records, config)
        
        # Test with the best performing model
        if config['save_models'] and model_paths:
            # Determine best model based on final performance
            last_n = 20  # Use last N episodes for evaluation
            mean_scores = [np.mean(scores[-last_n:]) if len(scores) >= last_n else np.mean(scores) 
                          for scores in all_score_records]
            best_run_idx = np.argmax(mean_scores)
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
        
        score_record, action_restore, model = train_single_run(config)
        
        # Save results if enabled
        if config['save_models']:
            save_dir, model_path = save_results([score_record], [action_restore], [model], config)
        
        # Plot training curve
        if config['plot_training_curve']:
            plot_training_curve(score_record)
        
        # Test with trained model
        if config['save_models'] and model_path:
            print(f"\n{'='*40}")
            print(f"Testing with trained model...")
            print(f"{'='*40}")
            test_and_visualize(config, model_path[0])  # Use first path from list


if __name__ == '__main__':
    main()
