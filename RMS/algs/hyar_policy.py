import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import math
import sys
import os
import datetime
import matplotlib.pyplot as plt
from itertools import product
sys.path.append(os.path.join(os.path.dirname(__file__), '../env'))
import env as ENV

# Import DDPG-LIRL components for sample generation
from ddpg_lirl_pi import MuNet as DDPGMuNet, OrnsteinUhlenbeckNoise, action_projection

# =======================
# HYPERPARAMETERS CONFIG
# =======================
CONFIG = {
    # Learning parameters
    'lr_policy': 0.0005,  # Learning rate for policy network
    'lr_q': 0.001,  # Learning rate for Q network
    'lr_vae': 0.001,  # Learning rate for Conditional VAE
    'gamma': 0.98,
    'batch_size': 128,
    'buffer_limit': 1000000,
    'tau': 0.005,  # for target network soft update
    
    # Environment parameters
    'num_of_jobs': 100,
    'num_of_robots': 5,
    'alpha': 0.5,
    'beta': 0.5,
    'num_of_episodes': 1000,
    'max_steps_per_episode': 550,  # Maximum steps per episode during training
    
    # Network architecture
    'policy_hidden_dim': 128,
    'q_hidden_dim1': 128,
    'q_hidden_dim2': 64,
    'latent_dim': 32,  # Latent dimension for VAE
    'vae_hidden_dim': 128,  # Hidden dimension for VAE
    
    # Stage 1: VAE training parameters
    'stage1_episodes': 200,  # Episodes for collecting samples
    'vae_train_epochs': 100,  # Epochs for VAE training
    'vae_kl_weight': 0.001,  # KL divergence weight in VAE loss
    
    # Stage 2: RL training parameters
    'memory_threshold': 500,
    'training_iterations': 20,
    
    # Multi-run training parameters
    'enable_multi_run': True,
    'seeds': [3047, 294, 714, 1092, 1386, 2856, 42, 114514, 2025, 1993],
    'num_runs': 10,
    
    # Testing parameters
    'max_test_steps': 100,
    
    # Output parameters
    'print_interval': 10,
    'enable_gantt_plots': False,
    'plot_training_curve': True,
    'save_models': True,
}


def get_valid_action_mask(env):
    """Get mask for valid actions based on current environment state"""
    # Get valid jobs mask
    job_mask = torch.zeros(CONFIG['num_of_jobs'])
    for job_id, job in enumerate(env.task_set):
        # Check if job is not finished
        finished = all(task.state for task in job)
        if not finished:
            job_mask[job_id] = 1.0
    
    # Get valid robots mask
    robot_mask = torch.zeros(CONFIG['num_of_robots'])
    for robot_id, state in enumerate(env.robot_state):
        if state == 1:  # Robot is available
            robot_mask[robot_id] = 1.0
    
    return job_mask, robot_mask


def apply_action_mask(job_logits, robot_logits, job_mask, robot_mask):
    """Apply mask to action logits to prevent invalid actions"""
    # Apply mask by setting invalid actions to very negative value
    masked_job_logits = job_logits.clone()
    masked_robot_logits = robot_logits.clone()
    
    # For jobs
    if job_mask.sum() > 0:  # At least one valid job
        masked_job_logits[job_mask == 0] = -1e8
    else:
        # If no valid jobs, keep original logits (will handle in decode_action)
        pass
    
    # For robots
    if robot_mask.sum() > 0:  # At least one valid robot
        masked_robot_logits[robot_mask == 0] = -1e8
    else:
        # If no valid robots, keep original logits
        pass
    
    return masked_job_logits, masked_robot_logits


class ConditionalVAE(nn.Module):
    """Conditional VAE for mapping implicit actions to explicit actions"""
    def __init__(self, state_size, implicit_action_size, explicit_action_size):
        super(ConditionalVAE, self).__init__()
        self.state_size = state_size
        self.implicit_action_size = implicit_action_size
        self.explicit_action_size = explicit_action_size
        
        # Encoder: encodes explicit action + state to latent space
        encoder_input_dim = explicit_action_size + state_size
        self.encoder_fc1 = nn.Linear(encoder_input_dim, CONFIG['vae_hidden_dim'])
        self.encoder_fc2 = nn.Linear(CONFIG['vae_hidden_dim'], CONFIG['vae_hidden_dim'])
        self.fc_mu = nn.Linear(CONFIG['vae_hidden_dim'], CONFIG['latent_dim'])
        self.fc_logvar = nn.Linear(CONFIG['vae_hidden_dim'], CONFIG['latent_dim'])
        
        # Decoder: decodes implicit action + state to explicit action
        decoder_input_dim = implicit_action_size + state_size
        self.decoder_fc1 = nn.Linear(decoder_input_dim, CONFIG['vae_hidden_dim'])
        self.decoder_fc2 = nn.Linear(CONFIG['vae_hidden_dim'], CONFIG['vae_hidden_dim'])
        self.decoder_out_job = nn.Linear(CONFIG['vae_hidden_dim'], CONFIG['num_of_jobs'])
        self.decoder_out_robot = nn.Linear(CONFIG['vae_hidden_dim'], CONFIG['num_of_robots'])
        self.decoder_out_param = nn.Linear(CONFIG['vae_hidden_dim'], 1)
        
    def encode(self, explicit_action, state):
        """Encode explicit action conditioned on state"""
        x = torch.cat([explicit_action, state], dim=1)
        h = F.relu(self.encoder_fc1(x))
        h = F.relu(self.encoder_fc2(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, implicit_action, state):
        """Decode implicit action conditioned on state to explicit action"""
        x = torch.cat([implicit_action, state], dim=1)
        h = F.relu(self.decoder_fc1(x))
        h = F.relu(self.decoder_fc2(h))
        
        # Output job and robot probabilities and continuous parameter
        job_logits = self.decoder_out_job(h)
        robot_logits = self.decoder_out_robot(h)
        param = torch.sigmoid(self.decoder_out_param(h))
        
        return job_logits, robot_logits, param
    
    def forward(self, implicit_action, explicit_action, state):
        """Full forward pass for training"""
        # Encode explicit action
        mu, logvar = self.encode(explicit_action, state)
        z = self.reparameterize(mu, logvar)
        
        # Decode from implicit action
        job_logits, robot_logits, param = self.decode(implicit_action, state)
        
        return job_logits, robot_logits, param, mu, logvar
    
    def decode_with_mask(self, implicit_action, state, job_mask, robot_mask):
        """Decode implicit action with action masking"""
        job_logits, robot_logits, param = self.decode(implicit_action, state)
        
        # Apply masking
        batch_size = job_logits.shape[0]
        for i in range(batch_size):
            if job_mask is not None:
                job_logits[i][job_mask == 0] = -1e8
            if robot_mask is not None:
                robot_logits[i][robot_mask == 0] = -1e8
        
        return job_logits, robot_logits, param


class ImplicitPolicyNet(nn.Module):
    """Policy network that outputs implicit (latent) actions"""
    def __init__(self, state_size, action_size):
        super(ImplicitPolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_size, CONFIG['policy_hidden_dim'])
        self.fc2 = nn.Linear(CONFIG['policy_hidden_dim'], CONFIG['policy_hidden_dim'])
        self.fc_out = nn.Linear(CONFIG['policy_hidden_dim'], action_size)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # Output bounded implicit actions
        action = torch.sigmoid(self.fc_out(x))
        return action


class QNet(nn.Module):
    """Q-network for evaluating state-action pairs"""
    def __init__(self, state_size, action_size):
        super(QNet, self).__init__()
        # Input: state + explicit action (job_id, robot_id, param)
        self.fc_s = nn.Linear(state_size, CONFIG['q_hidden_dim2'])
        self.fc_a = nn.Linear(action_size, CONFIG['q_hidden_dim2'])
        self.fc_q = nn.Linear(CONFIG['q_hidden_dim2'] * 2, CONFIG['q_hidden_dim1'])
        self.fc_out = nn.Linear(CONFIG['q_hidden_dim1'], 1)
        
    def forward(self, state, action):
        h1 = F.relu(self.fc_s(state))
        h2 = F.relu(self.fc_a(action))
        cat = torch.cat([h1, h2], dim=1)
        q = F.relu(self.fc_q(cat))
        q = self.fc_out(q)
        return q


class HybridReplayBuffer():
    """Replay buffer for hybrid action space"""
    def __init__(self):
        self.buffer = collections.deque(maxlen=CONFIG['buffer_limit'])

    def put(self, transition):
        self.buffer.append(transition)
        
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, implicit_a_lst, explicit_a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], [], []

        for transition in mini_batch:
            s, implicit_a, explicit_a, r, s_prime, done = transition
            s_lst.append(s)
            implicit_a_lst.append(implicit_a)
            explicit_a_lst.append(explicit_a)
            r_lst.append(r)
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0 
            done_mask_lst.append([done_mask])
        
        # Convert to tensors
        s_tensor = torch.FloatTensor(np.array(s_lst))
        implicit_a_tensor = torch.FloatTensor(np.array(implicit_a_lst))
        explicit_a_tensor = torch.FloatTensor(np.array(explicit_a_lst))
        r_tensor = torch.FloatTensor(np.array(r_lst))
        s_prime_tensor = torch.FloatTensor(np.array(s_prime_lst))
        done_mask_tensor = torch.FloatTensor(np.array(done_mask_lst))

        return s_tensor, implicit_a_tensor, explicit_a_tensor, r_tensor, s_prime_tensor, done_mask_tensor
    
    def size(self):
        return len(self.buffer)


def collect_samples_stage1(env, num_episodes=200):
    """Stage 1: Collect samples using DDPG-LIRL policy"""
    print(f"\n=== Stage 1: Collecting samples using DDPG-LIRL ===")
    
    # Initialize DDPG policy
    state_size = len(env.state)
    action_size = len(env.action)
    ddpg_policy = DDPGMuNet(state_size, action_size)
    ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(action_size))
    
    samples = []
    
    for episode in range(num_episodes):
        s = env.reset()
        done = False
        episode_samples = []
        step_count = 0
        
        while not done and step_count < CONFIG['max_steps_per_episode']:
            # Get implicit action from DDPG
            state_tensor = torch.from_numpy(s).float()
            implicit_action = ddpg_policy(state_tensor)
            implicit_action = torch.clamp(implicit_action + torch.from_numpy(ou_noise()).float(), 0, 1)
            
            # Get explicit action through projection
            explicit_action = action_projection(env, implicit_action)
            
            # Execute action
            s_prime, r, done = env.step(explicit_action)
            
            # Store sample: (state, implicit_action, explicit_action)
            episode_samples.append({
                'state': s,
                'implicit_action': implicit_action.detach().numpy(),
                'explicit_action': explicit_action
            })
            
            s = s_prime
            step_count += 1
        
        samples.extend(episode_samples)
        
        if (episode + 1) % 50 == 0:
            print(f"  Collected {episode + 1} episodes, {len(samples)} samples")
    
    print(f"  Total samples collected: {len(samples)}")
    return samples


def train_vae_stage1(vae, samples, epochs=100):
    """Stage 1: Train VAE on collected samples"""
    print(f"\n=== Stage 1: Training Conditional VAE ===")
    
    vae_optimizer = optim.Adam(vae.parameters(), lr=CONFIG['lr_vae'])
    
    # Convert samples to tensors - first convert to numpy arrays
    states = np.array([s['state'] for s in samples])
    implicit_actions = np.array([s['implicit_action'] for s in samples])
    
    # Then convert to tensors
    states = torch.FloatTensor(states)
    implicit_actions = torch.FloatTensor(implicit_actions)
    
    # Convert explicit actions to tensor format
    explicit_actions = []
    for s in samples:
        job_id, robot_id, param = s['explicit_action']
        # Create one-hot encoding for job and robot
        job_one_hot = torch.zeros(CONFIG['num_of_jobs'])
        robot_one_hot = torch.zeros(CONFIG['num_of_robots'])
        job_one_hot[job_id] = 1
        robot_one_hot[robot_id] = 1
        explicit_action = torch.cat([job_one_hot, robot_one_hot, torch.tensor([param])])
        explicit_actions.append(explicit_action)
    explicit_actions = torch.stack(explicit_actions)
    
    # Training loop
    batch_size = CONFIG['batch_size']
    num_samples = len(samples)
    
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_recon_loss = 0
        epoch_kl_loss = 0
        
        # Shuffle indices
        indices = torch.randperm(num_samples)
        
        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            
            state_batch = states[batch_indices]
            implicit_batch = implicit_actions[batch_indices]
            explicit_batch = explicit_actions[batch_indices]
            
            # Extract job, robot, and param from explicit actions
            job_target = explicit_batch[:, :CONFIG['num_of_jobs']].argmax(dim=1)
            robot_target = explicit_batch[:, CONFIG['num_of_jobs']:CONFIG['num_of_jobs']+CONFIG['num_of_robots']].argmax(dim=1)
            param_target = explicit_batch[:, -1:]
            
            # Forward pass
            job_logits, robot_logits, param_pred, mu, logvar = vae(implicit_batch, explicit_batch, state_batch)
            
            # Reconstruction loss
            job_loss = F.cross_entropy(job_logits, job_target)
            robot_loss = F.cross_entropy(robot_logits, robot_target)
            param_loss = F.mse_loss(param_pred, param_target)
            recon_loss = job_loss + robot_loss + param_loss
            
            # KL divergence loss
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
            
            # Total loss
            loss = recon_loss + CONFIG['vae_kl_weight'] * kl_loss
            
            vae_optimizer.zero_grad()
            loss.backward()
            vae_optimizer.step()
            
            epoch_loss += loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += kl_loss.item()
        
        if (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / (num_samples // batch_size)
            avg_recon = epoch_recon_loss / (num_samples // batch_size)
            avg_kl = epoch_kl_loss / (num_samples // batch_size)
            print(f"  Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Recon={avg_recon:.4f}, KL={avg_kl:.4f}")
    
    print("  VAE training completed!")
    return vae


def decode_action(vae, implicit_action, state, env):
    """Decode implicit action to explicit action using trained VAE"""
    with torch.no_grad():
        job_logits, robot_logits, param = vae.decode(implicit_action, state)
        
        # Get valid action masks
        job_mask, robot_mask = get_valid_action_mask(env)
        
        # Apply masks to logits
        masked_job_logits, masked_robot_logits = apply_action_mask(
            job_logits.squeeze(), robot_logits.squeeze(), job_mask, robot_mask
        )
        
        # Get job and robot from masked logits
        if job_mask.sum() > 0 and robot_mask.sum() > 0:
            job_probs = F.softmax(masked_job_logits, dim=-1)
            robot_probs = F.softmax(masked_robot_logits, dim=-1)
            
            job_id = job_probs.argmax().item()
            robot_id = robot_probs.argmax().item()
        else:
            # Fallback: if no valid actions, return first valid or default
            valid_jobs = [i for i, m in enumerate(job_mask) if m > 0]
            valid_robots = [i for i, m in enumerate(robot_mask) if m > 0]
            
            job_id = valid_jobs[0] if valid_jobs else 0
            robot_id = valid_robots[0] if valid_robots else 0
        
        param_value = param.squeeze().item()
        
        return [job_id, robot_id, param_value]


def train_hyar_stage2(policy, policy_target, q, q_target, vae, memory, 
                     policy_optimizer, q_optimizer):
    """Stage 2: Train HyAR with decoded actions and masking"""
    s, implicit_a, explicit_a, r, s_prime, done_mask = memory.sample(CONFIG['batch_size'])
    
    # Create explicit action tensor for Q-network
    explicit_a_tensor = explicit_a
    
    # Compute target Q-values
    with torch.no_grad():
        # Get next implicit action
        next_implicit = policy_target(s_prime)
        
        # Decode to explicit action
        next_job_logits, next_robot_logits, next_param = vae.decode(next_implicit, s_prime)
        
        # Note: We don't apply mask here since we're using batch data
        # In actual environment interaction, masks would be applied
        
        # Create explicit action representation for Q-network
        next_job_probs = F.softmax(next_job_logits, dim=-1)
        next_robot_probs = F.softmax(next_robot_logits, dim=-1)
        next_explicit = torch.cat([next_job_probs, next_robot_probs, next_param], dim=1)
        
        # Compute target Q-value
        target_q = q_target(s_prime, next_explicit)
        target = r.unsqueeze(1) + CONFIG['gamma'] * target_q * done_mask
    
    # Update Q-network
    current_q = q(s, explicit_a_tensor)
    q_loss = F.mse_loss(current_q, target)
    
    q_optimizer.zero_grad()
    q_loss.backward()
    q_optimizer.step()
    
    # Update policy network
    # Get implicit actions from policy
    implicit_actions = policy(s)
    
    # Decode to explicit actions
    job_logits, robot_logits, param = vae.decode(implicit_actions, s)
    job_probs = F.softmax(job_logits, dim=-1)
    robot_probs = F.softmax(robot_logits, dim=-1)
    explicit_actions = torch.cat([job_probs, robot_probs, param], dim=1)
    
    # Compute policy loss
    policy_q = q(s, explicit_actions)
    policy_loss = -policy_q.mean()
    
    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_optimizer.step()


def soft_update(net, net_target):
    """Soft update target network"""
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * (1.0 - CONFIG['tau']) + param.data * CONFIG['tau'])


def main(config=None):
    """Main training function for HyAR"""
    if config is None:
        config = CONFIG
    
    # Environment setup
    env = ENV.Env(config['num_of_jobs'], config['num_of_robots'], config['alpha'], config['beta'])
    state_size = len(env.state)
    implicit_action_size = 3  # Same as DDPG output
    explicit_action_size = config['num_of_jobs'] + config['num_of_robots'] + 1
    
    # ========== Stage 1: Train VAE ==========
    print(f"\n{'='*60}")
    print(f"Starting HyAR Training - Stage 1: VAE Training")
    print(f"{'='*60}")
    
    # Collect samples
    samples = collect_samples_stage1(env, config['stage1_episodes'])
    
    # Initialize and train VAE
    vae = ConditionalVAE(state_size, implicit_action_size, explicit_action_size)
    vae = train_vae_stage1(vae, samples, config['vae_train_epochs'])
    vae.eval()  # Set to evaluation mode
    
    # ========== Stage 2: RL Training ==========
    print(f"\n{'='*60}")
    print(f"Starting HyAR Training - Stage 2: RL Training with Action Masking")
    print(f"{'='*60}")
    
    # Initialize networks
    policy = ImplicitPolicyNet(state_size, implicit_action_size)
    policy_target = ImplicitPolicyNet(state_size, implicit_action_size)
    policy_target.load_state_dict(policy.state_dict())
    
    q = QNet(state_size, explicit_action_size)
    q_target = QNet(state_size, explicit_action_size)
    q_target.load_state_dict(q.state_dict())
    
    # Optimizers
    policy_optimizer = optim.Adam(policy.parameters(), lr=config['lr_policy'])
    q_optimizer = optim.Adam(q.parameters(), lr=config['lr_q'])
    
    # Training components
    memory = HybridReplayBuffer()
    
    # Training variables
    score_record = []
    action_record = []
    
    print(f"\nStarting Stage 2 RL training:")
    print(f"Jobs: {config['num_of_jobs']}, Robots: {config['num_of_robots']}")
    print(f"Episodes: {config['num_of_episodes']}")
    
    for n_epi in range(config['num_of_episodes']):
        s = env.reset()
        done = False
        episode_reward = 0
        episode_actions = []
        step_count = 0
        
        while not done and step_count < config['max_steps_per_episode']:
            # Get state tensor
            state_tensor = torch.FloatTensor(s).unsqueeze(0)
            
            # Get implicit action from policy
            with torch.no_grad():
                implicit_action = policy(state_tensor)
                
                # Decode to explicit action with masking
                explicit_action = decode_action(vae, implicit_action, state_tensor, env)
            
            episode_actions.append(explicit_action)
            
            # Execute action
            s_prime, r, done = env.step(explicit_action)
            
            # Create explicit action tensor for storage
            job_id, robot_id, param = explicit_action
            job_one_hot = torch.zeros(config['num_of_jobs'])
            robot_one_hot = torch.zeros(config['num_of_robots'])
            job_one_hot[job_id] = 1
            robot_one_hot[robot_id] = 1
            explicit_tensor = torch.cat([job_one_hot, robot_one_hot, torch.tensor([param])]).numpy()
            
            # Store transition
            memory.put((s, implicit_action.squeeze().numpy(), explicit_tensor, r, s_prime, done))
            
            s = s_prime
            episode_reward += r
            step_count += 1
            
            # Optional: Real-time Gantt chart plotting
            if config['enable_gantt_plots'] and n_epi % 10 == 0:
                try:
                    env.render(f"HyAR Stage 2 Training - Episode {n_epi}")
                except:
                    pass
        
        # Check if terminated due to max steps
        if step_count >= config['max_steps_per_episode'] and not done:
            print(f"Episode {n_epi}: Reached max steps ({config['max_steps_per_episode']})")
        
        score_record.append(episode_reward)
        action_record.append(episode_actions)
        
        # Training update
        if memory.size() > config['memory_threshold']:
            for _ in range(config['training_iterations']):
                train_hyar_stage2(policy, policy_target, q, q_target, vae,
                                 memory, policy_optimizer, q_optimizer)
                
                # Soft update target networks
                soft_update(policy, policy_target)
                soft_update(q, q_target)
        
        if n_epi % config['print_interval'] == 0 and n_epi != 0:
            print(f"Episode {n_epi}: Average Score = {np.mean(score_record[-config['print_interval']:]):.4f}")
    
    return score_record, action_record, [policy, q, vae, policy_target, q_target]


def test_and_visualize(config=None, model_paths=None):
    """Test trained HyAR model with action masking"""
    if config is None:
        config = CONFIG
        
    print("\n=== Starting HyAR Testing and Visualization ===")
    
    # Create environment
    env = ENV.Env(config['num_of_jobs'], config['num_of_robots'], config['alpha'], config['beta'])
    state_size = len(env.state)
    implicit_action_size = 3
    explicit_action_size = config['num_of_jobs'] + config['num_of_robots'] + 1
    
    # Load trained models
    policy = ImplicitPolicyNet(state_size, implicit_action_size)
    vae = ConditionalVAE(state_size, implicit_action_size, explicit_action_size)
    
    if model_paths and len(model_paths) >= 3:
        try:
            policy.load_state_dict(torch.load(model_paths[0]))
            vae.load_state_dict(torch.load(model_paths[2]))
            policy.eval()
            vae.eval()
            print(f"Successfully loaded models")
        except Exception as e:
            print(f"Error loading models: {e}")
            print("Using randomly initialized models")
    else:
        print("Warning: Model paths not provided, using random initialization")
    
    # Reset environment
    s = env.reset()
    done = False
    step = 0
    total_reward = 0
    
    print(f"\nStarting scheduling - Jobs: {config['num_of_jobs']}, Robots: {config['num_of_robots']}")
    print(f"Action masking enabled for valid jobs and robots")
    print("-" * 50)
    
    # Execute scheduling process
    while not done and step < config['max_test_steps']:
        # Get state tensor
        state_tensor = torch.FloatTensor(s).unsqueeze(0)
        
        # Get action with masking
        with torch.no_grad():
            implicit_action = policy(state_tensor)
            action = decode_action(vae, implicit_action, state_tensor, env)
        
        # Get current masks for display
        job_mask, robot_mask = get_valid_action_mask(env)
        valid_jobs = [i for i, m in enumerate(job_mask) if m > 0]
        valid_robots = [i for i, m in enumerate(robot_mask) if m > 0]
        
        print(f"Step {step+1}: Job{action[0]}, Robot{action[1]}, Param{action[2]:.3f}")
        print(f"  Valid jobs: {valid_jobs[:10]}{'...' if len(valid_jobs) > 10 else ''}")
        print(f"  Valid robots: {valid_robots}")
        
        # Execute action
        s_prime, reward, done = env.step(action)
        
        print(f"  Reward: {reward:.4f}")
        total_reward += reward
        s = s_prime
        step += 1
        
        # Real-time Gantt chart update
        if config['enable_gantt_plots']:
            try:
                env.render(f"HyAR Scheduling - Step {step}")
                print(f"  Gantt chart updated")
            except Exception as e:
                print(f"  Error plotting Gantt chart: {e}")
        
        if done:
            print(f"\nAll tasks completed! Total steps: {step}")
            break
    
    # Print final results
    print(f"\n=== Scheduling Results Summary ===")
    print(f"Total steps: {step}")
    print(f"Total reward: {total_reward:.4f}")
    print(f"Average reward: {total_reward/step:.4f}")
    print(f"Makespan: {env.future_time:.2f}")
    
    # Final Gantt chart
    print(f"\n=== Drawing Final Gantt Chart ===")
    try:
        env.render(f"HyAR Final Results (Jobs:{config['num_of_jobs']}, Robots:{config['num_of_robots']})")
        print("Final Gantt chart generated successfully!")
    except Exception as e:
        print(f"Error generating final Gantt chart: {e}")
    
    return total_reward, step


def multi_run_training(config=None):
    """Execute multiple training runs with different seeds"""
    if config is None:
        config = CONFIG
    
    all_score_records = []
    all_action_records = []
    all_models = []
    
    print(f"\n{'='*80}")
    print(f"Starting Multi-Run HyAR Training")
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
        score_record, action_record, models = main(config)
        
        # Store results
        all_score_records.append(score_record)
        all_action_records.append(action_record)
        all_models.append(models)
        
        print(f"Run {run_idx + 1} completed - Final Score: {score_record[-1]:.4f}")
        
    print(f"\n{'='*60}")
    print(f"All {len(config['seeds'])} runs completed!")
    print(f"{'='*60}")
    
    return all_score_records, all_action_records, all_models


def save_results(score_records, action_records, models_restore, config):
    """Save training results and models"""
    if not config['save_models']:
        return None, None
        
    # Create save directory with timestamp
    alg_name = "hyar_vae"
    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Check if multi-run or single-run
    if len(score_records) > 1:  # Multi-run
        save_dir = f"{alg_name}_multi_run_{now_str}"
    else:  # Single run
        save_dir = f"{alg_name}_{now_str}"
    os.makedirs(save_dir, exist_ok=True)

    if len(score_records) > 1:  # Multi-run save
        # Save training data for all runs
        np.save(os.path.join(save_dir, f"{alg_name}_all_scores_{now_str}.npy"), score_records)
        np.save(os.path.join(save_dir, f"{alg_name}_all_actions_{now_str}.npy"), action_records)

        # Save models from all runs
        model_paths = []
        for run_idx, models in enumerate(models_restore):
            policy_net, q_net, vae, policy_target, q_target = models
            run_save_dir = os.path.join(save_dir, f"run_{run_idx+1}_seed_{config['seeds'][run_idx]}")
            os.makedirs(run_save_dir, exist_ok=True)
            
            policy_path = os.path.join(run_save_dir, f"{alg_name}_policy_{now_str}.pth")
            torch.save(policy_net.state_dict(), policy_path)
            torch.save(q_net.state_dict(), os.path.join(run_save_dir, f"{alg_name}_q_{now_str}.pth"))
            torch.save(vae.state_dict(), os.path.join(run_save_dir, f"{alg_name}_vae_{now_str}.pth"))
            torch.save(policy_target.state_dict(), os.path.join(run_save_dir, f"{alg_name}_policy_target_{now_str}.pth"))
            torch.save(q_target.state_dict(), os.path.join(run_save_dir, f"{alg_name}_q_target_{now_str}.pth"))
            
            model_paths.append([policy_path, 
                              os.path.join(run_save_dir, f"{alg_name}_q_{now_str}.pth"),
                              os.path.join(run_save_dir, f"{alg_name}_vae_{now_str}.pth")])
        
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
        
        # Find best run
        best_run_idx = np.argmax([scores[-1] for scores in score_records])
        return save_dir, model_paths[best_run_idx]
        
    else:  # Single run save
        # Save training data
        np.save(os.path.join(save_dir, f"{alg_name}_scores_{now_str}.npy"), score_records[0])
        np.save(os.path.join(save_dir, f"{alg_name}_actions_{now_str}.npy"), action_records[0])

        # Save models
        models = models_restore[0]
        policy_net, q_net, vae, policy_target, q_target = models
        policy_path = os.path.join(save_dir, f"{alg_name}_policy_{now_str}.pth")
        q_path = os.path.join(save_dir, f"{alg_name}_q_{now_str}.pth")
        vae_path = os.path.join(save_dir, f"{alg_name}_vae_{now_str}.pth")
        
        torch.save(policy_net.state_dict(), policy_path)
        torch.save(q_net.state_dict(), q_path)
        torch.save(vae.state_dict(), vae_path)
        torch.save(policy_target.state_dict(), os.path.join(save_dir, f"{alg_name}_policy_target_{now_str}.pth"))
        torch.save(q_target.state_dict(), os.path.join(save_dir, f"{alg_name}_q_target_{now_str}.pth"))
        
        print(f"Results saved to directory: {save_dir}")
        return save_dir, [policy_path, q_path, vae_path]


def plot_multi_run_training_curves(all_score_records, config=None, save_path=None):
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
    
    plt.title(f'HyAR Multi-Run Training Curves (Jobs={config["num_of_jobs"]}, Robots={config["num_of_robots"]})')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Training curves saved to: {save_path}")
    
    if config['plot_training_curve']:
        plt.show()
    else:
        plt.close()


def plot_single_run_training_curve(score_record, config=None, save_path=None):
    """Plot training curve for a single run"""
    if config is None:
        config = CONFIG
        
    plt.figure(figsize=(10, 6))
    
    x = range(len(score_record))
    plt.plot(x, score_record, 'b-', linewidth=1, alpha=0.8)
    
    # Add moving average
    window_size = min(50, len(score_record) // 10)
    if window_size > 1:
        moving_avg = np.convolve(score_record, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(score_record)), moving_avg, 'r-', linewidth=2, label=f'Moving Avg ({window_size} eps)')
    
    plt.title(f'HyAR Training Curve (Jobs={config["num_of_jobs"]}, Robots={config["num_of_robots"]})')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Training curve saved to: {save_path}")
    
    if config['plot_training_curve']:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='HyAR Algorithm for Robot Task Scheduling')
    parser.add_argument('--jobs', type=int, default=CONFIG['num_of_jobs'], help='Number of jobs')
    parser.add_argument('--robots', type=int, default=CONFIG['num_of_robots'], help='Number of robots')
    parser.add_argument('--alpha', type=float, default=CONFIG['alpha'], help='Alpha parameter')
    parser.add_argument('--beta', type=float, default=CONFIG['beta'], help='Beta parameter')
    parser.add_argument('--episodes', type=int, default=CONFIG['num_of_episodes'], help='Number of episodes')
    parser.add_argument('--max-steps', type=int, default=CONFIG['max_steps_per_episode'], 
                       help='Maximum steps per episode during training')
    parser.add_argument('--test-only', action='store_true', help='Run test only (skip training)')
    parser.add_argument('--model-paths', nargs='+', type=str, default=None, help='Paths to saved models')
    parser.add_argument('--multi-run', action='store_true', default=CONFIG['enable_multi_run'], 
                       help='Run multiple training sessions')
    parser.add_argument('--single-run', action='store_true', help='Force single run training')
    
    args = parser.parse_args()
    
    # Update CONFIG
    config = CONFIG.copy()
    config.update({
        'num_of_jobs': args.jobs,
        'num_of_robots': args.robots,
        'alpha': args.alpha,
        'beta': args.beta,
        'num_of_episodes': args.episodes,
        'max_steps_per_episode': args.max_steps,
        'enable_multi_run': args.multi_run and not args.single_run
    })
    
    print(f"\n{'='*60}")
    print(f"HyAR Algorithm for Robot Task Scheduling")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  Jobs: {config['num_of_jobs']}")
    print(f"  Robots: {config['num_of_robots']}")
    print(f"  Alpha: {config['alpha']}")
    print(f"  Beta: {config['beta']}")
    print(f"  Episodes: {config['num_of_episodes']}")
    print(f"  Max steps per episode: {config['max_steps_per_episode']}")
    print(f"  Multi-run mode: {config['enable_multi_run']}")
    print(f"{'='*60}")
    
    if args.test_only:
        # Test only mode
        test_and_visualize(config, args.model_paths)
    elif config['enable_multi_run']:
        # Multi-run training mode
        all_score_records, all_action_records, all_models = multi_run_training(config)
        
        # Save results
        save_dir, model_paths = None, None
        if config['save_models']:
            save_dir, model_paths = save_results(all_score_records, all_action_records, all_models, config)
            
            # Plot and save multi-run training curves
            if save_dir:
                plot_path = os.path.join(save_dir, "training_curves.png")
                plot_multi_run_training_curves(all_score_records, config, plot_path)
        
        # Test with the best performing model
        if model_paths:
            best_run_idx = np.argmax([scores[-1] for scores in all_score_records])
            print(f"\n{'='*40}")
            print(f"Testing with best model (Run {best_run_idx+1}, Final score: {all_score_records[best_run_idx][-1]:.4f})...")
            print(f"{'='*40}")
            test_and_visualize(config, model_paths)
    else:
        # Single run training mode
        if config['seeds']:
            seed = config['seeds'][0]
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            print(f"Using random seed: {seed}")
        
        score_record, action_record, models = main(config)
        
        # Save results if enabled
        save_dir, model_paths = save_results([score_record], [action_record], [models], config)
        
        # Plot and save single run training curve
        if save_dir:
            plot_path = os.path.join(save_dir, "training_curve.png")
            plot_single_run_training_curve(score_record, config, plot_path)
        
        # Test with trained model
        if model_paths:
            print(f"\n{'='*40}")
            print(f"Testing with trained model (Final score: {score_record[-1]:.4f})...")
            print(f"{'='*40}")
            test_and_visualize(config, model_paths)
