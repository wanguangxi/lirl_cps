"""
PDQN (Parameterized Deep Q-Network) for Protein Crystallization Screening
Based on the structure of lirl.py

PDQN combines:
- DQN for discrete action selection (protocol selection)
- Continuous parameter networks for action parameters (composition, temperature, time)

KEY DIFFERENCE FROM LIRL:
- PDQN has NO action correction/projection mechanism
- Agent learns valid actions purely through constraint penalties (negative rewards)
- Invalid actions receive penalties based on constraint violations
- Over time, the agent learns to avoid constraint-violating actions
"""

import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import os
import datetime
import matplotlib.pyplot as plt
import json

# Add environment path
sys.path.append(os.path.join(os.path.dirname(__file__), '../env'))
from cced_crystallization_env import make_protein_crystallization_spec, ProteinCrystallizationBaseEnv

# =======================
# DEVICE CONFIGURATION
# =======================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =======================
# HYPERPARAMETERS CONFIG
# =======================
CONFIG = {
    # Learning parameters
    'lr_q': 0.001,              # Q-network learning rate
    'lr_param': 0.0005,         # Parameter network learning rate
    'gamma': 0.98,              # Discount factor
    'batch_size': 128,          # Batch size for training
    'buffer_limit': 100000,     # Replay buffer size
    'tau': 0.005,               # Soft update coefficient
    
    # Environment parameters
    'batch_size_env': 2,        # Droplets per step
    'horizon': 25,              # Episode length
    'seed': 42,
    
    # Network architecture
    'q_hidden_dim1': 256,       # Q-network first hidden layer
    'q_hidden_dim2': 128,       # Q-network second hidden layer
    'param_hidden_dim1': 256,   # Parameter network first hidden layer
    'param_hidden_dim2': 128,   # Parameter network second hidden layer
    
    # Training parameters
    'num_of_episodes': 500,
    'memory_threshold': 1000,   # Minimum buffer size before training
    'training_iterations': 10,  # Training iterations per step
    'epsilon_start': 1.0,       # Initial exploration rate
    'epsilon_end': 0.01,        # Final exploration rate
    'epsilon_decay': 0.995,     # Exploration decay rate
    'print_interval': 10,
    
    # Constraint penalty (for learning from violations)
    'constraint_penalty': 0.5,  # Penalty for constraint violations
    
    # Output parameters
    'plot_training_curve': True,
    'save_models': True,
}


class ReplayBuffer:
    """Experience replay buffer for PDQN"""
    def __init__(self, device):
        self.buffer = collections.deque(maxlen=CONFIG['buffer_limit'])
        self.device = device

    def put(self, transition):
        """Store a transition in the buffer"""
        self.buffer.append(transition)
        
    def sample(self, n):
        """Sample a batch of transitions"""
        mini_batch = random.sample(self.buffer, n)
        s_lst, discrete_a_lst, param_a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], [], []

        for transition in mini_batch:
            s, discrete_a, param_a, r, s_prime, done = transition
            s_lst.append(s)
            discrete_a_lst.append(discrete_a)
            param_a_lst.append(param_a)
            r_lst.append(r)
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0 
            done_mask_lst.append([done_mask])
        
        # Move tensors to device
        s_tensor = torch.FloatTensor(np.array(s_lst)).to(self.device)
        discrete_a_tensor = torch.LongTensor(np.array(discrete_a_lst)).to(self.device)
        param_a_tensor = torch.FloatTensor(np.array(param_a_lst)).to(self.device)
        r_tensor = torch.FloatTensor(np.array(r_lst)).to(self.device)
        s_prime_tensor = torch.FloatTensor(np.array(s_prime_lst)).to(self.device)
        done_mask_tensor = torch.FloatTensor(np.array(done_mask_lst)).to(self.device)

        return s_tensor, discrete_a_tensor, param_a_tensor, r_tensor, s_prime_tensor, done_mask_tensor
    
    def size(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    """
    Q-Network for PDQN
    Takes state and continuous parameters as input
    Outputs Q-values for each discrete action (protocol for each droplet)
    """
    def __init__(self, state_size, param_size, n_discrete_actions):
        super(QNetwork, self).__init__()
        self.n_discrete_actions = n_discrete_actions
        
        # State encoder
        self.fc_s = nn.Linear(state_size, CONFIG['q_hidden_dim1'])
        
        # Parameter encoder
        self.fc_param = nn.Linear(param_size, CONFIG['q_hidden_dim2'])
        
        # Combined layers
        self.fc_combined = nn.Linear(CONFIG['q_hidden_dim1'] + CONFIG['q_hidden_dim2'], CONFIG['q_hidden_dim2'])
        
        # Output layer - Q-values for each discrete action
        self.fc_out = nn.Linear(CONFIG['q_hidden_dim2'], n_discrete_actions)

    def forward(self, state, params):
        """
        Forward pass
        Args:
            state: [batch_size, state_size]
            params: [batch_size, param_size] - continuous action parameters
        Returns:
            q_values: [batch_size, n_discrete_actions]
        """
        h_s = F.relu(self.fc_s(state))
        h_param = F.relu(self.fc_param(params))
        h_combined = torch.cat([h_s, h_param], dim=1)
        h = F.relu(self.fc_combined(h_combined))
        q_values = self.fc_out(h)
        return q_values


class ParameterNetwork(nn.Module):
    """
    Parameter Network for PDQN
    Generates continuous action parameters (composition, temperature, time)
    """
    def __init__(self, state_size, param_size):
        super(ParameterNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_size, CONFIG['param_hidden_dim1'])
        self.fc2 = nn.Linear(CONFIG['param_hidden_dim1'], CONFIG['param_hidden_dim2'])
        self.fc_out = nn.Linear(CONFIG['param_hidden_dim2'], param_size)

    def forward(self, state):
        """
        Forward pass
        Args:
            state: [batch_size, state_size]
        Returns:
            params: [batch_size, param_size] - parameters in [0, 1]
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        params = torch.sigmoid(self.fc_out(x))  # Output in [0, 1]
        return params


class OrnsteinUhlenbeckNoise:
    """Ornstein-Uhlenbeck noise for exploration in continuous parameter space"""
    def __init__(self, mu, theta=0.15, dt=0.05, sigma=0.2):
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
    
    def reset(self):
        self.x_prev = np.zeros_like(self.mu)


def check_feasible(spec, k, u) -> tuple:
    """
    Check if an action satisfies all constraints
    Returns: (is_feasible, violation_count, violation_details)
    """
    R = spec.R
    p = u[:R]
    T = float(u[R])
    tau = float(u[R + 1])
    
    violations = []
    
    # Check component bounds
    if np.any(p < 0):
        violations.append("negative_component")
    if np.any(p > spec.p_max):
        violations.append("component_exceeds_max")
    
    # Check simplex constraint
    if abs(float(np.sum(p)) - 1.0) > 0.01:
        violations.append("simplex_violation")
    
    # Check T bounds
    if not (spec.T_bounds[0] <= T <= spec.T_bounds[1]):
        violations.append("temperature_out_of_bounds")
    
    # Check tau bounds
    if not (spec.tau_bounds[0] <= tau <= spec.tau_bounds[1]):
        violations.append("time_out_of_bounds")
    
    # Check protocol-specific constraints
    proto = spec.protocols[int(k)]
    if proto.G is not None and proto.G.size > 0:
        constraint_violations = proto.G @ u - proto.h
        if np.any(constraint_violations > 1e-3):
            violations.append("protocol_constraint_violation")
    
    return len(violations) == 0, len(violations), violations


class PDQNAgent:
    """
    PDQN Agent for Protein Crystallization Screening
    Combines discrete protocol selection with continuous parameter optimization
    """
    def __init__(self, state_size, n_protocols, batch_size_env, param_size_per_droplet, config, device):
        self.state_size = state_size
        self.n_protocols = n_protocols          # K protocols
        self.batch_size_env = batch_size_env    # B droplets
        self.param_size_per_droplet = param_size_per_droplet  # d = R + 2
        self.config = config
        self.device = device
        
        # Total discrete actions: one protocol per droplet = K^B or simplified to K*B
        # Simplified: treat as B independent choices, each from K protocols
        self.n_discrete_actions = n_protocols * batch_size_env
        
        # Total continuous parameters: B * d
        self.param_size = batch_size_env * param_size_per_droplet
        
        # Q-Network and target
        self.q_network = QNetwork(state_size, self.param_size, self.n_discrete_actions).to(device)
        self.q_target = QNetwork(state_size, self.param_size, self.n_discrete_actions).to(device)
        self.q_target.load_state_dict(self.q_network.state_dict())
        
        # Parameter Network and target
        self.param_network = ParameterNetwork(state_size, self.param_size).to(device)
        self.param_target = ParameterNetwork(state_size, self.param_size).to(device)
        self.param_target.load_state_dict(self.param_network.state_dict())
        
        # Optimizers
        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=config['lr_q'])
        self.param_optimizer = optim.Adam(self.param_network.parameters(), lr=config['lr_param'])
        
        # Exploration
        self.epsilon = config['epsilon_start']
        self.ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(self.param_size))
        
    def select_action(self, state, spec, training=True):
        """
        Select action using epsilon-greedy for discrete and OU noise for continuous
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Get continuous parameters
            params = self.param_network(state_tensor)
            
            if training:
                # Add OU noise to parameters
                noise = torch.from_numpy(self.ou_noise()).float().to(self.device)
                noise_scale = max(0.1, self.epsilon)
                params = params + noise * noise_scale
                params = torch.clamp(params, 0, 1)
            
            # Epsilon-greedy for discrete action selection
            if training and random.random() < self.epsilon:
                # Random protocol for each droplet
                discrete_action = random.randint(0, self.n_discrete_actions - 1)
            else:
                # Select best discrete action using Q-network
                q_values = self.q_network(state_tensor, params)
                discrete_action = q_values.argmax(dim=1).item()
        
        params_np = params.squeeze().cpu().numpy()
        
        # Convert to environment action format
        k_vec, u_mat = self._convert_to_env_action(discrete_action, params_np, spec)
        
        return discrete_action, params_np, k_vec, u_mat
    
    def _convert_to_env_action(self, discrete_action, params, spec):
        """
        Convert discrete action index and continuous parameters to environment action
        
        PDQN Philosophy: 
        - NO action correction/projection
        - Agent learns valid actions through constraint penalties
        """
        B = self.batch_size_env
        K = self.n_protocols
        d = self.param_size_per_droplet
        R = spec.R
        
        # Decode discrete action to protocol indices for each droplet
        # discrete_action encodes: droplet_idx * K + protocol_idx
        k_vec = np.zeros(B, dtype=int)
        for j in range(B):
            # Extract protocol for droplet j
            k_vec[j] = (discrete_action // (K ** j)) % K
        
        # Reshape continuous parameters to (B, d)
        u_mat = params.reshape(B, d)
        
        # Scale parameters to appropriate ranges (NO PROJECTION, just scaling)
        u_mat_scaled = np.zeros_like(u_mat)
        for j in range(B):
            # Component fractions: scale from [0,1] to [0, p_max]
            # Then normalize to sum to 1 (soft constraint, not hard projection)
            p = u_mat[j, :R] * spec.p_max
            p_sum = np.sum(p)
            if p_sum > 0:
                p = p / p_sum  # Normalize to simplex
            else:
                p = np.ones(R) / R  # Default equal distribution
            u_mat_scaled[j, :R] = p
            
            # Temperature: scale from [0,1] to T_bounds
            u_mat_scaled[j, R] = spec.T_bounds[0] + u_mat[j, R] * (spec.T_bounds[1] - spec.T_bounds[0])
            
            # Time: scale from [0,1] to tau_bounds
            u_mat_scaled[j, R+1] = spec.tau_bounds[0] + u_mat[j, R+1] * (spec.tau_bounds[1] - spec.tau_bounds[0])
        
        return k_vec, u_mat_scaled
    
    def train(self, memory):
        """Train Q-network and Parameter network"""
        if memory.size() < self.config['memory_threshold']:
            return None
        
        # Sample batch
        s, discrete_a, param_a, r, s_prime, done_mask = memory.sample(self.config['batch_size'])
        
        # ===== Update Q-Network =====
        with torch.no_grad():
            # Get target parameters for next states
            next_params = self.param_target(s_prime)
            
            # Get target Q-values
            next_q_values = self.q_target(s_prime, next_params)
            max_next_q = next_q_values.max(dim=1, keepdim=True)[0]
            
            # Compute target
            target = r.unsqueeze(1) + self.config['gamma'] * max_next_q * done_mask
        
        # Current Q-values
        current_q_all = self.q_network(s, param_a)
        current_q = current_q_all.gather(1, discrete_a.unsqueeze(1))
        
        # Q-network loss
        q_loss = F.smooth_l1_loss(current_q, target)
        
        self.q_optimizer.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.q_optimizer.step()
        
        # ===== Update Parameter Network =====
        predicted_params = self.param_network(s)
        q_values_for_params = self.q_network(s, predicted_params)
        param_loss = -q_values_for_params.max(dim=1)[0].mean()
        
        self.param_optimizer.zero_grad()
        param_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.param_network.parameters(), 1.0)
        self.param_optimizer.step()
        
        return {
            'q_loss': q_loss.item(),
            'param_loss': param_loss.item(),
            'avg_q_value': current_q.mean().item(),
        }
    
    def soft_update(self):
        """Soft update target networks"""
        tau = self.config['tau']
        for param_target, param in zip(self.q_target.parameters(), self.q_network.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)
        for param_target, param in zip(self.param_target.parameters(), self.param_network.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(
            self.config['epsilon_end'],
            self.epsilon * self.config['epsilon_decay']
        )


def main(config=None):
    """Main training function for PDQN"""
    if config is None:
        config = CONFIG
    
    # Print device information
    print(f"Using device: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Environment setup
    spec = make_protein_crystallization_spec(
        seed=config['seed'], 
        batch_size=config['batch_size_env'], 
        horizon=config['horizon']
    )
    env = ProteinCrystallizationBaseEnv(spec)
    
    state_size = env.observation_space.shape[0]
    n_protocols = spec.K
    batch_size_env = spec.batch_size
    param_size_per_droplet = spec.R + 2  # d = R + 2
    
    # Initialize agent
    agent = PDQNAgent(
        state_size, n_protocols, batch_size_env, 
        param_size_per_droplet, config, DEVICE
    )
    
    # Training components
    memory = ReplayBuffer(DEVICE)
    score_record = []
    best_quality_record = []
    
    # Constraint violation tracking
    total_violations = 0
    total_steps = 0
    
    print(f"\nStarting PDQN training for Protein Crystallization:")
    print(f"Protocols: {n_protocols}, Droplets/step: {batch_size_env}, Horizon: {config['horizon']}")
    print(f"Episodes: {config['num_of_episodes']}")
    print(f"State size: {state_size}, Discrete actions: {agent.n_discrete_actions}, Param size: {agent.param_size}")
    print(f"NOTE: PDQN does NOT use action projection - learns from constraint penalties")

    for n_epi in range(config['num_of_episodes']):
        s, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0.0
        episode_violations = 0
        agent.ou_noise.reset()
        
        while not (done or truncated):
            # Select action
            discrete_a, param_a, k_vec, u_mat = agent.select_action(s, spec, training=True)
            
            # Check constraint violations (for monitoring, not for projection)
            step_violations = 0
            for j in range(batch_size_env):
                is_feasible, viol_count, _ = check_feasible(spec, k_vec[j], u_mat[j])
                if not is_feasible:
                    step_violations += viol_count
            
            episode_violations += step_violations
            total_violations += step_violations
            total_steps += batch_size_env
            
            # Execute action (with potential constraint violations)
            action = {"k": k_vec, "u": u_mat}
            s_prime, r, done, truncated, info = env.step(action)
            
            # Apply constraint penalty if violations occurred
            if step_violations > 0:
                r = r - config['constraint_penalty'] * step_violations
            
            # Store experience
            memory.put((s, discrete_a, param_a, r, s_prime, done))
            
            s = s_prime
            episode_reward += r
        
        score_record.append(episode_reward)
        best_quality_record.append(info.get('best_quality', 0))
        
        # Training updates
        if memory.size() > config['memory_threshold']:
            for _ in range(config['training_iterations']):
                agent.train(memory)
                agent.soft_update()
        
        # Decay exploration
        agent.decay_epsilon()
        
        # Print progress
        if n_epi % config['print_interval'] == 0 and n_epi != 0:
            avg_score = np.mean(score_record[-config['print_interval']:])
            running_cvr = total_violations / max(1, total_steps)
            print(f"Episode {n_epi}: Avg Score = {avg_score:.4f}, "
                  f"Best Quality = {info.get('best_quality', 0):.4f}, "
                  f"Epi Violations = {episode_violations}, "
                  f"Running CVR = {running_cvr:.4f}, "
                  f"Epsilon = {agent.epsilon:.4f}")
    
    # Final statistics
    print(f"\n{'='*60}")
    print(f"PDQN Training Completed!")
    print(f"{'='*60}")
    print(f"Total constraint violations: {total_violations}")
    print(f"Total action steps: {total_steps}")
    print(f"Overall violation rate: {(total_violations / max(1, total_steps)):.4f}")
    print(f"{'='*60}")
    
    # Move models to CPU for saving
    agent.q_network.cpu()
    agent.q_target.cpu()
    agent.param_network.cpu()
    agent.param_target.cpu()
    
    return score_record, best_quality_record, agent, total_violations


def save_results(score_records, agent, config, total_violations):
    """Save training results and models"""
    if not config['save_models']:
        return
    
    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_base_dir = "/home/one/LIRL-CPS-main/Protein_Crystallization_Screening/exp"
    save_dir = os.path.join(exp_base_dir, f"pdqn_crystallization_{now_str}")
    os.makedirs(save_dir, exist_ok=True)

    # Save scores
    np.save(os.path.join(save_dir, "scores.npy"), score_records)
    
    # Save models
    torch.save(agent.q_network.state_dict(), os.path.join(save_dir, "q_network.pth"))
    torch.save(agent.param_network.state_dict(), os.path.join(save_dir, "param_network.pth"))
    
    # Save config
    config_to_save = config.copy()
    config_to_save['device'] = str(DEVICE)
    config_to_save['total_violations'] = total_violations
    with open(os.path.join(save_dir, "config.json"), 'w') as f:
        json.dump(config_to_save, f, indent=2)
    
    # Save training curve
    if config['plot_training_curve']:
        plt.figure(figsize=(10, 6))
        plt.plot(score_records, label='Episode Reward')
        window = min(20, len(score_records) // 5) if len(score_records) > 10 else 1
        if window > 1:
            moving_avg = np.convolve(score_records, np.ones(window)/window, mode='valid')
            plt.plot(range(window-1, len(score_records)), moving_avg, 'r-', linewidth=2, label=f'Moving Avg (window={window})')
        plt.title("PDQN Training Curve - Protein Crystallization (No Projection)")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, "training_curve.png"), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Training curve saved to: {os.path.join(save_dir, 'training_curve.png')}")
    
    print(f"Results saved to: {save_dir}")


if __name__ == "__main__":
    score_record, agent, total_violations = main(CONFIG)
    save_results(score_record, agent, CONFIG, total_violations)
