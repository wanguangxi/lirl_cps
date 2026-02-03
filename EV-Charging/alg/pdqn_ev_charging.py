"""
PDQN (Parameterized Deep Q-Network) for EV Charging Station Control
Based on the structure of lirl.py (DDPG-LIRL)

PDQN combines:
- DQN for discrete action selection (which station, which vehicle)
- Continuous parameter networks for action parameters (charging power)

KEY DIFFERENCE FROM LIRL:
- PDQN has NO action correction/projection mechanism
- Agent learns valid actions purely through constraint penalties (negative rewards)
- Invalid actions (e.g., occupied station, non-existent vehicle) receive penalties
- Over time, the agent learns to avoid constraint-violating actions
- This is the core philosophy of PDQN: learning from environmental feedback
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
import sys
import os
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import pandas as pd
from datetime import datetime, timedelta
sys.path.append(os.path.join(os.path.dirname(__file__), '../env'))
from ev import EVChargingEnv

# =======================
# HYPERPARAMETERS CONFIG
# =======================
CONFIG = {
    # Learning parameters
    'lr_q': 0.001,              # Q-network learning rate
    'lr_param': 0.0001,         # Parameter network learning rate
    'gamma': 0.98,              # Discount factor
    'batch_size': 128,          # Batch size for training
    'buffer_limit': 1000000,    # Replay buffer size
    'tau': 0.005,               # Soft update coefficient
    
    # Environment parameters
    'n_stations': 5,            # Number of charging stations
    'p_max': 150.0,             # Maximum power per station
    'arrival_rate': 0.75,       # Vehicle arrival rate
    'num_of_episodes': 200,     # Number of training episodes
    
    # Network architecture
    'q_hidden_dim1': 128,       # Q-network first hidden layer
    'q_hidden_dim2': 64,        # Q-network second hidden layer
    'param_hidden_dim1': 64,    # Parameter network first hidden layer
    'param_hidden_dim2': 32,    # Parameter network second hidden layer
    
    # Training parameters
    'memory_threshold': 500,    # Minimum buffer size before training
    'training_iterations': 20,  # Training iterations per step
    'epsilon_start': 1.0,       # Initial exploration rate
    'epsilon_end': 0.01,        # Final exploration rate
    'epsilon_decay': 0.995,     # Exploration decay rate
    
    # Multi-run training parameters
    'enable_multi_run': True,
    'seeds': [3047, 294, 714, 1092, 1386, 2856, 42, 114514, 2025, 1993],
    'num_runs': 10,
    
    # Testing parameters
    'max_test_steps': 288,      # Maximum test steps (one day)
    
    # Output parameters
    'print_interval': 10,
    'enable_gantt_plots': False,
    'plot_training_curve': True,
    'save_models': True,
}


class ReplayBuffer:
    """Experience replay buffer for PDQN"""
    def __init__(self):
        self.buffer = collections.deque(maxlen=CONFIG['buffer_limit'])

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
        
        s_tensor = torch.FloatTensor(np.array(s_lst))
        discrete_a_tensor = torch.LongTensor(np.array(discrete_a_lst))
        param_a_tensor = torch.FloatTensor(np.array(param_a_lst))
        r_tensor = torch.FloatTensor(np.array(r_lst))
        s_prime_tensor = torch.FloatTensor(np.array(s_prime_lst))
        done_mask_tensor = torch.FloatTensor(np.array(done_mask_lst))

        return s_tensor, discrete_a_tensor, param_a_tensor, r_tensor, s_prime_tensor, done_mask_tensor
    
    def size(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    """
    Q-Network for PDQN
    Takes state and continuous parameters as input
    Outputs Q-values for each discrete action
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
    Generates continuous action parameters for each discrete action
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
            params: [batch_size, param_size] - normalized parameters in [0, 1]
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        params = torch.sigmoid(self.fc_out(x))  # Output in [0, 1]
        return params


class OrnsteinUhlenbeckNoise:
    """Ornstein-Uhlenbeck noise for exploration in continuous parameter space"""
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
    
    def reset(self):
        self.x_prev = np.zeros_like(self.mu)


class PDQNAgent:
    """
    PDQN Agent for EV Charging Control
    Combines discrete action selection with continuous parameter optimization
    """
    def __init__(self, state_size, n_discrete_actions, param_size, config):
        self.state_size = state_size
        self.n_discrete_actions = n_discrete_actions  # Number of station-vehicle combinations
        self.param_size = param_size  # Continuous parameters (e.g., power)
        self.config = config
        
        # Q-Network and target
        self.q_network = QNetwork(state_size, param_size, n_discrete_actions)
        self.q_target = QNetwork(state_size, param_size, n_discrete_actions)
        self.q_target.load_state_dict(self.q_network.state_dict())
        
        # Parameter Network and target
        self.param_network = ParameterNetwork(state_size, param_size)
        self.param_target = ParameterNetwork(state_size, param_size)
        self.param_target.load_state_dict(self.param_network.state_dict())
        
        # Optimizers
        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=config['lr_q'])
        self.param_optimizer = optim.Adam(self.param_network.parameters(), lr=config['lr_param'])
        
        # Exploration
        self.epsilon = config['epsilon_start']
        self.ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(param_size))
        
    def select_action(self, state, env, training=True):
        """
        Select action using epsilon-greedy for discrete and OU noise for continuous
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            # Get continuous parameters
            params = self.param_network(state_tensor)
            
            if training:
                # Add OU noise to parameters
                noise = torch.from_numpy(self.ou_noise()).float()
                noise_scale = max(0.1, 1.0 - self.epsilon)  # Decrease noise over time
                params = params + noise * noise_scale
                params = torch.clamp(params, 0, 1)
            
            # Epsilon-greedy for discrete action selection
            if training and random.random() < self.epsilon:
                discrete_action = random.randint(0, self.n_discrete_actions - 1)
            else:
                # Select best discrete action using Q-network
                q_values = self.q_network(state_tensor, params)
                discrete_action = q_values.argmax(dim=1).item()
        
        params_np = params.squeeze().numpy()
        
        # Ensure params_np is always 1D array
        if params_np.ndim == 0:
            params_np = np.array([params_np.item()])
        
        # Convert to environment action format
        env_action = self._convert_to_env_action(discrete_action, params_np, env)
        
        return discrete_action, params_np, env_action
    
    def _convert_to_env_action(self, discrete_action, params, env):
        """
        Convert discrete action index and continuous parameters to environment action
        
        PDQN Philosophy: 
        - NO action correction/projection (unlike LIRL)
        - Agent learns valid actions through constraint penalties
        - Discrete action directly maps to station-vehicle pair
        """
        # Decode discrete action to station and vehicle indices
        # discrete_action ranges from 0 to (n_stations * max_vehicles - 1)
        station_id = discrete_action % env.n_stations
        vehicle_id = discrete_action // env.n_stations
        
        # Ensure indices are within valid range
        vehicle_id = min(vehicle_id, env.max_vehicles - 1)
        
        # If selected vehicle doesn't exist, find any existing vehicle
        # (This handles the case where action space > actual vehicles)
        if env.vehicles[vehicle_id] is None:
            # Find first existing vehicle
            found = False
            for i in range(env.max_vehicles):
                if env.vehicles[i] is not None:
                    vehicle_id = i
                    found = True
                    break
            # If no vehicle exists at all, use vehicle_id 0 
            # (environment will handle constraint violation)
            if not found:
                vehicle_id = 0
        
        # Convert normalized power [0,1] to actual power [50, 150]
        # Handle both scalar and array params
        if isinstance(params, np.ndarray) and params.size > 0:
            power_normalized = np.clip(params.flat[0], 0, 1)
        elif np.isscalar(params):
            power_normalized = np.clip(float(params), 0, 1)
        else:
            power_normalized = 0.5
        power = 50.0 + power_normalized * 100.0
        power = np.clip(power, 50.0, min(150.0, env.p_max))
        
        # Return action directly - NO CORRECTION
        # If this action violates constraints, environment will:
        # 1. Return a negative reward (penalty)
        # 2. Set constraint_violation info
        # Agent learns to avoid such actions through these penalties
        return {
            'station_id': station_id,
            'vehicle_id': vehicle_id,
            'power': np.array([power], dtype=np.float32)
        }
    
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
        # Generate parameters
        predicted_params = self.param_network(s)
        
        # Get Q-values for predicted parameters
        q_values_for_params = self.q_network(s, predicted_params)
        
        # Maximize the maximum Q-value (equivalent to maximizing expected return)
        param_loss = -q_values_for_params.max(dim=1)[0].mean()
        
        self.param_optimizer.zero_grad()
        param_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.param_network.parameters(), 1.0)
        self.param_optimizer.step()
        
        return {
            'q_loss': q_loss.item(),
            'param_loss': param_loss.item(),
            'avg_q_value': current_q.mean().item(),
            'avg_target': target.mean().item()
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
    
    # Environment setup
    env = EVChargingEnv(
        n_stations=config['n_stations'],
        p_max=config['p_max'],
        arrival_rate=config['arrival_rate']
    )
    
    state_size = env.observation_space.shape[0]
    n_discrete_actions = config['n_stations'] * env.max_vehicles  # Station x Vehicle combinations
    param_size = 1  # Power parameter (normalized)
    
    # Initialize agent
    agent = PDQNAgent(state_size, n_discrete_actions, param_size, config)
    
    # Training components
    memory = ReplayBuffer()
    score_record = []
    action_restore = []
    
    # Episode statistics tracking
    episode_stats = {
        'fully_charged_vehicles': [],
        'cumulative_revenue': [],
        'energy_delivered': [],
        'average_power': [],
        'station_utilization': [],
        'total_arrivals': [],
        'vehicles_charged': [],
        'vehicles_left_uncharged': [],
        'charging_success_rate': []
    }
    
    # Constraint violation tracking
    constraint_violations = {
        'total_violations': 0,
        'episode_violations': [],
        'violation_rate': [],
        'violation_types': {},
        'violation_details': []
    }
    
    print(f"Starting PDQN training for EV Charging Station:")
    print(f"Stations: {config['n_stations']}, Max Power: {config['p_max']}kW")
    print(f"Episodes: {config['num_of_episodes']}")
    print(f"State size: {state_size}, Discrete actions: {n_discrete_actions}, Param size: {param_size}")
    
    for n_epi in range(config['num_of_episodes']):
        s = env.reset()
        done = False
        action_eps = []
        step = 0
        episode_reward = 0
        episode_violations = 0
        agent.ou_noise.reset()  # Reset noise at start of episode
        
        while not done:
            # Select action
            discrete_a, param_a, env_action = agent.select_action(s, env, training=True)
            
            # Execute action
            s_prime, r, done, info = env.step(env_action)
            
            # Track constraint violations
            violation_info = info.get('constraint_violation', None)
            if violation_info and violation_info['has_violation']:
                episode_violations += 1
                constraint_violations['total_violations'] += 1
                
                violation_type = violation_info['violation_type']
                if violation_type not in constraint_violations['violation_types']:
                    constraint_violations['violation_types'][violation_type] = 0
                constraint_violations['violation_types'][violation_type] += 1
                
                constraint_violations['violation_details'].append({
                    'episode': n_epi,
                    'step': step,
                    'violation_type': violation_type,
                    'violation_details': violation_info['violation_details'],
                    'attempted_action': violation_info['attempted_action'],
                    'reward': r
                })
                
                # print(f"Episode {n_epi}, Step {step+1} - {violation_type}: {violation_info['violation_details']}")
            
            # Store experience
            memory.put((s, discrete_a, param_a, r, s_prime, done))
            
            s = s_prime
            step += 1
            episode_reward += r
            action_eps.append({'discrete': discrete_a, 'param': param_a})
        
        action_restore.append(action_eps)
        score_record.append(episode_reward)
        
        # Training updates
        if memory.size() > config['memory_threshold']:
            for _ in range(config['training_iterations']):
                agent.train(memory)
                agent.soft_update()
        
        # Decay exploration
        agent.decay_epsilon()
        
        # Calculate episode statistics
        fully_charged_count = info.get('episode_charged_count', 0)
        total_arrivals = info.get('episode_arrivals', 1)
        
        # Calculate cumulative revenue
        cumulative_revenue = 0.0
        if hasattr(env, 'charging_records'):
            for record in env.charging_records:
                if record['start_step'] >= 0 and record['end_step'] <= env.current_step:
                    hour = (record['start_step'] * 5 // 60) % 24
                    price_multiplier = env._get_price_multiplier(hour)
                    revenue = record['energy'] * env.base_price * price_multiplier
                    
                    damage_delta = record.get('damage_delta') or record.get('lifetime_damage_delta')
                    if damage_delta is not None:
                        cumulative_revenue += revenue - damage_delta * 5
                    else:
                        cumulative_revenue += revenue
        
        if cumulative_revenue == 0:
            cumulative_revenue = max(0, info['total_energy'] * env.base_price - info['total_cost'])
        
        # Vehicle statistics
        current_vehicles_count = info['num_vehicles']
        episode_charged_vehicles = fully_charged_count
        episode_uncharged_left = max(0, total_arrivals - episode_charged_vehicles - current_vehicles_count)
        charging_success_rate = (episode_charged_vehicles / total_arrivals * 100) if total_arrivals > 0 else 0
        
        # Record statistics
        episode_stats['fully_charged_vehicles'].append(fully_charged_count)
        episode_stats['cumulative_revenue'].append(cumulative_revenue)
        episode_stats['energy_delivered'].append(info['total_energy'])
        episode_stats['total_arrivals'].append(total_arrivals)
        episode_stats['vehicles_charged'].append(episode_charged_vehicles)
        episode_stats['vehicles_left_uncharged'].append(episode_uncharged_left)
        episode_stats['charging_success_rate'].append(charging_success_rate)
        
        # Average power
        if hasattr(env, 'charging_records') and len(env.charging_records) > 0:
            avg_power = np.mean([record['power'] for record in env.charging_records])
        else:
            avg_power = 0.0
        episode_stats['average_power'].append(avg_power)
        
        # Station utilization
        total_possible_time = env.n_stations * env.current_step
        actual_usage_time = sum(1 for i in range(env.n_stations) if env.station_status[i] == 0) * env.current_step
        utilization_rate = (actual_usage_time / total_possible_time * 100) if total_possible_time > 0 else 0
        episode_stats['station_utilization'].append(utilization_rate)
        
        # Constraint violation stats
        constraint_violations['episode_violations'].append(episode_violations)
        violation_rate = episode_violations / step if step > 0 else 0
        constraint_violations['violation_rate'].append(violation_rate)
        
        # Print progress
        if n_epi % config['print_interval'] == 0 and n_epi != 0:
            avg_score = np.mean(score_record[-config['print_interval']:])
            avg_violations = np.mean(constraint_violations['episode_violations'][-config['print_interval']:])
            avg_violation_rate = np.mean(constraint_violations['violation_rate'][-config['print_interval']:]) * 100
            
            avg_charged_vehicles = np.mean(episode_stats['fully_charged_vehicles'][-config['print_interval']:])
            avg_revenue = np.mean(episode_stats['cumulative_revenue'][-config['print_interval']:])
            avg_energy = np.mean(episode_stats['energy_delivered'][-config['print_interval']:])
            avg_total_arrivals = np.mean(episode_stats['total_arrivals'][-config['print_interval']:])
            avg_uncharged_left = np.mean(episode_stats['vehicles_left_uncharged'][-config['print_interval']:])
            avg_success_rate = np.mean(episode_stats['charging_success_rate'][-config['print_interval']:])
            
            print(f"Episode {n_epi}: Average Score = {avg_score:.4f}, "
                  f"Vehicles: {info['num_vehicles']}, Energy: {info['total_energy']:.2f}kWh")
            print(f"  Performance - Charged Vehicles: {avg_charged_vehicles:.1f}, "
                  f"Revenue: {avg_revenue:.2f}, Avg Energy: {avg_energy:.2f}kWh")
            print(f"  Vehicle Flow - Total Arrivals: {avg_total_arrivals:.1f}, "
                  f"Charged: {avg_charged_vehicles:.1f}, Left Uncharged: {avg_uncharged_left:.1f}, "
                  f"Success Rate: {avg_success_rate:.1f}%")
            print(f"  Constraint Violations - Total: {constraint_violations['total_violations']}, "
                  f"Episode: {episode_violations}, Avg: {avg_violations:.1f}, Rate: {avg_violation_rate:.1f}%")
            print(f"  Epsilon: {agent.epsilon:.4f}")
    
    # Print final statistics
    total_steps = sum(len(actions) for actions in action_restore)
    final_violation_rate = (constraint_violations['total_violations'] / total_steps * 100) if total_steps > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"PDQN Training Completed!")
    print(f"{'='*60}")
    print(f"Total constraint violations: {constraint_violations['total_violations']}")
    print(f"Total steps taken: {total_steps}")
    print(f"Overall violation rate: {final_violation_rate:.2f}%")
    print(f"Average violations per episode: {np.mean(constraint_violations['episode_violations']):.2f}")
    print(f"Max violations in single episode: {np.max(constraint_violations['episode_violations'])}")
    print(f"Episodes with zero violations: {sum(1 for v in constraint_violations['episode_violations'] if v == 0)}")
    
    if constraint_violations['violation_types']:
        print(f"\nViolation Types Breakdown:")
        for violation_type, count in constraint_violations['violation_types'].items():
            percentage = (count / constraint_violations['total_violations']) * 100 if constraint_violations['total_violations'] > 0 else 0
            print(f"  {violation_type}: {count} ({percentage:.1f}%)")
    
    if episode_stats['fully_charged_vehicles']:
        print(f"\nEpisode Performance Statistics:")
        print(f"  Total vehicles charged: {sum(episode_stats['fully_charged_vehicles'])}")
        print(f"  Average vehicles per episode: {np.mean(episode_stats['fully_charged_vehicles']):.2f}")
        print(f"  Total cumulative revenue: {sum(episode_stats['cumulative_revenue']):.2f}")
        print(f"  Average revenue per episode: {np.mean(episode_stats['cumulative_revenue']):.2f}")
        print(f"  Average energy per episode: {np.mean(episode_stats['energy_delivered']):.2f} kWh")
        print(f"  Average station utilization: {np.mean(episode_stats['station_utilization']):.1f}%")
        
        print(f"\nVehicle Flow Statistics:")
        print(f"  Total vehicle arrivals: {sum(episode_stats['total_arrivals'])}")
        print(f"  Average arrivals per episode: {np.mean(episode_stats['total_arrivals']):.2f}")
        print(f"  Total vehicles charged: {sum(episode_stats['vehicles_charged'])}")
        print(f"  Total vehicles left uncharged: {sum(episode_stats['vehicles_left_uncharged'])}")
        print(f"  Overall charging success rate: {np.mean(episode_stats['charging_success_rate']):.1f}%")
        
        best_success_episode = np.argmax(episode_stats['charging_success_rate'])
        worst_success_episode = np.argmin(episode_stats['charging_success_rate'])
        print(f"  Best episode (Episode {best_success_episode}): "
              f"{episode_stats['total_arrivals'][best_success_episode]} arrivals, "
              f"{episode_stats['vehicles_charged'][best_success_episode]} charged, "
              f"{episode_stats['charging_success_rate'][best_success_episode]:.1f}% success rate")
        print(f"  Worst episode (Episode {worst_success_episode}): "
              f"{episode_stats['total_arrivals'][worst_success_episode]} arrivals, "
              f"{episode_stats['vehicles_charged'][worst_success_episode]} charged, "
              f"{episode_stats['charging_success_rate'][worst_success_episode]:.1f}% success rate")
    
    print(f"{'='*60}")
    
    return score_record, action_restore, agent, constraint_violations, episode_stats


def test_and_visualize(config=None, agent=None):
    """Test trained PDQN model and visualize results"""
    if config is None:
        config = CONFIG
    
    print("\n=== Starting PDQN Testing for EV Charging ===")
    
    # Create environment
    env = EVChargingEnv(
        n_stations=config['n_stations'],
        p_max=config['p_max'],
        arrival_rate=config['arrival_rate']
    )
    
    if agent is None:
        print("Warning: No agent provided, creating random agent")
        state_size = env.observation_space.shape[0]
        n_discrete_actions = config['n_stations'] * env.max_vehicles
        param_size = 1
        agent = PDQNAgent(state_size, n_discrete_actions, param_size, config)
    
    # Reset environment
    s = env.reset()
    done = False
    step = 0
    total_reward = 0
    
    print(f"\nStarting EV charging scheduling - Stations: {config['n_stations']}")
    print("-" * 50)
    
    while not done and step < config['max_test_steps']:
        # Select action without exploration
        discrete_a, param_a, env_action = agent.select_action(s, env, training=False)
        
        # Execute action
        s_prime, reward, done, info = env.step(env_action)
        
        violation_info = info.get('constraint_violation', None)
        if violation_info and violation_info['has_violation']:
            print(f"Step {step+1}: CONSTRAINT VIOLATION - {violation_info['violation_type']}")
        
        total_reward += reward
        s = s_prime
        step += 1
    
    print(f"\n=== Charging Station Results Summary ===")
    print(f"Total steps: {step}")
    print(f"Total reward: {total_reward:.4f}")
    print(f"Average reward per step: {total_reward/step:.4f}")
    print(f"Total energy delivered: {info['total_energy']:.2f} kWh")
    print(f"Total cost: {info['total_cost']:.2f}")
    print(f"Total lifetime damage: {info['total_lifetime_damage']:.4f}")
    
    return total_reward, step


def multi_run_training(config=None):
    """Execute multiple training runs with different seeds"""
    if config is None:
        config = CONFIG
    
    all_score_records = []
    all_action_restores = []
    all_agents = []
    all_constraint_violations = []
    all_episode_stats = []
    
    print(f"\n{'='*80}")
    print(f"Starting Multi-Run PDQN Training")
    print(f"Seeds: {config['seeds']}")
    print(f"Total runs: {len(config['seeds'])}")
    print(f"{'='*80}")
    
    for run_idx, seed in enumerate(config['seeds']):
        print(f"\n{'='*60}")
        print(f"Run {run_idx + 1}/{len(config['seeds'])} - Seed: {seed}")
        print(f"{'='*60}")
        
        # Set random seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        # Run training
        score_record, action_restore, agent, constraint_violations, episode_stats = main(config)
        
        # Store results
        all_score_records.append(score_record)
        all_action_restores.append(action_restore)
        all_agents.append(agent)
        all_constraint_violations.append(constraint_violations)
        all_episode_stats.append(episode_stats)
        
        print(f"Run {run_idx + 1} completed - Final Score: {score_record[-1]:.4f}")
        print(f"  Total violations: {constraint_violations['total_violations']}")
    
    print(f"\n{'='*60}")
    print(f"All {len(config['seeds'])} runs completed!")
    print(f"{'='*60}")
    
    # Print multi-run summary
    print(f"\nMulti-Run Constraint Violation Summary:")
    total_violations_all_runs = sum(cv['total_violations'] for cv in all_constraint_violations)
    total_steps_all_runs = sum(sum(len(actions) for actions in action_restore) 
                              for action_restore in all_action_restores)
    
    print(f"Total violations across all runs: {total_violations_all_runs}")
    print(f"Total steps across all runs: {total_steps_all_runs}")
    print(f"Overall violation rate: {(total_violations_all_runs / total_steps_all_runs * 100):.2f}%")
    
    for i, cv in enumerate(all_constraint_violations):
        run_steps = sum(len(actions) for actions in all_action_restores[i])
        run_rate = (cv['total_violations'] / run_steps * 100) if run_steps > 0 else 0
        print(f"Run {i+1}: {cv['total_violations']} violations, {run_rate:.2f}% rate")
    
    return all_score_records, all_action_restores, all_agents, all_constraint_violations, all_episode_stats


def evaluate_multi_run_results(all_score_records, config=None):
    """Evaluate and analyze results from multiple runs"""
    if config is None:
        config = CONFIG
    
    print(f"\n{'='*60}")
    print(f"Multi-Run Training Results Analysis")
    print(f"{'='*60}")
    
    final_scores = [scores[-1] for scores in all_score_records]
    mean_scores = [np.mean(scores[-20:]) if len(scores) >= 20 else np.mean(scores) for scores in all_score_records]
    
    print(f"Final Episode Scores:")
    for i, (seed, score) in enumerate(zip(config['seeds'], final_scores)):
        print(f"  Run {i+1} (Seed {seed}): {score:.4f}")
    
    print(f"\nLast 20 Episodes Average Scores:")
    for i, (seed, score) in enumerate(zip(config['seeds'], mean_scores)):
        print(f"  Run {i+1} (Seed {seed}): {score:.4f}")
    
    print(f"\nOverall Statistics:")
    print(f"  Mean Final Score: {np.mean(final_scores):.4f} ± {np.std(final_scores):.4f}")
    print(f"  Best Final Score: {np.max(final_scores):.4f}")
    print(f"  Worst Final Score: {np.min(final_scores):.4f}")
    print(f"  Mean of Last 20 Episodes: {np.mean(mean_scores):.4f} ± {np.std(mean_scores):.4f}")
    
    return {
        'final_scores': final_scores,
        'mean_scores': mean_scores,
        'overall_mean': np.mean(final_scores),
        'overall_std': np.std(final_scores),
        'best_score': np.max(final_scores),
        'worst_score': np.min(final_scores)
    }


def save_results(score_records, action_restores, agents, config):
    """Save training results and models"""
    if not config['save_models']:
        return None, None
    
    alg_name = "pdqn"
    now_str = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"{alg_name}_{now_str}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Save training data
    np.save(os.path.join(save_dir, f"{alg_name}_scores_{now_str}.npy"), score_records, allow_pickle=True)
    
    # Save action restores
    with open(os.path.join(save_dir, f"{alg_name}_actions_{now_str}.pkl"), 'wb') as f:
        pickle.dump(action_restores, f)
    
    # Save models
    model_paths = []
    for idx, agent in enumerate(agents):
        run_save_dir = os.path.join(save_dir, f"run_{idx+1}_seed_{config['seeds'][idx]}")
        os.makedirs(run_save_dir, exist_ok=True)
        
        q_path = os.path.join(run_save_dir, f"{alg_name}_q_network_{now_str}.pth")
        param_path = os.path.join(run_save_dir, f"{alg_name}_param_network_{now_str}.pth")
        
        torch.save(agent.q_network.state_dict(), q_path)
        torch.save(agent.q_target.state_dict(), os.path.join(run_save_dir, f"{alg_name}_q_target_{now_str}.pth"))
        torch.save(agent.param_network.state_dict(), param_path)
        torch.save(agent.param_target.state_dict(), os.path.join(run_save_dir, f"{alg_name}_param_target_{now_str}.pth"))
        
        model_paths.append(q_path)
    
    # Save configuration
    import json
    config_path = os.path.join(save_dir, f"config_{now_str}.json")
    with open(config_path, 'w') as f:
        json_config = {k: (v.tolist() if isinstance(v, np.ndarray) else v) 
                       for k, v in config.items() if not callable(v)}
        json.dump(json_config, f, indent=2)
    
    print(f"Results saved to directory: {save_dir}")
    return save_dir, model_paths


def save_multi_run_vehicle_flow_statistics(all_episode_stats, all_constraint_violations, config, save_dir):
    """Save multi-run vehicle flow statistics"""
    import csv
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save runs summary
    runs_summary_path = os.path.join(save_dir, f'pdqn_multi_run_summary_{timestamp}.csv')
    with open(runs_summary_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'Run', 'Seed', 'Total_Arrivals', 'Total_Charged', 'Total_Left_Uncharged',
            'Success_Rate_%', 'Avg_Revenue', 'Avg_Energy_kWh', 'Total_Violations', 'Violation_Rate_%'
        ])
        
        for i, (episode_stats, constraint_violations) in enumerate(zip(all_episode_stats, all_constraint_violations)):
            writer.writerow([
                i + 1,
                config['seeds'][i],
                sum(episode_stats['total_arrivals']),
                sum(episode_stats['vehicles_charged']),
                sum(episode_stats['vehicles_left_uncharged']),
                round(np.mean(episode_stats['charging_success_rate']), 2),
                round(np.mean(episode_stats['cumulative_revenue']), 2),
                round(np.mean(episode_stats['energy_delivered']), 2),
                constraint_violations['total_violations'],
                round(np.mean(constraint_violations['violation_rate']) * 100, 2)
            ])
    
    # Save overall summary
    overall_summary_path = os.path.join(save_dir, f'pdqn_overall_summary_{timestamp}.csv')
    with open(overall_summary_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        all_success_rates = [np.mean(stats['charging_success_rate']) for stats in all_episode_stats]
        all_revenues = [np.mean(stats['cumulative_revenue']) for stats in all_episode_stats]
        all_energies = [np.mean(stats['energy_delivered']) for stats in all_episode_stats]
        all_violations = [cv['total_violations'] for cv in all_constraint_violations]
        
        writer.writerow(['Metric', 'Mean', 'Std', 'Min', 'Max'])
        writer.writerow(['Success Rate (%)', round(np.mean(all_success_rates), 2),
                        round(np.std(all_success_rates), 2),
                        round(min(all_success_rates), 2),
                        round(max(all_success_rates), 2)])
        writer.writerow(['Revenue', round(np.mean(all_revenues), 2),
                        round(np.std(all_revenues), 2),
                        round(min(all_revenues), 2),
                        round(max(all_revenues), 2)])
        writer.writerow(['Energy (kWh)', round(np.mean(all_energies), 2),
                        round(np.std(all_energies), 2),
                        round(min(all_energies), 2),
                        round(max(all_energies), 2)])
        writer.writerow(['Violations', round(np.mean(all_violations), 2),
                        round(np.std(all_violations), 2),
                        min(all_violations),
                        max(all_violations)])
    
    print(f"Statistics saved to: {runs_summary_path}")
    print(f"Statistics saved to: {overall_summary_path}")
    
    return runs_summary_path, overall_summary_path


def plot_training_curve(score_records, save_dir=None):
    """Plot training curve"""
    plt.figure(figsize=(10, 6))
    plt.plot(score_records)
    plt.title('PDQN Training Score over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.grid(True)
    
    if save_dir:
        save_path = os.path.join(save_dir, 'pdqn_training_curve.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curve saved to: {save_path}")
    
    plt.show()


def plot_multi_run_training_curves(all_score_records, config=None, save_dir=None):
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
    
    plt.title('PDQN Multi-Run Training Curves')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    if save_dir:
        save_path = os.path.join(save_dir, 'pdqn_multi_run_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Multi-run curves saved to: {save_path}")
    
    plt.show()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='PDQN for EV Charging Station')
    parser.add_argument('--stations', type=int, default=CONFIG['n_stations'], help='Number of charging stations')
    parser.add_argument('--power', type=float, default=CONFIG['p_max'], help='Maximum power per station')
    parser.add_argument('--arrival-rate', type=float, default=CONFIG['arrival_rate'], help='Vehicle arrival rate')
    parser.add_argument('--episodes', type=int, default=CONFIG['num_of_episodes'], help='Number of episodes')
    parser.add_argument('--test-only', action='store_true', help='Run test only')
    parser.add_argument('--multi-run', action='store_true', default=CONFIG['enable_multi_run'], help='Multi-run training')
    parser.add_argument('--single-run', action='store_true', help='Force single run')
    parser.add_argument('--seeds', nargs='+', type=int, default=CONFIG['seeds'], help='Random seeds')
    
    args = parser.parse_args()
    
    # Update config
    config = CONFIG.copy()
    config.update({
        'n_stations': args.stations,
        'p_max': args.power,
        'arrival_rate': args.arrival_rate,
        'num_of_episodes': args.episodes,
        'seeds': args.seeds,
        'enable_multi_run': args.multi_run and not args.single_run
    })
    
    print(f"\n{'='*60}")
    print(f"PDQN for EV Charging Station")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  Stations: {config['n_stations']}")
    print(f"  Max Power: {config['p_max']} kW")
    print(f"  Arrival Rate: {config['arrival_rate']}")
    print(f"  Episodes: {config['num_of_episodes']}")
    print(f"  Seeds: {config['seeds']}")
    print(f"  Test only: {args.test_only}")
    print(f"  Multi-run mode: {config['enable_multi_run']}")
    print(f"{'='*60}")
    
    if args.test_only:
        test_and_visualize(config)
    elif config['enable_multi_run']:
        # Multi-run training
        all_score_records, all_action_restores, all_agents, all_constraint_violations, all_episode_stats = multi_run_training(config)
        
        # Evaluate results
        stats = evaluate_multi_run_results(all_score_records, config)
        
        # Save results
        if config['save_models']:
            save_dir, model_paths = save_results(all_score_records, all_action_restores, all_agents, config)
            save_multi_run_vehicle_flow_statistics(all_episode_stats, all_constraint_violations, config, save_dir)
        
        # Plot curves
        if config['plot_training_curve']:
            plot_multi_run_training_curves(all_score_records, config, save_dir if config['save_models'] else None)
        
        # Test best model
        if config['save_models'] and all_agents:
            best_run_idx = np.argmax([scores[-1] for scores in all_score_records])
            print(f"\n{'='*40}")
            print(f"Testing with best model (Run {best_run_idx+1})...")
            print(f"{'='*40}")
            test_and_visualize(config, all_agents[best_run_idx])
    else:
        # Single run training
        if config['seeds']:
            seed = config['seeds'][0]
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            print(f"Using random seed: {seed}")
        
        score_record, action_restore, agent, constraint_violations, episode_stats = main(config)
        
        # Save results
        if config['save_models']:
            save_dir, model_paths = save_results([score_record], [action_restore], [agent], config)
        
        # Plot curve
        if config['plot_training_curve']:
            plot_training_curve(score_record, save_dir if config['save_models'] else None)
        
        # Test model
        print(f"\n{'='*40}")
        print(f"Testing trained PDQN model...")
        print(f"{'='*40}")
        test_and_visualize(config, agent)

