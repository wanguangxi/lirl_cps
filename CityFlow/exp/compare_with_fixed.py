"""
Comparison experiment: LIRL with different action-space configurations

Compare two action-space configurations:
1. Dynamic green duration: min_duration=10, max_duration=60 (current setting)
2. Fixed green duration: fixed duration (50s in this config)

Key metrics:
- Throughput
- Average travel time

Runs on the City_3_5 scenario.
"""

import os
import sys
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import collections
from datetime import datetime
from typing import Dict, List
import argparse

# Add project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "env"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "algs"))

from cityflow_multi_env import CityFlowMultiIntersectionEnv, get_default_config

# Optional plotting libs
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False
    print("[WARNING] matplotlib is not installed; plotting will be skipped.")

# =======================
# Device detection
# =======================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[DEVICE] Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"[DEVICE] GPU: {torch.cuda.get_device_name(0)}")


# =======================
# Configuration
# =======================
BASE_CONFIG = {
    # Learning parameters
    'lr_mu': 0.0003,
    'lr_q': 0.001,
    'gamma': 0.99,
    'batch_size': 128 if torch.cuda.is_available() else 64,
    'buffer_limit': 100000,
    'tau': 0.005,
    
    # Environment parameters
    'episode_length': 3600,
    'ctrl_interval': 10,
    'min_green': 10,
    
    # Network architecture
    'hidden_dim1': 256,
    'hidden_dim2': 128,
    
    # Training parameters
    'memory_threshold': 500,
    'training_iterations': 10,
}

# Two action-space configurations
ACTION_SPACE_CONFIGS = {
    "Dynamic green duration": {
        "name": "Dynamic Duration (10-60s)",
        "min_duration": 10,
        "max_duration": 60,
        "description": "Green duration can be adjusted dynamically within 10–60 seconds."
    },
    "Fixed green duration": {
        "name": "Fixed Duration (50s)",
        "min_duration": 50,
        "max_duration": 50,
        "description": "Green duration is fixed (50 seconds in this config)."
    }
}


# =======================
# Neural networks
# =======================

class ActorNetwork(nn.Module):
    """Actor network."""
    def __init__(self, state_size, action_size, hidden1=256, hidden2=128):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, action_size)
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))


class CriticNetwork(nn.Module):
    """Critic network."""
    def __init__(self, state_size, action_size, hidden1=256, hidden2=128):
        super().__init__()
        self.fc_s = nn.Linear(state_size, hidden1)
        self.fc_a = nn.Linear(action_size, hidden1)
        self.fc2 = nn.Linear(hidden1 * 2, hidden2)
        self.fc_out = nn.Linear(hidden2, 1)
        self._init_weights()
    
    def _init_weights(self):
        for m in [self.fc_s, self.fc_a, self.fc2, self.fc_out]:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, x, a):
        h1 = F.relu(self.fc_s(x))
        h2 = F.relu(self.fc_a(a))
        cat = torch.cat([h1, h2], dim=1)
        q = F.relu(self.fc2(cat))
        return self.fc_out(q)


class ReplayBuffer:
    """Replay buffer."""
    def __init__(self, buffer_limit=100000):
        self.buffer = collections.deque(maxlen=buffer_limit)
    
    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        return random.sample(self.buffer, min(n, len(self.buffer)))
    
    def size(self):
        return len(self.buffer)


class OrnsteinUhlenbeckNoise:
    """Ornstein–Uhlenbeck (OU) noise."""
    def __init__(self, mu, theta=0.15, dt=0.01, sigma=0.2):
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


class ConstraintAwareProjector:
    """Constraint-aware action projector."""
    def __init__(self, num_intersections, num_phases, min_duration, max_duration, min_green=10):
        self.num_intersections = num_intersections
        self.num_phases = num_phases
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.min_green = min_green
        self.num_duration_options = max_duration - min_duration + 1
    
    def project(self, continuous_action, env=None):
        """Project a continuous action into a discrete action that satisfies constraints."""
        if isinstance(continuous_action, torch.Tensor):
            continuous_action = continuous_action.detach().cpu().numpy()
        
        a_ = np.clip(continuous_action.flatten(), 0, 1)
        
        expected_dim = self.num_intersections * 2
        if len(a_) < expected_dim:
            padded = np.ones(expected_dim) * 0.5
            padded[:len(a_)] = a_
            a_ = padded
        elif len(a_) > expected_dim:
            a_ = a_[:expected_dim]
        
        discrete_action = np.zeros(self.num_intersections * 2, dtype=np.int64)
        
        if env is None:
            return self._simple_project(a_)
        
        try:
            current_phases = env.current_phases.copy()
            phase_elapsed = env.phase_elapsed.copy()
            target_durations = env.target_durations.copy()
            valid_phases = env.valid_phases.copy()
            intersection_ids = env.intersection_ids.copy()
        except Exception:
            return self._simple_project(a_)
        
        for i, inter_id in enumerate(intersection_ids):
            phase_prob = a_[i * 2]
            duration_prob = a_[i * 2 + 1]
            
            cur_phase = current_phases.get(inter_id, 0)
            elapsed = phase_elapsed.get(inter_id, 0.0)
            target_duration = target_durations.get(inter_id, self.min_duration)
            inter_valid_phases = valid_phases.get(inter_id, [True] * self.num_phases)
            
            desired_phase = int(phase_prob * (self.num_phases - 1) + 0.5)
            desired_phase = np.clip(desired_phase, 0, self.num_phases - 1)
            
            can_switch = (elapsed >= self.min_green) and (elapsed >= target_duration)
            
            if desired_phase != cur_phase:
                if can_switch and inter_valid_phases[desired_phase]:
                    selected_phase = desired_phase
                else:
                    selected_phase = cur_phase
            else:
                selected_phase = cur_phase
            
            # Duration parameter
            desired_duration_idx = int(duration_prob * (self.num_duration_options - 1) + 0.5)
            desired_duration_idx = np.clip(desired_duration_idx, 0, self.num_duration_options - 1)
            
            min_duration_idx = max(0, self.min_green - self.min_duration)
            selected_duration_idx = max(desired_duration_idx, min_duration_idx)
            
            discrete_action[i * 2] = selected_phase
            discrete_action[i * 2 + 1] = selected_duration_idx
        
        return discrete_action
    
    def _simple_project(self, a_):
        discrete_action = np.zeros(self.num_intersections * 2, dtype=np.int64)
        for i in range(self.num_intersections):
            phase_prob = a_[i * 2]
            duration_prob = a_[i * 2 + 1]
            discrete_action[i * 2] = int(np.clip(phase_prob * (self.num_phases - 1) + 0.5, 0, self.num_phases - 1))
            discrete_action[i * 2 + 1] = int(np.clip(duration_prob * (self.num_duration_options - 1) + 0.5, 0, self.num_duration_options - 1))
        return discrete_action


# =======================
# LIRL Agent
# =======================

class LIRLAgent:
    """LIRL (DDPG) Agent"""
    def __init__(self, state_size, num_intersections, num_phases, config, 
                 min_duration, max_duration):
        self.config = config
        self.num_intersections = num_intersections
        self.num_phases = num_phases
        self.min_duration = min_duration
        self.max_duration = max_duration
        
        action_size = num_intersections * 2
        
        self.actor = ActorNetwork(state_size, action_size,
                                   config['hidden_dim1'], config['hidden_dim2']).to(DEVICE)
        self.actor_target = ActorNetwork(state_size, action_size,
                                          config['hidden_dim1'], config['hidden_dim2']).to(DEVICE)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = CriticNetwork(state_size, action_size,
                                     config['hidden_dim1'], config['hidden_dim2']).to(DEVICE)
        self.critic_target = CriticNetwork(state_size, action_size,
                                            config['hidden_dim1'], config['hidden_dim2']).to(DEVICE)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config['lr_mu'])
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config['lr_q'])
        
        self.memory = ReplayBuffer(config['buffer_limit'])
        self.ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(action_size))
        self.noise_scale = 1.0
        
        self.projector = ConstraintAwareProjector(
            num_intersections, num_phases,
            min_duration, max_duration,
            config['min_green']
        )
        self.env = None
    
    def set_env(self, env):
        self.env = env
    
    def select_action(self, state, deterministic=False):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            action = self.actor(state_t).squeeze(0).cpu().numpy()
        
        if not deterministic:
            noise = self.ou_noise() * self.noise_scale
            action = np.clip(action + noise, 0, 1)
        
        env_action = self.projector.project(action, self.env)
        return env_action, action
    
    def store(self, state, action, reward, next_state, done):
        self.memory.put((state, action, reward, next_state, done))
    
    def train_step(self):
        if self.memory.size() < self.config['memory_threshold']:
            return
        
        batch = self.memory.sample(self.config['batch_size'])
        states = torch.FloatTensor([t[0] for t in batch]).to(DEVICE)
        actions = torch.FloatTensor([t[1] for t in batch]).to(DEVICE)
        rewards = torch.FloatTensor([[t[2]] for t in batch]).to(DEVICE)
        next_states = torch.FloatTensor([t[3] for t in batch]).to(DEVICE)
        dones = torch.FloatTensor([[1.0 - t[4]] for t in batch]).to(DEVICE)
        
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target = rewards + self.config['gamma'] * target_q * dones
        
        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        actor_loss = -self.critic(states, self.actor(states)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # Soft update
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.config['tau'] * param.data + (1 - self.config['tau']) * target_param.data)
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.config['tau'] * param.data + (1 - self.config['tau']) * target_param.data)
    
    def decay_noise(self, episode, total_episodes):
        self.noise_scale = max(0.1, 1.0 - episode / (total_episodes * 0.8))
    
    def reset_noise(self):
        self.ou_noise.reset()
    
    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path, map_location=DEVICE)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])


# =======================
# Training and evaluation
# =======================

def create_environment(config_path: str, base_config: Dict, min_duration: int, max_duration: int):
    """Create environment."""
    env_config = get_default_config(config_path)
    env_config.update({
        "episode_length": base_config['episode_length'],
        "ctrl_interval": base_config['ctrl_interval'],
        "min_green": base_config['min_green'],
        "min_duration": min_duration,
        "max_duration": max_duration,
        "verbose_violations": False,
        "log_violations": True,
    })
    return CityFlowMultiIntersectionEnv(env_config)


def train_agent(agent, env, num_episodes: int, print_interval: int = 50) -> Dict:
    """Train the agent."""
    agent.set_env(env)
    
    episode_rewards = []
    episode_travel_times = []
    episode_throughputs = []
    
    for ep in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        done = False
        agent.reset_noise()
        
        while not done:
            env_action, action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(env_action)
            done = terminated or truncated
            
            agent.store(state, action, reward, next_state, done)
            episode_reward += reward
            state = next_state
        
        # Train
        for _ in range(10):
            agent.train_step()
        
        agent.decay_noise(ep, num_episodes)
        
        # Record stats
        episode_rewards.append(episode_reward)
        avg_tt = info.get('average_travel_time', 0)
        episode_travel_times.append(avg_tt)
        
        flow_stats = info.get('intersection_flow', {})
        throughput = sum(s.get('throughput', 0) for s in flow_stats.values())
        episode_throughputs.append(throughput)
        
        if (ep + 1) % print_interval == 0:
            avg_reward = np.mean(episode_rewards[-print_interval:])
            avg_tt_val = np.mean(episode_travel_times[-print_interval:])
            avg_tp = np.mean(episode_throughputs[-print_interval:])
            print(f"    Episode {ep+1}/{num_episodes}: R={avg_reward:.0f}, "
                  f"TT={avg_tt_val:.0f}s, TP={avg_tp:.0f}")
    
    return {
        'episode_rewards': episode_rewards,
        'episode_travel_times': episode_travel_times,
        'episode_throughputs': episode_throughputs,
    }


def evaluate_agent(agent, env, num_episodes: int = 10) -> Dict:
    """Evaluate the agent."""
    agent.set_env(env)
    
    all_rewards = []
    all_travel_times = []
    all_throughputs = []
    
    for ep in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            env_action, _ = agent.select_action(state, deterministic=True)
            state, reward, terminated, truncated, info = env.step(env_action)
            done = terminated or truncated
            episode_reward += reward
        
        all_rewards.append(episode_reward)
        all_travel_times.append(info.get('average_travel_time', 0))
        
        flow_stats = info.get('intersection_flow', {})
        throughput = sum(s.get('throughput', 0) for s in flow_stats.values())
        all_throughputs.append(throughput)
    
    return {
        'mean_reward': np.mean(all_rewards),
        'std_reward': np.std(all_rewards),
        'mean_travel_time': np.mean(all_travel_times),
        'std_travel_time': np.std(all_travel_times),
        'mean_throughput': np.mean(all_throughputs),
        'std_throughput': np.std(all_throughputs),
        'all_rewards': all_rewards,
        'all_travel_times': all_travel_times,
        'all_throughputs': all_throughputs,
    }


# =======================
# Plotting
# =======================

def plot_learning_curves(results: Dict, output_dir: str):
    """Plot learning curve comparison."""
    if not HAS_PLOT:
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    colors = {'Dynamic green duration': '#2ecc71', 'Fixed green duration': '#e74c3c'}
    
    metrics = [
        ('episode_throughputs', 'Throughput', 'Throughput Learning Curve'),
        ('episode_travel_times', 'Average Travel Time (s)', 'Travel Time Learning Curve'),
        ('episode_rewards', 'Episode Reward', 'Reward Learning Curve'),
    ]
    
    for ax, (metric_key, ylabel, title) in zip(axes, metrics):
        for config_name, data in results.items():
            values = data['training'][metric_key]
            window = min(20, len(values) // 10) if len(values) > 20 else 1
            if window > 1:
                smoothed = np.convolve(values, np.ones(window)/window, mode='valid')
                x = np.arange(window-1, len(values))
            else:
                smoothed = values
                x = np.arange(len(values))
            
            ax.plot(x, smoothed, label=config_name, color=colors.get(config_name, '#3498db'), linewidth=2)
        
        ax.set_xlabel('Episode')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'learning_curves.png'), dpi=150)
    plt.close()
    print("  Learning curves saved: learning_curves.png")


def plot_evaluation_comparison(results: Dict, output_dir: str):
    """Plot evaluation comparison bar chart."""
    if not HAS_PLOT:
        return
    
    configs = list(results.keys())
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    colors = ['#2ecc71', '#e74c3c']
    
    # Throughput comparison
    ax = axes[0]
    throughputs = [results[c]['evaluation']['mean_throughput'] for c in configs]
    throughput_stds = [results[c]['evaluation']['std_throughput'] for c in configs]
    bars = ax.bar(configs, throughputs, yerr=throughput_stds, capsize=8, 
                  color=colors, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Throughput (vehicles)', fontsize=12)
    ax.set_title('Throughput Comparison', fontsize=14)
    ax.tick_params(axis='x', labelsize=11)
    
    # Add value labels
    for bar, val, std in zip(bars, throughputs, throughput_stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 100,
                f'{val:.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Travel Time Comparison
    ax = axes[1]
    travel_times = [results[c]['evaluation']['mean_travel_time'] for c in configs]
    travel_time_stds = [results[c]['evaluation']['std_travel_time'] for c in configs]
    bars = ax.bar(configs, travel_times, yerr=travel_time_stds, capsize=8,
                  color=colors, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Average Travel Time (s)', fontsize=12)
    ax.set_title('Travel Time Comparison', fontsize=14)
    ax.tick_params(axis='x', labelsize=11)
    
    # Add value labels
    for bar, val, std in zip(bars, travel_times, travel_time_stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 10,
                f'{val:.0f}s', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'evaluation_comparison.png'), dpi=150)
    plt.close()
    print("  Evaluation comparison plot saved: evaluation_comparison.png")


def plot_box_comparison(results: Dict, output_dir: str):
    """Plot boxplot comparison."""
    if not HAS_PLOT:
        return
    
    configs = list(results.keys())
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    colors = ['#2ecc71', '#e74c3c']
    
    # Throughput boxplot
    ax = axes[0]
    throughput_data = [results[c]['evaluation']['all_throughputs'] for c in configs]
    bp = ax.boxplot(throughput_data, labels=configs, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('Throughput (vehicles)', fontsize=12)
    ax.set_title('Throughput Distribution', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Travel Time Box Plot
    ax = axes[1]
    travel_time_data = [results[c]['evaluation']['all_travel_times'] for c in configs]
    bp = ax.boxplot(travel_time_data, labels=configs, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('Average Travel Time (s)', fontsize=12)
    ax.set_title('Travel Time Distribution', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'box_comparison.png'), dpi=150)
    plt.close()
    print("  Boxplot saved: box_comparison.png")


# =======================
# Main
# =======================

def main():
    parser = argparse.ArgumentParser(description="LIRL action-space configuration comparison experiment")
    parser.add_argument("--config", type=str,
                       default=os.path.join(PROJECT_ROOT, "examples/City_3_5/config.json"),
                       help="CityFlow config file path")
    parser.add_argument("--train-episodes", type=int, default=500,
                       help="Number of training episodes")
    parser.add_argument("--eval-episodes", type=int, default=10,
                       help="Number of evaluation episodes")
    parser.add_argument("--output-dir", type=str,
                       default=os.path.join(PROJECT_ROOT, "outputs/compare_with_fixed"),
                       help="Output directory")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"\n{'#'*70}")
    print("# LIRL action-space configuration comparison")
    print(f"# Output dir: {run_dir}")
    print(f"# Train episodes: {args.train_episodes}")
    print(f"# Eval episodes: {args.eval_episodes}")
    print(f"{'#'*70}")
    
    results = {}
    
    for config_name, action_config in ACTION_SPACE_CONFIGS.items():
        print(f"\n{'='*70}")
        print(f"Config: {config_name}")
        print(f"  {action_config['description']}")
        print(f"  min_duration={action_config['min_duration']}, max_duration={action_config['max_duration']}")
        print(f"{'='*70}")
        
        min_dur = action_config['min_duration']
        max_dur = action_config['max_duration']
        
        # Create environment
        env = create_environment(args.config, BASE_CONFIG, min_dur, max_dur)
        
        state_size = env.observation_space.shape[0]
        num_intersections = env.num_intersections
        num_phases = env.num_phases
        
        print("\n  Environment info:")
        print(f"    State dim: {state_size}")
        print(f"    Intersections: {num_intersections}")
        print(f"    Action space: {env.action_space}")
        
        # Create agent
        agent = LIRLAgent(
            state_size=state_size,
            num_intersections=num_intersections,
            num_phases=num_phases,
            config=BASE_CONFIG,
            min_duration=min_dur,
            max_duration=max_dur
        )
        
        # Train
        print(f"\n  Start training ({args.train_episodes} episodes)...")
        training_results = train_agent(
            agent=agent,
            env=env,
            num_episodes=args.train_episodes,
            print_interval=max(1, args.train_episodes // 10)
        )
        
        # Save model
        model_path = os.path.join(run_dir, f"{config_name.lower().replace(' ', '_')}_model.pt")
        agent.save(model_path)
        print(f"  Model saved: {model_path}")
        
        # Evaluate
        print(f"\n  Start evaluation ({args.eval_episodes} episodes)...")
        eval_results = evaluate_agent(agent, env, args.eval_episodes)
        
        env.close()
        
        results[config_name] = {
            'config': action_config,
            'training': training_results,
            'evaluation': eval_results,
            'model_path': model_path,
        }
        
        print(f"\n  Evaluation results:")
        print(f"    Mean throughput: {eval_results['mean_throughput']:.1f} ± {eval_results['std_throughput']:.1f}")
        print(f"    Mean travel time: {eval_results['mean_travel_time']:.1f} ± {eval_results['std_travel_time']:.1f} s")
    
    # Plot comparison figures
    print(f"\n{'='*70}")
    print("Generating comparison figures...")
    print(f"{'='*70}")
    
    plot_learning_curves(results, run_dir)
    plot_evaluation_comparison(results, run_dir)
    plot_box_comparison(results, run_dir)
    
    # Save results
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        return obj
    
    results_serializable = convert_to_serializable(results)
    
    results_path = os.path.join(run_dir, "results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results_serializable, f, indent=2, ensure_ascii=False)
    print(f"Full results saved: {results_path}")
    
    # Summary
    summary = {
        'experiment_info': {
            'timestamp': timestamp,
            'train_episodes': args.train_episodes,
            'eval_episodes': args.eval_episodes,
        },
        'comparison': {}
    }
    
    for config_name, data in results.items():
        summary['comparison'][config_name] = {
            'mean_throughput': float(data['evaluation']['mean_throughput']),
            'std_throughput': float(data['evaluation']['std_throughput']),
            'mean_travel_time': float(data['evaluation']['mean_travel_time']),
            'std_travel_time': float(data['evaluation']['std_travel_time']),
        }
    
    summary_path = os.path.join(run_dir, "summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Summary saved: {summary_path}")
    
    # Print final comparison
    print(f"\n{'='*70}")
    print("Experiment result comparison")
    print(f"{'='*70}")
    print(f"{'Config':<25} {'Mean throughput':<20} {'Mean travel time':<20}")
    print(f"{'-'*70}")
    
    for config_name, data in results.items():
        eval_data = data['evaluation']
        tp_str = f"{eval_data['mean_throughput']:.0f} ± {eval_data['std_throughput']:.0f}"
        tt_str = f"{eval_data['mean_travel_time']:.0f} ± {eval_data['std_travel_time']:.0f}s"
        print(f"{config_name:<20} {tp_str:<20} {tt_str:<20}")
    
    # Improvement ratio
    if 'Dynamic green duration' in results and 'Fixed green duration' in results:
        dynamic_tp = results['Dynamic green duration']['evaluation']['mean_throughput']
        fixed_tp = results['Fixed green duration']['evaluation']['mean_throughput']
        dynamic_tt = results['Dynamic green duration']['evaluation']['mean_travel_time']
        fixed_tt = results['Fixed green duration']['evaluation']['mean_travel_time']
        
        tp_improvement = (dynamic_tp - fixed_tp) / fixed_tp * 100 if fixed_tp > 0 else 0
        tt_improvement = (fixed_tt - dynamic_tt) / fixed_tt * 100 if fixed_tt > 0 else 0
        
        print(f"\n{'='*70}")
        print("Improvement of dynamic duration vs fixed duration:")
        print(f"  Throughput: {tp_improvement:+.1f}%")
        print(f"  Travel time reduction: {tt_improvement:+.1f}%")
    
    print(f"\n{'='*70}")
    print("Done!")
    print(f"Output dir: {run_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
