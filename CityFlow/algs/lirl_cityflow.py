"""
DDPG for CityFlow Multi-Intersection Traffic Signal Control

Adapted to the new action space:
- Phase control (discrete: 0 ~ num_phases-1)
- Green duration control (continuous: min_duration ~ max_duration, 1-second granularity)

Based on the DDPG-LIRL framework.
"""

import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
import datetime as dt
from datetime import datetime

# Add environment path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "env"))
from cityflow_multi_env import CityFlowMultiIntersectionEnv, get_default_config

# =======================
# HYPERPARAMETERS CONFIG
# =======================
CONFIG = {
    # Learning parameters
    'lr_mu': 0.0003,
    'lr_q': 0.001,
    'gamma': 0.99,
    'batch_size': 64,
    'buffer_limit': 100000,
    'tau': 0.005,  # for target network soft update
    
    # Environment parameters
    'episode_length': 3600,
    'ctrl_interval': 10,
    'min_green': 10,
    'min_duration': 10,
    'max_duration': 60,
    'num_of_episodes': 200,
    
    # Network architecture
    'hidden_dim1': 256,
    'hidden_dim2': 128,
    'critic_hidden': 64,
    
    # Training parameters
    'memory_threshold': 500,
    'training_iterations': 10,
    'noise_params': {'theta': 0.15, 'dt': 0.01, 'sigma': 0.2},
    
    # Multi-run training parameters
    'enable_multi_run': False,
    'seeds': [42, 123, 456],
    'num_runs': 3,
    
    # Output parameters
    'print_interval': 10,
    'save_models': True,
    'output_dir': './outputs/ddpg_cityflow',
}


class ReplayBuffer:
    """Replay buffer."""
    def __init__(self, buffer_limit=None):
        limit = buffer_limit or CONFIG['buffer_limit']
        self.buffer = collections.deque(maxlen=limit)

    def put(self, transition):
        self.buffer.append(transition)
        
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append(r)
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0 
            done_mask_lst.append([done_mask])
        
        return (
            torch.FloatTensor(np.array(s_lst)),
            torch.FloatTensor(np.array(a_lst)),
            torch.FloatTensor(np.array(r_lst)).unsqueeze(1),
            torch.FloatTensor(np.array(s_prime_lst)),
            torch.FloatTensor(np.array(done_mask_lst))
        )
    
    def size(self):
        return len(self.buffer)


class ActorNetwork(nn.Module):
    """
    Actor network (policy).
    
    Output: continuous action vector
    - 2 values per intersection: [phase_prob, duration_prob]
    - phase_prob: 0~1, mapped to a discrete phase
    - duration_prob: 0~1, mapped to a continuous duration
    """
    def __init__(self, state_size, action_size, hidden1=None, hidden2=None):
        super(ActorNetwork, self).__init__()
        hidden1 = hidden1 or CONFIG['hidden_dim1']
        hidden2 = hidden2 or CONFIG['hidden_dim2']
        
        self.fc1 = nn.Linear(state_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, action_size)
        
        # Initialize weights
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
        # Use sigmoid to constrain outputs to [0, 1]
        x = torch.sigmoid(self.fc3(x))
        return x


class CriticNetwork(nn.Module):
    """
    Critic network (Q network).
    
    Input: state + action
    Output: Q value
    """
    def __init__(self, state_size, action_size, hidden1=None, hidden2=None, critic_hidden=None):
        super(CriticNetwork, self).__init__()
        hidden1 = hidden1 or CONFIG['hidden_dim1']
        hidden2 = hidden2 or CONFIG['hidden_dim2']
        critic_hidden = critic_hidden or CONFIG['critic_hidden']
        
        self.fc_s = nn.Linear(state_size, hidden1)
        self.fc_a = nn.Linear(action_size, hidden1)
        self.fc2 = nn.Linear(hidden1 * 2, hidden2)
        self.fc3 = nn.Linear(hidden2, critic_hidden)
        self.fc_out = nn.Linear(critic_hidden, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc_s.weight)
        nn.init.xavier_uniform_(self.fc_a.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc_out.weight)

    def forward(self, x, a):
        h1 = F.relu(self.fc_s(x))
        h2 = F.relu(self.fc_a(a))
        cat = torch.cat([h1, h2], dim=1)
        q = F.relu(self.fc2(cat))
        q = F.relu(self.fc3(q))
        q = self.fc_out(q)
        return q


class OrnsteinUhlenbeckNoise:
    """Ornstein-Uhlenbeck noise for exploration."""
    def __init__(self, mu, theta=None, dt=None, sigma=None):
        self.theta = theta or CONFIG['noise_params']['theta']
        self.dt = dt or CONFIG['noise_params']['dt']
        self.sigma = sigma or CONFIG['noise_params']['sigma']
        self.mu = mu
        self.x_prev = np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x
    
    def reset(self):
        self.x_prev = np.zeros_like(self.mu)


class CityFlowActionProjector:
    """
    Project the actor's continuous output to a valid discrete action space for CityFlow.
    
    Uses a projection algorithm similar to `ddpg_lirl_pi.py`:
    1. Take an environment state snapshot
    2. Filter feasible actions based on constraints
    3. Compute costs combining network preference and traffic state
    4. Choose the best action while satisfying constraints
    
    Network output: [phase_0_prob, duration_0_prob, phase_1_prob, duration_1_prob, ...]
    Env action:     [phase_0, duration_idx_0, phase_1, duration_idx_1, ...]
    """
    def __init__(self, num_intersections, num_phases, num_duration_options, 
                 min_green=10, duration_options=None):
        self.num_intersections = num_intersections
        self.num_phases = num_phases
        self.num_duration_options = num_duration_options
        self.min_green = min_green
        self.duration_options = duration_options or list(range(10, 61))
    
    def project(self, continuous_action, env=None):
        """
        Project a continuous action to a valid discrete action.
        
        Args:
            continuous_action: numpy array, shape (num_intersections * 2,)
                              2 values per intersection: [phase_prob, duration_prob] in [0, 1]
            env: CityFlowMultiIntersectionEnv (optional; enables constraint-aware projection)
        
        Returns:
            discrete_action: numpy array, shape (num_intersections * 2,)
                            [phase_idx, duration_idx, ...]
        """
        if isinstance(continuous_action, torch.Tensor):
            continuous_action = continuous_action.detach().cpu().numpy()
        
        # Ensure a 1-D array within [0, 1]
        a_ = np.clip(continuous_action.flatten(), 0, 1)
        
        # Check action dimension
        expected_dim = self.num_intersections * 2
        if len(a_) < expected_dim:
            # Pad to expected dim with 0.5
            padded = np.ones(expected_dim) * 0.5
            padded[:len(a_)] = a_
            a_ = padded
        elif len(a_) > expected_dim:
            a_ = a_[:expected_dim]
        
        discrete_action = np.zeros(self.num_intersections * 2, dtype=np.int64)
        
        # If env is not provided, use simple projection
        if env is None:
            return self._simple_project(a_)
        
        # ========== Constraint-aware projection ==========
        # 1. Take an environment state snapshot
        try:
            current_phases = env.current_phases.copy()
            phase_elapsed = env.phase_elapsed.copy()
            target_durations = env.target_durations.copy()
            valid_phases = env.valid_phases.copy()
            intersection_ids = env.intersection_ids.copy()
            
            # Get traffic state (queue lengths)
            lane_waiting = env.eng.get_lane_waiting_vehicle_count()
        except Exception as e:
            # If state fetch fails, fall back to simple projection
            print(f"[WARNING] Failed to get env state: {e}; falling back to simple projection")
            return self._simple_project(a_)
        
        # 2. Constraint-aware projection per intersection
        for i, inter_id in enumerate(intersection_ids):
            phase_prob = a_[i * 2]
            duration_prob = a_[i * 2 + 1]
            
            cur_phase = current_phases.get(inter_id, 0)
            elapsed = phase_elapsed.get(inter_id, 0.0)
            target_duration = target_durations.get(inter_id, self.duration_options[0])
            inter_valid_phases = valid_phases.get(inter_id, [True] * self.num_phases)
            
            # 2.1 Queue by direction (for cost computation)
            queue_by_dir = {"N": 0, "E": 0, "S": 0, "W": 0}
            if hasattr(env, 'in_lanes') and inter_id in env.in_lanes:
                for direction in ["N", "E", "S", "W"]:
                    lanes = env.in_lanes[inter_id].get(direction, [])
                    queue_by_dir[direction] = sum(lane_waiting.get(lane, 0) for lane in lanes)
            total_queue = sum(queue_by_dir.values())
            
            # 2.2 Determine feasible phases
            feasible_phases = []
            for p in range(self.num_phases):
                if inter_valid_phases[p]:
                    # Current phase is always feasible
                    if p == cur_phase:
                        feasible_phases.append(p)
                    # Other phases must satisfy switching conditions
                    elif elapsed >= self.min_green and elapsed >= target_duration:
                        feasible_phases.append(p)
            
            # If no feasible phase (should not happen), keep current phase
            if not feasible_phases:
                feasible_phases = [cur_phase]
            
            # 2.3 Cost for each feasible phase (network preference + traffic state)
            phase_costs = []
            for p in feasible_phases:
                # Network preference cost: distance to desired phase
                desired_phase = int(phase_prob * (self.num_phases - 1) + 0.5)
                preference_cost = abs(p - desired_phase) / max(self.num_phases - 1, 1)
                
                # Switching cost: small penalty if switching
                switch_cost = 0.1 if p != cur_phase else 0.0
                
                # Traffic-state cost: based on queue length (optional; not used in simple version)
                # traffic_cost = -total_queue * 0.01 if p == cur_phase else 0
                
                total_cost = preference_cost + switch_cost
                phase_costs.append((p, total_cost))
            
            # 2.4 Choose the phase with minimum cost
            phase_costs.sort(key=lambda x: x[1])
            selected_phase = phase_costs[0][0]
            
            # 2.5 Determine green duration
            # - If staying in current phase, you can update the target duration
            # - If switching phase, set a new target duration
            desired_duration_idx = int(duration_prob * (self.num_duration_options - 1) + 0.5)
            desired_duration_idx = np.clip(desired_duration_idx, 0, self.num_duration_options - 1)
            
            # Ensure duration is not below min green time
            min_duration_idx = 0
            for idx, dur in enumerate(self.duration_options):
                if dur >= self.min_green:
                    min_duration_idx = idx
                    break
            
            selected_duration_idx = max(desired_duration_idx, min_duration_idx)
            
            # 2.6 Store result
            discrete_action[i * 2] = selected_phase
            discrete_action[i * 2 + 1] = selected_duration_idx
        
        return discrete_action
    
    def _simple_project(self, a_):
        """Simple projection (used when env state is unavailable)."""
        discrete_action = np.zeros(self.num_intersections * 2, dtype=np.int64)
        
        for i in range(self.num_intersections):
            phase_prob = a_[i * 2]
            duration_prob = a_[i * 2 + 1]
            
            # Phase: [0, 1] -> [0, num_phases-1]
            phase_idx = int(phase_prob * (self.num_phases - 1) + 0.5)
            phase_idx = np.clip(phase_idx, 0, self.num_phases - 1)
            
            # Duration: [0, 1] -> [0, num_duration_options-1]
            duration_idx = int(duration_prob * (self.num_duration_options - 1) + 0.5)
            duration_idx = np.clip(duration_idx, 0, self.num_duration_options - 1)
            
            discrete_action[i * 2] = phase_idx
            discrete_action[i * 2 + 1] = duration_idx
        
        return discrete_action
    
    def project_with_safety_check(self, continuous_action, env):
        """
        Projection with safety checks (recommended).
        
        Ensures the returned action is always valid even if the network output is abnormal.
        """
        try:
            action = self.project(continuous_action, env)
            
            # Final safety check
            for i in range(self.num_intersections):
                # Phase range check
                if action[i * 2] < 0 or action[i * 2] >= self.num_phases:
                    action[i * 2] = 0  # fall back to default phase
                
                # Duration range check
                if action[i * 2 + 1] < 0 or action[i * 2 + 1] >= self.num_duration_options:
                    action[i * 2 + 1] = 0  # fall back to minimum duration
            
            return action
            
        except Exception as e:
            print(f"[WARNING] Action projection failed: {e}; returning default action")
            # Return a default action (keep current / minimum)
            default_action = np.zeros(self.num_intersections * 2, dtype=np.int64)
            return default_action
    
    def continuous_to_discrete_batch(self, continuous_actions, env=None):
        """Batch conversion."""
        batch_size = continuous_actions.shape[0]
        discrete_actions = np.zeros((batch_size, self.num_intersections * 2), dtype=np.int64)
        
        for b in range(batch_size):
            discrete_actions[b] = self.project(continuous_actions[b], env)
        
        return discrete_actions


def train_step(actor, actor_target, critic, critic_target, memory, 
               critic_optimizer, actor_optimizer, config=None):
    """Run one training step."""
    config = config or CONFIG
    
    s, a, r, s_prime, done_mask = memory.sample(config['batch_size'])
    
    # Critic update
    with torch.no_grad():
        target_actions = actor_target(s_prime)
        target_q = critic_target(s_prime, target_actions)
        target = r + config['gamma'] * target_q * done_mask
    
    current_q = critic(s, a)
    critic_loss = F.mse_loss(current_q, target)
    
    critic_optimizer.zero_grad()
    critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
    critic_optimizer.step()
    
    # Actor update
    actor_loss = -critic(s, actor(s)).mean()
    
    actor_optimizer.zero_grad()
    actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
    actor_optimizer.step()
    
    return critic_loss.item(), actor_loss.item()


def soft_update(net, net_target, tau=None):
    """Soft-update target network."""
    tau = tau or CONFIG['tau']
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)


def main(config=None, cityflow_config_path=None):
    """Main training entry."""
    config = config or CONFIG.copy()
    
    # Set CityFlow config path
    if cityflow_config_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cityflow_config_path = os.path.join(script_dir, "../examples/City_3_5/config.json")
    
    # Create environment config
    env_config = get_default_config(cityflow_config_path)
    env_config["episode_length"] = config['episode_length']
    env_config["ctrl_interval"] = config['ctrl_interval']
    env_config["min_green"] = config['min_green']
    env_config["min_duration"] = config['min_duration']
    env_config["max_duration"] = config['max_duration']
    env_config["verbose_violations"] = False
    env_config["log_violations"] = True
    
    # Create environment
    env = CityFlowMultiIntersectionEnv(env_config)
    
    # Get space dimensions
    state_size = env.observation_space.shape[0]
    # Continuous action: 2 values per intersection (phase_prob, duration_prob)
    continuous_action_size = env.num_intersections * 2
    
    print(f"\n{'='*60}")
    print("DDPG for CityFlow Traffic Signal Control")
    print(f"{'='*60}")
    print(f"Intersections: {env.num_intersections}")
    print(f"Phases per intersection: {env.num_phases}")
    print(f"Green duration range: [{env.min_duration}, {env.max_duration}] s")
    print(f"State dim: {state_size}")
    print(f"Continuous action dim: {continuous_action_size}")
    print(f"Discrete action space: {env.action_space}")
    print(f"{'='*60}\n")
    
    # Create action projector (constraint-aware projection)
    duration_options = list(range(env.min_duration, env.max_duration + 1))
    action_projector = CityFlowActionProjector(
        num_intersections=env.num_intersections,
        num_phases=env.num_phases,
        num_duration_options=env.num_duration_options,
        min_green=env.min_green,
        duration_options=duration_options
    )
    
    # Create networks
    actor = ActorNetwork(state_size, continuous_action_size)
    actor_target = ActorNetwork(state_size, continuous_action_size)
    actor_target.load_state_dict(actor.state_dict())
    
    critic = CriticNetwork(state_size, continuous_action_size)
    critic_target = CriticNetwork(state_size, continuous_action_size)
    critic_target.load_state_dict(critic.state_dict())
    
    # Optimizers
    actor_optimizer = optim.Adam(actor.parameters(), lr=config['lr_mu'])
    critic_optimizer = optim.Adam(critic.parameters(), lr=config['lr_q'])
    
    # Noise and replay buffer
    ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(continuous_action_size))
    memory = ReplayBuffer()
    
    # Training logs
    episode_rewards = []
    episode_travel_times = []
    episode_throughputs = []
    episode_violations = []
    
    # Create output directory
    output_dir = config.get('output_dir', './outputs/ddpg_cityflow')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"Training started. Output dir: {run_dir}")
    print(f"Total episodes: {config['num_of_episodes']}\n")
    
    for n_epi in range(config['num_of_episodes']):
        s, info = env.reset()
        done = False
        episode_reward = 0
        step = 0
        
        ou_noise.reset()
        
        while not done:
            # Get continuous action
            with torch.no_grad():
                state_tensor = torch.FloatTensor(s).unsqueeze(0)
                continuous_action = actor(state_tensor).squeeze(0).numpy()
            
            # Add exploration noise (decays over training)
            noise_scale = max(0.1, 1.0 - n_epi / (config['num_of_episodes'] * 0.8))
            noise = ou_noise() * noise_scale
            continuous_action = np.clip(continuous_action + noise, 0, 1)
            
            # Constraint-aware projection to discrete action
            discrete_action = action_projector.project_with_safety_check(continuous_action, env)
            
            # Step env
            s_prime, reward, terminated, truncated, info = env.step(discrete_action)
            done = terminated or truncated
            
            # Store transition (using continuous action)
            memory.put((s, continuous_action, reward, s_prime, done))
            
            episode_reward += reward
            s = s_prime
            step += 1
        
        # Train networks
        if memory.size() > config['memory_threshold']:
            for _ in range(config['training_iterations']):
                critic_loss, actor_loss = train_step(
                    actor, actor_target, critic, critic_target,
                    memory, critic_optimizer, actor_optimizer, config
                )
            
            # Soft update target networks
            soft_update(actor, actor_target)
            soft_update(critic, critic_target)
        
        # Record stats
        episode_rewards.append(episode_reward)
        avg_travel_time = info.get('average_travel_time', 0)
        episode_travel_times.append(avg_travel_time)
        
        flow_stats = info.get('intersection_flow', {})
        total_throughput = sum(s.get('throughput', 0) for s in flow_stats.values())
        episode_throughputs.append(total_throughput)
        
        total_violations = info.get('total_violations', {})
        total_viol = sum(total_violations.values())
        episode_violations.append(total_viol)
        
        # Print progress
        if (n_epi + 1) % config['print_interval'] == 0:
            avg_reward = np.mean(episode_rewards[-config['print_interval']:])
            avg_tt = np.mean(episode_travel_times[-config['print_interval']:])
            avg_tp = np.mean(episode_throughputs[-config['print_interval']:])
            avg_viol = np.mean(episode_violations[-config['print_interval']:])
            
            print(f"Episode {n_epi+1}/{config['num_of_episodes']}: "
                  f"Reward={avg_reward:.1f}, "
                  f"AvgTravelTime={avg_tt:.1f}s, "
                  f"Throughput={avg_tp:.0f}, "
                  f"Violations={avg_viol:.1f}")
    
    # Save model
    if config.get('save_models', True):
        model_path = os.path.join(run_dir, "ddpg_cityflow_final.pt")
        torch.save({
            'actor_state_dict': actor.state_dict(),
            'critic_state_dict': critic.state_dict(),
            'actor_target_state_dict': actor_target.state_dict(),
            'critic_target_state_dict': critic_target.state_dict(),
            'config': config,
        }, model_path)
        print(f"\nModel saved: {model_path}")
    
    # Save training stats
    import json
    stats_path = os.path.join(run_dir, "training_stats.json")
    with open(stats_path, 'w') as f:
        json.dump({
            'episode_rewards': episode_rewards,
            'episode_travel_times': episode_travel_times,
            'episode_throughputs': episode_throughputs,
            'episode_violations': episode_violations,
        }, f, indent=2)
    print(f"Training stats saved: {stats_path}")
    
    env.close()
    
    return {
        'actor': actor,
        'critic': critic,
        'episode_rewards': episode_rewards,
        'episode_travel_times': episode_travel_times,
        'run_dir': run_dir,
    }


def evaluate(model_path, cityflow_config_path=None, n_episodes=5, render=True):
    """Evaluate a trained model."""
    # Load model
    checkpoint = torch.load(model_path)
    config = checkpoint.get('config', CONFIG)
    
    # Set CityFlow config path
    if cityflow_config_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cityflow_config_path = os.path.join(script_dir, "../examples/City_3_5/config.json")
    
    # Create environment
    env_config = get_default_config(cityflow_config_path)
    env_config["episode_length"] = config.get('episode_length', 3600)
    env_config["ctrl_interval"] = config.get('ctrl_interval', 10)
    env_config["min_green"] = config.get('min_green', 10)
    env_config["min_duration"] = config.get('min_duration', 10)
    env_config["max_duration"] = config.get('max_duration', 60)
    env_config["verbose_violations"] = render
    
    env = CityFlowMultiIntersectionEnv(env_config, render_mode="human" if render else None)
    
    state_size = env.observation_space.shape[0]
    continuous_action_size = env.num_intersections * 2
    
    # Load actor network
    actor = ActorNetwork(state_size, continuous_action_size)
    actor.load_state_dict(checkpoint['actor_state_dict'])
    actor.eval()
    
    # Create action projector (constraint-aware projection)
    duration_options = list(range(env.min_duration, env.max_duration + 1))
    action_projector = CityFlowActionProjector(
        num_intersections=env.num_intersections,
        num_phases=env.num_phases,
        num_duration_options=env.num_duration_options,
        min_green=env.min_green,
        duration_options=duration_options
    )
    
    print(f"\n{'='*60}")
    print("DDPG model evaluation")
    print(f"{'='*60}")
    print(f"Model path: {model_path}")
    print(f"Eval episodes: {n_episodes}")
    
    episode_rewards = []
    episode_travel_times = []
    
    for ep in range(n_episodes):
        s, info = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(s).unsqueeze(0)
                continuous_action = actor(state_tensor).squeeze(0).numpy()
            
            # Constraint-aware projection to discrete action
            discrete_action = action_projector.project_with_safety_check(continuous_action, env)
            s, reward, terminated, truncated, info = env.step(discrete_action)
            done = terminated or truncated
            episode_reward += reward
            
            if render:
                env.render()
        
        episode_rewards.append(episode_reward)
        avg_travel_time = info.get('average_travel_time', 0)
        episode_travel_times.append(avg_travel_time)
        
        print(f"Episode {ep+1}/{n_episodes}: "
              f"Reward={episode_reward:.1f}, "
              f"AvgTravelTime={avg_travel_time:.1f}s")
        
        if render:
            env.print_intersection_flow_summary()
            env.print_violation_summary()
    
    env.close()
    
    print(f"\n{'='*60}")
    print("Evaluation results")
    print(f"{'='*60}")
    print(f"Mean reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Mean travel time: {np.mean(episode_travel_times):.2f}s")
    
    return {
        'episode_rewards': episode_rewards,
        'episode_travel_times': episode_travel_times,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DDPG for CityFlow Traffic Signal Control")
    parser.add_argument("--mode", type=str, default="evaluate", choices=["train", "evaluate"],
                       help="Run mode")
    parser.add_argument("--config", type=str, default="../examples/City_3_5/config.json",
                       help="CityFlow config file path")
    parser.add_argument("--model", type=str, default="../outputs/ddpg_cityflow/run_20260113_125816/ddpg_cityflow_final.pt",
                       help="Model file path (required for evaluate)")
    parser.add_argument("--episodes", type=int, default=200,
                       help="Number of training episodes")
    parser.add_argument("--episode-length", type=int, default=3600,
                       help="Simulation length per episode (seconds)")
    parser.add_argument("--min-duration", type=int, default=10,
                       help="Min green duration (seconds)")
    parser.add_argument("--max-duration", type=int, default=60,
                       help="Max green duration (seconds)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Resolve paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Config path
    if not os.path.isabs(args.config):
        config_path = os.path.join(script_dir, args.config)
    else:
        config_path = args.config
    
    if args.mode == "train":
        # Update config
        config = CONFIG.copy()
        config['num_of_episodes'] = args.episodes
        config['episode_length'] = args.episode_length
        config['min_duration'] = args.min_duration
        config['max_duration'] = args.max_duration
        
        results = main(config=config, cityflow_config_path=config_path)
        
        print("\nTraining finished!")
        print(f"Final mean reward: {np.mean(results['episode_rewards'][-10:]):.2f}")
        print(f"Final mean travel time: {np.mean(results['episode_travel_times'][-10:]):.2f}s")
        
    elif args.mode == "evaluate":
        if args.model is None:
            print("Error: evaluate mode requires --model")
            sys.exit(1)
        
        if not os.path.isabs(args.model):
            model_path = os.path.join(script_dir, args.model)
        else:
            model_path = args.model
        
        evaluate(model_path, cityflow_config_path=config_path, n_episodes=5, render=True)
    
    print("\nDone!")

