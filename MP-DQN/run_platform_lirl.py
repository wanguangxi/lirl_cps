#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LIRL Agent for Platform Environment
====================================
Train on Platform environment using LIRL algorithm

LIRL (Learning with Integer and Real-valued Actions via Lagrangian relaxation)
Uses action projection function to map continuous network output to mixed discrete-continuous action space
"""

import os
import click
import time
import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
import gym_platform
from gym.wrappers import Monitor
from common import ClickPythonLiteralOption
from common.wrappers import ScaledParameterisedActionWrapper
from common.platform_domain import PlatformFlattenedActionWrapper
from common.wrappers import ScaledStateWrapper

# Try importing scipy, use numpy implementation if unavailable
try:
    from scipy.optimize import minimize, linear_sum_assignment
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available, using numpy fallback implementations")


# =======================
# Device Configuration
# =======================
device = torch.device("cpu")


# =======================
# Action Projection Function
# =======================
class ActionProjection:
    """
    LIRL Action Projection Class
    Projects network output continuous actions to valid discrete-continuous mixed action space
    
    Platform environment action space:
    - 3 discrete actions: run(0), hop(1), leap(2)
    - Continuous parameters: each action requires 1 parameter
    """
    
    def __init__(self, action_space, use_qp=True):
        """
        Args:
            action_space: Environment action space
            use_qp: Whether to use QP for continuous parameter projection
        """
        self.use_qp = use_qp
        self.num_actions = 3
        
        self.action_param_sizes = [1, 1, 1]
        self.action_param_offsets = [0, 1, 2, 3]
        
        self.param_min = -1.0
        self.param_max = 1.0
        
        self.reset_timings()
    
    def reset_timings(self):
        """Reset timing statistics"""
        self.discrete_selection_times = []
        self.qp_times = []
        self.total_projection_times = []
    
    def project(self, action_probs, action_params, record_timing=True):
        """
        Project network output to valid action space
        
        Args:
            action_probs: Discrete action probability distribution [3]
            action_params: Continuous action parameters [3]
            record_timing: Whether to record timing
        
        Returns:
            discrete_action: Selected discrete action index
            continuous_params: Projected continuous parameters
            timing_info: Timing information dictionary
        """
        total_start = time.perf_counter()
        
        if isinstance(action_probs, torch.Tensor):
            action_probs = action_probs.detach().cpu().numpy()
        if isinstance(action_params, torch.Tensor):
            action_params = action_params.detach().cpu().numpy()
        
        action_probs = np.asarray(action_probs).flatten()
        action_params = np.asarray(action_params).flatten()
        
        discrete_start = time.perf_counter()
        discrete_action = self._select_discrete_action(action_probs)
        discrete_time = time.perf_counter() - discrete_start
        
        qp_start = time.perf_counter()
        projected_params = self._project_continuous_params(action_params, discrete_action)
        qp_time = time.perf_counter() - qp_start
        
        total_time = time.perf_counter() - total_start
        
        if record_timing:
            self.discrete_selection_times.append(discrete_time)
            self.qp_times.append(qp_time)
            self.total_projection_times.append(total_time)
        
        timing_info = {
            'discrete_selection_time': discrete_time,
            'qp_time': qp_time,
            'total_time': total_time
        }
        
        return discrete_action, projected_params, timing_info
    
    def _select_discrete_action(self, action_probs):
        """
        Select discrete action
        Uses cost matrix-based method (simplified Hungarian algorithm idea)
        
        For Platform environment, we build a cost matrix to select optimal action
        """
        cost_matrix = np.zeros((1, self.num_actions))
        for i in range(self.num_actions):
            cost_matrix[0, i] = 1.0 - action_probs[i]
        
        if SCIPY_AVAILABLE:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            return col_ind[0]
        else:
            return np.argmin(cost_matrix[0])
    
    def _project_continuous_params(self, action_params, discrete_action):
        """
        Project continuous parameters to valid range
        Uses QP (Quadratic Programming) to solve for nearest feasible solution
        
        Objective: min ||x - v||^2
        Constraints: param_min <= x <= param_max
        
        Args:
            action_params: Original continuous parameters [3]
            discrete_action: Selected discrete action
        
        Returns:
            Projected parameters (only returns parameters needed for the selected action)
        """
        start_idx = self.action_param_offsets[discrete_action]
        end_idx = self.action_param_offsets[discrete_action + 1]
        params_for_action = action_params[start_idx:end_idx]
        
        if self.use_qp:
            projected = self._solve_qp(params_for_action)
        else:
            projected = np.clip(params_for_action, self.param_min, self.param_max)
        
        return projected
    
    def _solve_qp(self, v):
        """
        Solve parameter projection using QP
        
        Objective function: min 0.5 * ||x - v||^2
        Constraints: param_min <= x <= param_max
        
        For simple box constraints, the optimal solution is to project v to [min,max] range
        i.e., x* = clip(v, min, max)
        
        Args:
            v: Original parameter vector
        
        Returns:
            Projected parameter vector
        """
        v = np.asarray(v, dtype=np.float64)
        n = len(v)
        
        if n == 0:
            return v
        
        if SCIPY_AVAILABLE and self.use_qp:
            def objective(x):
                return 0.5 * np.sum((x - v) ** 2)
            
            def gradient(x):
                return x - v
            
            bounds = [(self.param_min, self.param_max) for _ in range(n)]
            
            x0 = np.clip(v, self.param_min, self.param_max)
            
            try:
                result = minimize(
                    objective,
                    x0,
                    method='L-BFGS-B',
                    jac=gradient,
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-8}
                )
                
                if result.success:
                    return result.x
                else:
                    return np.clip(v, self.param_min, self.param_max)
            except Exception:
                return np.clip(v, self.param_min, self.param_max)
        else:
            return np.clip(v, self.param_min, self.param_max)
    
    def project_with_constraints(self, action_probs, action_params, 
                                  action_mask=None, param_constraints=None):
        """
        Action projection with additional constraints
        
        Args:
            action_probs: Discrete action probabilities
            action_params: Continuous parameters
            action_mask: Optional action mask [3], True means action is available
            param_constraints: Optional parameter constraints dictionary
        
        Returns:
            discrete_action, continuous_params, timing_info
        """
        if action_mask is not None:
            masked_probs = action_probs.copy()
            masked_probs[~action_mask] = -np.inf
            action_probs = masked_probs
        
        return self.project(action_probs, action_params)
    
    def get_timing_statistics(self):
        """Get timing statistics"""
        stats = {}
        
        if self.discrete_selection_times:
            stats['discrete_selection'] = {
                'mean': np.mean(self.discrete_selection_times),
                'std': np.std(self.discrete_selection_times),
                'min': np.min(self.discrete_selection_times),
                'max': np.max(self.discrete_selection_times),
                'count': len(self.discrete_selection_times)
            }
        
        if self.qp_times:
            stats['qp'] = {
                'mean': np.mean(self.qp_times),
                'std': np.std(self.qp_times),
                'min': np.min(self.qp_times),
                'max': np.max(self.qp_times),
                'count': len(self.qp_times)
            }
        
        if self.total_projection_times:
            stats['total_projection'] = {
                'mean': np.mean(self.total_projection_times),
                'std': np.std(self.total_projection_times),
                'min': np.min(self.total_projection_times),
                'max': np.max(self.total_projection_times),
                'count': len(self.total_projection_times)
            }
        
        return stats
    
    def print_timing_summary(self):
        """Print timing summary"""
        stats = self.get_timing_statistics()
        
        print("\n" + "="*60)
        print("Action Projection Timing Summary")
        print("="*60)
        
        if 'discrete_selection' in stats:
            ds = stats['discrete_selection']
            print(f"\nDiscrete Action Selection (Hungarian):")
            print(f"  Mean: {ds['mean']*1000:.4f} ms")
            print(f"  Std: {ds['std']*1000:.4f} ms")
        
        if 'qp' in stats:
            qp = stats['qp']
            print(f"\nQP Solving (Continuous Parameter Projection):")
            print(f"  Mean: {qp['mean']*1000:.4f} ms")
            print(f"  Std: {qp['std']*1000:.4f} ms")
        
        if 'total_projection' in stats:
            tp = stats['total_projection']
            print(f"\nTotal Projection Time:")
            print(f"  Mean: {tp['mean']*1000:.4f} ms")
            print(f"  Std: {tp['std']*1000:.4f} ms")
            print(f"  Call Count: {tp['count']}")
        
        print("="*60)


# =======================
# Neural Network Definitions
# =======================
class ReplayBuffer:
    """Experience replay buffer"""
    def __init__(self, buffer_limit):
        self.buffer = collections.deque(maxlen=buffer_limit)
    
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
        
        s_tensor = torch.from_numpy(s_arr).to(device)
        a_tensor = torch.from_numpy(a_arr).to(device)
        r_tensor = torch.from_numpy(r_arr).to(device)
        s_prime_tensor = torch.from_numpy(s_prime_arr).to(device)
        done_tensor = torch.from_numpy(done_arr).to(device)
        
        return s_tensor, a_tensor, r_tensor, s_prime_tensor, done_tensor
    
    def size(self):
        return len(self.buffer)


class MuNet(nn.Module):
    """Actor Network - Outputs continuous actions"""
    def __init__(self, state_size, action_size, hidden_layers=(128,)):
        super(MuNet, self).__init__()
        self.layers = nn.ModuleList()
        
        last_size = state_size
        for hidden_size in hidden_layers:
            self.layers.append(nn.Linear(last_size, hidden_size))
            last_size = hidden_size
        
        self.action_output = nn.Linear(last_size, 3)
        self.param_output = nn.Linear(last_size, action_size)

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        
        action_probs = F.softmax(self.action_output(x), dim=-1)
        action_params = torch.tanh(self.param_output(x))
        
        return action_probs, action_params


class QNet(nn.Module):
    """Critic Network - Evaluates Q-values"""
    def __init__(self, state_size, action_size, hidden_layers=(128,)):
        super(QNet, self).__init__()
        input_size = state_size + 3 + action_size
        
        self.layers = nn.ModuleList()
        last_size = input_size
        for hidden_size in hidden_layers:
            self.layers.append(nn.Linear(last_size, hidden_size))
            last_size = hidden_size
        
        self.output_layer = nn.Linear(last_size, 1)

    def forward(self, state, action_probs, action_params):
        x = torch.cat([state, action_probs, action_params], dim=-1)
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.output_layer(x)


class OrnsteinUhlenbeckNoise:
    """OU noise for exploration"""
    def __init__(self, size, mu=0., theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.size = size
        self.reset()

    def reset(self):
        self.x_prev = np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) + \
            self.sigma * np.random.randn(self.size)
        self.x_prev = x
        return x


# =======================
# LIRL Agent
# =======================
class LIRLAgent:
    """LIRL Agent for Platform Environment"""
    
    def __init__(self, state_size, action_param_size, action_space=None, 
                 hidden_layers=(128,), lr_actor=1e-3, lr_critic=1e-3, 
                 gamma=0.9, tau=0.005, buffer_size=10000, batch_size=128, 
                 use_action_projection=True, seed=None):
        
        self.state_size = state_size
        self.action_param_size = action_param_size
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.use_action_projection = use_action_projection
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        self.mu = MuNet(state_size, action_param_size, hidden_layers).to(device)
        self.mu_target = MuNet(state_size, action_param_size, hidden_layers).to(device)
        self.mu_target.load_state_dict(self.mu.state_dict())
        
        self.q = QNet(state_size, action_param_size, hidden_layers).to(device)
        self.q_target = QNet(state_size, action_param_size, hidden_layers).to(device)
        self.q_target.load_state_dict(self.q.state_dict())
        
        self.mu_optimizer = optim.Adam(self.mu.parameters(), lr=lr_actor)
        self.q_optimizer = optim.Adam(self.q.parameters(), lr=lr_critic)
        
        self.memory = ReplayBuffer(buffer_size)
        
        self.noise = OrnsteinUhlenbeckNoise(action_param_size)
        
        self.action_projector = ActionProjection(action_space, use_qp=True)
        
        self.action_param_min = np.array([-1., -1., -1.])
        self.action_param_max = np.array([1., 1., 1.])
        
        self.epsilon = 1.0
        self.epsilon_final = 0.01
        self.epsilon_decay = 0.9995
        
    def act(self, state, add_noise=True):
        """
        Select action
        
        Uses LIRL action projection function to map network output to valid action space
        """
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).float().to(device)
            if len(state_tensor.shape) == 1:
                state_tensor = state_tensor.unsqueeze(0)
            
            action_probs, action_params = self.mu(state_tensor)
            action_probs = action_probs.cpu().numpy().squeeze()
            action_params = action_params.cpu().numpy().squeeze()
        
        if add_noise:
            if np.random.random() < self.epsilon:
                action_probs = np.random.dirichlet(np.ones(3))
                action_params = np.random.uniform(-1, 1, self.action_param_size)
            else:
                action_params = action_params + self.noise() * 0.1
                action_params = np.clip(action_params, -1, 1)
        
        if self.use_action_projection:
            action, act_param, timing_info = self.action_projector.project(
                action_probs, action_params, record_timing=True
            )
        else:
            action = np.argmax(action_probs)
            act_param = np.clip(action_params[action:action+1], -1, 1)
        
        return action, act_param, action_probs, action_params
    
    def step(self, state, action_probs, action_params, reward, next_state, done):
        """Store experience and learn"""
        combined_action = np.concatenate([action_probs, action_params])
        self.memory.put((state, combined_action, reward, next_state, done))
    
    def learn(self):
        """Learn from experience"""
        if self.memory.size() < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        action_probs = actions[:, :3]
        action_params = actions[:, 3:]
        
        with torch.no_grad():
            next_action_probs, next_action_params = self.mu_target(next_states)
            target_q = self.q_target(next_states, next_action_probs, next_action_params)
            target = rewards.unsqueeze(1) + self.gamma * target_q * dones
        
        current_q = self.q(states, action_probs, action_params)
        q_loss = F.mse_loss(current_q, target)
        
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        
        pred_action_probs, pred_action_params = self.mu(states)
        actor_loss = -self.q(states, pred_action_probs, pred_action_params).mean()
        
        self.mu_optimizer.zero_grad()
        actor_loss.backward()
        self.mu_optimizer.step()
        
        self._soft_update(self.mu, self.mu_target)
        self._soft_update(self.q, self.q_target)
    
    def _soft_update(self, source, target):
        """Soft update target network"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
    
    def decay_epsilon(self):
        """Decay epsilon"""
        self.epsilon = max(self.epsilon_final, self.epsilon * self.epsilon_decay)
    
    def reset_noise(self):
        """Reset noise"""
        self.noise.reset()
    
    def start_episode(self):
        """Start new episode"""
        self.reset_noise()
    
    def end_episode(self):
        """End episode"""
        self.decay_epsilon()
    
    def get_projection_stats(self):
        """Get action projection statistics"""
        return self.action_projector.get_timing_statistics()
    
    def print_projection_summary(self):
        """Print action projection summary"""
        self.action_projector.print_timing_summary()
    
    def save_models(self, prefix):
        """Save models"""
        torch.save(self.mu.state_dict(), '{}_mu.pth'.format(prefix))
        torch.save(self.q.state_dict(), '{}_q.pth'.format(prefix))
    
    def __str__(self):
        return (f"LIRL Agent (Platform)\n"
                f"State size: {self.state_size}\n"
                f"Action param size: {self.action_param_size}\n"
                f"Gamma: {self.gamma}\n"
                f"Tau: {self.tau}\n"
                f"Batch size: {self.batch_size}\n"
                f"Use action projection: {self.use_action_projection}\n"
                f"Epsilon: {self.epsilon:.4f}\n"
                f"Memory size: {self.memory.size()}")


def pad_action(act, act_param):
    """Convert action to format required by environment"""
    params = [np.zeros((1,), dtype=np.float32), np.zeros((1,), dtype=np.float32), np.zeros((1,), dtype=np.float32)]
    params[act][:] = act_param
    return (act, params)


def evaluate(env, agent, episodes=1000):
    """Evaluate agent"""
    returns = []
    timesteps = []
    for _ in range(episodes):
        state, _ = env.reset()
        terminal = False
        t = 0
        total_reward = 0.
        while not terminal:
            t += 1
            state = np.array(state, dtype=np.float32, copy=False)
            act, act_param, _, _ = agent.act(state, add_noise=False)
            action = pad_action(act, act_param)
            (state, _), reward, terminal, _ = env.step(action)
            total_reward += reward
        timesteps.append(t)
        returns.append(total_reward)
    return np.array(returns)


@click.command()
@click.option('--seed', default=1, help='Random seed.', type=int)
@click.option('--episodes', default=20000, help='Number of episodes.', type=int)
@click.option('--evaluation-episodes', default=1000, help='Episodes over which to evaluate after training.', type=int)
@click.option('--batch-size', default=128, help='Minibatch size.', type=int)
@click.option('--gamma', default=0.9, help='Discount factor.', type=float)
@click.option('--initial-memory-threshold', default=500, help='Number of transitions required to start learning.', type=int)
@click.option('--replay-memory-size', default=10000, help='Replay memory size in transitions.', type=int)
@click.option('--tau', default=0.005, help='Soft target network update averaging factor.', type=float)
@click.option('--learning-rate-actor', default=1e-3, help="Actor network learning rate.", type=float)
@click.option('--learning-rate-critic', default=1e-3, help="Critic network learning rate.", type=float)
@click.option('--scale-actions', default=True, help="Scale actions.", type=bool)
@click.option('--use-action-projection', default=True, help="Use LIRL action projection (Hungarian + QP).", type=bool)
@click.option('--save-dir', default="results/platform", help='Model save directory.', type=str)
@click.option('--save-freq', default=0, help='How often to save models (0 = never).', type=int)
@click.option('--layers', default="[128,]", help='Hidden layer sizes.', cls=ClickPythonLiteralOption)
@click.option('--render-freq', default=100, help='How often to render / save frames of an episode.', type=int)
@click.option('--visualise', default=True, help="Render game states.", type=bool)
@click.option('--title', default="LIRL", help="Prefix of output files", type=str)
def run(seed, episodes, evaluation_episodes, batch_size, gamma, initial_memory_threshold,
        replay_memory_size, tau, learning_rate_actor, learning_rate_critic,
        scale_actions, use_action_projection, layers, save_dir, save_freq, 
        render_freq, visualise, title):
    
    if save_freq > 0 and save_dir:
        save_dir_full = os.path.join(save_dir, title + "{}".format(str(seed)))
        os.makedirs(save_dir_full, exist_ok=True)
    
    if visualise:
        assert render_freq > 0
    
    env = gym.make('Platform-v0')
    
    initial_params_ = [3., 10., 400.]
    if scale_actions:
        for a in range(env.action_space.spaces[0].n):
            initial_params_[a] = 2. * (initial_params_[a] - env.action_space.spaces[1].spaces[a].low) / (
                        env.action_space.spaces[1].spaces[a].high - env.action_space.spaces[1].spaces[a].low) - 1.
    
    env = ScaledStateWrapper(env)
    env = PlatformFlattenedActionWrapper(env)
    if scale_actions:
        env = ScaledParameterisedActionWrapper(env)
    
    dir = os.path.join(save_dir, title)
    env = Monitor(env, directory=os.path.join(dir, str(seed)), video_callable=False, write_upon_reset=False, force=True)
    
    print(env.action_space)
    print(env.observation_space)
    
    env.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    
    state_size = env.observation_space.spaces[0].shape[0]
    action_param_size = 3
    
    print(f"State size: {state_size}")
    print(f"Action param size: {action_param_size}")
    
    agent = LIRLAgent(
        state_size=state_size,
        action_param_size=action_param_size,
        action_space=env.action_space,
        hidden_layers=layers,
        lr_actor=learning_rate_actor,
        lr_critic=learning_rate_critic,
        gamma=gamma,
        tau=tau,
        buffer_size=replay_memory_size,
        batch_size=batch_size,
        use_action_projection=use_action_projection,
        seed=seed
    )
    
    print(agent)
    
    max_steps = 250
    total_reward = 0.
    returns = []
    start_time = time.time()
    
    for i in range(episodes):
        if save_freq > 0 and save_dir and i % save_freq == 0:
            agent.save_models(os.path.join(save_dir, title + str(seed), str(i)))
        
        state, _ = env.reset()
        state = np.array(state, dtype=np.float32, copy=False)
        
        if visualise and i % render_freq == 0:
            env.render()
        
        agent.start_episode()
        episode_reward = 0.
        
        for j in range(max_steps):
            act, act_param, action_probs, action_params = agent.act(state)
            action = pad_action(act, act_param)
            
            ret = env.step(action)
            (next_state, steps), reward, terminal, _ = ret
            next_state = np.array(next_state, dtype=np.float32, copy=False)
            
            agent.step(state, action_probs, action_params, reward, next_state, terminal)
            
            if agent.memory.size() >= initial_memory_threshold:
                agent.learn()
            
            state = next_state
            episode_reward += reward
            
            if visualise and i % render_freq == 0:
                env.render()
            
            if terminal:
                break
        
        agent.end_episode()
        
        returns.append(episode_reward)
        total_reward += episode_reward
        
        if i % 100 == 0:
            print('{0:5s} R:{1:.4f} r100:{2:.4f}'.format(
                str(i), total_reward / (i + 1), np.array(returns[-100:]).mean()))
    
    end_time = time.time()
    print("Took %.2f seconds" % (end_time - start_time))
    env.close()
    
    if save_freq > 0 and save_dir:
        agent.save_models(os.path.join(save_dir, title + str(seed), str(i)))
    
    print(agent)
    
    if use_action_projection:
        agent.print_projection_summary()
    
    returns = env.get_episode_rewards()
    print("Ave. return =", sum(returns) / len(returns))
    print("Ave. last 100 episode return =", sum(returns[-100:]) / 100.)
    
    np.save(os.path.join(dir, title + "{}".format(str(seed))), returns)
    
    torch.save(agent.mu.state_dict(), os.path.join(dir, 'mu_{}.pth'.format(seed)))
    torch.save(agent.q.state_dict(), os.path.join(dir, 'q_{}.pth'.format(seed)))
    
    if evaluation_episodes > 0:
        print("Evaluating agent over {} episodes".format(evaluation_episodes))
        agent.epsilon = 0.
        evaluation_returns = evaluate(env, agent, evaluation_episodes)
        print("Ave. evaluation return =", sum(evaluation_returns) / len(evaluation_returns))
        np.save(os.path.join(dir, title + "{}e".format(str(seed))), evaluation_returns)


if __name__ == '__main__':
    run()

