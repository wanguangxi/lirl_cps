#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LIRL-PPO Agent for Goal Environment
====================================
LIRL implementation based on PPO algorithm

PPO (Proximal Policy Optimization) features:
- On-policy algorithm
- Uses clipped surrogate objective to limit policy update magnitude
- More stable training process
- Supports GAE (Generalized Advantage Estimation)
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
from torch.distributions import Categorical, Normal
import gym
import gym_goal
from gym_goal.envs.config import GOAL_WIDTH, PITCH_LENGTH, PITCH_WIDTH
from gym.wrappers import Monitor
from common import ClickPythonLiteralOption
from common.wrappers import ScaledParameterisedActionWrapper
from common.goal_domain import GoalFlattenedActionWrapper, GoalObservationWrapper
from common.wrappers import ScaledStateWrapper

# Try importing scipy
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
# Action Projection Function (same as DDPG version)
# =======================
class ActionProjection:
    """
    LIRL Action Projection Class
    Projects network output continuous actions to valid discrete-continuous mixed action space
    """
    
    def __init__(self, action_space, use_qp=True):
        self.use_qp = use_qp
        self.num_actions = 3
        self.action_param_sizes = [2, 1, 1]
        self.action_param_offsets = [0, 2, 3, 4]
        self.param_min = -1.0
        self.param_max = 1.0
        self.reset_timings()
    
    def reset_timings(self):
        self.discrete_selection_times = []
        self.qp_times = []
        self.total_projection_times = []
    
    def project(self, action_probs, action_params, record_timing=True):
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
        cost_matrix = np.zeros((1, self.num_actions))
        for i in range(self.num_actions):
            cost_matrix[0, i] = 1.0 - action_probs[i]
        
        if SCIPY_AVAILABLE:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            return col_ind[0]
        else:
            return np.argmin(cost_matrix[0])
    
    def _project_continuous_params(self, action_params, discrete_action):
        start_idx = self.action_param_offsets[discrete_action]
        end_idx = self.action_param_offsets[discrete_action + 1]
        params_for_action = action_params[start_idx:end_idx]
        
        if self.use_qp:
            projected = self._solve_qp(params_for_action)
        else:
            projected = np.clip(params_for_action, self.param_min, self.param_max)
        
        return projected
    
    def _solve_qp(self, v):
        v = np.asarray(v, dtype=np.float64)
        return np.clip(v, self.param_min, self.param_max)
    
    def get_timing_statistics(self):
        stats = {}
        if self.total_projection_times:
            stats['total_projection'] = {
                'mean': np.mean(self.total_projection_times),
                'count': len(self.total_projection_times)
            }
        return stats
    
    def print_timing_summary(self):
        stats = self.get_timing_statistics()
        if 'total_projection' in stats:
            tp = stats['total_projection']
            print(f"Action projection average time: {tp['mean']*1000:.4f} ms, call count: {tp['count']}")


# =======================
# PPO Network Definition
# =======================
class ActorCritic(nn.Module):
    """
    Actor-Critic Network for PPO
    
    Actor outputs:
    - Discrete action probability distribution (Categorical)
    - Continuous parameter mean and standard deviation (Gaussian)
    
    Critic outputs:
    - State value V(s)
    """
    def __init__(self, state_size, action_param_size, hidden_layers=(128, 64)):
        super(ActorCritic, self).__init__()
        
        self.state_size = state_size
        self.action_param_size = action_param_size
        self.num_discrete_actions = 3
        
        self.shared_layers = nn.ModuleList()
        last_size = state_size
        for hidden_size in hidden_layers:
            self.shared_layers.append(nn.Linear(last_size, hidden_size))
            last_size = hidden_size
        
        self.actor_discrete = nn.Linear(last_size, self.num_discrete_actions)
        
        self.actor_param_mean = nn.Linear(last_size, action_param_size)
        
        self.actor_param_log_std = nn.Parameter(torch.zeros(action_param_size))
        
        self.critic = nn.Linear(last_size, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for layer in self.shared_layers:
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            nn.init.zeros_(layer.bias)
        
        nn.init.orthogonal_(self.actor_discrete.weight, gain=0.01)
        nn.init.zeros_(self.actor_discrete.bias)
        
        nn.init.orthogonal_(self.actor_param_mean.weight, gain=0.01)
        nn.init.zeros_(self.actor_param_mean.bias)
        
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)
    
    def forward(self, state):
        x = state
        for layer in self.shared_layers:
            x = torch.tanh(layer(x))
        
        action_logits = self.actor_discrete(x)
        action_probs = F.softmax(action_logits, dim=-1)
        
        param_mean = self.actor_param_mean(x)
        param_std = torch.exp(self.actor_param_log_std).expand_as(param_mean)
        
        value = self.critic(x)
        
        return action_probs, param_mean, param_std, value
    
    def get_action_and_value(self, state, discrete_action=None, action_params=None):
        """
        Get action and value estimate
        
        If action is provided, compute log probability; otherwise sample new action
        """
        action_probs, param_mean, param_std, value = self.forward(state)
        
        discrete_dist = Categorical(action_probs)
        
        param_dist = Normal(param_mean, param_std)
        
        if discrete_action is None:
            discrete_action = discrete_dist.sample()
            action_params = param_dist.sample()
            action_params = torch.tanh(action_params)
        
        discrete_log_prob = discrete_dist.log_prob(discrete_action)
        
        raw_params = torch.atanh(torch.clamp(action_params, -0.999, 0.999))
        param_log_prob = param_dist.log_prob(raw_params).sum(dim=-1)
        param_log_prob = param_log_prob - torch.log(1 - action_params.pow(2) + 1e-6).sum(dim=-1)
        
        total_log_prob = discrete_log_prob + param_log_prob
        
        discrete_entropy = discrete_dist.entropy()
        param_entropy = param_dist.entropy().sum(dim=-1)
        total_entropy = discrete_entropy + param_entropy
        
        return discrete_action, action_probs, action_params, total_log_prob, total_entropy, value


# =======================
# Rollout Buffer
# =======================
class RolloutBuffer:
    """PPO experience buffer"""
    def __init__(self):
        self.states = []
        self.discrete_actions = []
        self.action_probs = []
        self.action_params = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
    
    def add(self, state, discrete_action, action_probs, action_params, reward, done, log_prob, value):
        self.states.append(state)
        self.discrete_actions.append(discrete_action)
        self.action_probs.append(action_probs)
        self.action_params.append(action_params)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
    
    def get(self):
        return (
            np.array(self.states, dtype=np.float32),
            np.array(self.discrete_actions),
            np.array(self.action_probs, dtype=np.float32),
            np.array(self.action_params, dtype=np.float32),
            np.array(self.rewards, dtype=np.float32),
            np.array(self.dones, dtype=np.float32),
            np.array(self.log_probs, dtype=np.float32),
            np.array(self.values, dtype=np.float32)
        )
    
    def clear(self):
        self.states = []
        self.discrete_actions = []
        self.action_probs = []
        self.action_params = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
    
    def size(self):
        return len(self.states)


# =======================
# LIRL-PPO Agent
# =======================
class LIRLPPOAgent:
    """
    LIRL Agent based on PPO
    
    PPO features:
    - Clipped surrogate objective
    - GAE (Generalized Advantage Estimation)
    - Value function clipping
    - Entropy bonus
    """
    
    def __init__(self, state_size, action_param_size, action_space=None,
                 hidden_layers=(128, 64), lr=3e-4, gamma=0.99, gae_lambda=0.95,
                 clip_epsilon=0.2, c1=0.5, c2=0.01, update_epochs=10,
                 batch_size=64, use_action_projection=True, seed=None):
        
        self.state_size = state_size
        self.action_param_size = action_param_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.c1 = c1  # Value loss coefficient
        self.c2 = c2  # Entropy coefficient
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.use_action_projection = use_action_projection
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        self.policy = ActorCritic(state_size, action_param_size, hidden_layers).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, eps=1e-5)
        
        self.buffer = RolloutBuffer()
        
        self.action_projector = ActionProjection(action_space, use_qp=True)
        
        self.total_steps = 0
    
    def act(self, state):
        """Select action"""
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).float().to(device)
            if len(state_tensor.shape) == 1:
                state_tensor = state_tensor.unsqueeze(0)
            
            discrete_action, action_probs, action_params, log_prob, _, value = \
                self.policy.get_action_and_value(state_tensor)
            
            discrete_action = discrete_action.cpu().numpy().item()
            action_probs = action_probs.cpu().numpy().squeeze()
            action_params = action_params.cpu().numpy().squeeze()
            log_prob = log_prob.cpu().numpy().item()
            value = value.cpu().numpy().item()
        
        if self.use_action_projection:
            projected_action, projected_params, _ = self.action_projector.project(
                action_probs, action_params, record_timing=True
            )
        else:
            projected_action = discrete_action
            if projected_action == 0:
                projected_params = np.clip(action_params[:2], -1, 1)
            elif projected_action == 1:
                projected_params = np.clip(action_params[2:3], -1, 1)
            else:
                projected_params = np.clip(action_params[3:4], -1, 1)
        
        return projected_action, projected_params, action_probs, action_params, log_prob, value
    
    def store_transition(self, state, discrete_action, action_probs, action_params, 
                         reward, done, log_prob, value):
        """Store transition"""
        self.buffer.add(state, discrete_action, action_probs, action_params, 
                        reward, done, log_prob, value)
        self.total_steps += 1
    
    def compute_gae(self, rewards, values, dones, next_value):
        """Compute GAE (Generalized Advantage Estimation)"""
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
        
        returns = advantages + values
        return advantages, returns
    
    def update(self, next_state):
        """PPO update"""
        if self.buffer.size() == 0:
            return {}
        
        states, discrete_actions, action_probs, action_params, rewards, dones, old_log_probs, values = \
            self.buffer.get()
        
        with torch.no_grad():
            next_state_tensor = torch.from_numpy(next_state).float().to(device).unsqueeze(0)
            _, _, _, _, _, next_value = self.policy.get_action_and_value(next_state_tensor)
            next_value = next_value.cpu().numpy().item()
        
        advantages, returns = self.compute_gae(rewards, values, dones, next_value)
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        states_t = torch.from_numpy(states).float().to(device)
        discrete_actions_t = torch.from_numpy(discrete_actions).long().to(device)
        action_params_t = torch.from_numpy(action_params).float().to(device)
        old_log_probs_t = torch.from_numpy(old_log_probs).float().to(device)
        advantages_t = torch.from_numpy(advantages).float().to(device)
        returns_t = torch.from_numpy(returns).float().to(device)
        old_values_t = torch.from_numpy(values).float().to(device)
        
        total_loss_list = []
        policy_loss_list = []
        value_loss_list = []
        entropy_list = []
        
        num_samples = len(states)
        indices = np.arange(num_samples)
        
        for epoch in range(self.update_epochs):
            np.random.shuffle(indices)
            
            for start in range(0, num_samples, self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                batch_states = states_t[batch_indices]
                batch_discrete_actions = discrete_actions_t[batch_indices]
                batch_action_params = action_params_t[batch_indices]
                batch_old_log_probs = old_log_probs_t[batch_indices]
                batch_advantages = advantages_t[batch_indices]
                batch_returns = returns_t[batch_indices]
                batch_old_values = old_values_t[batch_indices]
                
                _, _, _, new_log_probs, entropy, new_values = \
                    self.policy.get_action_and_value(
                        batch_states, batch_discrete_actions, batch_action_params
                    )
                
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                new_values = new_values.squeeze()
                value_clipped = batch_old_values + torch.clamp(
                    new_values - batch_old_values, -self.clip_epsilon, self.clip_epsilon
                )
                value_loss1 = (new_values - batch_returns).pow(2)
                value_loss2 = (value_clipped - batch_returns).pow(2)
                value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()
                
                entropy_loss = -entropy.mean()
                
                total_loss = policy_loss + self.c1 * value_loss + self.c2 * entropy_loss
                
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
                
                total_loss_list.append(total_loss.item())
                policy_loss_list.append(policy_loss.item())
                value_loss_list.append(value_loss.item())
                entropy_list.append(-entropy_loss.item())
        
        self.buffer.clear()
        
        return {
            'total_loss': np.mean(total_loss_list),
            'policy_loss': np.mean(policy_loss_list),
            'value_loss': np.mean(value_loss_list),
            'entropy': np.mean(entropy_list)
        }
    
    def get_projection_stats(self):
        return self.action_projector.get_timing_statistics()
    
    def print_projection_summary(self):
        self.action_projector.print_timing_summary()
    
    def __str__(self):
        return (f"LIRL-PPO Agent\n"
                f"State size: {self.state_size}\n"
                f"Action param size: {self.action_param_size}\n"
                f"Gamma: {self.gamma}\n"
                f"GAE Lambda: {self.gae_lambda}\n"
                f"Clip Epsilon: {self.clip_epsilon}\n"
                f"Update Epochs: {self.update_epochs}\n"
                f"Batch Size: {self.batch_size}\n"
                f"Use action projection: {self.use_action_projection}\n"
                f"Total Steps: {self.total_steps}")


def pad_action(act, act_param):
    """Convert action to format required by environment"""
    params = [np.zeros((2,)), np.zeros((1,)), np.zeros((1,))]
    params[act] = act_param
    return (act, params)


def evaluate(env, agent, episodes=1000):
    """Evaluate agent"""
    returns = []
    for _ in range(episodes):
        state, _ = env.reset()
        terminal = False
        total_reward = 0.
        while not terminal:
            state = np.array(state, dtype=np.float32, copy=False)
            act, act_param, _, _, _, _ = agent.act(state)
            action = pad_action(act, act_param)
            (state, _), reward, terminal, _ = env.step(action)
            total_reward += reward
        returns.append(total_reward)
    return np.array(returns)


@click.command()
@click.option('--seed', default=0, help='Random seed.', type=int)
@click.option('--episodes', default=50000, help='Number of episodes.', type=int)
@click.option('--evaluation-episodes', default=100, help='Episodes over which to evaluate after training.', type=int)
@click.option('--gamma', default=0.99, help='Discount factor.', type=float)
@click.option('--gae-lambda', default=0.95, help='GAE lambda.', type=float)
@click.option('--lr', default=3e-4, help='Learning rate.', type=float)
@click.option('--clip-epsilon', default=0.2, help='PPO clip epsilon.', type=float)
@click.option('--update-epochs', default=10, help='Number of PPO update epochs.', type=int)
@click.option('--batch-size', default=64, help='Minibatch size.', type=int)
@click.option('--rollout-steps', default=2048, help='Steps per rollout before update.', type=int)
@click.option('--c1', default=0.5, help='Value loss coefficient.', type=float)
@click.option('--c2', default=0.01, help='Entropy coefficient.', type=float)
@click.option('--reward-scale', default=1./50., help='Reward scaling factor.', type=float)
@click.option('--use-action-projection', default=True, help='Use LIRL action projection.', type=bool)
@click.option('--layers', default="(128,64)", help='Hidden layer sizes.', cls=ClickPythonLiteralOption)
@click.option('--save-dir', default="results/goal", help='Output directory.', type=str)
@click.option('--title', default="LIRL_PPO", help="Prefix of output files", type=str)
def run(seed, episodes, evaluation_episodes, gamma, gae_lambda, lr, clip_epsilon,
        update_epochs, batch_size, rollout_steps, c1, c2, reward_scale,
        use_action_projection, layers, save_dir, title):
    
    env = gym.make('Goal-v0')
    env = GoalObservationWrapper(env)
    env = GoalFlattenedActionWrapper(env)
    env = ScaledParameterisedActionWrapper(env)
    env = ScaledStateWrapper(env)
    
    dir = os.path.join(save_dir, title)
    os.makedirs(dir, exist_ok=True)
    env = Monitor(env, directory=os.path.join(dir, str(seed)), video_callable=False, 
                  write_upon_reset=False, force=True)
    
    print("="*60)
    print("LIRL-PPO for Goal Environment")
    print("="*60)
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Seed: {seed}")
    print(f"Episodes: {episodes}")
    print(f"Hidden layers: {layers}")
    print(f"PPO Clip Epsilon: {clip_epsilon}")
    print(f"Rollout Steps: {rollout_steps}")
    print("="*60)
    
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    
    state_size = env.observation_space.spaces[0].shape[0]
    action_param_size = 4
    
    agent = LIRLPPOAgent(
        state_size=state_size,
        action_param_size=action_param_size,
        action_space=env.action_space,
        hidden_layers=layers,
        lr=lr,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_epsilon=clip_epsilon,
        c1=c1,
        c2=c2,
        update_epochs=update_epochs,
        batch_size=batch_size,
        use_action_projection=use_action_projection,
        seed=seed
    )
    
    print(agent)
    
    max_steps = 150
    total_reward = 0.
    returns = []
    start_time = time.time()
    
    episode = 0
    step_count = 0
    
    while episode < episodes:
        state, _ = env.reset()
        state = np.array(state, dtype=np.float32, copy=False)
        episode_reward = 0.
        
        for _ in range(max_steps):
            action, act_param, action_probs, action_params, log_prob, value = agent.act(state)
            env_action = pad_action(action, act_param)
            
            (next_state, _), reward, terminal, _ = env.step(env_action)
            next_state = np.array(next_state, dtype=np.float32, copy=False)
            
            r = reward * reward_scale
            
            agent.store_transition(state, action, action_probs, action_params, 
                                   r, terminal, log_prob, value)
            
            state = next_state
            episode_reward += reward
            step_count += 1
            
            if step_count % rollout_steps == 0:
                update_info = agent.update(next_state)
            
            if terminal:
                break
        
        returns.append(episode_reward)
        total_reward += episode_reward
        episode += 1
        
        if episode % 100 == 0:
            avg_reward = total_reward / episode
            success_rate = (np.array(returns) == 50.).sum() / len(returns)
            print('{0:5s} R:{1:.5f} P(S):{2:.4f}'.format(
                str(episode), avg_reward, success_rate))
    
    end_time = time.time()
    print("Took %.2f seconds" % (end_time - start_time))
    env.close()
    print(agent)
    
    if use_action_projection:
        agent.print_projection_summary()
    
    returns = env.get_episode_rewards()
    np.save(os.path.join(dir, title + "{}".format(str(seed))), returns)
    
    torch.save(agent.policy.state_dict(), os.path.join(dir, 'policy_{}.pth'.format(seed)))
    
    if evaluation_episodes > 0:
        print("Evaluating agent over {} episodes".format(evaluation_episodes))
        evaluation_returns = evaluate(env, agent, evaluation_episodes)
        print("Ave. evaluation return =", sum(evaluation_returns) / len(evaluation_returns))
        print("Ave. evaluation prob. =", sum(evaluation_returns == 50.) / len(evaluation_returns))
        np.save(os.path.join(dir, title + "{}e".format(str(seed))), evaluation_returns)


if __name__ == '__main__':
    run()

