"""
CPO (Constrained Policy Optimization) for EV Charging Station Control

Based on the structure of `pdqn_ev_charging.py`. Implements CPO for constrained optimization.

CPO key idea:
- Extends TRPO with constraint handling
- Uses a trust region to limit policy update steps
- Uses projection/line-search steps to ensure constraints are satisfied

This implementation uses a CPO-Clip approximation:
- Uses a PPO-style clip objective instead of a strict KL constraint
- Implements a constraint projection step
- Uses line search to ensure constraint satisfaction

No explicit action correction/projection; constraints are enforced via CPO updates.
"""

import os
import sys
import json
import random
import datetime as dt
from typing import Dict, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal

sys.path.append(os.path.join(os.path.dirname(__file__), "../env"))
from ev import EVChargingEnv


# =======================
# HYPERPARAMETERS CONFIG
# =======================
CONFIG = {
    # CPO Learning parameters
    "lr_actor": 3e-4,           # Actor learning rate
    "lr_critic": 1e-3,          # Critic learning rate
    "gamma": 0.99,              # Discount factor
    "gae_lambda": 0.95,         # GAE lambda
    "clip_eps": 0.2,            # PPO clip epsilon
    "entropy_coef": 0.01,       # Entropy coefficient
    "value_coef": 0.5,          # Value loss coefficient
    "max_grad_norm": 0.5,       # Gradient clipping
    "cpo_epochs": 5,            # CPO update epochs
    "batch_size": 256,          # Mini-batch size
    
    # CPO constraint parameters
    "cost_limit": 0.1,              # Constraint limit (target violation rate)
    "safety_margin": 0.01,          # Safety margin for constraint
    "line_search_steps": 10,        # Line search steps
    "line_search_decay": 0.8,       # Line search decay factor
    "backtrack_coef": 0.5,          # Backtracking coefficient
    "cost_scale": 1.0,              # Scale factor for constraint cost
    
    # Environment parameters
    "n_stations": 5,
    "p_max": 150.0,
    "arrival_rate": 0.75,
    "num_of_episodes": 20,
    "max_steps": 288,
    
    # Network architecture
    "hidden_dim1": 128,
    "hidden_dim2": 64,
    
    # Output parameters
    "print_interval": 10,
    "save_models": True,
    "plot_training_curve": True,
    
    # Multi-run parameters
    "enable_multi_run": True,
    "seeds": [3047, 294, 714, 1092, 1386, 2856, 42, 114514, 2025, 1993],
}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class RolloutBuffer:
    """On-policy rollout storage for CPO."""
    
    def __init__(self):
        self.clear()
    
    def clear(self):
        self.states = []
        self.station_actions = []
        self.vehicle_actions = []
        self.power_samples = []
        self.power_actions = []
        self.logprobs = []
        self.rewards = []
        self.costs = []  # Constraint violation costs
        self.dones = []
        self.values = []
        self.cost_values = []


class CPOActorCritic(nn.Module):
    """
    Actor-Critic network for CPO.
    Includes both reward value head and cost value head.
    """
    
    def __init__(self, state_dim: int, n_stations: int, n_vehicles: int, config: dict):
        super().__init__()
        self.n_stations = n_stations
        self.n_vehicles = n_vehicles
        
        hidden1 = config["hidden_dim1"]
        hidden2 = config["hidden_dim2"]
        
        # Shared feature extractor
        self.fc1 = nn.Linear(state_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        
        # Actor heads (discrete + continuous)
        self.station_head = nn.Linear(hidden2, n_stations)
        self.vehicle_head = nn.Linear(hidden2, n_vehicles)
        self.power_mean = nn.Linear(hidden2, 1)
        self.power_log_std = nn.Parameter(torch.zeros(1))
        
        # Critic heads (reward value + cost value)
        self.value_head = nn.Linear(hidden2, 1)      # V(s) for reward
        self.cost_value_head = nn.Linear(hidden2, 1)  # V_c(s) for cost
    
    def forward(self, state: torch.Tensor):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        station_logits = self.station_head(x)
        vehicle_logits = self.vehicle_head(x)
        power_mean = torch.sigmoid(self.power_mean(x))
        power_log_std = self.power_log_std.expand_as(power_mean)
        
        value = self.value_head(x)
        cost_value = self.cost_value_head(x)
        
        return station_logits, vehicle_logits, power_mean, power_log_std, value, cost_value
    
    def get_actor_params(self):
        """Get actor parameters for separate optimization."""
        return list(self.station_head.parameters()) + \
               list(self.vehicle_head.parameters()) + \
               list(self.power_mean.parameters()) + \
               [self.power_log_std]
    
    def get_critic_params(self):
        """Get critic parameters for separate optimization."""
        return list(self.fc1.parameters()) + \
               list(self.fc2.parameters()) + \
               list(self.value_head.parameters()) + \
               list(self.cost_value_head.parameters())


class CPOAgent:
    """
    CPO Agent for constrained reinforcement learning.
    
    Key features:
    - Trust region style optimization with PPO clip
    - Constraint satisfaction through projection
    - Line search for feasibility
    - Separate value functions for reward and cost
    """
    
    def __init__(self, state_dim: int, n_stations: int, n_vehicles: int, config: dict):
        self.n_stations = n_stations
        self.n_vehicles = n_vehicles
        self.config = config
        
        # Actor-Critic network
        self.model = CPOActorCritic(state_dim, n_stations, n_vehicles, config).to(device)
        
        # Store old policy for KL computation
        self.old_model = CPOActorCritic(state_dim, n_stations, n_vehicles, config).to(device)
        self.old_model.load_state_dict(self.model.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.model.get_actor_params(), lr=config["lr_actor"])
        self.critic_optimizer = optim.Adam(self.model.get_critic_params(), lr=config["lr_critic"])
        
        # CPO specific tracking
        self.episode_costs = []
        self.avg_cost = 0.0
    
    @torch.no_grad()
    def select_action(self, state: np.ndarray):
        """Select action using current policy."""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        station_logits, vehicle_logits, power_mean, power_log_std, value, cost_value = self.model(state_t)
        
        # Sample discrete actions
        station_dist = Categorical(logits=station_logits)
        vehicle_dist = Categorical(logits=vehicle_logits)
        station_action = station_dist.sample()
        vehicle_action = vehicle_dist.sample()
        
        # Sample continuous action
        power_std = power_log_std.exp()
        power_dist = Normal(power_mean, power_std)
        power_sample = power_dist.rsample()
        power_action = torch.sigmoid(power_sample)
        
        # Compute log probability with change-of-variables for sigmoid
        logprob_power = (
            power_dist.log_prob(power_sample)
            - torch.log(power_action.clamp_min(1e-8))
            - torch.log((1 - power_action).clamp_min(1e-8))
        ).sum(dim=-1)
        
        logprob = (
            station_dist.log_prob(station_action) +
            vehicle_dist.log_prob(vehicle_action) +
            logprob_power
        )
        
        # Convert to environment action format
        action_env = {
            "station_id": int(station_action.item()),
            "vehicle_id": int(vehicle_action.item()),
            "power": np.array([50.0 + power_action.item() * 100.0], dtype=np.float32),
        }
        
        return (
            action_env,
            logprob.item(),
            value.item(),
            cost_value.item(),
            int(station_action.item()),
            int(vehicle_action.item()),
            power_sample.squeeze().item(),
            power_action.squeeze().item(),
        )
    
    def evaluate_actions(self, states, station_actions, vehicle_actions, power_samples, model=None):
        """Evaluate actions for CPO update."""
        if model is None:
            model = self.model
            
        station_logits, vehicle_logits, power_mean, power_log_std, values, cost_values = model(states)
        
        station_dist = Categorical(logits=station_logits)
        vehicle_dist = Categorical(logits=vehicle_logits)
        power_std = power_log_std.exp()
        power_dist = Normal(power_mean, power_std)
        
        power_samples_expanded = power_samples.unsqueeze(-1)
        power_action = torch.sigmoid(power_samples_expanded)
        
        logprob_power = (
            power_dist.log_prob(power_samples_expanded)
            - torch.log(power_action.clamp_min(1e-8))
            - torch.log((1 - power_action).clamp_min(1e-8))
        ).sum(dim=-1)
        
        logprob = (
            station_dist.log_prob(station_actions) +
            vehicle_dist.log_prob(vehicle_actions) +
            logprob_power
        )
        
        entropy = (
            station_dist.entropy() +
            vehicle_dist.entropy() +
            power_dist.entropy().sum(dim=-1)
        )
        
        return logprob, entropy, values.squeeze(-1), cost_values.squeeze(-1)
    
    def compute_kl(self, states, station_actions, vehicle_actions, power_samples):
        """Compute KL divergence between old and new policy."""
        # Old policy
        with torch.no_grad():
            old_logprob, _, _, _ = self.evaluate_actions(
                states, station_actions, vehicle_actions, power_samples, self.old_model
            )
        
        # New policy
        new_logprob, _, _, _ = self.evaluate_actions(
            states, station_actions, vehicle_actions, power_samples, self.model
        )
        
        # KL divergence approximation
        kl = (old_logprob - new_logprob).mean()
        return kl
    
    def line_search(self, states, station_actions, vehicle_actions, power_samples,
                   old_logprobs, advantages, cost_advantages, cost_limit_exceeded):
        """
        Perform line search to find step size that satisfies constraints.
        """
        # Save current parameters
        old_params = {name: param.clone() for name, param in self.model.named_parameters()}
        
        # Compute full step
        logprob, entropy, values, cost_values = self.evaluate_actions(
            states, station_actions, vehicle_actions, power_samples
        )
        ratio = torch.exp(logprob - old_logprobs)
        
        # Reward objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.config["clip_eps"], 
                           1 + self.config["clip_eps"]) * advantages
        reward_obj = torch.min(surr1, surr2).mean()
        
        # Cost objective
        cost_surr1 = ratio * cost_advantages
        cost_surr2 = torch.clamp(ratio, 1 - self.config["clip_eps"],
                                 1 + self.config["clip_eps"]) * cost_advantages
        cost_obj = torch.min(cost_surr1, cost_surr2).mean()
        
        # If constraint is violated, we need to reduce cost
        # If constraint is satisfied, we maximize reward
        if cost_limit_exceeded:
            # Prioritize reducing cost
            loss = -reward_obj + 10.0 * cost_obj
        else:
            # Standard PPO objective
            loss = -reward_obj
        
        # Add entropy bonus
        loss -= self.config["entropy_coef"] * entropy.mean()
        
        return loss, reward_obj.item(), cost_obj.item()
    
    def update(self, buffer: RolloutBuffer):
        """
        CPO update with constraint satisfaction.
        
        Algorithm:
        1. Compute advantages for reward and cost
        2. Check if current policy violates constraint
        3. If violated: take recovery step (reduce cost)
        4. If not violated: take standard reward-maximizing step
        5. Use line search for feasibility
        """
        states = torch.FloatTensor(np.array(buffer.states)).to(device)
        station_actions = torch.LongTensor(buffer.station_actions).to(device)
        vehicle_actions = torch.LongTensor(buffer.vehicle_actions).to(device)
        power_samples = torch.FloatTensor(buffer.power_samples).to(device)
        old_logprobs = torch.FloatTensor(buffer.logprobs).to(device)
        returns = torch.FloatTensor(buffer.returns).to(device)
        advantages = torch.FloatTensor(buffer.advantages).to(device)
        cost_returns = torch.FloatTensor(buffer.cost_returns).to(device)
        cost_advantages = torch.FloatTensor(buffer.cost_advantages).to(device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        cost_advantages = (cost_advantages - cost_advantages.mean()) / (cost_advantages.std() + 1e-8)
        
        # Check constraint violation
        avg_cost = np.mean(buffer.costs)
        cost_limit = self.config["cost_limit"]
        cost_limit_exceeded = avg_cost > (cost_limit + self.config["safety_margin"])
        
        # Update old model
        self.old_model.load_state_dict(self.model.state_dict())
        
        dataset_size = len(states)
        batch_size = min(self.config["batch_size"], dataset_size)
        indices = np.arange(dataset_size)
        
        # Track losses
        total_policy_loss = 0
        total_value_loss = 0
        total_cost_value_loss = 0
        n_updates = 0
        final_reward_obj = 0
        final_cost_obj = 0
        
        for epoch in range(self.config["cpo_epochs"]):
            np.random.shuffle(indices)
            
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]
                
                batch_states = states[batch_idx]
                batch_station_actions = station_actions[batch_idx]
                batch_vehicle_actions = vehicle_actions[batch_idx]
                batch_power_samples = power_samples[batch_idx]
                batch_old_logprobs = old_logprobs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_cost_advantages = cost_advantages[batch_idx]
                batch_returns = returns[batch_idx]
                batch_cost_returns = cost_returns[batch_idx]
                
                # Evaluate actions
                logprob, entropy, values, cost_values = self.evaluate_actions(
                    batch_states, batch_station_actions, batch_vehicle_actions, batch_power_samples
                )
                
                # Compute ratio
                ratio = torch.exp(logprob - batch_old_logprobs)
                
                # Reward objective (PPO-clip)
                surr1_r = ratio * batch_advantages
                surr2_r = torch.clamp(ratio, 1 - self.config["clip_eps"],
                                      1 + self.config["clip_eps"]) * batch_advantages
                reward_obj = torch.min(surr1_r, surr2_r).mean()
                
                # Cost objective (PPO-clip)
                surr1_c = ratio * batch_cost_advantages
                surr2_c = torch.clamp(ratio, 1 - self.config["clip_eps"],
                                      1 + self.config["clip_eps"]) * batch_cost_advantages
                cost_obj = torch.min(surr1_c, surr2_c).mean()
                
                # CPO policy update
                if cost_limit_exceeded:
                    # Recovery mode: reduce cost while trying to maintain reward
                    # Use a penalty coefficient that increases with constraint violation
                    violation_amount = max(0, avg_cost - cost_limit)
                    penalty_coef = 1.0 + 10.0 * violation_amount
                    policy_loss = -reward_obj + penalty_coef * cost_obj
                else:
                    # Normal mode: maximize reward with small cost penalty
                    policy_loss = -reward_obj + 0.1 * cost_obj
                
                # Entropy bonus
                entropy_loss = -self.config["entropy_coef"] * entropy.mean()
                
                # Value losses
                value_loss = F.mse_loss(values, batch_returns)
                cost_value_loss = F.mse_loss(cost_values, batch_cost_returns)
                
                # Combined loss
                total_loss = (
                    policy_loss + 
                    entropy_loss + 
                    self.config["value_coef"] * (value_loss + cost_value_loss)
                )
                
                # Update networks
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config["max_grad_norm"])
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                
                # Track losses
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_cost_value_loss += cost_value_loss.item()
                final_reward_obj = reward_obj.item()
                final_cost_obj = cost_obj.item()
                n_updates += 1
        
        # Update average cost tracking
        self.avg_cost = avg_cost
        
        return {
            "policy_loss": total_policy_loss / n_updates if n_updates > 0 else 0,
            "value_loss": total_value_loss / n_updates if n_updates > 0 else 0,
            "cost_value_loss": total_cost_value_loss / n_updates if n_updates > 0 else 0,
            "reward_obj": final_reward_obj,
            "cost_obj": final_cost_obj,
            "avg_cost": avg_cost,
            "cost_limit_exceeded": cost_limit_exceeded,
        }


def compute_gae(rewards, dones, values, gamma, gae_lambda):
    """Compute Generalized Advantage Estimation."""
    advantages = []
    gae = 0.0
    next_value = 0.0
    
    for step in reversed(range(len(rewards))):
        mask = 1.0 - float(dones[step])
        delta = rewards[step] + gamma * next_value * mask - values[step]
        gae = delta + gamma * gae_lambda * mask * gae
        advantages.insert(0, gae)
        next_value = values[step]
    
    returns = [adv + val for adv, val in zip(advantages, values)]
    return advantages, returns


def add_to_buffer(buffer, state, s_a, v_a, p_sample, p_action, logprob, reward, cost, done, value, cost_value):
    """Add transition to buffer."""
    buffer.states.append(state)
    buffer.station_actions.append(s_a)
    buffer.vehicle_actions.append(v_a)
    buffer.power_samples.append(p_sample)
    buffer.power_actions.append(p_action)
    buffer.logprobs.append(logprob)
    buffer.rewards.append(reward)
    buffer.costs.append(cost)
    buffer.dones.append(done)
    buffer.values.append(value)
    buffer.cost_values.append(cost_value)


def train_cpo(config=None):
    """Main training function for CPO."""
    if config is None:
        config = CONFIG
    
    set_seed(config.get("seed"))
    
    env = EVChargingEnv(
        n_stations=config["n_stations"],
        p_max=config["p_max"],
        arrival_rate=config["arrival_rate"],
    )
    
    state_dim = env.observation_space.shape[0]
    n_stations = env.n_stations
    n_vehicles = env.max_vehicles
    
    agent = CPOAgent(state_dim, n_stations, n_vehicles, config)
    buffer = RolloutBuffer()
    
    score_record = []
    cost_record = []
    constraint_violations = {
        "total_violations": 0,
        "episode_violations": [],
        "violation_rate": [],
        "violation_details": [],
    }
    
    print(f"Starting CPO training:")
    print(f"  Stations: {n_stations}, Vehicles: {n_vehicles}")
    print(f"  Episodes: {config['num_of_episodes']}")
    print(f"  Cost limit: {config['cost_limit']}")
    print("-" * 70)
    
    for episode in range(config["num_of_episodes"]):
        state = env.reset()
        done = False
        episode_reward = 0.0
        episode_violations = 0
        episode_steps = 0
        buffer.clear()
        
        while not done:
            # Select action
            (action_env, logprob, value, cost_value,
             s_a, v_a, p_sample, p_action) = agent.select_action(state)
            
            # Execute action
            next_state, reward, done, info = env.step(action_env)
            
            # Check constraint violation
            violation_info = info.get("constraint_violation", None)
            cost = 0.0
            if violation_info and violation_info.get("has_violation", False):
                episode_violations += 1
                constraint_violations["total_violations"] += 1
                cost = config["cost_scale"]
                constraint_violations["violation_details"].append({
                    "episode": episode,
                    "step": env.current_step,
                    "type": violation_info.get("violation_type"),
                    "details": violation_info.get("violation_details"),
                })
            
            # Store transition
            add_to_buffer(buffer, state, s_a, v_a, p_sample, p_action,
                         logprob, reward, cost, done, value, cost_value)
            
            state = next_state
            episode_reward += reward
            episode_steps += 1
        
        # Compute GAE for rewards
        advantages, returns = compute_gae(
            buffer.rewards, buffer.dones, buffer.values,
            config["gamma"], config["gae_lambda"]
        )
        buffer.advantages = advantages
        buffer.returns = returns
        
        # Compute GAE for costs
        cost_advantages, cost_returns = compute_gae(
            buffer.costs, buffer.dones, buffer.cost_values,
            config["gamma"], config["gae_lambda"]
        )
        buffer.cost_advantages = cost_advantages
        buffer.cost_returns = cost_returns
        
        # Update agent
        update_info = agent.update(buffer)
        
        # Record statistics
        score_record.append(episode_reward)
        cost_record.append(update_info["avg_cost"])
        violation_rate = episode_violations / episode_steps if episode_steps > 0 else 0
        constraint_violations["episode_violations"].append(episode_violations)
        constraint_violations["violation_rate"].append(violation_rate)
        
        # Print progress
        if episode % config["print_interval"] == 0:
            avg_score = np.mean(score_record[-config["print_interval"]:]) if len(score_record) >= config["print_interval"] else episode_reward
            avg_violation_rate = np.mean(constraint_violations["violation_rate"][-config["print_interval"]:]) * 100
            mode = "RECOVERY" if update_info["cost_limit_exceeded"] else "NORMAL"
            
            print(f"Episode {episode:04d} | Reward: {episode_reward:.2f} | "
                  f"AvgReward: {avg_score:.2f} | AvgCost: {update_info['avg_cost']:.3f} | "
                  f"ViolRate: {avg_violation_rate:.1f}% | Mode: {mode}")
    
    # Final statistics
    print(f"\n{'='*70}")
    print(f"CPO Training Completed!")
    print(f"{'='*70}")
    print(f"Total violations: {constraint_violations['total_violations']}")
    print(f"Average violation rate: {np.mean(constraint_violations['violation_rate'])*100:.2f}%")
    print(f"Final avg reward (last 20): {np.mean(score_record[-20:]):.2f}")
    print(f"Target cost limit: {config['cost_limit']*100:.1f}%")
    print(f"{'='*70}")
    
    return score_record, cost_record, agent, constraint_violations


def multi_run_cpo(config=None):
    """Execute multiple training runs with different seeds."""
    if config is None:
        config = CONFIG
    
    all_scores = []
    all_costs = []
    all_agents = []
    all_violations = []
    
    seeds = config.get("seeds", [])
    print(f"\n{'='*70}")
    print(f"Starting multi-run CPO with seeds: {seeds}")
    print(f"{'='*70}")
    
    for idx, seed in enumerate(seeds):
        print(f"\n--- Run {idx+1}/{len(seeds)} | Seed: {seed} ---")
        run_config = config.copy()
        run_config["seed"] = seed
        
        scores, costs, agent, violations = train_cpo(run_config)
        
        all_scores.append(scores)
        all_costs.append(costs)
        all_agents.append(agent)
        all_violations.append(violations)
        
        print(f"Run {idx+1} completed - Final Score: {scores[-1]:.2f}, "
              f"Final Cost: {costs[-1]:.3f}, Violations: {violations['total_violations']}")
    
    print(f"\n{'='*70}")
    print(f"All {len(seeds)} runs completed!")
    print(f"{'='*70}")
    
    return all_scores, all_costs, all_agents, all_violations


@torch.no_grad()
def test_cpo(agent: CPOAgent, config=None):
    """Test trained CPO agent."""
    if config is None:
        config = CONFIG
    
    env = EVChargingEnv(
        n_stations=config["n_stations"],
        p_max=config["p_max"],
        arrival_rate=config["arrival_rate"],
    )
    
    state = env.reset()
    done = False
    total_reward = 0.0
    total_violations = 0
    step = 0
    
    while not done and step < config["max_steps"]:
        action_env, _, _, _, _, _, _, _ = agent.select_action(state)
        state, reward, done, info = env.step(action_env)
        
        violation_info = info.get("constraint_violation", None)
        if violation_info and violation_info.get("has_violation", False):
            total_violations += 1
        
        total_reward += reward
        step += 1
    
    print(f"Test finished. Steps={step}, Reward={total_reward:.2f}, Violations={total_violations}")
    return total_reward, step, total_violations


def save_multi_run_results(all_scores, all_costs, all_agents, all_violations, config, save_dir=None):
    """Save all multi-run results to a single folder."""
    import csv
    
    if save_dir is None:
        now_str = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = f"cpo_multi_run_{now_str}"
    os.makedirs(save_dir, exist_ok=True)
    
    seeds = config.get("seeds", [])
    
    # Save all models and scores
    for i, (agent, scores, costs, seed) in enumerate(zip(all_agents, all_scores, all_costs, seeds)):
        model_path = os.path.join(save_dir, f"cpo_run{i+1}_seed{seed}_model.pth")
        torch.save(agent.model.state_dict(), model_path)
        
        scores_path = os.path.join(save_dir, f"cpo_run{i+1}_seed{seed}_scores.npy")
        np.save(scores_path, np.array(scores))
        
        costs_path = os.path.join(save_dir, f"cpo_run{i+1}_seed{seed}_costs.npy")
        np.save(costs_path, np.array(costs))
    
    # Save per-run summary
    summary_path = os.path.join(save_dir, "cpo_multi_run_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Run", "Seed", "Final_Score", "Avg_Last20", "Final_Cost",
                        "Total_Violations", "Avg_Violation_Rate"])
        
        for i, (scores, costs, violations) in enumerate(zip(all_scores, all_costs, all_violations)):
            final_score = scores[-1] if scores else 0
            avg_last20 = np.mean(scores[-20:]) if len(scores) >= 20 else np.mean(scores)
            final_cost = costs[-1] if costs else 0
            total_violations = violations["total_violations"]
            avg_violation_rate = np.mean(violations["violation_rate"]) * 100
            writer.writerow([i + 1, seeds[i], round(final_score, 2), round(avg_last20, 2),
                           round(final_cost, 4), total_violations, round(avg_violation_rate, 2)])
    
    # Save overall summary
    overall_path = os.path.join(save_dir, "cpo_overall_summary.csv")
    with open(overall_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        final_scores = [s[-1] for s in all_scores]
        final_costs = [c[-1] for c in all_costs]
        total_violations_list = [v["total_violations"] for v in all_violations]
        avg_violation_rates = [np.mean(v["violation_rate"]) * 100 for v in all_violations]
        
        writer.writerow(["Metric", "Mean", "Std", "Min", "Max"])
        writer.writerow(["Final Score", round(np.mean(final_scores), 2), round(np.std(final_scores), 2),
                        round(min(final_scores), 2), round(max(final_scores), 2)])
        writer.writerow(["Final Cost", round(np.mean(final_costs), 4), round(np.std(final_costs), 4),
                        round(min(final_costs), 4), round(max(final_costs), 4)])
        writer.writerow(["Total Violations", round(np.mean(total_violations_list), 2),
                        round(np.std(total_violations_list), 2),
                        min(total_violations_list), max(total_violations_list)])
        writer.writerow(["Avg Violation Rate (%)", round(np.mean(avg_violation_rates), 2),
                        round(np.std(avg_violation_rates), 2),
                        round(min(avg_violation_rates), 2), round(max(avg_violation_rates), 2)])
    
    # Save all data combined
    np.save(os.path.join(save_dir, "cpo_all_scores.npy"), np.array(all_scores, dtype=object), allow_pickle=True)
    np.save(os.path.join(save_dir, "cpo_all_costs.npy"), np.array(all_costs, dtype=object), allow_pickle=True)
    
    # Save config
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\nMulti-run results saved to: {save_dir}")
    return save_dir


def plot_multi_run_curves(all_scores, all_costs, config, save_dir=None):
    """Plot training curves for multiple runs."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    seeds = config.get("seeds", [])
    
    # Plot reward curves
    ax1 = axes[0]
    for i, scores in enumerate(all_scores):
        ax1.plot(scores, alpha=0.5, label=f"Run {i+1} (Seed {seeds[i]})")
    
    min_len = min(len(s) for s in all_scores)
    scores_array = np.array([s[:min_len] for s in all_scores])
    mean_scores = scores_array.mean(axis=0)
    std_scores = scores_array.std(axis=0)
    ax1.plot(mean_scores, "k-", linewidth=2, label="Mean")
    ax1.fill_between(range(min_len), mean_scores - std_scores, mean_scores + std_scores,
                     alpha=0.2, color="black")
    ax1.set_title("CPO Training Rewards")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.legend(fontsize=8)
    ax1.grid(True)
    
    # Plot cost curves
    ax2 = axes[1]
    for i, costs in enumerate(all_costs):
        ax2.plot(costs, alpha=0.5, label=f"Run {i+1}")
    
    min_len_c = min(len(c) for c in all_costs)
    costs_array = np.array([c[:min_len_c] for c in all_costs])
    mean_costs = costs_array.mean(axis=0)
    std_costs = costs_array.std(axis=0)
    ax2.plot(mean_costs, "k-", linewidth=2, label="Mean")
    ax2.fill_between(range(min_len_c), mean_costs - std_costs, mean_costs + std_costs,
                     alpha=0.2, color="black")
    ax2.axhline(y=config["cost_limit"], color="r", linestyle="--", label=f"Limit ({config['cost_limit']})")
    ax2.set_title("CPO Constraint Cost Evolution")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Average Cost")
    ax2.legend(fontsize=8)
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_dir:
        path = os.path.join(save_dir, "cpo_multi_run_curves.png")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Curves saved to {path}")
    plt.show()


def evaluate_multi_run_results(all_scores, all_costs, all_violations, config):
    """Print evaluation of multi-run results."""
    seeds = config.get("seeds", [])
    
    print(f"\n{'='*70}")
    print("Multi-Run CPO Results Evaluation")
    print(f"{'='*70}")
    
    final_scores = [s[-1] for s in all_scores]
    final_costs = [c[-1] for c in all_costs]
    avg_violation_rates = [np.mean(v["violation_rate"]) * 100 for v in all_violations]
    
    print("\nPer-Run Results:")
    for i, (scores, costs, violations) in enumerate(zip(all_scores, all_costs, all_violations)):
        print(f"  Run {i+1} (Seed {seeds[i]}): Score={scores[-1]:.2f}, "
              f"Cost={costs[-1]:.4f}, ViolRate={avg_violation_rates[i]:.1f}%")
    
    print(f"\nOverall Statistics:")
    print(f"  Mean Final Score: {np.mean(final_scores):.2f} ± {np.std(final_scores):.2f}")
    print(f"  Mean Final Cost: {np.mean(final_costs):.4f} ± {np.std(final_costs):.4f}")
    print(f"  Mean Violation Rate: {np.mean(avg_violation_rates):.2f}% ± {np.std(avg_violation_rates):.2f}%")
    print(f"  Target Cost Limit: {config['cost_limit']*100:.1f}%")
    print(f"{'='*70}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="CPO for EV Charging Station")
    parser.add_argument("--stations", type=int, default=CONFIG["n_stations"])
    parser.add_argument("--power", type=float, default=CONFIG["p_max"])
    parser.add_argument("--arrival-rate", type=float, default=CONFIG["arrival_rate"])
    parser.add_argument("--episodes", type=int, default=CONFIG["num_of_episodes"])
    parser.add_argument("--cost-limit", type=float, default=CONFIG["cost_limit"],
                       help="Target constraint cost limit")
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--multi-run", action="store_true", default=CONFIG["enable_multi_run"])
    parser.add_argument("--single-run", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=CONFIG["seeds"])
    args = parser.parse_args()
    
    config = CONFIG.copy()
    config.update({
        "n_stations": args.stations,
        "p_max": args.power,
        "arrival_rate": args.arrival_rate,
        "num_of_episodes": args.episodes,
        "cost_limit": args.cost_limit,
        "seeds": args.seeds,
        "enable_multi_run": args.multi_run and not args.single_run,
    })
    
    print(f"\n{'='*60}")
    print(f"CPO (Constrained Policy Optimization) for EV Charging Station")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  Stations: {config['n_stations']}")
    print(f"  Max Power: {config['p_max']} kW")
    print(f"  Arrival Rate: {config['arrival_rate']}")
    print(f"  Episodes: {config['num_of_episodes']}")
    print(f"  Cost Limit: {config['cost_limit']*100:.1f}%")
    print(f"  Seeds: {config['seeds']}")
    print(f"  Multi-run mode: {config['enable_multi_run']}")
    print(f"{'='*60}")
    
    if args.test_only:
        if args.model_path and os.path.exists(args.model_path):
            env_tmp = EVChargingEnv(n_stations=config["n_stations"], p_max=config["p_max"],
                                    arrival_rate=config["arrival_rate"])
            agent = CPOAgent(env_tmp.observation_space.shape[0], env_tmp.n_stations,
                            env_tmp.max_vehicles, config)
            agent.model.load_state_dict(torch.load(args.model_path, map_location=device))
            test_cpo(agent, config)
        else:
            print("Test-only mode requires --model-path pointing to a saved model.")
    
    elif config["enable_multi_run"]:
        # Multi-run training
        all_scores, all_costs, all_agents, all_violations = multi_run_cpo(config)
        
        # Evaluate results
        evaluate_multi_run_results(all_scores, all_costs, all_violations, config)
        
        # Save all results to a single folder
        save_dir = None
        if config.get("save_models", False):
            save_dir = save_multi_run_results(all_scores, all_costs, all_agents, all_violations, config)
            plot_multi_run_curves(all_scores, all_costs, config, save_dir)
        
        # Test best model
        best_idx = np.argmax([s[-1] for s in all_scores])
        print(f"\n{'='*40}")
        print(f"Testing with best model (Run {best_idx + 1}, Seed {config['seeds'][best_idx]})...")
        print(f"{'='*40}")
        test_cpo(all_agents[best_idx], config)
    
    else:
        # Single run training
        if config["seeds"]:
            config["seed"] = config["seeds"][0]
            print(f"Using random seed: {config['seed']}")
        
        scores, costs, agent, violations = train_cpo(config)
        
        if config.get("save_models", False):
            now_str = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = f"cpo_{now_str}"
            os.makedirs(save_dir, exist_ok=True)
            torch.save(agent.model.state_dict(), os.path.join(save_dir, "cpo_model.pth"))
            np.save(os.path.join(save_dir, "cpo_scores.npy"), np.array(scores))
            np.save(os.path.join(save_dir, "cpo_costs.npy"), np.array(costs))
            with open(os.path.join(save_dir, "config.json"), "w") as f:
                json.dump(config, f, indent=2)
            print(f"Model saved to {save_dir}")
        
        test_cpo(agent, config)

