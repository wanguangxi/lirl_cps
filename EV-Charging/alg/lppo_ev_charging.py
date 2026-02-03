"""
Lagrangian PPO (LPPO) for EV Charging Station Control

Based on the structure of `pdqn_ev_charging.py` and `hppo_ev_charging.py`.
Uses a Lagrangian method to handle constrained optimization.

Key idea:
- Standard PPO maximizes expected return
- A Lagrange multiplier \u03bb penalizes constraint violations
- Update both policy parameters \u03b8 and \u03bb
- Objective: \u03bcax_\u03b8 min_\u03bb E[reward] - \u03bb * (E[constraint_cost] - threshold)

No explicit action correction/projection; constraints are enforced via Lagrangian penalties.
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
    # PPO Learning parameters
    "lr_actor": 3e-4,           # Actor learning rate
    "lr_critic": 1e-3,          # Critic learning rate
    "lr_lambda": 5e-3,          # Lagrangian multiplier learning rate
    "gamma": 0.99,              # Discount factor
    "gae_lambda": 0.95,         # GAE lambda
    "clip_eps": 0.2,            # PPO clip epsilon
    "entropy_coef": 0.01,       # Entropy coefficient
    "value_coef": 0.5,          # Value loss coefficient
    "max_grad_norm": 0.5,       # Gradient clipping
    "ppo_epochs": 5,            # PPO update epochs
    "batch_size": 256,          # Mini-batch size
    
    # Lagrangian constraint parameters
    "constraint_threshold": 0.1,    # Target constraint violation rate (10%)
    "lambda_init": 0.1,             # Initial Lagrangian multiplier
    "lambda_max": 10.0,             # Maximum Lagrangian multiplier
    "lambda_min": 0.0,              # Minimum Lagrangian multiplier
    "cost_scale": 1.0,              # Scale factor for constraint cost
    
    # Environment parameters
    "n_stations": 5,
    "p_max": 150.0,
    "arrival_rate": 0.75,
    "num_of_episodes": 2,
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
    """On-policy rollout storage for Lagrangian PPO."""
    
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
        self.cost_values = []  # Value estimates for cost
    
    def add(self, state, station_id, vehicle_id, power_sample, power_action,
            logprob, reward, cost, done, value, cost_value):
        self.states.append(state)
        self.station_actions.append(station_id)
        self.vehicle_actions.append(vehicle_id)
        self.power_samples.append(power_sample)
        self.power_actions.append(power_action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.costs.append(cost)
        self.dones.append(done)
        self.values.append(value)
        self.cost_values.append(cost_value)


class LagrangianActorCritic(nn.Module):
    """
    Actor-Critic network for Lagrangian PPO.
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


class LPPOAgent:
    """
    Lagrangian PPO Agent for constrained reinforcement learning.
    
    Key features:
    - Dual optimization: max_θ min_λ objective
    - Learnable Lagrangian multiplier λ
    - Separate value functions for reward and cost
    """
    
    def __init__(self, state_dim: int, n_stations: int, n_vehicles: int, config: dict):
        self.n_stations = n_stations
        self.n_vehicles = n_vehicles
        self.config = config
        
        # Actor-Critic network
        self.model = LagrangianActorCritic(state_dim, n_stations, n_vehicles, config).to(device)
        
        # Lagrangian multiplier (learnable)
        self.log_lambda = torch.tensor(
            np.log(max(config["lambda_init"], 1e-8)),
            requires_grad=True,
            device=device
        )
        
        # Optimizers
        self.actor_optimizer = optim.Adam(
            list(self.model.station_head.parameters()) +
            list(self.model.vehicle_head.parameters()) +
            list(self.model.power_mean.parameters()) +
            [self.model.power_log_std],
            lr=config["lr_actor"]
        )
        self.critic_optimizer = optim.Adam(
            list(self.model.fc1.parameters()) +
            list(self.model.fc2.parameters()) +
            list(self.model.value_head.parameters()) +
            list(self.model.cost_value_head.parameters()),
            lr=config["lr_critic"]
        )
        self.lambda_optimizer = optim.Adam([self.log_lambda], lr=config["lr_lambda"])
    
    @property
    def lambda_value(self):
        """Get current Lagrangian multiplier value (clamped)."""
        return torch.clamp(
            self.log_lambda.exp(),
            self.config["lambda_min"],
            self.config["lambda_max"]
        )
    
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
    
    def evaluate_actions(self, states, station_actions, vehicle_actions, power_samples):
        """Evaluate actions for PPO update."""
        station_logits, vehicle_logits, power_mean, power_log_std, values, cost_values = self.model(states)
        
        station_dist = Categorical(logits=station_logits)
        vehicle_dist = Categorical(logits=vehicle_logits)
        power_std = power_log_std.exp()
        power_dist = Normal(power_mean, power_std)
        
        power_samples = power_samples.unsqueeze(-1)
        power_action = torch.sigmoid(power_samples)
        
        logprob_power = (
            power_dist.log_prob(power_samples)
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
    
    def update(self, buffer: RolloutBuffer):
        """
        Lagrangian PPO update.
        
        1. Update critic (value functions)
        2. Update actor with Lagrangian penalty
        3. Update Lagrangian multiplier
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
        
        dataset_size = len(states)
        batch_size = min(self.config["batch_size"], dataset_size)
        indices = np.arange(dataset_size)
        
        # Track losses for logging
        total_policy_loss = 0
        total_value_loss = 0
        total_cost_value_loss = 0
        total_lambda_loss = 0
        n_updates = 0
        
        for _ in range(self.config["ppo_epochs"]):
            np.random.shuffle(indices)
            
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]
                
                logprob, entropy, values, cost_values = self.evaluate_actions(
                    states[batch_idx],
                    station_actions[batch_idx],
                    vehicle_actions[batch_idx],
                    power_samples[batch_idx],
                )
                
                # PPO clipped objective
                ratio = torch.exp(logprob - old_logprobs[batch_idx])
                
                # Reward advantage
                surr1_reward = ratio * advantages[batch_idx]
                surr2_reward = torch.clamp(ratio, 1 - self.config["clip_eps"], 
                                           1 + self.config["clip_eps"]) * advantages[batch_idx]
                reward_obj = torch.min(surr1_reward, surr2_reward)
                
                # Cost advantage (constraint)
                surr1_cost = ratio * cost_advantages[batch_idx]
                surr2_cost = torch.clamp(ratio, 1 - self.config["clip_eps"],
                                         1 + self.config["clip_eps"]) * cost_advantages[batch_idx]
                cost_obj = torch.min(surr1_cost, surr2_cost)
                
                # Lagrangian objective: maximize reward - λ * cost
                lambda_val = self.lambda_value.detach()
                policy_loss = -(reward_obj - lambda_val * cost_obj).mean()
                
                # Entropy bonus
                entropy_loss = -self.config["entropy_coef"] * entropy.mean()
                
                # Value losses
                value_loss = F.mse_loss(values, returns[batch_idx])
                cost_value_loss = F.mse_loss(cost_values, cost_returns[batch_idx])
                
                # Combined loss (actor + critic)
                total_loss = (
                    policy_loss + 
                    entropy_loss + 
                    self.config["value_coef"] * (value_loss + cost_value_loss)
                )
                
                # Single backward pass
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
                n_updates += 1
        
        # Update Lagrangian multiplier
        # λ increases if constraint violated, decreases otherwise
        avg_cost = np.mean(buffer.costs)
        constraint_violation = avg_cost - self.config["constraint_threshold"]
        
        # Gradient ascent on λ: λ = λ + lr * (avg_cost - threshold)
        lambda_loss = -self.log_lambda * constraint_violation
        self.lambda_optimizer.zero_grad()
        lambda_loss.backward()
        self.lambda_optimizer.step()
        
        # Clamp log_lambda to reasonable range
        with torch.no_grad():
            self.log_lambda.clamp_(
                np.log(max(self.config["lambda_min"], 1e-8)),
                np.log(self.config["lambda_max"])
            )
        
        return {
            "policy_loss": total_policy_loss / n_updates if n_updates > 0 else 0,
            "value_loss": total_value_loss / n_updates if n_updates > 0 else 0,
            "cost_value_loss": total_cost_value_loss / n_updates if n_updates > 0 else 0,
            "lambda": self.lambda_value.item(),
            "avg_cost": avg_cost,
            "constraint_violation": constraint_violation,
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


def train_lppo(config=None):
    """Main training function for Lagrangian PPO."""
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
    
    agent = LPPOAgent(state_dim, n_stations, n_vehicles, config)
    buffer = RolloutBuffer()
    
    score_record = []
    lambda_record = []
    constraint_violations = {
        "total_violations": 0,
        "episode_violations": [],
        "violation_rate": [],
        "violation_details": [],
    }
    
    print(f"Starting Lagrangian PPO training:")
    print(f"  Stations: {n_stations}, Vehicles: {n_vehicles}")
    print(f"  Episodes: {config['num_of_episodes']}")
    print(f"  Constraint threshold: {config['constraint_threshold']}")
    print(f"  Initial λ: {config['lambda_init']}")
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
                cost = config["cost_scale"]  # Binary cost for violation
                constraint_violations["violation_details"].append({
                    "episode": episode,
                    "step": env.current_step,
                    "type": violation_info.get("violation_type"),
                    "details": violation_info.get("violation_details"),
                })
            
            # Store transition
            buffer.add(state, s_a, v_a, p_sample, p_action,
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
        lambda_record.append(update_info["lambda"])
        violation_rate = episode_violations / episode_steps if episode_steps > 0 else 0
        constraint_violations["episode_violations"].append(episode_violations)
        constraint_violations["violation_rate"].append(violation_rate)
        
        # Print progress
        if episode % config["print_interval"] == 0:
            avg_score = np.mean(score_record[-config["print_interval"]:]) if len(score_record) >= config["print_interval"] else episode_reward
            avg_violation_rate = np.mean(constraint_violations["violation_rate"][-config["print_interval"]:]) * 100
            
            print(f"Episode {episode:04d} | Reward: {episode_reward:.2f} | "
                  f"AvgReward: {avg_score:.2f} | λ: {update_info['lambda']:.4f} | "
                  f"ViolRate: {avg_violation_rate:.1f}%")
    
    # Final statistics
    print(f"\n{'='*70}")
    print(f"Lagrangian PPO Training Completed!")
    print(f"{'='*70}")
    print(f"Final λ: {agent.lambda_value.item():.4f}")
    print(f"Total violations: {constraint_violations['total_violations']}")
    print(f"Average violation rate: {np.mean(constraint_violations['violation_rate'])*100:.2f}%")
    print(f"Final avg reward (last 20): {np.mean(score_record[-20:]):.2f}")
    print(f"{'='*70}")
    
    return score_record, lambda_record, agent, constraint_violations


def multi_run_lppo(config=None):
    """Execute multiple training runs with different seeds."""
    if config is None:
        config = CONFIG
    
    all_scores = []
    all_lambdas = []
    all_agents = []
    all_violations = []
    
    seeds = config.get("seeds", [])
    print(f"\n{'='*70}")
    print(f"Starting multi-run Lagrangian PPO with seeds: {seeds}")
    print(f"{'='*70}")
    
    for idx, seed in enumerate(seeds):
        print(f"\n--- Run {idx+1}/{len(seeds)} | Seed: {seed} ---")
        run_config = config.copy()
        run_config["seed"] = seed
        
        scores, lambdas, agent, violations = train_lppo(run_config)
        
        all_scores.append(scores)
        all_lambdas.append(lambdas)
        all_agents.append(agent)
        all_violations.append(violations)
        
        print(f"Run {idx+1} completed - Final Score: {scores[-1]:.2f}, "
              f"Final λ: {lambdas[-1]:.4f}, Violations: {violations['total_violations']}")
    
    print(f"\n{'='*70}")
    print(f"All {len(seeds)} runs completed!")
    print(f"{'='*70}")
    
    return all_scores, all_lambdas, all_agents, all_violations


@torch.no_grad()
def test_lppo(agent: LPPOAgent, config=None):
    """Test trained Lagrangian PPO agent."""
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


def save_multi_run_results(all_scores, all_lambdas, all_agents, all_violations, config, save_dir=None):
    """Save all multi-run results to a single folder."""
    import csv
    
    if save_dir is None:
        now_str = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = f"lppo_multi_run_{now_str}"
    os.makedirs(save_dir, exist_ok=True)
    
    seeds = config.get("seeds", [])
    
    # Save all models and scores
    for i, (agent, scores, lambdas, seed) in enumerate(zip(all_agents, all_scores, all_lambdas, seeds)):
        model_path = os.path.join(save_dir, f"lppo_run{i+1}_seed{seed}_model.pth")
        torch.save(agent.model.state_dict(), model_path)
        
        scores_path = os.path.join(save_dir, f"lppo_run{i+1}_seed{seed}_scores.npy")
        np.save(scores_path, np.array(scores))
        
        lambdas_path = os.path.join(save_dir, f"lppo_run{i+1}_seed{seed}_lambdas.npy")
        np.save(lambdas_path, np.array(lambdas))
    
    # Save per-run summary
    summary_path = os.path.join(save_dir, "lppo_multi_run_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Run", "Seed", "Final_Score", "Avg_Last20", "Final_Lambda", 
                        "Total_Violations", "Avg_Violation_Rate"])
        
        for i, (scores, lambdas, violations) in enumerate(zip(all_scores, all_lambdas, all_violations)):
            final_score = scores[-1] if scores else 0
            avg_last20 = np.mean(scores[-20:]) if len(scores) >= 20 else np.mean(scores)
            final_lambda = lambdas[-1] if lambdas else 0
            total_violations = violations["total_violations"]
            avg_violation_rate = np.mean(violations["violation_rate"]) * 100
            writer.writerow([i + 1, seeds[i], round(final_score, 2), round(avg_last20, 2),
                           round(final_lambda, 4), total_violations, round(avg_violation_rate, 2)])
    
    # Save overall summary
    overall_path = os.path.join(save_dir, "lppo_overall_summary.csv")
    with open(overall_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        final_scores = [s[-1] for s in all_scores]
        final_lambdas = [l[-1] for l in all_lambdas]
        total_violations_list = [v["total_violations"] for v in all_violations]
        avg_violation_rates = [np.mean(v["violation_rate"]) * 100 for v in all_violations]
        
        writer.writerow(["Metric", "Mean", "Std", "Min", "Max"])
        writer.writerow(["Final Score", round(np.mean(final_scores), 2), round(np.std(final_scores), 2),
                        round(min(final_scores), 2), round(max(final_scores), 2)])
        writer.writerow(["Final Lambda", round(np.mean(final_lambdas), 4), round(np.std(final_lambdas), 4),
                        round(min(final_lambdas), 4), round(max(final_lambdas), 4)])
        writer.writerow(["Total Violations", round(np.mean(total_violations_list), 2),
                        round(np.std(total_violations_list), 2),
                        min(total_violations_list), max(total_violations_list)])
        writer.writerow(["Avg Violation Rate (%)", round(np.mean(avg_violation_rates), 2),
                        round(np.std(avg_violation_rates), 2),
                        round(min(avg_violation_rates), 2), round(max(avg_violation_rates), 2)])
    
    # Save all scores and lambdas combined
    np.save(os.path.join(save_dir, "lppo_all_scores.npy"), np.array(all_scores, dtype=object), allow_pickle=True)
    np.save(os.path.join(save_dir, "lppo_all_lambdas.npy"), np.array(all_lambdas, dtype=object), allow_pickle=True)
    
    # Save config
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\nMulti-run results saved to: {save_dir}")
    return save_dir


def plot_multi_run_curves(all_scores, all_lambdas, config, save_dir=None):
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
    ax1.set_title("Lagrangian PPO Training Rewards")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.legend(fontsize=8)
    ax1.grid(True)
    
    # Plot lambda curves
    ax2 = axes[1]
    for i, lambdas in enumerate(all_lambdas):
        ax2.plot(lambdas, alpha=0.5, label=f"Run {i+1}")
    
    min_len_l = min(len(l) for l in all_lambdas)
    lambdas_array = np.array([l[:min_len_l] for l in all_lambdas])
    mean_lambdas = lambdas_array.mean(axis=0)
    std_lambdas = lambdas_array.std(axis=0)
    ax2.plot(mean_lambdas, "k-", linewidth=2, label="Mean")
    ax2.fill_between(range(min_len_l), mean_lambdas - std_lambdas, mean_lambdas + std_lambdas,
                     alpha=0.2, color="black")
    ax2.axhline(y=config["constraint_threshold"], color="r", linestyle="--", label="Threshold")
    ax2.set_title("Lagrangian Multiplier (λ) Evolution")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("λ")
    ax2.legend(fontsize=8)
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_dir:
        path = os.path.join(save_dir, "lppo_multi_run_curves.png")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Curves saved to {path}")
    plt.show()


def evaluate_multi_run_results(all_scores, all_lambdas, all_violations, config):
    """Print evaluation of multi-run results."""
    seeds = config.get("seeds", [])
    
    print(f"\n{'='*70}")
    print("Multi-Run Lagrangian PPO Results Evaluation")
    print(f"{'='*70}")
    
    final_scores = [s[-1] for s in all_scores]
    final_lambdas = [l[-1] for l in all_lambdas]
    avg_violation_rates = [np.mean(v["violation_rate"]) * 100 for v in all_violations]
    
    print("\nPer-Run Results:")
    for i, (scores, lambdas, violations) in enumerate(zip(all_scores, all_lambdas, all_violations)):
        print(f"  Run {i+1} (Seed {seeds[i]}): Score={scores[-1]:.2f}, "
              f"λ={lambdas[-1]:.4f}, ViolRate={avg_violation_rates[i]:.1f}%")
    
    print(f"\nOverall Statistics:")
    print(f"  Mean Final Score: {np.mean(final_scores):.2f} ± {np.std(final_scores):.2f}")
    print(f"  Mean Final λ: {np.mean(final_lambdas):.4f} ± {np.std(final_lambdas):.4f}")
    print(f"  Mean Violation Rate: {np.mean(avg_violation_rates):.2f}% ± {np.std(avg_violation_rates):.2f}%")
    print(f"  Target Threshold: {config['constraint_threshold']*100:.1f}%")
    print(f"{'='*70}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Lagrangian PPO for EV Charging Station")
    parser.add_argument("--stations", type=int, default=CONFIG["n_stations"])
    parser.add_argument("--power", type=float, default=CONFIG["p_max"])
    parser.add_argument("--arrival-rate", type=float, default=CONFIG["arrival_rate"])
    parser.add_argument("--episodes", type=int, default=CONFIG["num_of_episodes"])
    parser.add_argument("--constraint-threshold", type=float, default=CONFIG["constraint_threshold"],
                       help="Target constraint violation rate")
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
        "constraint_threshold": args.constraint_threshold,
        "seeds": args.seeds,
        "enable_multi_run": args.multi_run and not args.single_run,
    })
    
    print(f"\n{'='*60}")
    print(f"Lagrangian PPO for EV Charging Station")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  Stations: {config['n_stations']}")
    print(f"  Max Power: {config['p_max']} kW")
    print(f"  Arrival Rate: {config['arrival_rate']}")
    print(f"  Episodes: {config['num_of_episodes']}")
    print(f"  Constraint Threshold: {config['constraint_threshold']*100:.1f}%")
    print(f"  Seeds: {config['seeds']}")
    print(f"  Multi-run mode: {config['enable_multi_run']}")
    print(f"{'='*60}")
    
    if args.test_only:
        if args.model_path and os.path.exists(args.model_path):
            env_tmp = EVChargingEnv(n_stations=config["n_stations"], p_max=config["p_max"],
                                    arrival_rate=config["arrival_rate"])
            agent = LPPOAgent(env_tmp.observation_space.shape[0], env_tmp.n_stations,
                             env_tmp.max_vehicles, config)
            agent.model.load_state_dict(torch.load(args.model_path, map_location=device))
            test_lppo(agent, config)
        else:
            print("Test-only mode requires --model-path pointing to a saved model.")
    
    elif config["enable_multi_run"]:
        # Multi-run training
        all_scores, all_lambdas, all_agents, all_violations = multi_run_lppo(config)
        
        # Evaluate results
        evaluate_multi_run_results(all_scores, all_lambdas, all_violations, config)
        
        # Save all results to a single folder
        save_dir = None
        if config.get("save_models", False):
            save_dir = save_multi_run_results(all_scores, all_lambdas, all_agents, all_violations, config)
            plot_multi_run_curves(all_scores, all_lambdas, config, save_dir)
        
        # Test best model
        best_idx = np.argmax([s[-1] for s in all_scores])
        print(f"\n{'='*40}")
        print(f"Testing with best model (Run {best_idx + 1}, Seed {config['seeds'][best_idx]})...")
        print(f"{'='*40}")
        test_lppo(all_agents[best_idx], config)
    
    else:
        # Single run training
        if config["seeds"]:
            config["seed"] = config["seeds"][0]
            print(f"Using random seed: {config['seed']}")
        
        scores, lambdas, agent, violations = train_lppo(config)
        
        if config.get("save_models", False):
            now_str = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = f"lppo_{now_str}"
            os.makedirs(save_dir, exist_ok=True)
            torch.save(agent.model.state_dict(), os.path.join(save_dir, "lppo_model.pth"))
            np.save(os.path.join(save_dir, "lppo_scores.npy"), np.array(scores))
            np.save(os.path.join(save_dir, "lppo_lambdas.npy"), np.array(lambdas))
            with open(os.path.join(save_dir, "config.json"), "w") as f:
                json.dump(config, f, indent=2)
            print(f"Model saved to {save_dir}")
        
        test_lppo(agent, config)

