"""
HPPO (Hybrid Proximal Policy Optimization) for EV Charging Station Control

Based on the structure of `pdqn_ev_charging.py`. Uses PPO to handle a hybrid action:
discrete (station_id, vehicle_id) and continuous (power). There is no explicit
action correction/projection; constraints are learned via environment rewards/penalties.
"""

import os
import sys
import math
import json
import random
import datetime as dt
from typing import Dict, Tuple

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
    # Learning parameters
    "lr": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_eps": 0.2,
    "entropy_coef": 0.01,
    "value_coef": 0.5,
    "max_grad_norm": 0.5,
    "ppo_epochs": 5,
    "batch_size": 256,

    # Environment parameters
    "n_stations": 5,
    "p_max": 150.0,
    "arrival_rate": 0.75,
    "num_of_episodes": 20,
    "max_steps": 288,  # a day

    # Network
    "hidden_dim1": 128,
    "hidden_dim2": 64,

    # Output
    "print_interval": 10,
    "save_models": True,
    "plot_training_curve": True,
    # Multi-run
    "enable_multi_run": True,
    "seeds": [3047, 294, 714, 1092, 1386, 2856, 42, 114514, 2025, 1993],
}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class RolloutBuffer:
    """On-policy rollout storage for PPO."""

    def __init__(self):
        self.clear()

    def clear(self):
        self.states = []
        self.station_actions = []
        self.vehicle_actions = []
        self.power_samples = []      # pre-sigmoid samples
        self.power_actions = []      # sigmoid outputs in [0,1]
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def add(self, state, station_id, vehicle_id, power_sample, power_action,
            logprob, reward, done, value):
        self.states.append(state)
        self.station_actions.append(station_id)
        self.vehicle_actions.append(vehicle_id)
        self.power_samples.append(power_sample)
        self.power_actions.append(power_action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)


class HybridActorCritic(nn.Module):
    """Shared backbone with discrete logits and continuous head."""

    def __init__(self, state_dim: int, n_stations: int, n_vehicles: int):
        super().__init__()
        self.n_stations = n_stations
        self.n_vehicles = n_vehicles

        self.fc1 = nn.Linear(state_dim, CONFIG["hidden_dim1"])
        self.fc2 = nn.Linear(CONFIG["hidden_dim1"], CONFIG["hidden_dim2"])

        self.station_head = nn.Linear(CONFIG["hidden_dim2"], n_stations)
        self.vehicle_head = nn.Linear(CONFIG["hidden_dim2"], n_vehicles)

        # continuous power head
        self.power_mean = nn.Linear(CONFIG["hidden_dim2"], 1)
        self.power_log_std = nn.Parameter(torch.zeros(1))

        self.value_head = nn.Linear(CONFIG["hidden_dim2"], 1)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        station_logits = self.station_head(x)
        vehicle_logits = self.vehicle_head(x)

        power_mean = torch.sigmoid(self.power_mean(x))  # [0,1] mean
        power_log_std = self.power_log_std.expand_as(power_mean)

        value = self.value_head(x)
        return station_logits, vehicle_logits, power_mean, power_log_std, value


class HPPOAgent:
    def __init__(self, state_dim: int, n_stations: int, n_vehicles: int):
        self.n_stations = n_stations
        self.n_vehicles = n_vehicles
        self.model = HybridActorCritic(state_dim, n_stations, n_vehicles).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=CONFIG["lr"])

    @torch.no_grad()
    def select_action(self, state: np.ndarray):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        station_logits, vehicle_logits, power_mean, power_log_std, value = self.model(state_t)

        station_dist = Categorical(logits=station_logits)
        vehicle_dist = Categorical(logits=vehicle_logits)

        power_std = power_log_std.exp()
        power_dist = Normal(power_mean, power_std)

        station_action = station_dist.sample()
        vehicle_action = vehicle_dist.sample()

        # Reparameterized sample for power, bounded by sigmoid
        power_sample = power_dist.rsample()
        power_action = torch.sigmoid(power_sample)

        # Log-prob with change-of-variables for sigmoid
        logprob_power = (
            power_dist.log_prob(power_sample)
            - torch.log(power_action.clamp_min(1e-8))
            - torch.log((1 - power_action).clamp_min(1e-8))
        ).sum(dim=-1)

        logprob = station_dist.log_prob(station_action) \
            + vehicle_dist.log_prob(vehicle_action) \
            + logprob_power

        action_env = {
            "station_id": int(station_action.item()),
            "vehicle_id": int(vehicle_action.item()),
            "power": np.array([50.0 + power_action.item() * 100.0], dtype=np.float32),
        }

        return (
            action_env,
            logprob.item(),
            value.item(),
            int(station_action.item()),
            int(vehicle_action.item()),
            power_sample.squeeze().item(),
            power_action.squeeze().item(),
        )

    def evaluate_actions(self, states, station_actions, vehicle_actions, power_samples):
        station_logits, vehicle_logits, power_mean, power_log_std, values = self.model(states)

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

        logprob = station_dist.log_prob(station_actions) \
            + vehicle_dist.log_prob(vehicle_actions) \
            + logprob_power

        entropy = station_dist.entropy() + vehicle_dist.entropy() + power_dist.entropy().sum(dim=-1)

        return logprob, entropy, values.squeeze(-1)

    def update(self, buffer: RolloutBuffer):
        states = torch.FloatTensor(np.array(buffer.states)).to(device)
        station_actions = torch.LongTensor(buffer.station_actions).to(device)
        vehicle_actions = torch.LongTensor(buffer.vehicle_actions).to(device)
        power_samples = torch.FloatTensor(buffer.power_samples).to(device)
        old_logprobs = torch.FloatTensor(buffer.logprobs).to(device)
        returns = torch.FloatTensor(buffer.returns).to(device)
        advantages = torch.FloatTensor(buffer.advantages).to(device)

        # normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_size = len(states)
        batch_size = min(CONFIG["batch_size"], dataset_size)
        indices = np.arange(dataset_size)

        for _ in range(CONFIG["ppo_epochs"]):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]

                logprob, entropy, values = self.evaluate_actions(
                    states[batch_idx],
                    station_actions[batch_idx],
                    vehicle_actions[batch_idx],
                    power_samples[batch_idx],
                )

                ratio = torch.exp(logprob - old_logprobs[batch_idx])
                surr1 = ratio * advantages[batch_idx]
                surr2 = torch.clamp(ratio, 1 - CONFIG["clip_eps"], 1 + CONFIG["clip_eps"]) * advantages[batch_idx]
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(values, returns[batch_idx])
                entropy_loss = -CONFIG["entropy_coef"] * entropy.mean()

                loss = policy_loss + CONFIG["value_coef"] * value_loss + entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), CONFIG["max_grad_norm"])
                self.optimizer.step()


def compute_gae(rewards, dones, values, gamma, lam):
    advantages = []
    gae = 0.0
    next_value = 0.0
    for step in reversed(range(len(rewards))):
        mask = 1.0 - float(dones[step])
        delta = rewards[step] + gamma * next_value * mask - values[step]
        gae = delta + gamma * lam * mask * gae
        advantages.insert(0, gae)
        next_value = values[step]
    returns = [adv + val for adv, val in zip(advantages, values)]
    return advantages, returns


def train_hppo(config=None):
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

    agent = HPPOAgent(state_dim, n_stations, n_vehicles)
    buffer = RolloutBuffer()

    score_record = []
    constraint_violations = {
        "total_violations": 0,
        "episode_violations": [],
        "violation_details": [],
    }

    print(f"Starting HPPO training: stations={n_stations}, vehicles={n_vehicles}, episodes={config['num_of_episodes']}")

    for episode in range(config["num_of_episodes"]):
        state = env.reset()
        done = False
        episode_reward = 0.0
        episode_violations = 0
        buffer.clear()

        while not done:
            action_env, logprob, value, s_a, v_a, p_sample, p_action = agent.select_action(state)
            next_state, reward, done, info = env.step(action_env)

            violation_info = info.get("constraint_violation", None)
            if violation_info and violation_info.get("has_violation", False):
                episode_violations += 1
                constraint_violations["total_violations"] += 1
                constraint_violations["violation_details"].append({
                    "episode": episode,
                    "step": env.current_step,
                    "type": violation_info.get("violation_type"),
                    "details": violation_info.get("violation_details"),
                    "action": violation_info.get("attempted_action"),
                })

            buffer.add(state, s_a, v_a, p_sample, p_action, logprob, reward, done, value)

            state = next_state
            episode_reward += reward

        # GAE
        advantages, returns = compute_gae(
            buffer.rewards,
            buffer.dones,
            buffer.values,
            config["gamma"],
            config["gae_lambda"],
        )
        buffer.advantages = advantages
        buffer.returns = returns

        agent.update(buffer)

        score_record.append(episode_reward)
        constraint_violations["episode_violations"].append(episode_violations)

        if episode % config["print_interval"] == 0:
            avg_score = np.mean(score_record[-config["print_interval"] :]) if len(score_record) >= config["print_interval"] else episode_reward
            print(
                f"Episode {episode:04d} | Reward: {episode_reward:.3f} | "
                f"AvgReward: {avg_score:.3f} | Violations: {episode_violations}"
            )

    return score_record, agent, constraint_violations


def multi_run_hppo(config=None):
    """Execute multiple training runs with different seeds.
    All results are saved to a single folder after all runs complete.
    """
    if config is None:
        config = CONFIG

    all_scores = []
    all_agents = []
    all_violations = []

    seeds = config.get("seeds", [])
    print(f"\n{'='*70}")
    print(f"Starting multi-run HPPO with seeds: {seeds}")
    print(f"{'='*70}")

    for idx, seed in enumerate(seeds):
        print(f"\n--- Run {idx+1}/{len(seeds)} | Seed: {seed} ---")
        run_config = config.copy()
        run_config["seed"] = seed
        scores, agent, violations = train_hppo(run_config)

        all_scores.append(scores)
        all_agents.append(agent)
        all_violations.append(violations)
        
        print(f"Run {idx+1} completed - Final Score: {scores[-1]:.4f}, Violations: {violations['total_violations']}")

    print(f"\n{'='*70}")
    print(f"All {len(seeds)} runs completed!")
    print(f"{'='*70}")

    return all_scores, all_agents, all_violations


@torch.no_grad()
def test_hppo(agent: HPPOAgent, config=None):
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
    step = 0

    while not done and step < config["max_steps"]:
        action_env, _, _, _, _, _, _ = agent.select_action(state)
        state, reward, done, info = env.step(action_env)
        total_reward += reward
        step += 1

    print(f"Test finished. Steps={step}, TotalReward={total_reward:.3f}")
    return total_reward, step


def save_model(agent: HPPOAgent, score_record, config, save_dir=None):
    if not config.get("save_models", False):
        return None
    if save_dir is None:
        now_str = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = f"hppo_{now_str}"
    os.makedirs(save_dir, exist_ok=True)

    torch.save(agent.model.state_dict(), os.path.join(save_dir, "hppo_actor_critic.pth"))
    np.save(os.path.join(save_dir, "hppo_scores.npy"), np.array(score_record))
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"Model and scores saved to {save_dir}")
    return save_dir


def plot_training_curve(score_record, save_dir=None):
    if not CONFIG["plot_training_curve"]:
        return
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(score_record)
    plt.title("HPPO Training Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)

    if save_dir:
        path = os.path.join(save_dir, "hppo_training_curve.png")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Curve saved to {path}")
    plt.show()


def plot_multi_run_curves(all_scores, config, save_dir=None):
    """Plot training curves for multiple runs with mean and std."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 8))
    
    seeds = config.get("seeds", [])
    for i, scores in enumerate(all_scores):
        plt.plot(scores, alpha=0.5, label=f"Run {i+1} (Seed {seeds[i]})")
    
    # Calculate mean and std
    min_len = min(len(s) for s in all_scores)
    scores_array = np.array([s[:min_len] for s in all_scores])
    mean_scores = scores_array.mean(axis=0)
    std_scores = scores_array.std(axis=0)
    
    plt.plot(mean_scores, "k-", linewidth=2, label="Mean")
    plt.fill_between(range(min_len), mean_scores - std_scores, mean_scores + std_scores,
                     alpha=0.2, color="black")
    
    plt.title("HPPO Multi-Run Training Curves")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    if save_dir:
        path = os.path.join(save_dir, "hppo_multi_run_curves.png")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Multi-run curves saved to {path}")
    plt.show()


def save_multi_run_results(all_scores, all_agents, all_violations, config, save_dir=None):
    """Save all multi-run results (models, scores, stats) to a single folder."""
    import csv
    
    if save_dir is None:
        now_str = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = f"hppo_multi_run_{now_str}"
    os.makedirs(save_dir, exist_ok=True)
    
    seeds = config.get("seeds", [])
    
    # Save all models to the same folder
    for i, (agent, scores, seed) in enumerate(zip(all_agents, all_scores, seeds)):
        # Save model
        model_path = os.path.join(save_dir, f"hppo_run{i+1}_seed{seed}_model.pth")
        torch.save(agent.model.state_dict(), model_path)
        
        # Save scores for each run
        scores_path = os.path.join(save_dir, f"hppo_run{i+1}_seed{seed}_scores.npy")
        np.save(scores_path, np.array(scores))
    
    # Save per-run summary
    summary_path = os.path.join(save_dir, "hppo_multi_run_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Run", "Seed", "Final_Score", "Avg_Last20", "Total_Violations", "Avg_Violation_Rate"])
        
        for i, (scores, violations) in enumerate(zip(all_scores, all_violations)):
            final_score = scores[-1] if scores else 0
            avg_last20 = np.mean(scores[-20:]) if len(scores) >= 20 else np.mean(scores)
            total_violations = violations["total_violations"]
            avg_violation_rate = np.mean([v / 288 * 100 for v in violations["episode_violations"]]) if violations["episode_violations"] else 0
            writer.writerow([i + 1, seeds[i], round(final_score, 2), round(avg_last20, 2), 
                           total_violations, round(avg_violation_rate, 2)])
    
    # Save overall summary
    overall_path = os.path.join(save_dir, "hppo_overall_summary.csv")
    with open(overall_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        final_scores = [s[-1] for s in all_scores]
        avg_last20s = [np.mean(s[-20:]) if len(s) >= 20 else np.mean(s) for s in all_scores]
        total_violations_list = [v["total_violations"] for v in all_violations]
        
        writer.writerow(["Metric", "Mean", "Std", "Min", "Max"])
        writer.writerow(["Final Score", round(np.mean(final_scores), 2), round(np.std(final_scores), 2),
                        round(min(final_scores), 2), round(max(final_scores), 2)])
        writer.writerow(["Avg Last 20", round(np.mean(avg_last20s), 2), round(np.std(avg_last20s), 2),
                        round(min(avg_last20s), 2), round(max(avg_last20s), 2)])
        writer.writerow(["Total Violations", round(np.mean(total_violations_list), 2), 
                        round(np.std(total_violations_list), 2),
                        min(total_violations_list), max(total_violations_list)])
    
    # Save all scores combined
    all_scores_path = os.path.join(save_dir, "hppo_all_scores.npy")
    np.save(all_scores_path, np.array(all_scores, dtype=object), allow_pickle=True)
    
    # Save config
    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\nMulti-run results saved to: {save_dir}")
    print(f"  - {len(all_agents)} models saved")
    print(f"  - {len(all_scores)} score files saved")
    print(f"  - Summary CSVs saved")
    
    return save_dir


def evaluate_multi_run_results(all_scores, all_violations, config):
    """Print evaluation of multi-run results."""
    seeds = config.get("seeds", [])
    
    print(f"\n{'='*70}")
    print("Multi-Run HPPO Results Evaluation")
    print(f"{'='*70}")
    
    final_scores = [s[-1] for s in all_scores]
    avg_last20s = [np.mean(s[-20:]) if len(s) >= 20 else np.mean(s) for s in all_scores]
    
    print("\nPer-Run Results:")
    for i, (scores, violations) in enumerate(zip(all_scores, all_violations)):
        print(f"  Run {i+1} (Seed {seeds[i]}): Final={scores[-1]:.2f}, "
              f"AvgLast20={avg_last20s[i]:.2f}, Violations={violations['total_violations']}")
    
    print(f"\nOverall Statistics:")
    print(f"  Mean Final Score: {np.mean(final_scores):.2f} ± {np.std(final_scores):.2f}")
    print(f"  Best Final Score: {max(final_scores):.2f}")
    print(f"  Worst Final Score: {min(final_scores):.2f}")
    print(f"  Mean of Last 20 Episodes: {np.mean(avg_last20s):.2f} ± {np.std(avg_last20s):.2f}")
    
    total_violations_list = [v["total_violations"] for v in all_violations]
    print(f"  Mean Violations: {np.mean(total_violations_list):.2f} ± {np.std(total_violations_list):.2f}")
    print(f"{'='*70}")
    
    return {
        "final_scores": final_scores,
        "avg_last20s": avg_last20s,
        "total_violations": total_violations_list
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="HPPO for EV Charging Station")
    parser.add_argument("--stations", type=int, default=CONFIG["n_stations"])
    parser.add_argument("--power", type=float, default=CONFIG["p_max"])
    parser.add_argument("--arrival-rate", type=float, default=CONFIG["arrival_rate"])
    parser.add_argument("--episodes", type=int, default=CONFIG["num_of_episodes"])
    parser.add_argument("--test-only", action="store_true", help="Only run test (requires trained model)")
    parser.add_argument("--model-path", type=str, default=None, help="Path to a trained actor-critic model")
    parser.add_argument("--multi-run", action="store_true", default=CONFIG["enable_multi_run"],
                       help="Run multiple training sessions with different seeds")
    parser.add_argument("--single-run", action="store_true", help="Force single run (override config)")
    parser.add_argument("--seeds", nargs="+", type=int, default=CONFIG["seeds"],
                       help="Random seeds for multi-run training")
    args = parser.parse_args()

    config = CONFIG.copy()
    config.update({
        "n_stations": args.stations,
        "p_max": args.power,
        "arrival_rate": args.arrival_rate,
        "num_of_episodes": args.episodes,
        "seeds": args.seeds,
        "enable_multi_run": args.multi_run and not args.single_run,
    })

    print(f"\n{'='*60}")
    print(f"HPPO for EV Charging Station")
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
        if args.model_path and os.path.exists(args.model_path):
            state_dim_dummy = EVChargingEnv(n_stations=config["n_stations"], p_max=config["p_max"],
                                            arrival_rate=config["arrival_rate"]).observation_space.shape[0]
            agent = HPPOAgent(state_dim_dummy, config["n_stations"],
                              EVChargingEnv(n_stations=config["n_stations"], p_max=config["p_max"],
                                            arrival_rate=config["arrival_rate"]).max_vehicles)
            agent.model.load_state_dict(torch.load(args.model_path, map_location=device))
            test_hppo(agent, config)
        else:
            print("Test-only mode requires --model-path pointing to a saved model.")
    
    elif config["enable_multi_run"]:
        # Multi-run training
        all_scores, all_agents, all_violations = multi_run_hppo(config)
        
        # Evaluate results
        evaluate_multi_run_results(all_scores, all_violations, config)
        
        # Save all results to a single folder
        save_dir = None
        if config.get("save_models", False):
            save_dir = save_multi_run_results(all_scores, all_agents, all_violations, config)
            plot_multi_run_curves(all_scores, config, save_dir)
        
        # Test best model
        best_idx = np.argmax([s[-1] for s in all_scores])
        print(f"\n{'='*40}")
        print(f"Testing with best model (Run {best_idx + 1}, Seed {config['seeds'][best_idx]})...")
        print(f"{'='*40}")
        test_hppo(all_agents[best_idx], config)
    
    else:
        # Single run training
        if config["seeds"]:
            config["seed"] = config["seeds"][0]
            print(f"Using random seed: {config['seed']}")
        
        scores, agent, violations = train_hppo(config)
        save_dir = save_model(agent, scores, config)
        plot_training_curve(scores, save_dir)
        test_hppo(agent, config)

