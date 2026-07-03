"""
HPPO (Hybrid Proximal Policy Optimization) for Protein Crystallization Screening

Uses PPO to handle hybrid action space:
- Discrete: protocol selection for each droplet
- Continuous: composition, temperature, time parameters

KEY DIFFERENCE FROM LIRL:
- HPPO has NO action correction/projection mechanism
- Agent learns valid actions through constraint penalties (negative rewards)
- Over time, the agent learns to avoid constraint-violating actions
"""

import os
import sys
import random
import datetime
import json
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal
import matplotlib.pyplot as plt

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
    "lr": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_eps": 0.2,
    "entropy_coef": 0.01,
    "value_coef": 0.5,
    "max_grad_norm": 0.5,
    "ppo_epochs": 5,
    "batch_size": 64,

    # Environment parameters
    "batch_size_env": 2,        # Droplets per step
    "horizon": 25,              # Episode length
    "seed": 42,
    "num_of_episodes": 500,

    # Network
    "hidden_dim1": 256,
    "hidden_dim2": 128,

    # Constraint penalty
    "constraint_penalty": 0.5,

    # Output
    "print_interval": 10,
    "save_models": True,
    "plot_training_curve": True,
}


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
        self.protocol_actions = []      # Discrete protocol indices for each droplet
        self.cont_samples = []          # Pre-sigmoid continuous samples
        self.cont_actions = []          # Sigmoid outputs in [0,1]
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def add(self, state, protocol_action, cont_sample, cont_action,
            logprob, reward, done, value):
        self.states.append(state)
        self.protocol_actions.append(protocol_action)
        self.cont_samples.append(cont_sample)
        self.cont_actions.append(cont_action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)


class HybridActorCritic(nn.Module):
    """Shared backbone with discrete protocol logits and continuous parameter heads."""

    def __init__(self, state_dim: int, n_protocols: int, batch_size_env: int, cont_dim_per_droplet: int):
        super().__init__()
        self.n_protocols = n_protocols
        self.batch_size_env = batch_size_env
        self.cont_dim_per_droplet = cont_dim_per_droplet
        self.total_cont_dim = batch_size_env * cont_dim_per_droplet

        # Shared backbone
        self.fc1 = nn.Linear(state_dim, CONFIG["hidden_dim1"])
        self.fc2 = nn.Linear(CONFIG["hidden_dim1"], CONFIG["hidden_dim2"])

        # Discrete heads: one protocol head per droplet
        self.protocol_heads = nn.ModuleList([
            nn.Linear(CONFIG["hidden_dim2"], n_protocols) for _ in range(batch_size_env)
        ])

        # Continuous head: mean and log_std for all continuous parameters
        self.cont_mean = nn.Linear(CONFIG["hidden_dim2"], self.total_cont_dim)
        self.cont_log_std = nn.Parameter(torch.zeros(self.total_cont_dim))

        # Value head
        self.value_head = nn.Linear(CONFIG["hidden_dim2"], 1)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        # Protocol logits for each droplet
        protocol_logits = [head(x) for head in self.protocol_heads]

        # Continuous parameters
        cont_mean = self.cont_mean(x)
        cont_log_std = self.cont_log_std.expand_as(cont_mean)

        # Value
        value = self.value_head(x)

        return protocol_logits, cont_mean, cont_log_std, value


class HPPOAgent:
    def __init__(self, state_dim: int, n_protocols: int, batch_size_env: int, cont_dim_per_droplet: int):
        self.n_protocols = n_protocols
        self.batch_size_env = batch_size_env
        self.cont_dim_per_droplet = cont_dim_per_droplet
        self.total_cont_dim = batch_size_env * cont_dim_per_droplet

        self.model = HybridActorCritic(
            state_dim, n_protocols, batch_size_env, cont_dim_per_droplet
        ).to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=CONFIG["lr"])

    @torch.no_grad()
    def select_action(self, state: np.ndarray, spec):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        protocol_logits_list, cont_mean, cont_log_std, value = self.model(state_t)

        # Sample discrete protocol for each droplet
        protocol_actions = []
        logprob_protocols = 0.0
        for i, logits in enumerate(protocol_logits_list):
            dist = Categorical(logits=logits)
            action = dist.sample()
            protocol_actions.append(action.item())
            logprob_protocols += dist.log_prob(action)

        # Sample continuous parameters
        cont_std = cont_log_std.exp()
        cont_dist = Normal(cont_mean, cont_std)
        cont_sample = cont_dist.rsample()
        cont_action = torch.sigmoid(cont_sample)

        # Log-prob with change-of-variables for sigmoid
        logprob_cont = (
            cont_dist.log_prob(cont_sample)
            - torch.log(cont_action.clamp_min(1e-8))
            - torch.log((1 - cont_action).clamp_min(1e-8))
        ).sum(dim=-1)

        total_logprob = logprob_protocols + logprob_cont

        # Convert to environment action format
        k_vec, u_mat = self._convert_to_env_action(
            protocol_actions, 
            cont_action.squeeze().cpu().numpy(), 
            spec
        )

        return (
            k_vec,
            u_mat,
            total_logprob.item(),
            value.item(),
            np.array(protocol_actions),
            cont_sample.squeeze().cpu().numpy(),
            cont_action.squeeze().cpu().numpy(),
        )

    def _convert_to_env_action(self, protocol_actions, cont_params, spec):
        """Convert network outputs to environment action format (NO PROJECTION)."""
        B = self.batch_size_env
        d = self.cont_dim_per_droplet
        R = spec.R

        k_vec = np.array(protocol_actions, dtype=int)
        
        # Reshape continuous parameters to (B, d)
        cont_params = cont_params.reshape(B, d)
        
        # Scale parameters to appropriate ranges
        u_mat = np.zeros((B, d), dtype=np.float32)
        for j in range(B):
            # Component fractions: scale from [0,1] to [0, p_max], then normalize to simplex
            p = cont_params[j, :R] * spec.p_max
            p_sum = np.sum(p)
            if p_sum > 0:
                p = p / p_sum
            else:
                p = np.ones(R) / R
            u_mat[j, :R] = p
            
            # Temperature: scale from [0,1] to T_bounds
            u_mat[j, R] = spec.T_bounds[0] + cont_params[j, R] * (spec.T_bounds[1] - spec.T_bounds[0])
            
            # Time: scale from [0,1] to tau_bounds
            u_mat[j, R+1] = spec.tau_bounds[0] + cont_params[j, R+1] * (spec.tau_bounds[1] - spec.tau_bounds[0])

        return k_vec, u_mat

    def evaluate_actions(self, states, protocol_actions, cont_samples):
        """Evaluate log probs and entropy for given actions."""
        protocol_logits_list, cont_mean, cont_log_std, values = self.model(states)

        # Protocol log probs
        logprob_protocols = torch.zeros(states.shape[0], device=DEVICE)
        entropy_protocols = torch.zeros(states.shape[0], device=DEVICE)
        
        for i, logits in enumerate(protocol_logits_list):
            dist = Categorical(logits=logits)
            logprob_protocols += dist.log_prob(protocol_actions[:, i])
            entropy_protocols += dist.entropy()

        # Continuous log probs
        cont_std = cont_log_std.exp()
        cont_dist = Normal(cont_mean, cont_std)
        
        cont_action = torch.sigmoid(cont_samples)
        logprob_cont = (
            cont_dist.log_prob(cont_samples)
            - torch.log(cont_action.clamp_min(1e-8))
            - torch.log((1 - cont_action).clamp_min(1e-8))
        ).sum(dim=-1)

        total_logprob = logprob_protocols + logprob_cont
        total_entropy = entropy_protocols + cont_dist.entropy().sum(dim=-1)

        return total_logprob, total_entropy, values.squeeze(-1)

    def update(self, buffer: RolloutBuffer):
        states = torch.FloatTensor(np.array(buffer.states)).to(DEVICE)
        protocol_actions = torch.LongTensor(np.array(buffer.protocol_actions)).to(DEVICE)
        cont_samples = torch.FloatTensor(np.array(buffer.cont_samples)).to(DEVICE)
        old_logprobs = torch.FloatTensor(buffer.logprobs).to(DEVICE)
        returns = torch.FloatTensor(buffer.returns).to(DEVICE)
        advantages = torch.FloatTensor(buffer.advantages).to(DEVICE)

        # Normalize advantages
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
                    protocol_actions[batch_idx],
                    cont_samples[batch_idx],
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
    """Compute Generalized Advantage Estimation."""
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


def check_feasible(spec, k, u) -> tuple:
    """Check if an action satisfies all constraints."""
    R = spec.R
    p = u[:R]
    T = float(u[R])
    tau = float(u[R + 1])
    
    violations = []
    
    if np.any(p < 0):
        violations.append("negative_component")
    if np.any(p > spec.p_max):
        violations.append("component_exceeds_max")
    if abs(float(np.sum(p)) - 1.0) > 0.01:
        violations.append("simplex_violation")
    if not (spec.T_bounds[0] <= T <= spec.T_bounds[1]):
        violations.append("temperature_out_of_bounds")
    if not (spec.tau_bounds[0] <= tau <= spec.tau_bounds[1]):
        violations.append("time_out_of_bounds")
    
    proto = spec.protocols[int(k)]
    if proto.G is not None and proto.G.size > 0:
        constraint_violations = proto.G @ u - proto.h
        if np.any(constraint_violations > 1e-3):
            violations.append("protocol_constraint_violation")
    
    return len(violations) == 0, len(violations), violations


def main(config=None):
    """Main training function for HPPO."""
    if config is None:
        config = CONFIG

    set_seed(config.get("seed"))

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

    state_dim = env.observation_space.shape[0]
    n_protocols = spec.K
    batch_size_env = spec.batch_size
    cont_dim_per_droplet = spec.R + 2  # d = R + 2

    # Initialize agent
    agent = HPPOAgent(state_dim, n_protocols, batch_size_env, cont_dim_per_droplet)
    buffer = RolloutBuffer()

    score_record = []
    best_quality_record = []
    total_violations = 0
    total_steps = 0

    print(f"\nStarting HPPO training for Protein Crystallization:")
    print(f"Protocols: {n_protocols}, Droplets/step: {batch_size_env}, Horizon: {config['horizon']}")
    print(f"Episodes: {config['num_of_episodes']}")
    print(f"State dim: {state_dim}, Cont dim per droplet: {cont_dim_per_droplet}")
    print(f"NOTE: HPPO does NOT use action projection - learns from constraint penalties")

    for episode in range(config["num_of_episodes"]):
        state, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0.0
        episode_violations = 0
        buffer.clear()

        while not (done or truncated):
            k_vec, u_mat, logprob, value, protocol_actions, cont_sample, cont_action = \
                agent.select_action(state, spec)

            # Check constraint violations
            step_violations = 0
            for j in range(batch_size_env):
                is_feasible, viol_count, _ = check_feasible(spec, k_vec[j], u_mat[j])
                if not is_feasible:
                    step_violations += viol_count

            episode_violations += step_violations
            total_violations += step_violations
            total_steps += batch_size_env

            # Execute action
            action = {"k": k_vec, "u": u_mat}
            next_state, reward, done, truncated, info = env.step(action)

            # Apply constraint penalty
            if step_violations > 0:
                reward = reward - config["constraint_penalty"] * step_violations

            buffer.add(state, protocol_actions, cont_sample, cont_action,
                      logprob, reward, done, value)

            state = next_state
            episode_reward += reward

        # Compute GAE and update
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
        best_quality_record.append(info.get('best_quality', 0))

        if episode % config["print_interval"] == 0 and episode != 0:
            avg_score = np.mean(score_record[-config["print_interval"]:])
            running_cvr = total_violations / max(1, total_steps)
            print(
                f"Episode {episode}: Avg Score = {avg_score:.4f}, "
                f"Best Quality = {info.get('best_quality', 0):.4f}, "
                f"Epi Violations = {episode_violations}, "
                f"Running CVR = {running_cvr:.4f}"
            )

    # Final statistics
    print(f"\n{'='*60}")
    print(f"HPPO Training Completed!")
    print(f"{'='*60}")
    print(f"Total constraint violations: {total_violations}")
    print(f"Total action steps: {total_steps}")
    print(f"Overall violation rate: {(total_violations / max(1, total_steps)):.4f}")
    print(f"{'='*60}")

    # Move model to CPU for saving
    agent.model.cpu()

    return score_record, best_quality_record, agent, total_violations


def save_results(score_records, agent, config, total_violations):
    """Save training results and models."""
    if not config['save_models']:
        return

    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_base_dir = "/home/one/LIRL-CPS-main/Protein_Crystallization_Screening/exp"
    save_dir = os.path.join(exp_base_dir, f"hppo_crystallization_{now_str}")
    os.makedirs(save_dir, exist_ok=True)

    # Save scores
    np.save(os.path.join(save_dir, "scores.npy"), score_records)

    # Save model
    torch.save(agent.model.state_dict(), os.path.join(save_dir, "hppo_model.pth"))

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
            plt.plot(range(window-1, len(score_records)), moving_avg, 'r-', linewidth=2, 
                    label=f'Moving Avg (window={window})')
        plt.title("HPPO Training Curve - Protein Crystallization (No Projection)")
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
