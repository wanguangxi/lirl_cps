"""
Lagrangian PPO (LPPO) for Protein Crystallization Screening

核心思想：
- 标准PPO最大化期望回报
- 拉格朗日乘子λ用于惩罚约束违反
- 同时更新策略参数θ和拉格朗日乘子λ
- 目标: max_θ min_λ E[reward] - λ * (E[constraint_cost] - threshold)

没有动作修正/投影，通过拉格朗日惩罚学习满足约束的策略。
"""

import os
import sys
import json
import random
import datetime
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
    "batch_size": 64,           # Mini-batch size
    
    # Lagrangian constraint parameters
    "constraint_threshold": 0.05,   # Target constraint violation rate (5%)
    "lambda_init": 1.0,             # Initial Lagrangian multiplier
    "lambda_max": 50.0,             # Maximum Lagrangian multiplier
    "lambda_min": 0.01,             # Minimum Lagrangian multiplier
    "cost_scale": 1.0,              # Scale factor for constraint cost
    
    # Environment parameters
    "batch_size_env": 2,        # Droplets per step
    "horizon": 25,              # Episode length
    "seed": 42,
    "num_of_episodes": 500,
    
    # Network architecture
    "hidden_dim1": 256,
    "hidden_dim2": 128,
    
    # Output parameters
    "print_interval": 10,
    "save_models": True,
    "plot_training_curve": True,
}


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
        self.protocol_actions = []
        self.cont_samples = []
        self.cont_actions = []
        self.logprobs = []
        self.rewards = []
        self.costs = []  # Constraint violation costs
        self.dones = []
        self.values = []
        self.cost_values = []  # Value estimates for cost


class LagrangianActorCritic(nn.Module):
    """
    Actor-Critic network for Lagrangian PPO.
    Includes both reward value head and cost value head.
    """
    
    def __init__(self, state_dim: int, n_protocols: int, batch_size_env: int,
                 cont_dim_per_droplet: int, config: dict):
        super().__init__()
        self.n_protocols = n_protocols
        self.batch_size_env = batch_size_env
        self.cont_dim_per_droplet = cont_dim_per_droplet
        self.total_cont_dim = batch_size_env * cont_dim_per_droplet
        
        hidden1 = config["hidden_dim1"]
        hidden2 = config["hidden_dim2"]
        
        # Shared feature extractor
        self.fc1 = nn.Linear(state_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        
        # Actor heads: protocol head per droplet + continuous parameters
        self.protocol_heads = nn.ModuleList([
            nn.Linear(hidden2, n_protocols) for _ in range(batch_size_env)
        ])
        self.cont_mean = nn.Linear(hidden2, self.total_cont_dim)
        self.cont_log_std = nn.Parameter(torch.zeros(self.total_cont_dim))
        
        # Critic heads (reward value + cost value)
        self.value_head = nn.Linear(hidden2, 1)      # V(s) for reward
        self.cost_value_head = nn.Linear(hidden2, 1)  # V_c(s) for cost
    
    def forward(self, state: torch.Tensor):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        protocol_logits = [head(x) for head in self.protocol_heads]
        cont_mean = self.cont_mean(x)
        cont_log_std = self.cont_log_std.expand_as(cont_mean)
        
        value = self.value_head(x)
        cost_value = self.cost_value_head(x)
        
        return protocol_logits, cont_mean, cont_log_std, value, cost_value


class LPPOAgent:
    """
    Lagrangian PPO Agent for constrained reinforcement learning.
    
    Key features:
    - Dual optimization: max_θ min_λ objective
    - Learnable Lagrangian multiplier λ
    - Separate value functions for reward and cost
    """
    
    def __init__(self, state_dim: int, n_protocols: int, batch_size_env: int,
                 cont_dim_per_droplet: int, config: dict):
        self.n_protocols = n_protocols
        self.batch_size_env = batch_size_env
        self.cont_dim_per_droplet = cont_dim_per_droplet
        self.total_cont_dim = batch_size_env * cont_dim_per_droplet
        self.config = config
        
        # Actor-Critic network
        self.model = LagrangianActorCritic(
            state_dim, n_protocols, batch_size_env, cont_dim_per_droplet, config
        ).to(DEVICE)
        
        # Lagrangian multiplier (learnable)
        self.log_lambda = torch.tensor(
            np.log(max(config["lambda_init"], 1e-8)),
            requires_grad=True,
            device=DEVICE
        )
        
        # Optimizers
        actor_params = []
        for head in self.model.protocol_heads:
            actor_params.extend(list(head.parameters()))
        actor_params.extend(list(self.model.cont_mean.parameters()))
        actor_params.append(self.model.cont_log_std)
        
        self.actor_optimizer = optim.Adam(actor_params, lr=config["lr_actor"])
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
    def select_action(self, state: np.ndarray, spec):
        """Select action using current policy."""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        protocol_logits_list, cont_mean, cont_log_std, value, cost_value = self.model(state_t)
        
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
            cost_value.item(),
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
        """Evaluate actions for PPO update."""
        protocol_logits_list, cont_mean, cont_log_std, values, cost_values = self.model(states)
        
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
        
        return total_logprob, total_entropy, values.squeeze(-1), cost_values.squeeze(-1)
    
    def update(self, buffer: RolloutBuffer):
        """
        Lagrangian PPO update.
        
        1. Update critic (value functions)
        2. Update actor with Lagrangian penalty
        3. Update Lagrangian multiplier
        """
        states = torch.FloatTensor(np.array(buffer.states)).to(DEVICE)
        protocol_actions = torch.LongTensor(np.array(buffer.protocol_actions)).to(DEVICE)
        cont_samples = torch.FloatTensor(np.array(buffer.cont_samples)).to(DEVICE)
        old_logprobs = torch.FloatTensor(buffer.logprobs).to(DEVICE)
        returns = torch.FloatTensor(buffer.returns).to(DEVICE)
        advantages = torch.FloatTensor(buffer.advantages).to(DEVICE)
        cost_returns = torch.FloatTensor(buffer.cost_returns).to(DEVICE)
        cost_advantages = torch.FloatTensor(buffer.cost_advantages).to(DEVICE)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        cost_advantages = (cost_advantages - cost_advantages.mean()) / (cost_advantages.std() + 1e-8)
        
        dataset_size = len(states)
        batch_size = min(self.config["batch_size"], dataset_size)
        indices = np.arange(dataset_size)
        
        # Track losses
        total_policy_loss = 0
        total_value_loss = 0
        total_cost_value_loss = 0
        n_updates = 0
        
        for _ in range(self.config["ppo_epochs"]):
            np.random.shuffle(indices)
            
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]
                
                logprob, entropy, values, cost_values = self.evaluate_actions(
                    states[batch_idx],
                    protocol_actions[batch_idx],
                    cont_samples[batch_idx],
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
    """Main training function for Lagrangian PPO."""
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
    cont_dim_per_droplet = spec.R + 2
    
    # Initialize agent
    agent = LPPOAgent(state_dim, n_protocols, batch_size_env, cont_dim_per_droplet, config)
    buffer = RolloutBuffer()
    
    score_record = []
    best_quality_record = []
    lambda_record = []
    total_violations = 0
    total_steps = 0
    
    print(f"\nStarting Lagrangian PPO training for Protein Crystallization:")
    print(f"Protocols: {n_protocols}, Droplets/step: {batch_size_env}, Horizon: {config['horizon']}")
    print(f"Episodes: {config['num_of_episodes']}, Constraint Threshold: {config['constraint_threshold']*100:.1f}%")
    print(f"Initial λ: {config['lambda_init']}")
    print(f"NOTE: LPPO does NOT use action projection - learns via Lagrangian penalty")
    print("-" * 70)
    
    for episode in range(config["num_of_episodes"]):
        state, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0.0
        episode_violations = 0
        episode_steps = 0
        buffer.clear()
        
        while not (done or truncated):
            # Select action
            (k_vec, u_mat, logprob, value, cost_value,
             protocol_actions, cont_sample, cont_action) = agent.select_action(state, spec)
            
            # Check constraint violations
            step_violations = 0
            for j in range(batch_size_env):
                is_feasible, viol_count, _ = check_feasible(spec, k_vec[j], u_mat[j])
                if not is_feasible:
                    step_violations += viol_count
            
            episode_violations += step_violations
            total_violations += step_violations
            total_steps += batch_size_env
            episode_steps += 1
            
            # Compute constraint cost
            cost = config["cost_scale"] * step_violations / batch_size_env
            
            # Execute action
            action = {"k": k_vec, "u": u_mat}
            next_state, reward, done, truncated, info = env.step(action)
            
            # Store transition
            buffer.states.append(state)
            buffer.protocol_actions.append(protocol_actions)
            buffer.cont_samples.append(cont_sample)
            buffer.cont_actions.append(cont_action)
            buffer.logprobs.append(logprob)
            buffer.rewards.append(reward)
            buffer.costs.append(cost)
            buffer.dones.append(done)
            buffer.values.append(value)
            buffer.cost_values.append(cost_value)
            
            state = next_state
            episode_reward += reward
        
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
        best_quality_record.append(info.get('best_quality', 0))
        lambda_record.append(update_info["lambda"])
        
        # Print progress
        if episode % config["print_interval"] == 0 and episode != 0:
            avg_score = np.mean(score_record[-config["print_interval"]:])
            avg_lambda = np.mean(lambda_record[-config["print_interval"]:])
            running_cvr = total_violations / max(1, total_steps)
            
            print(f"Episode {episode:04d} | Avg Score: {avg_score:.4f} | "
                  f"Best Quality: {info.get('best_quality', 0):.4f} | "
                  f"λ: {update_info['lambda']:.4f} | "
                  f"Epi Viol: {episode_violations} | CVR: {running_cvr:.4f}")
    
    # Final statistics
    print(f"\n{'='*70}")
    print(f"Lagrangian PPO Training Completed!")
    print(f"{'='*70}")
    print(f"Final λ: {agent.lambda_value.item():.4f}")
    print(f"Total violations: {total_violations}")
    print(f"Total action steps: {total_steps}")
    print(f"Overall violation rate: {(total_violations / max(1, total_steps)):.4f}")
    print(f"Target threshold: {config['constraint_threshold']*100:.1f}%")
    print(f"{'='*70}")
    
    # Move model to CPU for saving
    agent.model.cpu()
    
    return score_record, best_quality_record, lambda_record, agent, total_violations


def save_results(score_records, lambda_records, agent, config, total_violations):
    """Save training results and models."""
    if not config['save_models']:
        return
    
    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_base_dir = "/home/one/LIRL-CPS-main/Protein_Crystallization_Screening/exp"
    save_dir = os.path.join(exp_base_dir, f"lppo_crystallization_{now_str}")
    os.makedirs(save_dir, exist_ok=True)

    # Save scores and lambdas
    np.save(os.path.join(save_dir, "scores.npy"), score_records)
    np.save(os.path.join(save_dir, "lambdas.npy"), lambda_records)

    # Save model
    torch.save(agent.model.state_dict(), os.path.join(save_dir, "lppo_model.pth"))

    # Save config
    config_to_save = config.copy()
    config_to_save['device'] = str(DEVICE)
    config_to_save['total_violations'] = total_violations
    config_to_save['final_lambda'] = float(lambda_records[-1]) if lambda_records else config["lambda_init"]
    with open(os.path.join(save_dir, "config.json"), 'w') as f:
        json.dump(config_to_save, f, indent=2)

    # Save training curves
    if config['plot_training_curve']:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Reward curve
        ax1 = axes[0]
        ax1.plot(score_records, label='Episode Reward')
        window = min(20, len(score_records) // 5) if len(score_records) > 10 else 1
        if window > 1:
            moving_avg = np.convolve(score_records, np.ones(window)/window, mode='valid')
            ax1.plot(range(window-1, len(score_records)), moving_avg, 'r-', linewidth=2,
                    label=f'Moving Avg (window={window})')
        ax1.set_title("Lagrangian PPO Training Reward - Protein Crystallization")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Reward")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Lambda curve
        ax2 = axes[1]
        ax2.plot(lambda_records, label='Lagrangian Multiplier λ')
        ax2.axhline(y=config["constraint_threshold"], color='r', linestyle='--',
                   label=f'Constraint Threshold ({config["constraint_threshold"]*100:.1f}%)')
        if window > 1:
            moving_avg_lambda = np.convolve(lambda_records, np.ones(window)/window, mode='valid')
            ax2.plot(range(window-1, len(lambda_records)), moving_avg_lambda, 'g-', linewidth=2,
                    label=f'Moving Avg (window={window})')
        ax2.set_title("Lagrangian Multiplier (λ) Evolution")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("λ")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "training_curves.png"), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Training curves saved to: {os.path.join(save_dir, 'training_curves.png')}")

    print(f"Results saved to: {save_dir}")


if __name__ == "__main__":
    score_record, lambda_record, agent, total_violations = main(CONFIG)
    save_results(score_record, lambda_record, agent, CONFIG, total_violations)
