"""
Algorithm Comparison Script for EV Charging Station Control

对比五种算法：
1. LIRL (DDPG-based)
2. PDQN (Parameterized DQN)
3. HPPO (Hybrid PPO)
4. LPPO (Lagrangian PPO)
5. CPO (Constrained Policy Optimization)

功能：
- 每种算法训练500回合，测试10回合
- 绘制训练曲线对比
- 对比各项指标差异
- 保存所有训练数据和模型
"""

import os
import sys
import json
import random
import datetime as dt
from typing import Dict, List, Tuple
import csv

import numpy as np
import torch
import matplotlib.pyplot as plt

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), "../alg"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../env"))

from ev import EVChargingEnv

# =======================
# COMPARISON CONFIG
# =======================
COMPARE_CONFIG = {
    # Training parameters
    "num_episodes": 1000,        # Training episodes per algorithm
    "num_test_episodes": 10,    # Test episodes
    "max_steps": 288,           # Max steps per episode
    
    # Environment parameters
    "n_stations": 5,
    "p_max": 150.0,
    "arrival_rate": 0.75,
    
    # Random seed for reproducibility
    "seed": 3047,
    
    # Output
    "save_dir": None,  # Auto-generated
    "print_interval": 50,
    
    # Test-only mode
    "test_only": False,         # If True, only run testing (requires model_dir)
    "model_dir": None,          # Directory containing pre-trained models
}


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def collect_test_metrics(env, info, step, episode_metrics):
    """Collect comprehensive metrics during testing."""
    # Track station utilization (occupied / total)
    available = info.get('available_stations', env.n_stations)
    occupied = env.n_stations - available
    episode_metrics['station_utilization_sum'] += occupied / env.n_stations
    episode_metrics['total_steps'] += 1


def run_comprehensive_test(env, action_fn, config, algorithm_name):
    """
    Run comprehensive testing and collect all metrics.
    
    Args:
        env: EVChargingEnv environment
        action_fn: Function that takes (state, env) and returns action
        config: Configuration dict
        algorithm_name: Name of algorithm for logging
    
    Returns:
        Dict with all test metrics
    """
    test_results = {
        'rewards': [],
        'violations': [],
        'total_energy': [],
        'total_cost': [],
        'total_damage': [],
        'station_utilization': [],
        'charging_success_rate': [],
        'violation_rate': [],
    }
    
    for ep in range(config['num_test_episodes']):
        state = env.reset()
        done = False
        episode_reward = 0
        episode_violations = 0
        step = 0
        station_util_sum = 0.0
        
        while not done and step < config['max_steps']:
            action = action_fn(state, env)
            next_state, reward, done, info = env.step(action)
            
            # Track violations
            violation_info = info.get('constraint_violation', None)
            if violation_info and violation_info.get('has_violation', False):
                episode_violations += 1
            
            # Track station utilization
            available = info.get('available_stations', env.n_stations)
            occupied = env.n_stations - available
            station_util_sum += occupied / env.n_stations
            
            episode_reward += reward
            state = next_state
            step += 1
        
        # Get final episode metrics from info
        total_energy = info.get('total_energy', 0.0)
        total_cost = info.get('total_cost', 0.0)
        total_damage = info.get('total_lifetime_damage', 0.0)
        episode_arrivals = info.get('episode_arrivals', 1)
        episode_charged = info.get('episode_charged_count', 0)
        
        # Calculate derived metrics
        station_utilization = station_util_sum / max(step, 1) * 100  # percentage
        charging_success_rate = episode_charged / max(episode_arrivals, 1) * 100  # percentage
        violation_rate = episode_violations / max(step, 1) * 100  # percentage
        
        # Store results
        test_results['rewards'].append(episode_reward)
        test_results['violations'].append(episode_violations)
        test_results['total_energy'].append(total_energy)
        test_results['total_cost'].append(total_cost)
        test_results['total_damage'].append(total_damage)
        test_results['station_utilization'].append(station_utilization)
        test_results['charging_success_rate'].append(charging_success_rate)
        test_results['violation_rate'].append(violation_rate)
        
        print(f"  Episode {ep+1}: Reward={episode_reward:.1f}, Energy={total_energy:.1f}kWh, "
              f"Success={charging_success_rate:.1f}%, Violations={episode_violations}")
    
    # Calculate averages
    test_metrics = {
        'test_rewards': test_results['rewards'],
        'test_violations': test_results['violations'],
        'avg_reward': np.mean(test_results['rewards']),
        'std_reward': np.std(test_results['rewards']),
        'avg_energy': np.mean(test_results['total_energy']),
        'avg_cost': np.mean(test_results['total_cost']),
        'avg_damage': np.mean(test_results['total_damage']),
        'avg_station_utilization': np.mean(test_results['station_utilization']),
        'avg_charging_success_rate': np.mean(test_results['charging_success_rate']),
        'avg_violation_rate': np.mean(test_results['violation_rate']),
    }
    
    print(f"\n  {algorithm_name} Test Summary:")
    print(f"    Avg Reward: {test_metrics['avg_reward']:.2f} ± {test_metrics['std_reward']:.2f}")
    print(f"    Avg Energy: {test_metrics['avg_energy']:.2f} kWh")
    print(f"    Avg Cost: {test_metrics['avg_cost']:.2f}")
    print(f"    Station Utilization: {test_metrics['avg_station_utilization']:.2f}%")
    print(f"    Charging Success Rate: {test_metrics['avg_charging_success_rate']:.2f}%")
    print(f"    Violation Rate: {test_metrics['avg_violation_rate']:.2f}%")
    print(f"    Avg Damage: {test_metrics['avg_damage']:.4f}")
    
    return test_metrics


# =======================
# Algorithm Runners
# =======================

def run_lirl(config: dict) -> Dict:
    """Run LIRL (DDPG-based) algorithm."""
    print("\n" + "="*70)
    print("Training LIRL (DDPG-based)")
    print("="*70)
    
    # Import action_projection_ev which is critical for LIRL's performance
    from lirl import main as lirl_main, test_and_visualize, MuNet, CONFIG as LIRL_CONFIG, action_projection_ev
    
    # Update config
    lirl_config = LIRL_CONFIG.copy()
    lirl_config.update({
        'n_stations': config['n_stations'],
        'p_max': config['p_max'],
        'arrival_rate': config['arrival_rate'],
        'num_of_episodes': config['num_episodes'],
        'print_interval': config['print_interval'],
        'save_models': False,
        'plot_training_curve': False,
        'enable_multi_run': False,
    })
    
    set_seed(config['seed'])
    
    # Train
    score_record, action_restore, models, constraint_violations, episode_stats = lirl_main(lirl_config)
    
    # Test with comprehensive metrics
    env = EVChargingEnv(
        n_stations=config['n_stations'],
        p_max=config['p_max'],
        arrival_rate=config['arrival_rate']
    )
    
    mu = models[0]  # Get trained policy network
    mu.eval()
    
    # Define action function for LIRL
    def lirl_action_fn(state, env):
        with torch.no_grad():
            a = mu(torch.from_numpy(state).float())
            a = torch.clamp(a, 0, 1)
            if len(a.shape) > 1:
                a = a.squeeze(0)
        return action_projection_ev(env, a)
    
    # Run comprehensive test
    test_metrics = run_comprehensive_test(env, lirl_action_fn, config, 'LIRL')
    
    return {
        'name': 'LIRL',
        'train_scores': score_record,
        'test_rewards': test_metrics['test_rewards'],
        'test_violations': test_metrics['test_violations'],
        'constraint_violations': constraint_violations,
        'episode_stats': episode_stats,
        'model': models,
        'test_metrics': test_metrics,
    }


def run_pdqn(config: dict) -> Dict:
    """Run PDQN algorithm."""
    print("\n" + "="*70)
    print("Training PDQN")
    print("="*70)
    
    from pdqn_ev_charging import main as pdqn_main, PDQNAgent, CONFIG as PDQN_CONFIG
    
    # Update config
    pdqn_config = PDQN_CONFIG.copy()
    pdqn_config.update({
        'n_stations': config['n_stations'],
        'p_max': config['p_max'],
        'arrival_rate': config['arrival_rate'],
        'num_of_episodes': config['num_episodes'],
        'print_interval': config['print_interval'],
        'save_models': False,
        'plot_training_curve': False,
        'enable_multi_run': False,
    })
    
    set_seed(config['seed'])
    
    # Train
    score_record, action_restore, agent, constraint_violations, episode_stats = pdqn_main(pdqn_config)
    
    # Test with comprehensive metrics
    env = EVChargingEnv(
        n_stations=config['n_stations'],
        p_max=config['p_max'],
        arrival_rate=config['arrival_rate']
    )
    
    # Define action function for PDQN
    def pdqn_action_fn(state, env):
        _, _, action = agent.select_action(state, env, training=False)
        return action
    
    # Run comprehensive test
    test_metrics = run_comprehensive_test(env, pdqn_action_fn, config, 'PDQN')
    
    return {
        'name': 'PDQN',
        'train_scores': score_record,
        'test_rewards': test_metrics['test_rewards'],
        'test_violations': test_metrics['test_violations'],
        'constraint_violations': constraint_violations,
        'episode_stats': episode_stats,
        'model': agent,
        'test_metrics': test_metrics,
    }


def run_hppo(config: dict) -> Dict:
    """Run HPPO algorithm."""
    print("\n" + "="*70)
    print("Training HPPO")
    print("="*70)
    
    from hppo_ev_charging import train_hppo, test_hppo, HPPOAgent, CONFIG as HPPO_CONFIG, set_seed as hppo_set_seed
    
    # Update config
    hppo_config = HPPO_CONFIG.copy()
    hppo_config.update({
        'n_stations': config['n_stations'],
        'p_max': config['p_max'],
        'arrival_rate': config['arrival_rate'],
        'num_of_episodes': config['num_episodes'],
        'print_interval': config['print_interval'],
        'save_models': False,
        'plot_training_curve': False,
        'enable_multi_run': False,
        'seed': config['seed'],
    })
    
    # Train
    score_record, agent, constraint_violations = train_hppo(hppo_config)
    
    # Test with comprehensive metrics
    env = EVChargingEnv(
        n_stations=config['n_stations'],
        p_max=config['p_max'],
        arrival_rate=config['arrival_rate']
    )
    
    # Define action function for HPPO
    def hppo_action_fn(state, env):
        action_env, _, _, _, _, _, _ = agent.select_action(state)
        return action_env
    
    # Run comprehensive test
    test_metrics = run_comprehensive_test(env, hppo_action_fn, config, 'HPPO')
    
    return {
        'name': 'HPPO',
        'train_scores': score_record,
        'test_rewards': test_metrics['test_rewards'],
        'test_violations': test_metrics['test_violations'],
        'constraint_violations': constraint_violations,
        'episode_stats': None,
        'model': agent,
        'test_metrics': test_metrics,
    }


def run_lppo(config: dict) -> Dict:
    """Run LPPO (Lagrangian PPO) algorithm."""
    print("\n" + "="*70)
    print("Training LPPO (Lagrangian PPO)")
    print("="*70)
    
    from lppo_ev_charging import train_lppo, LPPOAgent, CONFIG as LPPO_CONFIG
    
    # Update config
    lppo_config = LPPO_CONFIG.copy()
    lppo_config.update({
        'n_stations': config['n_stations'],
        'p_max': config['p_max'],
        'arrival_rate': config['arrival_rate'],
        'num_of_episodes': config['num_episodes'],
        'print_interval': config['print_interval'],
        'save_models': False,
        'plot_training_curve': False,
        'enable_multi_run': False,
        'seed': config['seed'],
    })
    
    # Train
    score_record, lambda_record, agent, constraint_violations = train_lppo(lppo_config)
    
    # Test with comprehensive metrics
    env = EVChargingEnv(
        n_stations=config['n_stations'],
        p_max=config['p_max'],
        arrival_rate=config['arrival_rate']
    )
    
    # Define action function for LPPO
    def lppo_action_fn(state, env):
        action_env, _, _, _, _, _, _, _ = agent.select_action(state)
        return action_env
    
    # Run comprehensive test
    test_metrics = run_comprehensive_test(env, lppo_action_fn, config, 'LPPO')
    
    return {
        'name': 'LPPO',
        'train_scores': score_record,
        'lambda_record': lambda_record,
        'test_rewards': test_metrics['test_rewards'],
        'test_violations': test_metrics['test_violations'],
        'constraint_violations': constraint_violations,
        'episode_stats': None,
        'model': agent,
        'test_metrics': test_metrics,
    }


def run_cpo(config: dict) -> Dict:
    """Run CPO algorithm."""
    print("\n" + "="*70)
    print("Training CPO")
    print("="*70)
    
    from cpo_ev_charging import train_cpo, CPOAgent, CONFIG as CPO_CONFIG
    
    # Update config
    cpo_config = CPO_CONFIG.copy()
    cpo_config.update({
        'n_stations': config['n_stations'],
        'p_max': config['p_max'],
        'arrival_rate': config['arrival_rate'],
        'num_of_episodes': config['num_episodes'],
        'print_interval': config['print_interval'],
        'save_models': False,
        'plot_training_curve': False,
        'enable_multi_run': False,
        'seed': config['seed'],
    })
    
    # Train
    score_record, cost_record, agent, constraint_violations = train_cpo(cpo_config)
    
    # Test with comprehensive metrics
    env = EVChargingEnv(
        n_stations=config['n_stations'],
        p_max=config['p_max'],
        arrival_rate=config['arrival_rate']
    )
    
    # Define action function for CPO
    def cpo_action_fn(state, env):
        action_env, _, _, _, _, _, _, _ = agent.select_action(state)
        return action_env
    
    # Run comprehensive test
    test_metrics = run_comprehensive_test(env, cpo_action_fn, config, 'CPO')
    
    return {
        'name': 'CPO',
        'train_scores': score_record,
        'cost_record': cost_record,
        'test_rewards': test_metrics['test_rewards'],
        'test_violations': test_metrics['test_violations'],
        'constraint_violations': constraint_violations,
        'episode_stats': None,
        'model': agent,
        'test_metrics': test_metrics,
    }


# =======================
# Test-Only Functions
# =======================

def test_only_lirl(config: dict, model_dir: str) -> Dict:
    """Test LIRL with pre-trained model."""
    print("\n" + "="*70)
    print("Testing LIRL (DDPG-based) - Loading pre-trained model")
    print("="*70)
    
    from lirl import MuNet, action_projection_ev, CONFIG as LIRL_CONFIG
    
    set_seed(config['seed'])
    
    env = EVChargingEnv(
        n_stations=config['n_stations'],
        p_max=config['p_max'],
        arrival_rate=config['arrival_rate']
    )
    
    # Load model
    state_dim = env.observation_space.shape[0]
    action_dim = 3
    mu = MuNet(state_dim, action_dim)
    
    model_path = os.path.join(model_dir, 'lirl_model_0.pth')
    if os.path.exists(model_path):
        mu.load_state_dict(torch.load(model_path))
        print(f"Loaded model from {model_path}")
    else:
        print(f"Warning: Model not found at {model_path}")
        return None
    
    mu.eval()
    
    # Define action function for LIRL
    def lirl_action_fn(state, env):
        with torch.no_grad():
            a = mu(torch.from_numpy(state).float())
            a = torch.clamp(a, 0, 1)
            if len(a.shape) > 1:
                a = a.squeeze(0)
        return action_projection_ev(env, a)
    
    # Run comprehensive test
    test_metrics = run_comprehensive_test(env, lirl_action_fn, config, 'LIRL')
    
    return {
        'name': 'LIRL',
        'train_scores': [],
        'test_rewards': test_metrics['test_rewards'],
        'test_violations': test_metrics['test_violations'],
        'constraint_violations': {'total_violations': 0},
        'episode_stats': None,
        'model': mu,
        'test_metrics': test_metrics,
    }


def test_only_pdqn(config: dict, model_dir: str) -> Dict:
    """Test PDQN with pre-trained model."""
    print("\n" + "="*70)
    print("Testing PDQN - Loading pre-trained model")
    print("="*70)
    
    from pdqn_ev_charging import QNetwork, ParameterNetwork, PDQNAgent, CONFIG as PDQN_CONFIG
    
    set_seed(config['seed'])
    
    env = EVChargingEnv(
        n_stations=config['n_stations'],
        p_max=config['p_max'],
        arrival_rate=config['arrival_rate']
    )
    
    # Create agent
    state_size = env.observation_space.shape[0]
    n_discrete = env.n_stations * env.max_vehicles
    param_size = 1
    
    pdqn_config = PDQN_CONFIG.copy()
    pdqn_config.update({
        'n_stations': config['n_stations'],
        'p_max': config['p_max'],
    })
    
    agent = PDQNAgent(state_size, n_discrete, param_size, pdqn_config)
    
    # Load models
    q_path = os.path.join(model_dir, 'pdqn_q_network.pth')
    param_path = os.path.join(model_dir, 'pdqn_param_network.pth')
    
    if os.path.exists(q_path) and os.path.exists(param_path):
        agent.q_network.load_state_dict(torch.load(q_path))
        agent.param_network.load_state_dict(torch.load(param_path))
        print(f"Loaded models from {model_dir}")
    else:
        print(f"Warning: Models not found in {model_dir}")
        return None
    
    # Define action function for PDQN
    def pdqn_action_fn(state, env):
        _, _, action = agent.select_action(state, env, training=False)
        return action
    
    # Run comprehensive test
    test_metrics = run_comprehensive_test(env, pdqn_action_fn, config, 'PDQN')
    
    return {
        'name': 'PDQN',
        'train_scores': [],
        'test_rewards': test_metrics['test_rewards'],
        'test_violations': test_metrics['test_violations'],
        'constraint_violations': {'total_violations': 0},
        'episode_stats': None,
        'model': agent,
        'test_metrics': test_metrics,
    }


def test_only_hppo(config: dict, model_dir: str) -> Dict:
    """Test HPPO with pre-trained model."""
    print("\n" + "="*70)
    print("Testing HPPO - Loading pre-trained model")
    print("="*70)
    
    from hppo_ev_charging import HPPOAgent, CONFIG as HPPO_CONFIG
    
    set_seed(config['seed'])
    
    env = EVChargingEnv(
        n_stations=config['n_stations'],
        p_max=config['p_max'],
        arrival_rate=config['arrival_rate']
    )
    
    # Create agent (HPPOAgent doesn't take config parameter)
    state_dim = env.observation_space.shape[0]
    agent = HPPOAgent(state_dim, env.n_stations, env.max_vehicles)
    
    # Load model
    model_path = os.path.join(model_dir, 'hppo_model.pth')
    if os.path.exists(model_path):
        agent.model.load_state_dict(torch.load(model_path))
        print(f"Loaded model from {model_path}")
    else:
        print(f"Warning: Model not found at {model_path}")
        return None
    
    agent.model.eval()
    
    # Define action function for HPPO
    def hppo_action_fn(state, env):
        action_env, _, _, _, _, _, _ = agent.select_action(state)
        return action_env
    
    # Run comprehensive test
    test_metrics = run_comprehensive_test(env, hppo_action_fn, config, 'HPPO')
    
    return {
        'name': 'HPPO',
        'train_scores': [],
        'test_rewards': test_metrics['test_rewards'],
        'test_violations': test_metrics['test_violations'],
        'constraint_violations': {'total_violations': 0},
        'episode_stats': None,
        'model': agent,
        'test_metrics': test_metrics,
    }


def test_only_lppo(config: dict, model_dir: str) -> Dict:
    """Test LPPO with pre-trained model."""
    print("\n" + "="*70)
    print("Testing LPPO - Loading pre-trained model")
    print("="*70)
    
    from lppo_ev_charging import LPPOAgent, CONFIG as LPPO_CONFIG
    
    set_seed(config['seed'])
    
    env = EVChargingEnv(
        n_stations=config['n_stations'],
        p_max=config['p_max'],
        arrival_rate=config['arrival_rate']
    )
    
    # Create agent
    state_dim = env.observation_space.shape[0]
    lppo_config = LPPO_CONFIG.copy()
    lppo_config.update({
        'n_stations': config['n_stations'],
        'p_max': config['p_max'],
    })
    
    agent = LPPOAgent(state_dim, env.n_stations, env.max_vehicles, lppo_config)
    
    # Load model
    model_path = os.path.join(model_dir, 'lppo_model.pth')
    if os.path.exists(model_path):
        agent.model.load_state_dict(torch.load(model_path))
        print(f"Loaded model from {model_path}")
    else:
        print(f"Warning: Model not found at {model_path}")
        return None
    
    agent.model.eval()
    
    # Define action function for LPPO
    def lppo_action_fn(state, env):
        action_env, _, _, _, _, _, _, _ = agent.select_action(state)
        return action_env
    
    # Run comprehensive test
    test_metrics = run_comprehensive_test(env, lppo_action_fn, config, 'LPPO')
    
    return {
        'name': 'LPPO',
        'train_scores': [],
        'test_rewards': test_metrics['test_rewards'],
        'test_violations': test_metrics['test_violations'],
        'constraint_violations': {'total_violations': 0},
        'episode_stats': None,
        'model': agent,
        'test_metrics': test_metrics,
    }


def test_only_cpo(config: dict, model_dir: str) -> Dict:
    """Test CPO with pre-trained model."""
    print("\n" + "="*70)
    print("Testing CPO - Loading pre-trained model")
    print("="*70)
    
    from cpo_ev_charging import CPOAgent, CONFIG as CPO_CONFIG
    
    set_seed(config['seed'])
    
    env = EVChargingEnv(
        n_stations=config['n_stations'],
        p_max=config['p_max'],
        arrival_rate=config['arrival_rate']
    )
    
    # Create agent
    state_dim = env.observation_space.shape[0]
    cpo_config = CPO_CONFIG.copy()
    cpo_config.update({
        'n_stations': config['n_stations'],
        'p_max': config['p_max'],
    })
    
    agent = CPOAgent(state_dim, env.n_stations, env.max_vehicles, cpo_config)
    
    # Load model
    model_path = os.path.join(model_dir, 'cpo_model.pth')
    if os.path.exists(model_path):
        agent.model.load_state_dict(torch.load(model_path))
        print(f"Loaded model from {model_path}")
    else:
        print(f"Warning: Model not found at {model_path}")
        return None
    
    agent.model.eval()
    
    # Define action function for CPO
    def cpo_action_fn(state, env):
        action_env, _, _, _, _, _, _, _ = agent.select_action(state)
        return action_env
    
    # Run comprehensive test
    test_metrics = run_comprehensive_test(env, cpo_action_fn, config, 'CPO')
    
    return {
        'name': 'CPO',
        'train_scores': [],
        'test_rewards': test_metrics['test_rewards'],
        'test_violations': test_metrics['test_violations'],
        'constraint_violations': {'total_violations': 0},
        'episode_stats': None,
        'model': agent,
        'test_metrics': test_metrics,
    }


def run_test_only(config: dict) -> List[Dict]:
    """Run test-only mode for all algorithms."""
    model_dir = config.get('model_dir')
    if not model_dir or not os.path.exists(model_dir):
        print(f"Error: model_dir '{model_dir}' does not exist!")
        return []
    
    print("="*70)
    print("TEST-ONLY MODE")
    print(f"Loading models from: {model_dir}")
    print("="*70)
    
    results = []
    
    # Test each algorithm
    test_funcs = [
        ('LIRL', test_only_lirl),
        ('PDQN', test_only_pdqn),
        ('HPPO', test_only_hppo),
        ('LPPO', test_only_lppo),
        ('CPO', test_only_cpo),
    ]
    
    for name, test_func in test_funcs:
        try:
            result = test_func(config, model_dir)
            if result is not None:
                results.append(result)
        except Exception as e:
            print(f"{name} test failed: {e}")
            import traceback
            traceback.print_exc()
    
    return results


# =======================
# Visualization
# =======================

def plot_training_curves(results: List[Dict], save_dir: str):
    """Plot training curves for all algorithms."""
    plt.figure(figsize=(14, 10))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Plot 1: Training Reward Curves
    plt.subplot(2, 2, 1)
    for i, result in enumerate(results):
        scores = result['train_scores']
        episodes = range(len(scores))
        plt.plot(episodes, scores, alpha=0.3, color=colors[i])
        
        # Moving average
        window = 20
        if len(scores) >= window:
            moving_avg = np.convolve(scores, np.ones(window)/window, mode='valid')
            plt.plot(range(window-1, len(scores)), moving_avg, 
                    color=colors[i], linewidth=2, label=result['name'])
        else:
            plt.plot(episodes, scores, color=colors[i], linewidth=2, label=result['name'])
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Reward Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Test Performance Comparison
    plt.subplot(2, 2, 2)
    names = [r['name'] for r in results]
    test_means = [np.mean(r['test_rewards']) for r in results]
    test_stds = [np.std(r['test_rewards']) for r in results]
    
    x = np.arange(len(names))
    bars = plt.bar(x, test_means, yerr=test_stds, capsize=5, color=colors)
    plt.xticks(x, names)
    plt.ylabel('Average Test Reward')
    plt.title('Test Performance Comparison')
    
    # Add value labels
    for bar, mean in zip(bars, test_means):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{mean:.1f}', ha='center', va='bottom', fontsize=9)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Constraint Violation Comparison
    plt.subplot(2, 2, 3)
    violation_rates = []
    for r in results:
        if 'constraint_violations' in r and r['constraint_violations']:
            rates = r['constraint_violations'].get('violation_rate', [])
            if rates:
                violation_rates.append(np.mean(rates) * 100)
            else:
                total_v = r['constraint_violations'].get('total_violations', 0)
                violation_rates.append(total_v / (len(r['train_scores']) * 288) * 100)
        else:
            violation_rates.append(0)
    
    bars = plt.bar(x, violation_rates, color=colors)
    plt.xticks(x, names)
    plt.ylabel('Average Violation Rate (%)')
    plt.title('Training Constraint Violation Rate')
    
    for bar, rate in zip(bars, violation_rates):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=9)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Test Violation Comparison
    plt.subplot(2, 2, 4)
    test_violation_means = [np.mean(r['test_violations']) for r in results]
    test_violation_stds = [np.std(r['test_violations']) for r in results]
    
    bars = plt.bar(x, test_violation_means, yerr=test_violation_stds, capsize=5, color=colors)
    plt.xticks(x, names)
    plt.ylabel('Average Test Violations')
    plt.title('Test Constraint Violations')
    
    for bar, mean in zip(bars, test_violation_means):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{mean:.1f}', ha='center', va='bottom', fontsize=9)
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'algorithm_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"Training curves saved to {os.path.join(save_dir, 'algorithm_comparison.png')}")
    plt.show()


def plot_detailed_training_curves(results: List[Dict], save_dir: str):
    """Plot detailed training curves with confidence intervals."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Find min length for alignment
    min_len = min(len(r['train_scores']) for r in results)
    
    # Plot individual algorithm curves
    for i, result in enumerate(results):
        ax = axes[i // 3, i % 3]
        scores = result['train_scores'][:min_len]
        episodes = range(len(scores))
        
        ax.plot(episodes, scores, alpha=0.4, color=colors[i])
        
        # Moving average
        window = 20
        if len(scores) >= window:
            moving_avg = np.convolve(scores, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(scores)), moving_avg, 
                   color=colors[i], linewidth=2)
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title(f'{result["name"]} Training Curve')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        final_20_avg = np.mean(scores[-20:]) if len(scores) >= 20 else np.mean(scores)
        ax.axhline(y=final_20_avg, color='red', linestyle='--', alpha=0.5)
        ax.text(len(scores)*0.7, final_20_avg, f'Avg(last20): {final_20_avg:.1f}', 
               fontsize=9, color='red')
    
    # Combined comparison in last subplot
    ax = axes[1, 2]
    for i, result in enumerate(results):
        scores = result['train_scores'][:min_len]
        window = 20
        if len(scores) >= window:
            moving_avg = np.convolve(scores, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(scores)), moving_avg,
                   color=colors[i], linewidth=2, label=result['name'])
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('All Algorithms Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'detailed_training_curves.png'), dpi=300, bbox_inches='tight')
    print(f"Detailed curves saved to {os.path.join(save_dir, 'detailed_training_curves.png')}")
    plt.show()


def plot_radar_chart(results: List[Dict], save_dir: str):
    """
    Plot radar chart comparing 4 key metrics across algorithms.
    
    Metrics:
    - Station Utilization (充电桩利用率)
    - Charging Success Rate (充电成功率)
    - Energy Delivered (交付能量)
    - Safety (100 - Violation Rate) (安全性/低违规率)
    """
    # Extract metrics from test_metrics
    algorithms = []
    metrics_data = []
    
    for result in results:
        if 'test_metrics' not in result:
            continue
        
        algorithms.append(result['name'])
        tm = result['test_metrics']
        
        metrics_data.append({
            'energy': tm.get('avg_energy', 0),
            'station_util': tm.get('avg_station_utilization', 0),
            'success_rate': tm.get('avg_charging_success_rate', 0),
            'violation_rate': tm.get('avg_violation_rate', 0),
        })
    
    if not algorithms:
        print("No test_metrics available for radar chart")
        return
    
    # Normalize metrics for radar chart (0-100 scale)
    # Higher is better for all normalized metrics
    max_energy = max(m['energy'] for m in metrics_data) if metrics_data else 1
    
    normalized_data = []
    for m in metrics_data:
        normalized_data.append({
            '充电桩利用率\nStation Utilization': m['station_util'],
            '充电成功率\nCharging Success': m['success_rate'],
            '交付能量\nEnergy Delivered': (m['energy'] / max_energy * 100) if max_energy > 0 else 0,
            '安全性(低违规)\nSafety': 100 - m['violation_rate'],  # Lower violation = better
        })
    
    # Radar chart
    categories = list(normalized_data[0].keys())
    num_vars = len(categories)
    
    # Create angles for radar chart
    angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
    angles += angles[:1]  # Complete the loop
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, (alg, data) in enumerate(zip(algorithms, normalized_data)):
        values = [data[cat] for cat in categories]
        values += values[:1]  # Complete the loop
        
        ax.plot(angles, values, 'o-', linewidth=2, label=alg, color=colors[i % len(colors)])
        ax.fill(angles, values, alpha=0.15, color=colors[i % len(colors)])
    
    # Set category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=10)
    
    # Set y-axis limits
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], size=8)
    
    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    plt.title('Algorithm Performance Comparison (Radar Chart)\n综合性能雷达图对比', size=14, y=1.08)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'radar_chart.png'), dpi=300, bbox_inches='tight')
    print(f"Radar chart saved to {os.path.join(save_dir, 'radar_chart.png')}")
    plt.show()
    
    # Also print the raw metrics table
    print("\n" + "="*100)
    print("KEY PERFORMANCE METRICS (关键性能指标)")
    print("="*100)
    print(f"\n{'Metric':<35}", end="")
    for alg in algorithms:
        print(f"{alg:>14}", end="")
    print()
    print("-"*100)
    
    metric_names = [
        ('Station Utilization (%) 充电桩利用率', 'station_util', '.2f'),
        ('Charging Success Rate (%) 充电成功率', 'success_rate', '.2f'),
        ('Energy Delivered (kWh) 交付能量', 'energy', '.2f'),
        ('Violation Rate (%) 违规率', 'violation_rate', '.2f'),
    ]
    
    for display_name, key, fmt in metric_names:
        print(f"{display_name:<35}", end="")
        for m in metrics_data:
            value = m[key]
            print(f"{value:>14{fmt}}", end="")
        print()
    
    print("="*100)
    
    # Save key performance metrics to CSV
    metrics_csv_path = os.path.join(save_dir, 'key_performance_metrics.csv')
    with open(metrics_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Header
        writer.writerow(['Metric'] + algorithms)
        # Data rows
        for display_name, key, fmt in metric_names:
            row = [display_name]
            for m in metrics_data:
                row.append(f"{m[key]:{fmt}}")
            writer.writerow(row)
    
    print(f"Key performance metrics saved to {metrics_csv_path}")
    
    # Also save as JSON for programmatic access
    metrics_json_path = os.path.join(save_dir, 'key_performance_metrics.json')
    metrics_json = {
        'algorithms': algorithms,
        'metrics': {}
    }
    for i, alg in enumerate(algorithms):
        metrics_json['metrics'][alg] = {
            'station_utilization': float(metrics_data[i]['station_util']),
            'charging_success_rate': float(metrics_data[i]['success_rate']),
            'energy_delivered': float(metrics_data[i]['energy']),
            'violation_rate': float(metrics_data[i]['violation_rate']),
        }
    
    with open(metrics_json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_json, f, indent=2, ensure_ascii=False)
    
    print(f"Key performance metrics (JSON) saved to {metrics_json_path}")


# =======================
# Results Analysis
# =======================

def analyze_results(results: List[Dict], config: dict) -> Dict:
    """Analyze and compare results from all algorithms."""
    analysis = {}
    
    for result in results:
        name = result['name']
        train_scores = result['train_scores']
        test_rewards = result['test_rewards']
        test_violations = result['test_violations']
        
        analysis[name] = {
            # Training metrics
            'train_final_score': train_scores[-1] if train_scores else 0,
            'train_avg_last20': np.mean(train_scores[-20:]) if len(train_scores) >= 20 else np.mean(train_scores),
            'train_max_score': np.max(train_scores) if train_scores else 0,
            'train_min_score': np.min(train_scores) if train_scores else 0,
            'train_std': np.std(train_scores) if train_scores else 0,
            
            # Test metrics
            'test_avg_reward': np.mean(test_rewards),
            'test_std_reward': np.std(test_rewards),
            'test_max_reward': np.max(test_rewards),
            'test_min_reward': np.min(test_rewards),
            
            # Constraint metrics
            'test_avg_violations': np.mean(test_violations),
            'test_std_violations': np.std(test_violations),
            'test_total_violations': sum(test_violations),
        }
        
        # Training violation rate
        if 'constraint_violations' in result and result['constraint_violations']:
            cv = result['constraint_violations']
            analysis[name]['train_total_violations'] = cv.get('total_violations', 0)
            rates = cv.get('violation_rate', [])
            if rates:
                analysis[name]['train_avg_violation_rate'] = np.mean(rates) * 100
            else:
                analysis[name]['train_avg_violation_rate'] = 0
        else:
            analysis[name]['train_total_violations'] = 0
            analysis[name]['train_avg_violation_rate'] = 0
    
    return analysis


def print_comparison_table(analysis: Dict):
    """Print comparison table."""
    print("\n" + "="*100)
    print("ALGORITHM COMPARISON RESULTS")
    print("="*100)
    
    # Header
    algorithms = list(analysis.keys())
    print(f"\n{'Metric':<30}", end="")
    for alg in algorithms:
        print(f"{alg:>14}", end="")
    print()
    print("-"*100)
    
    # Training metrics
    metrics = [
        ('Train Final Score', 'train_final_score', '.2f'),
        ('Train Avg (Last 20)', 'train_avg_last20', '.2f'),
        ('Train Max Score', 'train_max_score', '.2f'),
        ('Train Std', 'train_std', '.2f'),
        ('Train Violations', 'train_total_violations', 'd'),
        ('Train Violation Rate (%)', 'train_avg_violation_rate', '.2f'),
        ('Test Avg Reward', 'test_avg_reward', '.2f'),
        ('Test Std Reward', 'test_std_reward', '.2f'),
        ('Test Avg Violations', 'test_avg_violations', '.2f'),
    ]
    
    for metric_name, metric_key, fmt in metrics:
        print(f"{metric_name:<30}", end="")
        for alg in algorithms:
            value = analysis[alg].get(metric_key, 0)
            if fmt == 'd':
                print(f"{int(value):>14}", end="")
            else:
                print(f"{value:>14{fmt}}", end="")
        print()
    
    print("="*100)
    
    # Find best algorithm
    print("\nBest Algorithm by Metric:")
    print("-"*50)
    
    # Best test reward
    best_test = max(algorithms, key=lambda x: analysis[x]['test_avg_reward'])
    print(f"  Highest Test Reward: {best_test} ({analysis[best_test]['test_avg_reward']:.2f})")
    
    # Lowest violations
    best_violations = min(algorithms, key=lambda x: analysis[x]['test_avg_violations'])
    print(f"  Lowest Test Violations: {best_violations} ({analysis[best_violations]['test_avg_violations']:.2f})")
    
    # Best training convergence
    best_train = max(algorithms, key=lambda x: analysis[x]['train_avg_last20'])
    print(f"  Best Training Convergence: {best_train} ({analysis[best_train]['train_avg_last20']:.2f})")
    
    print("="*100)


def save_results(results: List[Dict], analysis: Dict, config: dict, save_dir: str):
    """Save all results to files."""
    
    # Save training scores
    for result in results:
        name = result['name']
        scores_path = os.path.join(save_dir, f'{name.lower()}_train_scores.npy')
        np.save(scores_path, np.array(result['train_scores']))
        
        test_path = os.path.join(save_dir, f'{name.lower()}_test_rewards.npy')
        np.save(test_path, np.array(result['test_rewards']))
        
        # Save model
        model = result.get('model')
        if model is not None:
            if isinstance(model, list):
                # LIRL returns list of models
                for i, m in enumerate(model):
                    if hasattr(m, 'state_dict'):
                        torch.save(m.state_dict(), os.path.join(save_dir, f'{name.lower()}_model_{i}.pth'))
            elif hasattr(model, 'model') and hasattr(model.model, 'state_dict'):
                # Agent with model attribute
                torch.save(model.model.state_dict(), os.path.join(save_dir, f'{name.lower()}_model.pth'))
            elif hasattr(model, 'q_network'):
                # PDQN agent
                torch.save(model.q_network.state_dict(), os.path.join(save_dir, f'{name.lower()}_q_network.pth'))
                torch.save(model.param_network.state_dict(), os.path.join(save_dir, f'{name.lower()}_param_network.pth'))
    
    # Save analysis summary
    summary_path = os.path.join(save_dir, 'comparison_summary.csv')
    with open(summary_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        algorithms = list(analysis.keys())
        metrics = list(analysis[algorithms[0]].keys())
        
        # Header
        writer.writerow(['Metric'] + algorithms)
        
        # Data
        for metric in metrics:
            row = [metric]
            for alg in algorithms:
                value = analysis[alg].get(metric, 0)
                if isinstance(value, float):
                    row.append(f'{value:.4f}')
                else:
                    row.append(str(value))
            writer.writerow(row)
    
    # Save detailed results as JSON
    json_results = {}
    for result in results:
        name = result['name']
        json_results[name] = {
            'train_scores': [float(s) for s in result['train_scores']],
            'test_rewards': [float(r) for r in result['test_rewards']],
            'test_violations': [int(v) for v in result['test_violations']],
            'analysis': {k: float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v 
                        for k, v in analysis[name].items()},
        }
    
    json_path = os.path.join(save_dir, 'comparison_results.json')
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    # Save config
    config_path = os.path.join(save_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nAll results saved to: {save_dir}")
    print(f"  - Training scores: {len(results)} files")
    print(f"  - Models: saved")
    print(f"  - Summary CSV: comparison_summary.csv")
    print(f"  - Detailed JSON: comparison_results.json")


# =======================
# Main
# =======================

def main(config=None):
    """Main comparison function."""
    if config is None:
        config = COMPARE_CONFIG.copy()
    
    # Check if test-only mode
    test_only = config.get('test_only', False)
    model_dir = config.get('model_dir')
    
    # Create save directory
    now_str = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    if test_only:
        save_dir = config.get('save_dir') or f"test_results_{now_str}"
    else:
        save_dir = config.get('save_dir') or f"algorithm_comparison_{now_str}"
    os.makedirs(save_dir, exist_ok=True)
    config['save_dir'] = save_dir
    
    print("="*70)
    if test_only:
        print("ALGORITHM TESTING FOR EV CHARGING STATION (TEST-ONLY MODE)")
    else:
        print("ALGORITHM COMPARISON FOR EV CHARGING STATION")
    print("="*70)
    print(f"\nConfiguration:")
    if not test_only:
        print(f"  Training Episodes: {config['num_episodes']}")
    print(f"  Test Episodes: {config['num_test_episodes']}")
    print(f"  Stations: {config['n_stations']}")
    print(f"  Max Power: {config['p_max']} kW")
    print(f"  Seed: {config['seed']}")
    if test_only:
        print(f"  Model Directory: {model_dir}")
    print(f"  Save Directory: {save_dir}")
    print("="*70)
    
    results = []
    
    if test_only:
        # Test-only mode: load pre-trained models and test
        if not model_dir:
            print("Error: --model-dir is required for test-only mode!")
            return
        results = run_test_only(config)
    else:
        # Full mode: train and test all algorithms
        try:
            results.append(run_lirl(config))
        except Exception as e:
            print(f"LIRL failed: {e}")
        
        try:
            results.append(run_pdqn(config))
        except Exception as e:
            print(f"PDQN failed: {e}")
        
        try:
            results.append(run_hppo(config))
        except Exception as e:
            print(f"HPPO failed: {e}")
        
        try:
            results.append(run_lppo(config))
        except Exception as e:
            print(f"LPPO failed: {e}")
        
        try:
            results.append(run_cpo(config))
        except Exception as e:
            print(f"CPO failed: {e}")
    
    if not results:
        print("No algorithms completed successfully!")
        return
    
    # Analyze results
    analysis = analyze_results(results, config)
    
    # Print comparison table
    print_comparison_table(analysis)
    
    # Plot results (skip training curves in test-only mode)
    if not test_only:
        plot_training_curves(results, save_dir)
        plot_detailed_training_curves(results, save_dir)
    else:
        # Plot test-only results
        plot_test_only_results(results, save_dir)
    
    # Plot radar chart for comprehensive metrics comparison
    plot_radar_chart(results, save_dir)
    
    # Save all results
    save_results(results, analysis, config, save_dir)
    
    return results, analysis


def plot_test_only_results(results: List[Dict], save_dir: str):
    """Plot test-only results."""
    plt.figure(figsize=(14, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    names = [r['name'] for r in results]
    x = np.arange(len(names))
    
    # Plot 1: Test Rewards
    plt.subplot(1, 2, 1)
    test_means = [np.mean(r['test_rewards']) for r in results]
    test_stds = [np.std(r['test_rewards']) for r in results]
    
    bars = plt.bar(x, test_means, yerr=test_stds, capsize=5, color=colors[:len(names)])
    plt.xticks(x, names)
    plt.ylabel('Average Test Reward')
    plt.title('Test Performance Comparison')
    
    for bar, mean in zip(bars, test_means):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{mean:.1f}', ha='center', va='bottom', fontsize=9)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Test Violations
    plt.subplot(1, 2, 2)
    test_violation_means = [np.mean(r['test_violations']) for r in results]
    test_violation_stds = [np.std(r['test_violations']) for r in results]
    
    bars = plt.bar(x, test_violation_means, yerr=test_violation_stds, capsize=5, color=colors[:len(names)])
    plt.xticks(x, names)
    plt.ylabel('Average Test Violations')
    plt.title('Test Constraint Violations')
    
    for bar, mean in zip(bars, test_violation_means):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{mean:.1f}', ha='center', va='bottom', fontsize=9)
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'test_results.png'), dpi=300, bbox_inches='tight')
    print(f"Test results saved to {os.path.join(save_dir, 'test_results.png')}")
    plt.show()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare algorithms for EV Charging Station")
    parser.add_argument("--episodes", type=int, default=COMPARE_CONFIG["num_episodes"],
                       help="Number of training episodes")
    parser.add_argument("--test-episodes", type=int, default=COMPARE_CONFIG["num_test_episodes"],
                       help="Number of test episodes")
    parser.add_argument("--stations", type=int, default=COMPARE_CONFIG["n_stations"])
    parser.add_argument("--power", type=float, default=COMPARE_CONFIG["p_max"])
    parser.add_argument("--arrival-rate", type=float, default=COMPARE_CONFIG["arrival_rate"])
    parser.add_argument("--seed", type=int, default=COMPARE_CONFIG["seed"])
    parser.add_argument("--test-only", action="store_true",
                       help="Run test-only mode (requires --model-dir)")
    parser.add_argument("--model-dir", type=str, default=None,
                       help="Directory containing pre-trained models (for test-only mode)")
    args = parser.parse_args()
    
    config = COMPARE_CONFIG.copy()
    config.update({
        "num_episodes": args.episodes,
        "num_test_episodes": args.test_episodes,
        "n_stations": args.stations,
        "p_max": args.power,
        "arrival_rate": args.arrival_rate,
        "seed": args.seed,
        "test_only": args.test_only,
        "model_dir": args.model_dir,
    })
    
    main(config)

