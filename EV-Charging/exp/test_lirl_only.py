"""
Quick test script for LIRL only - to verify action_projection_ev fix
"""

import os
import sys
import random
import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "../alg"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../env"))

from ev import EVChargingEnv
from lirl import main as lirl_main, action_projection_ev, CONFIG as LIRL_CONFIG

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def test_lirl_with_projection(mu, env, num_episodes=10, max_steps=288):
    """Test LIRL using action_projection_ev (correct method)"""
    mu.eval()
    test_rewards = []
    test_violations = []
    
    for ep in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        episode_violations = 0
        step = 0
        
        while not done and step < max_steps:
            with torch.no_grad():
                a = mu(torch.from_numpy(state).float())
                a = torch.clamp(a, 0, 1)
                if len(a.shape) > 1:
                    a = a.squeeze(0)
            
            # Use action_projection_ev (CORRECT - uses Hungarian algorithm)
            action = action_projection_ev(env, a)
            
            next_state, reward, done, info = env.step(action)
            
            violation_info = info.get('constraint_violation', None)
            if violation_info and violation_info.get('has_violation', False):
                episode_violations += 1
            
            episode_reward += reward
            state = next_state
            step += 1
        
        test_rewards.append(episode_reward)
        test_violations.append(episode_violations)
        print(f"  Episode {ep+1}: Reward = {episode_reward:.2f}, Violations = {episode_violations}")
    
    return test_rewards, test_violations

def test_lirl_simple(mu, env, num_episodes=10, max_steps=288):
    """Test LIRL using simple mapping (WRONG - previous buggy method)"""
    mu.eval()
    test_rewards = []
    test_violations = []
    
    for ep in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        episode_violations = 0
        step = 0
        
        while not done and step < max_steps:
            with torch.no_grad():
                a = mu(torch.from_numpy(state).float())
                a = torch.clamp(a, 0, 1)
                if len(a.shape) > 1:
                    a = a.squeeze(0)
            
            # Simple action conversion (WRONG - no action projection)
            action = {
                'station_id': int(a[0].item() * (env.n_stations - 1)),
                'vehicle_id': 0,
                'power': np.array([50.0 + a[2].item() * 100.0], dtype=np.float32)
            }
            
            # Find valid vehicle
            for i in range(env.max_vehicles):
                if env.vehicles[i] is not None and not env.vehicles[i]['charging']:
                    action['vehicle_id'] = i
                    break
            
            next_state, reward, done, info = env.step(action)
            
            violation_info = info.get('constraint_violation', None)
            if violation_info and violation_info.get('has_violation', False):
                episode_violations += 1
            
            episode_reward += reward
            state = next_state
            step += 1
        
        test_rewards.append(episode_reward)
        test_violations.append(episode_violations)
        print(f"  Episode {ep+1}: Reward = {episode_reward:.2f}, Violations = {episode_violations}")
    
    return test_rewards, test_violations

def main():
    # Config
    config = LIRL_CONFIG.copy()
    config.update({
        'n_stations': 5,
        'p_max': 150.0,
        'arrival_rate': 0.75,
        'num_of_episodes': 100,  # Quick training
        'print_interval': 20,
        'save_models': False,
        'plot_training_curve': False,
        'enable_multi_run': False,
    })
    
    set_seed(3047)
    
    print("="*70)
    print("Training LIRL (100 episodes for quick test)")
    print("="*70)
    
    # Train
    score_record, action_restore, models, constraint_violations, episode_stats = lirl_main(config)
    
    mu = models[0]  # Get trained policy network
    
    # Create test environment
    env = EVChargingEnv(
        n_stations=config['n_stations'],
        p_max=config['p_max'],
        arrival_rate=config['arrival_rate']
    )
    
    print("\n" + "="*70)
    print("TEST 1: Using action_projection_ev (CORRECT method)")
    print("="*70)
    rewards_correct, violations_correct = test_lirl_with_projection(mu, env, num_episodes=10)
    
    print("\n" + "="*70)
    print("TEST 2: Using simple mapping (BUGGY method)")
    print("="*70)
    rewards_buggy, violations_buggy = test_lirl_simple(mu, env, num_episodes=10)
    
    # Summary
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"\nTraining Performance:")
    print(f"  Final Score: {score_record[-1]:.2f}")
    print(f"  Avg (Last 20): {np.mean(score_record[-20:]):.2f}")
    
    print(f"\nTest with action_projection_ev (CORRECT):")
    print(f"  Avg Reward: {np.mean(rewards_correct):.2f} ± {np.std(rewards_correct):.2f}")
    print(f"  Avg Violations: {np.mean(violations_correct):.2f}")
    
    print(f"\nTest with simple mapping (BUGGY):")
    print(f"  Avg Reward: {np.mean(rewards_buggy):.2f} ± {np.std(rewards_buggy):.2f}")
    print(f"  Avg Violations: {np.mean(violations_buggy):.2f}")
    
    print(f"\nPerformance Gap:")
    print(f"  Reward Difference: {np.mean(rewards_correct) - np.mean(rewards_buggy):.2f}")
    print(f"  Violation Difference: {np.mean(violations_buggy) - np.mean(violations_correct):.2f}")
    print("="*70)

if __name__ == "__main__":
    main()

