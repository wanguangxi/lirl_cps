"""
Simplified Computation Time Comparison for RL Algorithms
Uses pre-trained models from baseline folder
"""

import os
import sys
import time
import io
import random
import datetime
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment, minimize

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '../env'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../algs'))

import env as ENV

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# =======================
# CONFIGURATION
# =======================
CONFIG = {
    'num_of_jobs': 100,
    'num_of_robots': 5,
    'max_operations': 5,
    'alpha': 0.5,
    'beta': 0.5,
    'hidden_dim1': 128,
    'hidden_dim2': 64,
    'hidden_dim': 256,
    'latent_dim': 32,
    'num_timing_episodes': 100,
    'warmup_iterations': 50,
    'max_hungarian_size': 50,
    'seed': 42,
}

BASELINE_DIR = '/home/one/LIRL-CPS-main/RMS/baseline'


# =======================
# NETWORK DEFINITIONS
# =======================

class LIRLMuNet(nn.Module):
    """LIRL Policy Network"""
    def __init__(self, state_size, action_size, hidden_dim1=128, hidden_dim2=64):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc_mu = nn.Linear(hidden_dim2, action_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.fc_mu(x))


class SACLagActor(nn.Module):
    """SAC-Lag Actor"""
    def __init__(self, state_dim, num_jobs, num_machines, max_operations, hidden_dim=256):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()
        )
        self.job_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, num_jobs))
        self.machine_head = nn.Sequential(nn.Linear(hidden_dim + num_jobs, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, num_machines))
        self.param_mean_head = nn.Sequential(nn.Linear(hidden_dim + num_jobs + num_machines, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, max_operations))

    def forward(self, state):
        features = self.feature_extractor(state)
        job_logits = self.job_head(features)
        job_probs = F.softmax(job_logits, dim=-1)
        machine_input = torch.cat([features, job_probs], dim=-1)
        machine_logits = self.machine_head(machine_input)
        machine_probs = F.softmax(machine_logits, dim=-1)
        param_input = torch.cat([features, job_probs, machine_probs], dim=-1)
        param_mean = self.param_mean_head(param_input)
        return {'job_logits': job_logits, 'machine_logits': machine_logits, 'param_mean': param_mean}


class CPOPolicyNetwork(nn.Module):
    """CPO Policy"""
    def __init__(self, state_dim, num_jobs, num_machines, max_operations, hidden_dim=256):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(0.1)
        )
        self.job_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, num_jobs))
        self.machine_head = nn.Sequential(nn.Linear(hidden_dim + num_jobs, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, num_machines))
        self.param_mean_head = nn.Sequential(nn.Linear(hidden_dim + num_jobs + num_machines, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, max_operations), nn.Tanh())
        self.job_validity_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, num_jobs), nn.Sigmoid())
        self.machine_validity_head = nn.Sequential(nn.Linear(hidden_dim + num_jobs, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, num_machines), nn.Sigmoid())

    def forward(self, state):
        features = self.feature_extractor(state)
        job_logits = self.job_head(features)
        job_validity = self.job_validity_head(features)
        job_logits = job_logits + torch.log(job_validity + 1e-8)
        job_probs = F.softmax(job_logits, dim=-1)
        machine_input = torch.cat([features, job_probs], dim=-1)
        machine_logits = self.machine_head(machine_input)
        machine_validity = self.machine_validity_head(machine_input)
        machine_logits = machine_logits + torch.log(machine_validity + 1e-8)
        machine_probs = F.softmax(machine_logits, dim=-1)
        param_input = torch.cat([features, job_probs, machine_probs], dim=-1)
        param_mean = self.param_mean_head(param_input)
        return {'job_logits': job_logits, 'machine_logits': machine_logits, 'param_mean': param_mean}


class HighLevelPolicy(nn.Module):
    """H-PPO High-Level Policy"""
    def __init__(self, state_dim, num_jobs, num_machines, hidden_dim=256):
        super().__init__()
        self.feature_extractor = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.job_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, num_jobs))
        self.machine_head = nn.Sequential(nn.Linear(hidden_dim + num_jobs, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, num_machines))

    def forward(self, state):
        features = self.feature_extractor(state)
        job_logits = self.job_head(features)
        job_probs = F.softmax(job_logits, dim=-1)
        machine_input = torch.cat([features, job_probs], dim=-1)
        machine_logits = self.machine_head(machine_input)
        return job_logits, machine_logits


class LowLevelPolicy(nn.Module):
    """H-PPO Low-Level Policy"""
    def __init__(self, state_dim, num_jobs, num_machines, max_operations, hidden_dim=256):
        super().__init__()
        input_dim = state_dim + num_jobs + num_machines
        self.network = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.mean_head = nn.Linear(hidden_dim, max_operations)

    def forward(self, state, job_idx, machine_idx, num_jobs, num_machines):
        job_one_hot = F.one_hot(job_idx, num_classes=num_jobs).float()
        machine_one_hot = F.one_hot(machine_idx, num_classes=num_machines).float()
        x = torch.cat([state, job_one_hot, machine_one_hot], dim=-1)
        features = self.network(x)
        return torch.sigmoid(self.mean_head(features))


class ImplicitPolicyNet(nn.Module):
    """HyAR Implicit Policy"""
    def __init__(self, state_size, action_size, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.fc_out(x))


class ConditionalVAE(nn.Module):
    """HyAR VAE - Compatible with trained model"""
    def __init__(self, state_size, implicit_action_size, num_jobs, num_robots, vae_hidden_dim=128, latent_dim=32):
        super().__init__()
        # Encoder (for compatibility)
        encoder_input_dim = num_jobs + num_robots + 1 + state_size  # explicit_action_size = num_jobs + num_robots + 1
        self.encoder_fc1 = nn.Linear(encoder_input_dim, vae_hidden_dim)
        self.encoder_fc2 = nn.Linear(vae_hidden_dim, vae_hidden_dim)
        self.fc_mu = nn.Linear(vae_hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(vae_hidden_dim, latent_dim)
        
        # Decoder
        decoder_input_dim = implicit_action_size + state_size
        self.decoder_fc1 = nn.Linear(decoder_input_dim, vae_hidden_dim)
        self.decoder_fc2 = nn.Linear(vae_hidden_dim, vae_hidden_dim)
        self.decoder_out_job = nn.Linear(vae_hidden_dim, num_jobs)
        self.decoder_out_robot = nn.Linear(vae_hidden_dim, num_robots)
        self.decoder_out_param = nn.Linear(vae_hidden_dim, 1)

    def decode(self, state, implicit_action):
        x = torch.cat([state, implicit_action], dim=1)
        h = F.relu(self.decoder_fc1(x))
        h = F.relu(self.decoder_fc2(h))
        job_logits = self.decoder_out_job(h)
        robot_logits = self.decoder_out_robot(h)
        param = torch.sigmoid(self.decoder_out_param(h))
        return job_logits, robot_logits, param


class PDQNParameterNetwork(nn.Module):
    """PDQN Parameter Network"""
    def __init__(self, state_dim, num_jobs, num_machines, param_dim, hidden_dim=256):
        super().__init__()
        input_dim = state_dim + num_jobs + num_machines
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.network = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, param_dim))

    def forward(self, state, job_idx, machine_idx):
        batch_size = state.shape[0]
        job_one_hot = F.one_hot(job_idx.long(), num_classes=self.num_jobs).float()
        machine_one_hot = F.one_hot(machine_idx.long(), num_classes=self.num_machines).float()
        x = torch.cat([state, job_one_hot, machine_one_hot], dim=-1)
        return torch.sigmoid(self.network(x))


# =======================
# HELPER FUNCTIONS
# =======================

def get_valid_jobs_and_robots(env):
    """Get valid jobs and robots from environment"""
    valid_robots = np.where(env.robot_state == 1)[0].tolist()
    valid_jobs = []
    task_state = env.task_state
    for job_id in range(env.num_of_jobs):
        start_idx = job_id * 5
        end_idx = start_idx + 5
        if not np.all(task_state[start_idx:end_idx] == 1):
            valid_jobs.append(job_id)
    return valid_jobs, valid_robots


def solve_hungarian(cost_matrix):
    """Hungarian algorithm"""
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return row_ind, col_ind


def solve_qp(v):
    """Simple QP solver"""
    v = np.asarray(v, dtype=np.float64)
    n_params = len(v)
    A = np.vstack([np.eye(n_params), -np.eye(n_params)])
    b = np.hstack([np.ones(n_params), np.zeros(n_params)])
    
    def objective(x):
        return 0.5 * np.sum((x - v)**2)
    
    def constraint_fun(x):
        return b - A @ x
    
    constraints = {'type': 'ineq', 'fun': constraint_fun}
    x0 = np.clip(v, 0.0, 1.0)
    
    try:
        from scipy.optimize import minimize
        result = minimize(objective, x0, constraints=constraints, method='SLSQP')
        if result.success:
            return float(result.x[0]) if len(result.x) > 0 else float(v[0])
    except:
        pass
    return float(np.clip(v[0] if len(v) > 0 else 0.0, 0.0, 1.0))


def lirl_action_projection(env, a_np, max_hungarian_size=50):
    """LIRL action projection with Hungarian + QP"""
    timing = {'hungarian': 0, 'qp': 0}
    
    valid_jobs, valid_robots = get_valid_jobs_and_robots(env)
    if len(valid_jobs) == 0 or len(valid_robots) == 0:
        return [0, 0, a_np[2] if len(a_np) > 2 else 0.0], timing
    
    job_preference = a_np[0]
    robot_preference = a_np[1]
    
    # Limit Hungarian size
    k_jobs = min(max_hungarian_size, len(valid_jobs))
    k_robots = min(max_hungarian_size, len(valid_robots))
    
    job_scores = [abs(job_preference - (j / len(env.task_set))) for j in valid_jobs]
    robot_scores = [abs(robot_preference - (r / len(env.robot_state))) for r in valid_robots]
    
    top_job_indices = np.argsort(job_scores)[:k_jobs]
    top_robot_indices = np.argsort(robot_scores)[:k_robots]
    
    selected_jobs = [valid_jobs[i] for i in top_job_indices]
    selected_robots = [valid_robots[i] for i in top_robot_indices]
    
    # Build cost matrix
    cost_matrix = np.zeros((len(selected_jobs), len(selected_robots)))
    for i, job_id in enumerate(selected_jobs):
        for j, robot_id in enumerate(selected_robots):
            cost_matrix[i, j] = abs(job_preference - (job_id / len(env.task_set))) + abs(robot_preference - (robot_id / len(env.robot_state)))
    
    # Hungarian algorithm
    start = time.perf_counter()
    row_ind, col_ind = solve_hungarian(cost_matrix)
    timing['hungarian'] = time.perf_counter() - start
    
    job_id = selected_jobs[row_ind[0]] if len(row_ind) > 0 else selected_jobs[0]
    robot_id = selected_robots[col_ind[0]] if len(col_ind) > 0 else selected_robots[0]
    
    # QP
    start = time.perf_counter()
    if len(a_np) > 2:
        param = solve_qp(a_np[2:])
    else:
        param = 0.0
    timing['qp'] = time.perf_counter() - start
    
    return [job_id, robot_id, param], timing


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def env_step_silent(env, action):
    """Execute env.step without printing warnings"""
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        return env.step(action)
    finally:
        sys.stdout = old_stdout


# =======================
# MAIN FUNCTION
# =======================

def main():
    print("=" * 80)
    print("COMPUTATION TIME COMPARISON (Pre-trained Models)")
    print("=" * 80)
    
    random.seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    torch.manual_seed(CONFIG['seed'])
    
    device = torch.device('cpu')
    print(f"\nDevice: CPU (for consistent timing)")
    
    # Create environment
    env = ENV.Env(CONFIG['num_of_jobs'], CONFIG['num_of_robots'], CONFIG['alpha'], CONFIG['beta'])
    state_size = len(env.state)
    action_size = len(env.action)
    print(f"State size: {state_size}, Action size: {action_size}")
    
    # Load models
    print("\n" + "=" * 60)
    print("Loading Pre-trained Models...")
    print("=" * 60)
    
    models = {}
    
    # LIRL
    print("\n  Loading LIRL...")
    lirl_model = LIRLMuNet(state_size, action_size, CONFIG['hidden_dim1'], CONFIG['hidden_dim2'])
    lirl_path = os.path.join(BASELINE_DIR, 'ddpg_lirl_pi_multi_run_20250827_200238/run_1_seed_3047/ddpg_lirl_pi_mu_20250827_200238.pth')
    if os.path.exists(lirl_path):
        lirl_model.load_state_dict(torch.load(lirl_path, map_location=device))
        print(f"    Loaded from: {lirl_path}")
    models['LIRL'] = {'model': lirl_model.to(device), 'type': 'lirl'}
    
    # SAC-Lag
    print("\n  Loading SAC-Lag...")
    saclag_model = SACLagActor(state_size, CONFIG['num_of_jobs'], CONFIG['num_of_robots'], CONFIG['max_operations'], CONFIG['hidden_dim'])
    saclag_path = os.path.join(BASELINE_DIR, 'sac_lag_multi_run_20250829_214641/run_1_seed_3047/sac_lag_model_20250829_214641.pt')
    if os.path.exists(saclag_path):
        checkpoint = torch.load(saclag_path, map_location=device)
        if 'actor_state_dict' in checkpoint:
            try:
                saclag_model.load_state_dict(checkpoint['actor_state_dict'], strict=False)
                print(f"    Loaded from: {saclag_path}")
            except Exception as e:
                print(f"    Warning: {e}")
    models['SAC-Lag'] = {'model': saclag_model.to(device), 'type': 'discrete'}
    
    # CPO
    print("\n  Loading CPO...")
    cpo_model = CPOPolicyNetwork(state_size, CONFIG['num_of_jobs'], CONFIG['num_of_robots'], CONFIG['max_operations'], CONFIG['hidden_dim'])
    cpo_path = os.path.join(BASELINE_DIR, 'cpo_multi_run_20250830_211611/run_1_seed_3047/cpo_model_20250830_211611.pt')
    if os.path.exists(cpo_path):
        checkpoint = torch.load(cpo_path, map_location=device)
        if 'policy_state_dict' in checkpoint:
            try:
                cpo_model.load_state_dict(checkpoint['policy_state_dict'], strict=False)
                print(f"    Loaded from: {cpo_path}")
            except Exception as e:
                print(f"    Warning: {e}")
    models['CPO'] = {'model': cpo_model.to(device), 'type': 'discrete'}
    
    # H-PPO
    print("\n  Loading H-PPO...")
    hppo_high = HighLevelPolicy(state_size, CONFIG['num_of_jobs'], CONFIG['num_of_robots'], CONFIG['hidden_dim'])
    hppo_low = LowLevelPolicy(state_size, CONFIG['num_of_jobs'], CONFIG['num_of_robots'], CONFIG['max_operations'], CONFIG['hidden_dim'])
    hppo_path = os.path.join(BASELINE_DIR, 'hppo_multi_run_20250828_104924/run_1_seed_3047/hppo_model_20250828_104924.pt')
    if os.path.exists(hppo_path):
        checkpoint = torch.load(hppo_path, map_location=device)
        if 'high_policy_state_dict' in checkpoint:
            try:
                hppo_high.load_state_dict(checkpoint['high_policy_state_dict'], strict=False)
                hppo_low.load_state_dict(checkpoint['low_policy_state_dict'], strict=False)
                print(f"    Loaded from: {hppo_path}")
            except Exception as e:
                print(f"    Warning: {e}")
    models['H-PPO'] = {'model': hppo_high.to(device), 'low': hppo_low.to(device), 'type': 'hppo'}
    
    # HyAR - Note: The trained policy outputs action_size (3), not latent_dim
    print("\n  Loading HyAR...")
    hyar_policy = ImplicitPolicyNet(state_size, action_size, CONFIG['hidden_dim1'])  # action_size=3
    hyar_vae = ConditionalVAE(state_size, action_size, CONFIG['num_of_jobs'], CONFIG['num_of_robots'], CONFIG['hidden_dim1'], CONFIG['latent_dim'])
    hyar_policy_path = os.path.join(BASELINE_DIR, 'hyar_vae_multi_run_20250831_172655/run_1_seed_3047/hyar_vae_policy_20250831_172655.pth')
    hyar_vae_path = os.path.join(BASELINE_DIR, 'hyar_vae_multi_run_20250831_172655/run_1_seed_3047/hyar_vae_vae_20250831_172655.pth')
    if os.path.exists(hyar_policy_path):
        try:
            hyar_policy.load_state_dict(torch.load(hyar_policy_path, map_location=device))
            print(f"    Loaded policy from: {hyar_policy_path}")
        except Exception as e:
            print(f"    Warning: {e}")
    if os.path.exists(hyar_vae_path):
        try:
            hyar_vae.load_state_dict(torch.load(hyar_vae_path, map_location=device), strict=False)
            print(f"    Loaded VAE from: {hyar_vae_path}")
        except Exception as e:
            print(f"    Warning: {e}")
    models['HyAR'] = {'model': hyar_policy.to(device), 'vae': hyar_vae.to(device), 'type': 'hyar'}
    
    # PDQN
    print("\n  Loading PDQN...")
    pdqn_param = PDQNParameterNetwork(state_size, CONFIG['num_of_jobs'], CONFIG['num_of_robots'], CONFIG['max_operations'], CONFIG['hidden_dim'])
    pdqn_path = os.path.join(BASELINE_DIR, 'pdqn_multi_run_20250903_052937/run_1_seed_3047/pdqn_model_20250903_052937.pt')
    if os.path.exists(pdqn_path):
        checkpoint = torch.load(pdqn_path, map_location=device)
        if 'param_network_state_dict' in checkpoint:
            try:
                pdqn_param.load_state_dict(checkpoint['param_network_state_dict'], strict=False)
                print(f"    Loaded from: {pdqn_path}")
            except Exception as e:
                print(f"    Warning: {e}")
    models['PDQN'] = {'model': pdqn_param.to(device), 'type': 'pdqn'}
    
    # Print parameter counts
    print("\n" + "=" * 60)
    print("Model Parameters:")
    print("=" * 60)
    for name, info in models.items():
        if name == 'H-PPO':
            params = count_parameters(info['model']) + count_parameters(info['low'])
        elif name == 'HyAR':
            params = count_parameters(info['model']) + count_parameters(info['vae'])
        else:
            params = count_parameters(info['model'])
        print(f"  {name}: {params:,}")
    
    # Run timing tests
    print("\n" + "=" * 60)
    print("Running Timing Tests...")
    print("=" * 60)
    
    all_results = {}
    
    for alg_name, alg_info in models.items():
        print(f"\n  Testing {alg_name}...")
        
        model = alg_info['model']
        model.eval()
        if 'low' in alg_info:
            alg_info['low'].eval()
        if 'vae' in alg_info:
            alg_info['vae'].eval()
        
        # Warmup
        dummy_state = torch.randn(1, state_size, device=device)
        for _ in range(CONFIG['warmup_iterations']):
            with torch.no_grad():
                if alg_info['type'] == 'pdqn':
                    # PDQN needs job and machine indices
                    dummy_job = torch.tensor([0])
                    dummy_machine = torch.tensor([0])
                    _ = model(dummy_state, dummy_job, dummy_machine)
                else:
                    _ = model(dummy_state)
        
        # Timing test
        env = ENV.Env(CONFIG['num_of_jobs'], CONFIG['num_of_robots'], CONFIG['alpha'], CONFIG['beta'])
        
        network_times = []
        postprocess_times = []
        total_times = []
        hungarian_times = []
        qp_times = []
        
        for episode in range(CONFIG['num_timing_episodes']):
            s = env.reset()
            done = False
            step = 0
            
            while not done and step < 500:
                total_start = time.perf_counter()
                
                # Network forward
                net_start = time.perf_counter()
                with torch.no_grad():
                    s_tensor = torch.from_numpy(s.astype(np.float32)).unsqueeze(0)
                    
                    if alg_info['type'] == 'lirl':
                        output = model(s_tensor)
                    elif alg_info['type'] == 'hppo':
                        job_logits, machine_logits = model(s_tensor)
                        output = {'job_logits': job_logits, 'machine_logits': machine_logits}
                    elif alg_info['type'] == 'hyar':
                        z = model(s_tensor)
                        job_logits, robot_logits, param = alg_info['vae'].decode(s_tensor, z)
                        output = {'job_logits': job_logits, 'machine_logits': robot_logits, 'param': param}
                    elif alg_info['type'] == 'pdqn':
                        output = None
                    else:  # discrete
                        output = model(s_tensor)
                
                network_time = time.perf_counter() - net_start
                network_times.append(network_time)
                
                # Post-processing
                pp_start = time.perf_counter()
                
                if alg_info['type'] == 'lirl':
                    a_np = output.squeeze(0).clamp(0, 1).numpy()
                    action, timing = lirl_action_projection(env, a_np, CONFIG['max_hungarian_size'])
                    hungarian_times.append(timing['hungarian'])
                    qp_times.append(timing['qp'])
                elif alg_info['type'] == 'pdqn':
                    valid_jobs, valid_robots = get_valid_jobs_and_robots(env)
                    if len(valid_jobs) > 0 and len(valid_robots) > 0:
                        job_idx = torch.tensor([valid_jobs[0]])
                        machine_idx = torch.tensor([valid_robots[0]])
                        with torch.no_grad():
                            params = model(s_tensor, job_idx, machine_idx)
                        action = [valid_jobs[0], valid_robots[0], params[0, 0].item()]
                    else:
                        action = [0, 0, 0.5]
                else:  # discrete, hppo, hyar
                    job_logits = output['job_logits']
                    machine_logits = output['machine_logits']
                    job_idx = job_logits.argmax(dim=-1).item()
                    machine_idx = machine_logits.argmax(dim=-1).item()
                    param = output.get('param_mean', output.get('param', torch.tensor([[0.5]])))[0, 0].item() if 'param_mean' in output or 'param' in output else 0.5
                    action = [job_idx, machine_idx, param]
                
                postprocess_time = time.perf_counter() - pp_start
                postprocess_times.append(postprocess_time)
                
                total_time = time.perf_counter() - total_start
                total_times.append(total_time)
                
                # Step
                s, _, done = env_step_silent(env, action)
                step += 1
            
            print(f"    Episode {episode + 1}: {step} steps")
        
        # Collect results
        result = {
            'network': {'mean': np.mean(network_times), 'std': np.std(network_times)},
            'postprocess': {'mean': np.mean(postprocess_times), 'std': np.std(postprocess_times)},
            'total': {'mean': np.mean(total_times), 'std': np.std(total_times)},
        }
        if hungarian_times:
            result['hungarian'] = {'mean': np.mean(hungarian_times), 'std': np.std(hungarian_times)}
            result['qp'] = {'mean': np.mean(qp_times), 'std': np.std(qp_times)}
        
        all_results[alg_name] = result
    
    # Print summary
    print("\n" + "=" * 100)
    print("TOTAL DECISION TIME SUMMARY (milliseconds)")
    print("=" * 100)
    print(f"\n{'Algorithm':<12} {'Network':<18} {'Postprocess':<18} {'Total':<18} {'PP %':<10}")
    print("-" * 100)
    
    for name, result in all_results.items():
        net_ms = result['network']['mean'] * 1000
        pp_ms = result['postprocess']['mean'] * 1000
        total_ms = result['total']['mean'] * 1000
        pp_pct = pp_ms / total_ms * 100 if total_ms > 0 else 0
        
        print(f"{name:<12} {net_ms:.4f}±{result['network']['std']*1000:.4f}       "
              f"{pp_ms:.4f}±{result['postprocess']['std']*1000:.4f}       "
              f"{total_ms:.4f}±{result['total']['std']*1000:.4f}       {pp_pct:.1f}%")
    
    # Ranking
    print("\n" + "-" * 80)
    print("RANKING (by Total Decision Time, fastest to slowest)")
    print("-" * 80)
    
    sorted_results = sorted(all_results.items(), key=lambda x: x[1]['total']['mean'])
    fastest = sorted_results[0][1]['total']['mean']
    
    for i, (name, result) in enumerate(sorted_results, 1):
        t = result['total']['mean']
        rel = t / fastest if fastest > 0 else 1.0
        print(f"  {i}. {name:10s}: {t*1000:.4f} ms ({rel:.2f}x)")
    
    # LIRL breakdown
    if 'LIRL' in all_results and 'hungarian' in all_results['LIRL']:
        print("\n" + "-" * 80)
        print("LIRL Time Breakdown")
        print("-" * 80)
        lirl = all_results['LIRL']
        net = lirl['network']['mean'] * 1000
        hung = lirl['hungarian']['mean'] * 1000
        qp = lirl['qp']['mean'] * 1000
        total = lirl['total']['mean'] * 1000
        
        print(f"  Network:    {net:.4f} ms ({net/total*100:.1f}%)")
        print(f"  Hungarian:  {hung:.4f} ms ({hung/total*100:.1f}%)")
        print(f"  QP:         {qp:.4f} ms ({qp/total*100:.1f}%)")
        print(f"  Total:      {total:.4f} ms")
    
    # Save results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(os.path.dirname(__file__), f'computation_time_results_{timestamp}.json')
    
    results_json = {'timestamp': timestamp, 'config': CONFIG, 'results': {}}
    for name, result in all_results.items():
        results_json['results'][name] = {
            'network_ms': result['network']['mean'] * 1000,
            'postprocess_ms': result['postprocess']['mean'] * 1000,
            'total_ms': result['total']['mean'] * 1000,
        }
        if 'hungarian' in result:
            results_json['results'][name]['hungarian_ms'] = result['hungarian']['mean'] * 1000
            results_json['results'][name]['qp_ms'] = result['qp']['mean'] * 1000
    
    with open(save_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"\nResults saved to: {save_path}")
    
    # Plot
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        algorithms = list(all_results.keys())
        total_times_ms = [all_results[a]['total']['mean'] * 1000 for a in algorithms]
        network_times_ms = [all_results[a]['network']['mean'] * 1000 for a in algorithms]
        pp_times_ms = [all_results[a]['postprocess']['mean'] * 1000 for a in algorithms]
        
        # Bar chart
        x = np.arange(len(algorithms))
        ax1 = axes[0]
        ax1.bar(x, network_times_ms, label='Network', color='steelblue')
        ax1.bar(x, pp_times_ms, bottom=network_times_ms, label='Postprocess', color='coral')
        ax1.set_xticks(x)
        ax1.set_xticklabels(algorithms, rotation=45, ha='right')
        ax1.set_ylabel('Time (ms)')
        ax1.set_title('Decision Time Breakdown')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Horizontal bar
        ax2 = axes[1]
        sorted_algs = sorted(all_results.items(), key=lambda x: x[1]['total']['mean'])
        names = [a[0] for a in sorted_algs]
        times = [a[1]['total']['mean'] * 1000 for a in sorted_algs]
        ax2.barh(names, times, color='teal')
        ax2.set_xlabel('Time (ms)')
        ax2.set_title('Total Decision Time (sorted)')
        ax2.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        fig_path = os.path.join(os.path.dirname(__file__), f'computation_time_{timestamp}.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {fig_path}")
        plt.show()
    except Exception as e:
        print(f"Warning: Could not create plot: {e}")
    
    print("\n" + "=" * 80)
    print("COMPUTATION TIME COMPARISON COMPLETE!")
    print("=" * 80)


if __name__ == '__main__':
    main()
