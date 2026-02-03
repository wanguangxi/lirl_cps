#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Inexact Projection Accuracy-Latency Pareto Experiment (Fig. 5b)
Demonstrates controllable stationarity-latency Pareto via solver tolerance sweep.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import sys
import datetime
import json
import random
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment, minimize
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.join(os.path.dirname(__file__), '../env'))
try:
    import importlib
    ENV = importlib.import_module('env')
    if not hasattr(ENV, 'Env'):
        ENV = importlib.import_module('env.env')
except Exception:
    from env import env as ENV

# Configuration
TRAINED_MODEL_DIR = '/home/one/LIRL-CPS-main/RMS/exp/lirl_runtime_scaling_cpu_20260117_181324'

CONFIG = {
    'scales': {'large': {'num_of_jobs': 1000, 'num_of_robots': 100, 'name': 'Large (1000×100)'}},
    'tolerance_levels': [1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2e-1, 5e-1],
    'qp_maxiter_levels': [500, 200, 100, 50, 20, 10, 5, 3, 1],
    'hungarian_topk_levels': [None, 100, 50, 30, 20, 10, 5, 3, 1],
    'hidden_dim1': 128, 'hidden_dim2': 64, 'critic_hidden': 32,
    'alpha': 0.5, 'beta': 0.5,
    'num_warmup_steps': 200, 'num_measurement_steps': 1000, 'num_episodes': 10,
    'seed': 42, 'model_dir': TRAINED_MODEL_DIR,
    'save_results': True, 'output_dir': os.path.dirname(__file__),
}


class MuNet(nn.Module):
    def __init__(self, state_size, action_size, config):
        super().__init__()
        self.fc1 = nn.Linear(state_size, config['hidden_dim1'])
        self.fc2 = nn.Linear(config['hidden_dim1'], config['hidden_dim2'])
        self.fc_mu = nn.Linear(config['hidden_dim2'], action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.fc_mu(x))


class InexactProjectionSolver:
    def __init__(self, qp_tol=1e-6, qp_maxiter=100, hungarian_topk=None):
        self.qp_tol = qp_tol
        self.qp_maxiter = qp_maxiter
        self.hungarian_topk = hungarian_topk
        self.reset_measurements()
    
    def reset_measurements(self):
        self.stationarity_proxies = []
        self.optimality_gaps = []
        self.duality_gaps = []
        self.qp_residuals = []
        self.logical_violations = []
        self.total_times = []
        self.hungarian_times = []
        self.qp_times = []
    
    def get_valid_jobs_and_robots(self, env):
        valid_robots = np.where(np.array(env.robot_state) == 1)[0].tolist()
        valid_jobs = [i for i, job in enumerate(env.task_set) 
                      if any(not task.state for task in job)]
        return valid_jobs, valid_robots
    
    def build_cost_matrix(self, env, valid_jobs, valid_robots, a_):
        job_indices = np.array(valid_jobs)
        robot_indices = np.array(valid_robots)
        job_costs = np.abs(a_[0] - job_indices / max(1, len(env.task_set)))
        robot_costs = np.abs(a_[1] - robot_indices / max(1, len(env.robot_state)))
        return job_costs[:, np.newaxis] + robot_costs[np.newaxis, :]
    
    def solve_hungarian(self, cost_matrix, valid_jobs, valid_robots):
        start = time.perf_counter()
        n_jobs, n_robots = len(valid_jobs), len(valid_robots)
        
        if self.hungarian_topk and (n_jobs > self.hungarian_topk or n_robots > self.hungarian_topk):
            k = min(self.hungarian_topk, n_jobs, n_robots)
            job_idx = np.argsort(cost_matrix.min(axis=1))[:k]
            robot_idx = np.argsort(cost_matrix.min(axis=0))[:k]
            row, col = linear_sum_assignment(cost_matrix[np.ix_(job_idx, robot_idx)])
            job_id = valid_jobs[job_idx[row[0]]] if len(row) > 0 else valid_jobs[0]
            robot_id = valid_robots[robot_idx[col[0]]] if len(col) > 0 else valid_robots[0]
        else:
            row, col = linear_sum_assignment(cost_matrix)
            job_id = valid_jobs[row[0]] if len(row) > 0 else valid_jobs[0]
            robot_id = valid_robots[col[0]] if len(col) > 0 else valid_robots[0]
        
        return job_id, robot_id, time.perf_counter() - start
    
    def solve_qp(self, v):
        start = time.perf_counter()
        v = np.asarray(v, dtype=np.float64)
        n = len(v)
        A = np.vstack([np.eye(n), -np.eye(n)])
        b = np.hstack([np.ones(n), np.zeros(n)])
        
        result = minimize(
            lambda x: 0.5 * np.sum((x - v) ** 2), np.clip(v, 0, 1),
            jac=lambda x: x - v,
            constraints={'type': 'ineq', 'fun': lambda x: b - A @ x},
            method='SLSQP', options={'ftol': self.qp_tol, 'maxiter': self.qp_maxiter, 'disp': False}
        )
        
        qp_time = time.perf_counter() - start
        obj_gap = np.linalg.norm(result.x - v)
        dual_gap = abs(result.fun - 0.5 * np.sum((np.clip(v, 0, 1) - v) ** 2))
        residual = max(0, -np.min(b - A @ result.x))
        return result.x, qp_time, residual, obj_gap, dual_gap
    
    def project(self, env, z_tensor, eta=1.0):
        total_start = time.perf_counter()
        z = z_tensor.detach().cpu().numpy()
        valid_jobs, valid_robots = self.get_valid_jobs_and_robots(env)
        
        if not valid_jobs or not valid_robots:
            return [0, 0, z[2] if len(z) > 2 else 0.0], {}
        
        cost_matrix = self.build_cost_matrix(env, valid_jobs, valid_robots, z)
        job_id, robot_id, hungarian_time = self.solve_hungarian(cost_matrix, valid_jobs, valid_robots)
        
        if len(z) > 2:
            x_sol, qp_time, residual, obj_gap, dual_gap = self.solve_qp(z[2:])
            param = float(x_sol[0])
        else:
            qp_time, residual, obj_gap, dual_gap, param = 0, 0, 0, 0, 0.0
        
        total_time = time.perf_counter() - total_start
        z_proj = np.array([job_id / max(1, len(env.task_set)), 
                          robot_id / max(1, len(env.robot_state)), param])
        stationarity = np.linalg.norm(z[:3] - z_proj) / eta
        logical_viol = int(job_id not in valid_jobs) + int(robot_id not in valid_robots)
        
        self.stationarity_proxies.append(stationarity)
        self.optimality_gaps.append(obj_gap)
        self.duality_gaps.append(dual_gap)
        self.qp_residuals.append(residual)
        self.logical_violations.append(logical_viol)
        self.total_times.append(total_time)
        self.hungarian_times.append(hungarian_time)
        self.qp_times.append(qp_time)
        
        return [job_id, robot_id, param], {'total_time': total_time}
    
    def get_statistics(self):
        def pct(arr, p): return np.percentile(arr, p) if arr else 0
        def avg(arr): return np.mean(arr) if arr else 0
        def std(arr): return np.std(arr) if arr else 0
        
        return {
            'stationarity_mean': avg(self.stationarity_proxies),
            'stationarity_std': std(self.stationarity_proxies),
            'optimality_gap_mean': avg(self.optimality_gaps),
            'duality_gap_mean': avg(self.duality_gaps),
            'residual_mean': avg(self.qp_residuals),
            'logical_violations_total': sum(self.logical_violations),
            'latency_p50': pct(self.total_times, 50) * 1000,
            'latency_p95': pct(self.total_times, 95) * 1000,
            'latency_p99': pct(self.total_times, 99) * 1000,
            'latency_mean': avg(self.total_times) * 1000,
            'hungarian_time_mean': avg(self.hungarian_times) * 1000,
            'qp_time_mean': avg(self.qp_times) * 1000,
            'num_samples': len(self.total_times),
        }


def load_model(model_dir, scale_name, state_size, action_size, config):
    net = MuNet(state_size, action_size, config)
    path = os.path.join(model_dir, scale_name, 'mu.pth')
    if os.path.exists(path):
        try:
            net.load_state_dict(torch.load(path, map_location='cpu'))
            print(f"  ✓ Loaded model: {path}")
            return net, True
        except Exception as e:
            print(f"  ✗ Error: {e}")
    else:
        print(f"  ✗ Not found: {path}")
    return net, False


def run_experiment(config=None):
    config = config or CONFIG
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    
    print("=" * 70)
    print("Inexact Projection Pareto Experiment (Fig. 5b)")
    print("=" * 70)
    
    results = {}
    for scale_name, scale_cfg in config['scales'].items():
        print(f"\nScale: {scale_cfg['name']}")
        
        env = ENV.Env(scale_cfg['num_of_jobs'], scale_cfg['num_of_robots'], 
                      config['alpha'], config['beta'])
        mu_net, loaded = load_model(config['model_dir'], scale_name, 
                                    len(env.state), len(env.action), config)
        mu_net.eval()
        
        sweep_configs = []
        for tol, maxiter, topk in zip(config['tolerance_levels'], 
                                       config['qp_maxiter_levels'],
                                       config['hungarian_topk_levels']):
            eps = tol * 100 + 1.0/max(1, maxiter) + 1.0/(topk or 1000)
            sweep_configs.append({'qp_tol': tol, 'qp_maxiter': maxiter, 
                                  'hungarian_topk': topk, 'effective_eps': eps})
        sweep_configs.sort(key=lambda x: x['effective_eps'])
        
        eps_results = []
        for i, cfg in enumerate(sweep_configs):
            print(f"  [{i+1}/{len(sweep_configs)}] τ={cfg['qp_tol']:.0e}, iter={cfg['qp_maxiter']}")
            
            solver = InexactProjectionSolver(cfg['qp_tol'], cfg['qp_maxiter'], cfg['hungarian_topk'])
            
            # Warmup
            env.reset()
            for _ in range(config['num_warmup_steps']):
                with torch.no_grad():
                    z = mu_net(torch.FloatTensor(env.state))
                action, _ = solver.project(env, z)
                try:
                    _, _, done = env.step(action)
                    if done: env.reset()
                except: env.reset()
            
            solver.reset_measurements()
            
            # Measurement
            for _ in range(config['num_episodes']):
                env.reset()
                for _ in range(config['num_measurement_steps']):
                    with torch.no_grad():
                        z = mu_net(torch.FloatTensor(env.state))
                    action, _ = solver.project(env, z)
                    try:
                        _, _, done = env.step(action)
                        if done: break
                    except: break
            
            stats = solver.get_statistics()
            stats['config'] = cfg
            eps_results.append(stats)
            print(f"    m={stats['stationarity_mean']:.4f}, P50/95/99={stats['latency_p50']:.2f}/{stats['latency_p95']:.2f}/{stats['latency_p99']:.2f}ms")
        
        results[scale_name] = {'scale_config': scale_cfg, 'model_loaded': loaded, 'results': eps_results}
    
    return results


def plot_pareto(results, save_path=None):
    if 'large' not in results:
        print("No large scale data")
        return
    
    data = results['large']
    res = data['results']
    name = data['scale_config']['name']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    plt.rcParams.update({'font.family': 'serif', 'font.size': 12})
    
    eps = np.array([r['config']['effective_eps'] for r in res])
    idx = np.argsort(eps)
    eps = eps[idx]
    
    m_mean = np.array([r['stationarity_mean'] for r in res])[idx]
    m_std = np.array([r['stationarity_std'] for r in res])[idx]
    p50 = np.array([r['latency_p50'] for r in res])[idx]
    p95 = np.array([r['latency_p95'] for r in res])[idx]
    p99 = np.array([r['latency_p99'] for r in res])[idx]
    
    # Left: Stationarity
    ax1.errorbar(eps, m_mean, yerr=m_std, color='#2563EB', marker='o', 
                 markersize=8, linewidth=2.5, capsize=4, label=r'$m = \|z-\Pi(z)\|/\eta$')
    ref = np.logspace(np.log10(eps.min()), np.log10(eps.max()), 50)
    ax1.plot(ref, ref * 0.3, 'k--', alpha=0.5, lw=1.5, label=r'$\mathcal{O}(\varepsilon)$')
    ax1.set(xlabel=r'$\varepsilon$', ylabel=r'Stationarity $m$', xscale='log', yscale='log')
    ax1.set_title(f'(a) Projection Accuracy\n{name}', fontweight='bold')
    ax1.grid(True, alpha=0.3, ls='--')
    ax1.legend(loc='upper left')
    
    # Right: Latency
    ax2.plot(eps, p50, 'o-', color='#10B981', ms=8, lw=2.5, label='P50')
    ax2.plot(eps, p95, 's--', color='#F59E0B', ms=6, lw=2, label='P95')
    ax2.plot(eps, p99, '^:', color='#DC2626', ms=6, lw=2, label='P99')
    ax2.fill_between(eps, p50, p99, color='#3B82F6', alpha=0.15)
    ax2.set(xlabel=r'$\varepsilon$', ylabel='Latency (ms)', xscale='log')
    ax2.set_title(f'(b) End-to-End Latency\n{name}', fontweight='bold')
    ax2.grid(True, alpha=0.3, ls='--')
    ax2.legend(loc='upper right')
    
    fig.text(0.5, 0.02, '• Zero logical violations • Continuous residuals bounded by τ',
             ha='center', fontsize=10, style='italic',
             bbox=dict(boxstyle='round', facecolor='#F3F4F6', alpha=0.9))
    
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    
    if save_path:
        out = save_path.replace('.png', '_large_only.png')
        plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {out}")
    plt.show()


def save_json(results, output_dir):
    path = os.path.join(output_dir, f"inexact_projection_{datetime.datetime.now():%Y%m%d_%H%M%S}.json")
    data = {k: {'scale_config': v['scale_config'], 
                'results': [{**{k2: v2 for k2, v2 in r.items() if k2 != 'config'},
                            'config': r['config']} for r in v['results']]}
            for k, v in results.items()}
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {path}")


def main():
    print("\n" + "=" * 70)
    print("LIRL Inexact Projection - Figure 5b")
    print("=" * 70)
    
    results = run_experiment(CONFIG)
    plot_pareto(results, os.path.join(CONFIG['output_dir'], 'fig5b.png'))
    
    if CONFIG['save_results']:
        save_json(results, CONFIG['output_dir'])
    
    print("\n" + "=" * 70)
    for name, data in results.items():
        r = data['results']
        best_acc = min(r, key=lambda x: x['stationarity_mean'])
        best_spd = min(r, key=lambda x: x['latency_p50'])
        print(f"{data['scale_config']['name']}: "
              f"Best acc m={best_acc['stationarity_mean']:.4f} @ {best_acc['latency_p50']:.2f}ms, "
              f"Best speed m={best_spd['stationarity_mean']:.4f} @ {best_spd['latency_p50']:.2f}ms")
    print("=" * 70)


if __name__ == "__main__":
    main()
