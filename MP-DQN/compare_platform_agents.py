#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Platform环境算法对比程序
====================
对比四种算法在Platform环境上的表现:
1. LIRL - Learning with Integer and Real-valued Actions
2. PADDPG - Parameterised Action DDPG
3. PDQN - Parameterised DQN
4. QPAMDP - Q-PAMDP

支持多seeds运行，绘制带置信区间的收敛曲线
"""

import os
import sys
import time
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
import argparse


# =======================
# 配置
# =======================
ALGORITHMS = {
    'LIRL': {
        'script': 'run_platform_lirl.py',
        'title': 'LIRL',
        'color': '#e74c3c',  # 红色
        'marker': 'o'
    },
    'PADDPG': {
        'script': 'run_platform_paddpg.py',
        'title': 'PADDPG',
        'color': '#3498db',  # 蓝色
        'marker': 's'
    },
    'PDQN': {
        'script': 'run_platform_pdqn.py',
        'title': 'PDDQN',  # Platform的PDQN默认title是PDDQN
        'color': '#2ecc71',  # 绿色
        'marker': '^'
    },
    'QPAMDP': {
        'script': 'run_platform_qpamdp.py',
        'title': 'QPAMDP',
        'color': '#9b59b6',  # 紫色
        'marker': 'd'
    }
}


def format_time(seconds):
    """格式化时间显示"""
    if seconds < 60:
        return f"{seconds:.0f}秒"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}分钟"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}小时"


def run_algorithm(algo_name, algo_config, episodes, seed, save_dir, python_path=None):
    """
    运行单个算法（实时输出）
    
    Args:
        algo_name: 算法名称
        algo_config: 算法配置
        episodes: 训练episode数
        seed: 随机种子
        save_dir: 保存目录
        python_path: Python解释器路径
    
    Returns:
        dict: 包含训练结果的字典
    """
    if python_path is None:
        python_path = sys.executable
    
    script_path = os.path.join(os.path.dirname(__file__), algo_config['script'])
    
    # 构建命令
    cmd = [
        python_path,
        '-u',  # 禁用Python输出缓冲，实现实时输出
        script_path,
        '--episodes', str(episodes),
        '--seed', str(seed),
        '--save-dir', save_dir,
        '--title', algo_config['title'],
        '--evaluation-episodes', '100'
    ]
    
    # PDQN特殊参数：禁用可视化
    if algo_name == 'PDQN':
        cmd.extend(['--visualise', 'False'])
    
    # LIRL特殊参数：禁用可视化
    if algo_name == 'LIRL':
        cmd.extend(['--visualise', 'False'])
    
    print(f"\n{'='*60}")
    print(f"Running {algo_name} with seed {seed}...")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    sys.stdout.flush()
    
    start_time = time.time()
    stdout_lines = []
    stderr_lines = []
    
    try:
        # 使用Popen实现实时输出
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1
        )
        
        # 实时读取并打印stdout
        while True:
            line = process.stdout.readline()
            if line:
                print(line, end='')
                sys.stdout.flush()
                stdout_lines.append(line)
            
            if process.poll() is not None:
                remaining_stdout = process.stdout.read()
                if remaining_stdout:
                    print(remaining_stdout, end='')
                    stdout_lines.append(remaining_stdout)
                break
        
        # 读取stderr
        stderr_output = process.stderr.read()
        if stderr_output:
            print("STDERR:", stderr_output)
            stderr_lines.append(stderr_output)
        
        training_time = time.time() - start_time
        returncode = process.returncode
        
        # 加载结果
        possible_paths = [
            os.path.join(save_dir, algo_config['title'], f"{algo_config['title']}{seed}.npy"),
            os.path.join(save_dir, algo_config['title'], str(seed), f"{algo_config['title']}{seed}.npy"),
        ]
        
        results_file = None
        for path in possible_paths:
            if os.path.exists(path):
                results_file = path
                break
        
        if results_file and os.path.exists(results_file):
            returns = np.load(results_file)
            print(f"Loaded results from: {results_file}")
        else:
            print(f"Warning: Results file not found. Tried: {possible_paths}")
            returns = np.array([])
        
        return {
            'success': returncode == 0,
            'returns': returns,
            'training_time': training_time,
            'stdout': ''.join(stdout_lines),
            'stderr': ''.join(stderr_lines)
        }
        
    except Exception as e:
        print(f"ERROR running {algo_name}: {e}")
        return {
            'success': False,
            'returns': np.array([]),
            'training_time': time.time() - start_time,
            'stdout': ''.join(stdout_lines),
            'stderr': str(e)
        }


def run_algorithm_multi_seeds(algo_name, algo_config, episodes, seeds, save_dir, python_path=None,
                               progress_info=None):
    """
    使用多个seeds运行单个算法
    
    Args:
        algo_name: 算法名称
        algo_config: 算法配置
        episodes: 训练episode数
        seeds: 随机种子列表
        save_dir: 保存目录
        python_path: Python解释器路径
        progress_info: 进度信息字典
    
    Returns:
        dict: 包含所有seeds训练结果的字典
    """
    all_returns = []
    all_training_times = []
    all_successes = []
    
    for seed_idx, seed in enumerate(seeds):
        # 计算总体进度
        if progress_info is not None:
            current_task = progress_info['completed'] + seed_idx + 1
            total_tasks = progress_info['total']
            
            if len(progress_info['elapsed_times']) > 0:
                avg_time_per_task = np.mean(progress_info['elapsed_times'])
                remaining_tasks = total_tasks - current_task + 1
                estimated_remaining = avg_time_per_task * remaining_tasks
                eta_str = f" | 预估剩余: {format_time(estimated_remaining)}"
            else:
                eta_str = ""
            
            progress_str = f"总进度: {current_task}/{total_tasks} ({100*current_task/total_tasks:.1f}%){eta_str}"
        else:
            progress_str = ""
        
        print(f"\n{'#'*70}")
        print(f"# {algo_name} - Seed {seed} ({seed_idx+1}/{len(seeds)})")
        if progress_str:
            print(f"# {progress_str}")
        print(f"{'#'*70}")
        sys.stdout.flush()
        
        result = run_algorithm(
            algo_name, 
            algo_config, 
            episodes, 
            seed, 
            save_dir,
            python_path
        )
        
        all_returns.append(result['returns'])
        all_training_times.append(result['training_time'])
        all_successes.append(result['success'])
        
        # 更新进度信息
        if progress_info is not None:
            progress_info['elapsed_times'].append(result['training_time'])
        
        print(f"\n>>> Seed {seed} 完成，耗时: {format_time(result['training_time'])}")
        sys.stdout.flush()
    
    return {
        'returns_per_seed': all_returns,
        'training_times': all_training_times,
        'successes': all_successes,
        'seeds': seeds
    }


def smooth_curve(data, window=100):
    """平滑曲线"""
    if len(data) < window:
        return data
    smoothed = np.convolve(data, np.ones(window)/window, mode='valid')
    return smoothed


def align_and_aggregate_returns(returns_list, smooth_window=100):
    """
    对齐并聚合多个seeds的returns
    
    Args:
        returns_list: 每个seed的returns列表
        smooth_window: 平滑窗口大小
    
    Returns:
        tuple: (episodes, mean_returns, std_returns)
    """
    valid_returns = [r for r in returns_list if len(r) > 0]
    
    if len(valid_returns) == 0:
        return np.array([]), np.array([]), np.array([])
    
    min_len = min(len(r) for r in valid_returns)
    aligned_returns = np.array([r[:min_len] for r in valid_returns])
    
    smoothed_returns = []
    for returns in aligned_returns:
        smoothed = smooth_curve(returns, window=smooth_window)
        smoothed_returns.append(smoothed)
    
    min_smoothed_len = min(len(s) for s in smoothed_returns)
    smoothed_returns = np.array([s[:min_smoothed_len] for s in smoothed_returns])
    
    mean_returns = np.mean(smoothed_returns, axis=0)
    std_returns = np.std(smoothed_returns, axis=0)
    
    episodes = np.arange(len(mean_returns)) + smooth_window // 2
    
    return episodes, mean_returns, std_returns


def calculate_metrics(returns):
    """计算单个seed的评估指标"""
    if len(returns) == 0:
        return {
            'mean_return': 0,
            'std_return': 0,
            'last_100_mean': 0,
            'last_100_std': 0,
            'max_return': 0,
            'min_return': 0,
            'median_return': 0
        }
    
    return {
        'mean_return': np.mean(returns),
        'std_return': np.std(returns),
        'last_100_mean': np.mean(returns[-100:]) if len(returns) >= 100 else np.mean(returns),
        'last_100_std': np.std(returns[-100:]) if len(returns) >= 100 else np.std(returns),
        'max_return': np.max(returns),
        'min_return': np.min(returns),
        'median_return': np.median(returns)
    }


def calculate_aggregated_metrics(returns_list):
    """
    计算多个seeds的聚合指标
    
    Args:
        returns_list: 每个seed的returns列表
    
    Returns:
        dict: 聚合指标字典
    """
    valid_returns = [r for r in returns_list if len(r) > 0]
    
    if len(valid_returns) == 0:
        return {
            'mean_return': 0,
            'std_return': 0,
            'last_100_mean': 0,
            'last_100_std': 0,
            'max_return': 0,
            'min_return': 0,
            'num_seeds': 0
        }
    
    # 计算每个seed的最后100 episodes指标
    last_100_means = []
    for returns in valid_returns:
        last_100 = returns[-100:] if len(returns) >= 100 else returns
        last_100_means.append(np.mean(last_100))
    
    all_means = [np.mean(r) for r in valid_returns]
    all_max = [np.max(r) for r in valid_returns]
    all_min = [np.min(r) for r in valid_returns]
    
    return {
        'mean_return': np.mean(all_means),
        'std_return': np.std(all_means),
        'last_100_mean': np.mean(last_100_means),
        'last_100_std': np.std(last_100_means),
        'max_return': np.mean(all_max),
        'min_return': np.mean(all_min),
        'num_seeds': len(valid_returns)
    }


def plot_comparison_multi_seeds(all_results, save_path, smooth_window=100):
    """
    绘制多seeds对比图（带置信区间）
    """
    valid_results = {k: v for k, v in all_results.items() 
                     if any(len(r) > 0 for r in v['returns_per_seed'])}
    
    if len(valid_results) == 0:
        print("Warning: No valid results to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 学习曲线对比 (带置信区间)
    ax1 = axes[0, 0]
    for algo_name, result in valid_results.items():
        config = ALGORITHMS[algo_name]
        episodes, mean_returns, std_returns = align_and_aggregate_returns(
            result['returns_per_seed'], smooth_window
        )
        
        if len(episodes) > 0:
            ax1.plot(episodes, mean_returns, label=algo_name, color=config['color'], linewidth=2)
            ax1.fill_between(episodes, mean_returns - std_returns, mean_returns + std_returns,
                           color=config['color'], alpha=0.2)
    
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Return (smoothed)', fontsize=12)
    ax1.set_title('Learning Curves Comparison (Mean ± Std)', fontsize=14)
    if len(valid_results) > 0:
        ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    
    # 2. 最终100 episodes平均回报对比
    ax2 = axes[0, 1]
    names = []
    means = []
    stds = []
    colors = []
    for algo_name, result in valid_results.items():
        names.append(algo_name)
        metrics = calculate_aggregated_metrics(result['returns_per_seed'])
        means.append(metrics['last_100_mean'])
        stds.append(metrics['last_100_std'])
        colors.append(ALGORITHMS[algo_name]['color'])
    
    if len(names) > 0:
        x = np.arange(len(names))
        bars = ax2.bar(x, means, yerr=stds, color=colors, capsize=5, alpha=0.8)
        ax2.set_xticks(x)
        ax2.set_xticklabels(names, fontsize=11)
        ax2.set_ylabel('Average Return', fontsize=12)
        ax2.set_title('Final 100 Episodes Average Return', fontsize=14)
        ax2.grid(axis='y', alpha=0.3)
        
        for bar, mean, std in zip(bars, means, stds):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.5,
                    f'{mean:.1f}±{std:.1f}', ha='center', va='bottom', fontsize=9)
    
    # 3. 回报箱线图
    ax3 = axes[1, 0]
    box_data = []
    box_labels = []
    box_colors = []
    for algo_name, result in valid_results.items():
        # 收集所有seeds的最后100 episodes
        all_last_100 = []
        for returns in result['returns_per_seed']:
            if len(returns) > 0:
                last_100 = returns[-100:] if len(returns) >= 100 else returns
                all_last_100.extend(last_100)
        if len(all_last_100) > 0:
            box_data.append(all_last_100)
            box_labels.append(algo_name)
            box_colors.append(ALGORITHMS[algo_name]['color'])
    
    if len(box_data) > 0:
        bp = ax3.boxplot(box_data, labels=box_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax3.set_ylabel('Return', fontsize=12)
        ax3.set_title('Final 100 Episodes Return Distribution (All Seeds)', fontsize=14)
        ax3.grid(axis='y', alpha=0.3)
    
    # 4. 训练时间对比
    ax4 = axes[1, 1]
    time_means = []
    time_stds = []
    for result in valid_results.values():
        times = np.array(result['training_times']) / 60
        time_means.append(np.mean(times))
        time_stds.append(np.std(times))
    
    if len(names) > 0:
        bars = ax4.bar(x, time_means, yerr=time_stds, color=colors, capsize=5, alpha=0.8)
        ax4.set_xticks(x)
        ax4.set_xticklabels(names, fontsize=11)
        ax4.set_ylabel('Training Time (minutes)', fontsize=12)
        ax4.set_title('Training Time Comparison', fontsize=14)
        ax4.grid(axis='y', alpha=0.3)
        
        for bar, t_mean, t_std in zip(bars, time_means, time_stds):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + t_std + 0.1,
                    f'{t_mean:.1f}±{t_std:.1f}m', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved to: {save_path}")


def plot_learning_curves_all_seeds(all_results, save_path, smooth_window=100):
    """绘制所有seeds的学习曲线"""
    valid_results = {k: v for k, v in all_results.items() 
                     if any(len(r) > 0 for r in v['returns_per_seed'])}
    
    if len(valid_results) == 0:
        print("Warning: No valid results for learning curves")
        return
    
    n_algos = len(valid_results)
    fig, axes = plt.subplots(1, n_algos, figsize=(5*n_algos, 5))
    
    if n_algos == 1:
        axes = [axes]
    
    for idx, (algo_name, result) in enumerate(valid_results.items()):
        ax = axes[idx]
        config = ALGORITHMS[algo_name]
        
        # 绘制每个seed的曲线
        for seed_idx, returns in enumerate(result['returns_per_seed']):
            if len(returns) > 0:
                smoothed = smooth_curve(returns, window=smooth_window)
                ax.plot(np.arange(len(smoothed)) + smooth_window//2, smoothed, 
                       color=config['color'], alpha=0.3, linewidth=1)
        
        # 绘制均值曲线
        episodes, mean_returns, std_returns = align_and_aggregate_returns(
            result['returns_per_seed'], smooth_window
        )
        if len(episodes) > 0:
            ax.plot(episodes, mean_returns, color=config['color'], 
                   linewidth=2.5, label='Mean')
            ax.fill_between(episodes, mean_returns - std_returns, mean_returns + std_returns,
                          color=config['color'], alpha=0.2)
        
        ax.set_xlabel('Episode', fontsize=11)
        ax.set_ylabel('Return', fontsize=11)
        ax.set_title(f'{algo_name}', fontsize=13, fontweight='bold')
        ax.grid(alpha=0.3)
        
        # 添加统计信息
        metrics = calculate_aggregated_metrics(result['returns_per_seed'])
        info_text = f"Seeds: {metrics['num_seeds']}\n"
        info_text += f"Last 100: {metrics['last_100_mean']:.1f}±{metrics['last_100_std']:.1f}\n"
        info_text += f"Max: {metrics['max_return']:.1f}"
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
               verticalalignment='top', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Learning curves (all seeds) saved to: {save_path}")


def plot_convergence_comparison(all_results, save_path, smooth_window=100):
    """绘制收敛曲线对比图"""
    valid_results = {k: v for k, v in all_results.items() 
                     if any(len(r) > 0 for r in v['returns_per_seed'])}
    
    if len(valid_results) == 0:
        print("Warning: No valid results for convergence comparison")
        return
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for algo_name, result in valid_results.items():
        config = ALGORITHMS[algo_name]
        episodes, mean_returns, std_returns = align_and_aggregate_returns(
            result['returns_per_seed'], smooth_window
        )
        
        if len(episodes) > 0:
            ax.plot(episodes, mean_returns, label=algo_name, 
                   color=config['color'], linewidth=2.5)
            ax.fill_between(episodes, mean_returns - std_returns, mean_returns + std_returns,
                          color=config['color'], alpha=0.2)
    
    ax.set_xlabel('Episode', fontsize=14)
    ax.set_ylabel('Return', fontsize=14)
    ax.set_title('Convergence Curves Comparison (Mean ± Std)', fontsize=16)
    ax.legend(fontsize=12, loc='lower right')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Convergence comparison saved to: {save_path}")


def generate_report_multi_seeds(all_results, save_path, episodes, seeds):
    """生成多seeds文本报告"""
    report = []
    report.append("="*70)
    report.append("Platform环境算法对比报告 (多Seeds)")
    report.append("="*70)
    report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"训练Episodes: {episodes}")
    report.append(f"Seeds数量: {len(seeds)}")
    report.append(f"Seeds列表: {seeds}")
    report.append("")
    
    # 汇总表格
    report.append("-"*70)
    report.append(f"{'算法':<12} {'最后100回报':<18} {'最大回报':<14} {'训练时间(分钟)':<16}")
    report.append("-"*70)
    
    for algo_name, result in all_results.items():
        metrics = calculate_aggregated_metrics(result['returns_per_seed'])
        time_mean = np.mean(result['training_times']) / 60
        time_std = np.std(result['training_times']) / 60
        
        report.append(
            f"{algo_name:<12} "
            f"{metrics['last_100_mean']:.1f}±{metrics['last_100_std']:.1f}".ljust(18) +
            f"{metrics['max_return']:.1f}".ljust(14) +
            f"{time_mean:.1f}±{time_std:.1f}".ljust(16)
        )
    
    report.append("-"*70)
    report.append("")
    
    # 详细指标
    report.append("详细指标 (跨Seeds统计):")
    report.append("")
    
    for algo_name, result in all_results.items():
        report.append(f"【{algo_name}】")
        metrics = calculate_aggregated_metrics(result['returns_per_seed'])
        
        report.append(f"  有效Seeds数: {metrics['num_seeds']}")
        report.append(f"  总体平均回报: {metrics['mean_return']:.4f} ± {metrics['std_return']:.4f}")
        report.append(f"  最后100回合平均回报: {metrics['last_100_mean']:.4f} ± {metrics['last_100_std']:.4f}")
        report.append(f"  平均最大回报: {metrics['max_return']:.2f}")
        report.append(f"  平均最小回报: {metrics['min_return']:.2f}")
        
        # 每个seed的详细信息
        report.append(f"  各Seed详情:")
        for seed_idx, (seed, returns) in enumerate(zip(result['seeds'], result['returns_per_seed'])):
            if len(returns) > 0:
                seed_metrics = calculate_metrics(returns)
                time_min = result['training_times'][seed_idx] / 60
                report.append(f"    Seed {seed}: 回报={seed_metrics['last_100_mean']:.2f}, "
                            f"最大={seed_metrics['max_return']:.2f}, "
                            f"时间={time_min:.1f}分钟")
            else:
                report.append(f"    Seed {seed}: 无有效数据")
        report.append("")
    
    report.append("="*70)
    
    report_text = "\n".join(report)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\nReport saved to: {save_path}")
    
    return report_text


def save_results_json_multi_seeds(all_results, save_path, episodes, seeds):
    """保存JSON格式结果（多seeds）"""
    data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'environment': 'Platform-v0',
        'episodes': episodes,
        'num_seeds': len(seeds),
        'seeds': seeds,
        'algorithms': {}
    }
    
    for algo_name, result in all_results.items():
        metrics = calculate_aggregated_metrics(result['returns_per_seed'])
        
        seed_metrics = []
        for seed_idx, (seed, returns) in enumerate(zip(result['seeds'], result['returns_per_seed'])):
            if len(returns) > 0:
                sm = calculate_metrics(returns)
                sm['seed'] = seed
                sm['training_time'] = result['training_times'][seed_idx]
                seed_metrics.append(sm)
        
        data['algorithms'][algo_name] = {
            'aggregated_metrics': metrics,
            'mean_training_time': np.mean(result['training_times']),
            'std_training_time': np.std(result['training_times']),
            'num_successful_seeds': len([r for r in result['returns_per_seed'] if len(r) > 0]),
            'seed_metrics': seed_metrics
        }
    
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"JSON results saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Compare Platform environment algorithms (multi-seeds)')
    parser.add_argument('--episodes', type=int, default=20000, help='Number of training episodes')
    parser.add_argument('--num-seeds', type=int, default=1, help='Number of seeds to run for each algorithm')
    parser.add_argument('--base-seed', type=int, default=0, help='Base random seed')
    parser.add_argument('--save-dir', type=str, default='results/platform_comparison', help='Save directory')
    parser.add_argument('--algorithms', type=str, nargs='+', default=None,
                       help='Algorithms to compare (default: all). Options: LIRL, PADDPG, PDQN, QPAMDP')
    parser.add_argument('--python-path', type=str, default=None, help='Python interpreter path')
    parser.add_argument('--smooth-window', type=int, default=100, help='Smoothing window for plotting')
    
    args = parser.parse_args()
    
    # 生成seeds列表
    seeds = list(range(args.base_seed, args.base_seed + args.num_seeds))
    
    # 创建保存目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.save_dir, f'compare_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)
    
    print("="*70)
    print("Platform环境算法对比实验 (多Seeds)")
    print("="*70)
    print(f"训练Episodes: {args.episodes}")
    print(f"Seeds数量: {args.num_seeds}")
    print(f"Seeds列表: {seeds}")
    print(f"保存目录: {save_dir}")
    print("="*70)
    
    # 选择要运行的算法
    if args.algorithms:
        algorithms_to_run = {k: v for k, v in ALGORITHMS.items() if k in args.algorithms}
    else:
        algorithms_to_run = ALGORITHMS
    
    print(f"将运行以下算法: {list(algorithms_to_run.keys())}")
    print(f"每个算法将运行 {args.num_seeds} 次 (seeds: {seeds})")
    
    # 计算总任务数
    total_tasks = len(algorithms_to_run) * args.num_seeds
    print(f"总任务数: {total_tasks} (= {len(algorithms_to_run)} 算法 × {args.num_seeds} seeds)")
    
    # 初始化进度跟踪
    progress_info = {
        'completed': 0,
        'total': total_tasks,
        'elapsed_times': [],
        'start_time': time.time()
    }
    
    # 运行所有算法
    all_results = {}
    
    for algo_idx, (algo_name, algo_config) in enumerate(algorithms_to_run.items()):
        print(f"\n{'*'*70}")
        print(f"* 开始运行 {algo_name} ({args.num_seeds} seeds)")
        print(f"* 算法进度: {algo_idx+1}/{len(algorithms_to_run)}")
        if len(progress_info['elapsed_times']) > 0:
            total_elapsed = time.time() - progress_info['start_time']
            avg_per_task = np.mean(progress_info['elapsed_times'])
            remaining_tasks = total_tasks - progress_info['completed']
            est_remaining = avg_per_task * remaining_tasks
            print(f"* 已运行: {format_time(total_elapsed)} | 预估剩余: {format_time(est_remaining)}")
        print(f"{'*'*70}")
        sys.stdout.flush()
        
        result = run_algorithm_multi_seeds(
            algo_name, 
            algo_config, 
            args.episodes, 
            seeds, 
            save_dir,
            args.python_path,
            progress_info
        )
        all_results[algo_name] = result
        
        # 更新已完成数
        progress_info['completed'] += args.num_seeds
        
        # 保存每个算法的结果
        valid_returns = [r for r in result['returns_per_seed'] if len(r) > 0]
        if len(valid_returns) > 0:
            for seed_idx, returns in enumerate(result['returns_per_seed']):
                if len(returns) > 0:
                    np.save(
                        os.path.join(save_dir, f'{algo_name}_seed{seeds[seed_idx]}_returns.npy'),
                        returns
                    )
    
    # 计算总耗时
    total_time = time.time() - progress_info['start_time']
    
    # 生成对比图
    print("\n" + "="*70)
    print("生成对比图...")
    print("="*70)
    
    plot_comparison_multi_seeds(
        all_results, 
        os.path.join(save_dir, 'comparison.png'),
        args.smooth_window
    )
    plot_learning_curves_all_seeds(
        all_results, 
        os.path.join(save_dir, 'learning_curves.png'),
        args.smooth_window
    )
    plot_convergence_comparison(
        all_results, 
        os.path.join(save_dir, 'convergence_comparison.png'),
        args.smooth_window
    )
    
    # 生成报告
    print("\n生成报告...")
    generate_report_multi_seeds(all_results, os.path.join(save_dir, 'report.txt'), args.episodes, seeds)
    save_results_json_multi_seeds(all_results, os.path.join(save_dir, 'results.json'), args.episodes, seeds)
    
    print("\n" + "="*70)
    print("对比实验完成!")
    print("="*70)
    print(f"总耗时: {format_time(total_time)}")
    print(f"完成任务: {progress_info['completed']}/{total_tasks}")
    print(f"平均每任务: {format_time(np.mean(progress_info['elapsed_times'])) if progress_info['elapsed_times'] else 'N/A'}")
    print(f"结果保存在: {save_dir}")
    print("生成的文件:")
    print(f"  - comparison.png: 综合对比图")
    print(f"  - learning_curves.png: 各算法学习曲线")
    print(f"  - convergence_comparison.png: 收敛曲线对比")
    print(f"  - report.txt: 文本报告")
    print(f"  - results.json: JSON格式结果")
    print("="*70)
    
    return all_results


if __name__ == '__main__':
    main()
