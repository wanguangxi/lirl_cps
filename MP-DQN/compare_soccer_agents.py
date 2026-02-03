#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Soccer环境算法对比程序
====================
对比四种算法在Soccer环境上的表现:
1. LIRL - Learning with Integer and Real-valued Actions
2. PADDPG - Parameterised Action DDPG
3. PDQN - Parameterised DQN
4. QPAMDP - Q-PAMDP
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
        'script': 'run_soccer_lirl.py',
        'title': 'LIRL',
        'color': '#e74c3c',  # 红色
        'marker': 'o'
    },
    'PADDPG': {
        'script': 'run_soccer_paddpg.py',
        'title': 'PADDPG',
        'color': '#3498db',  # 蓝色
        'marker': 's'
    },
    'PDQN': {
        'script': 'run_soccer_pdqn.py',
        'title': 'PDQN',
        'color': '#2ecc71',  # 绿色
        'marker': '^'
    },
    'QPAMDP': {
        'script': 'run_soccer_qpamdp.py',
        'title': 'QPAMDP',
        'color': '#9b59b6',  # 紫色
        'marker': 'd'
    }
}


def run_algorithm(algo_name, algo_config, episodes, seed, save_dir, python_path=None):
    """
    运行单个算法
    
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
        script_path,
        '--episodes', str(episodes),
        '--seed', str(seed),
        '--title', algo_config['title'],
        '--evaluation-episodes', '100'  # 减少评估episodes以加快速度
    ]
    
    # 为非QPAMDP算法添加save-dir参数
    if algo_name != 'QPAMDP':
        cmd.extend(['--save-dir', save_dir])
    
    print(f"\n{'='*60}")
    print(f"Running {algo_name}...")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Python 3.6兼容：使用stdout/stderr参数代替capture_output
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,  # Python 3.6兼容：代替text=True
            timeout=3600 * 4  # 4小时超时（Soccer环境可能需要更长时间）
        )
        
        training_time = time.time() - start_time
        
        # 打印输出
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        # 加载结果 - 尝试多个可能的路径
        # Soccer环境保存格式可能是 (returns, timesteps, goals) 或仅 returns
        possible_paths = [
            os.path.join(save_dir, algo_config['title'], f"{algo_config['title']}{seed}.npy"),
            os.path.join(save_dir, algo_config['title'], str(seed), f"{algo_config['title']}{seed}.npy"),
            # QPAMDP的默认保存路径
            os.path.join("results", "soccer", algo_config['title'], f"{algo_config['title']}{seed}.npy"),
        ]
        
        results_file = None
        for path in possible_paths:
            if os.path.exists(path):
                results_file = path
                break
        
        returns = np.array([])
        timesteps_arr = np.array([])
        goals_arr = np.array([])
        
        if results_file and os.path.exists(results_file):
            data = np.load(results_file)
            print(f"Loaded results from: {results_file}")
            print(f"Data shape: {data.shape}")
            
            # 检查数据格式
            if len(data.shape) == 1:
                # 仅returns
                returns = data
            elif len(data.shape) == 2 and data.shape[1] >= 3:
                # (returns, timesteps, goals)
                returns = data[:, 0]
                timesteps_arr = data[:, 1]
                goals_arr = data[:, 2]
            else:
                returns = data.flatten()
        else:
            print(f"Warning: Results file not found. Tried: {possible_paths}")
        
        return {
            'success': result.returncode == 0,
            'returns': returns,
            'timesteps': timesteps_arr,
            'goals': goals_arr,
            'training_time': training_time,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
        
    except subprocess.TimeoutExpired:
        print(f"ERROR: {algo_name} timed out after 4 hours")
        return {
            'success': False,
            'returns': np.array([]),
            'timesteps': np.array([]),
            'goals': np.array([]),
            'training_time': 14400,
            'stdout': '',
            'stderr': 'Timeout'
        }
    except Exception as e:
        print(f"ERROR running {algo_name}: {e}")
        return {
            'success': False,
            'returns': np.array([]),
            'timesteps': np.array([]),
            'goals': np.array([]),
            'training_time': time.time() - start_time,
            'stdout': '',
            'stderr': str(e)
        }


def smooth_curve(data, window=100):
    """
    平滑曲线
    """
    if len(data) < window:
        return data
    smoothed = np.convolve(data, np.ones(window)/window, mode='valid')
    return smoothed


def calculate_metrics(returns, timesteps=None, goals=None):
    """
    计算评估指标
    
    Args:
        returns: 回报数组
        timesteps: 时间步数组
        goals: 进球数组
    
    Returns:
        dict: 指标字典
    """
    if len(returns) == 0:
        return {
            'mean_return': 0,
            'std_return': 0,
            'goal_rate': 0,
            'last_100_mean': 0,
            'last_100_goal_rate': 0,
            'max_return': 0,
            'min_return': 0,
            'mean_timesteps': 0,
            'mean_timesteps_per_goal': 0
        }
    
    metrics = {
        'mean_return': np.mean(returns),
        'std_return': np.std(returns),
        'last_100_mean': np.mean(returns[-100:]) if len(returns) >= 100 else np.mean(returns),
        'max_return': np.max(returns),
        'min_return': np.min(returns)
    }
    
    # 计算进球率
    if goals is not None and len(goals) > 0:
        metrics['goal_rate'] = np.mean(goals)
        last_100_goals = goals[-100:] if len(goals) >= 100 else goals
        metrics['last_100_goal_rate'] = np.mean(last_100_goals)
    else:
        metrics['goal_rate'] = 0
        metrics['last_100_goal_rate'] = 0
    
    # 计算平均时间步
    if timesteps is not None and len(timesteps) > 0:
        metrics['mean_timesteps'] = np.mean(timesteps)
        # 计算进球的平均时间步
        if goals is not None and len(goals) > 0:
            goal_mask = np.array(goals) == 1
            if np.sum(goal_mask) > 0:
                metrics['mean_timesteps_per_goal'] = np.mean(np.array(timesteps)[goal_mask])
            else:
                metrics['mean_timesteps_per_goal'] = 0
        else:
            metrics['mean_timesteps_per_goal'] = 0
    else:
        metrics['mean_timesteps'] = 0
        metrics['mean_timesteps_per_goal'] = 0
    
    return metrics


def plot_comparison(all_results, save_path):
    """
    绘制对比图
    
    Args:
        all_results: 所有算法结果
        save_path: 图片保存路径
    """
    # 过滤出有效结果
    valid_results = {k: v for k, v in all_results.items() if len(v['returns']) > 0}
    
    if len(valid_results) == 0:
        print("Warning: No valid results to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 学习曲线对比 (平滑后)
    ax1 = axes[0, 0]
    for algo_name, result in valid_results.items():
        config = ALGORITHMS[algo_name]
        smoothed = smooth_curve(result['returns'], window=100)
        ax1.plot(smoothed, label=algo_name, color=config['color'], alpha=0.8)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Return (smoothed)')
    ax1.set_title('Learning Curves Comparison')
    if len(valid_results) > 0:
        ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 2. 最终100 episodes平均回报对比
    ax2 = axes[0, 1]
    names = []
    means = []
    stds = []
    colors = []
    for algo_name, result in valid_results.items():
        names.append(algo_name)
        last_100 = result['returns'][-100:] if len(result['returns']) >= 100 else result['returns']
        means.append(np.mean(last_100))
        stds.append(np.std(last_100))
        colors.append(ALGORITHMS[algo_name]['color'])
    
    if len(names) > 0:
        x = np.arange(len(names))
        bars = ax2.bar(x, means, yerr=stds, color=colors, capsize=5, alpha=0.8)
        ax2.set_xticks(x)
        ax2.set_xticklabels(names)
        ax2.set_ylabel('Average Return')
        ax2.set_title('Final 100 Episodes Average Return')
        ax2.grid(axis='y', alpha=0.3)
        
        # 添加数值标签
        for bar, mean in zip(bars, means):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{mean:.2f}', ha='center', va='bottom', fontsize=9)
    
    # 3. 进球率对比
    ax3 = axes[1, 0]
    goal_rates = []
    for algo_name, result in valid_results.items():
        if len(result['goals']) > 0:
            last_100_goals = result['goals'][-100:] if len(result['goals']) >= 100 else result['goals']
            gr = np.mean(last_100_goals) * 100
        else:
            gr = 0
        goal_rates.append(gr)
    
    if len(names) > 0:
        bars = ax3.bar(x, goal_rates, color=colors, alpha=0.8)
        ax3.set_xticks(x)
        ax3.set_xticklabels(names)
        ax3.set_ylabel('Goal Rate (%)')
        ax3.set_title('Final 100 Episodes Goal Rate')
        ax3.set_ylim(0, 100)
        ax3.grid(axis='y', alpha=0.3)
        
        for bar, gr in zip(bars, goal_rates):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{gr:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # 4. 训练时间对比
    ax4 = axes[1, 1]
    times = [result['training_time'] / 60 for result in valid_results.values()]  # 转换为分钟
    
    if len(names) > 0:
        bars = ax4.bar(x, times, color=colors, alpha=0.8)
        ax4.set_xticks(x)
        ax4.set_xticklabels(names)
        ax4.set_ylabel('Training Time (minutes)')
        ax4.set_title('Training Time Comparison')
        ax4.grid(axis='y', alpha=0.3)
        
        for bar, t in zip(bars, times):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{t:.1f}m', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved to: {save_path}")


def plot_learning_curves_detail(all_results, save_path):
    """
    绘制详细学习曲线
    """
    # 过滤有效结果
    valid_results = {k: v for k, v in all_results.items() if len(v['returns']) > 0}
    
    if len(valid_results) == 0:
        print("Warning: No valid results for detailed learning curves")
        return
    
    # 根据有效结果数量确定子图布局
    n_valid = len(valid_results)
    if n_valid == 1:
        fig, axes = plt.subplots(1, 1, figsize=(7, 5))
        axes = np.array([[axes]])
    elif n_valid == 2:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes = axes.reshape(1, 2)
    elif n_valid <= 4:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    else:
        rows = (n_valid + 1) // 2
        fig, axes = plt.subplots(rows, 2, figsize=(14, 5*rows))
    
    axes = np.atleast_2d(axes)
    
    for idx, (algo_name, result) in enumerate(valid_results.items()):
        row, col = idx // 2, idx % 2
        if row < axes.shape[0] and col < axes.shape[1]:
            ax = axes[row, col]
        else:
            continue
            
        config = ALGORITHMS[algo_name]
        # 原始数据
        ax.plot(result['returns'], alpha=0.3, color=config['color'], label='Raw')
        # 平滑数据
        smoothed = smooth_curve(result['returns'], window=100)
        ax.plot(np.arange(len(smoothed)) + 50, smoothed, 
               color=config['color'], linewidth=2, label='Smoothed (w=100)')
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Return')
        ax.set_title(f'{algo_name} Learning Curve')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 添加统计信息
        metrics = calculate_metrics(result['returns'], result['timesteps'], result['goals'])
        info_text = f"Mean: {metrics['mean_return']:.2f}\n"
        info_text += f"Goal Rate: {metrics['goal_rate']*100:.1f}%\n"
        info_text += f"Last 100: {metrics['last_100_mean']:.2f}"
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
               verticalalignment='top', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 隐藏多余的子图
    total_axes = axes.shape[0] * axes.shape[1]
    for idx in range(n_valid, total_axes):
        row, col = idx // 2, idx % 2
        if row < axes.shape[0] and col < axes.shape[1]:
            axes[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Detailed learning curves saved to: {save_path}")


def plot_goal_rate_curves(all_results, save_path):
    """
    绘制进球率学习曲线
    """
    valid_results = {k: v for k, v in all_results.items() if len(v['goals']) > 0}
    
    if len(valid_results) == 0:
        print("Warning: No goal data available for plotting")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for algo_name, result in valid_results.items():
        config = ALGORITHMS[algo_name]
        # 计算滑动窗口进球率
        goals = result['goals']
        if len(goals) >= 100:
            goal_rate = np.convolve(goals, np.ones(100)/100, mode='valid') * 100
            ax.plot(np.arange(len(goal_rate)) + 50, goal_rate, 
                   label=algo_name, color=config['color'], alpha=0.8)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Goal Rate (%, smoothed w=100)')
    ax.set_title('Goal Rate Learning Curves')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Goal rate curves saved to: {save_path}")


def generate_report(all_results, save_path, episodes, seed):
    """
    生成文本报告
    """
    report = []
    report.append("="*80)
    report.append("Soccer环境算法对比报告")
    report.append("="*80)
    report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"训练Episodes: {episodes}")
    report.append(f"随机种子: {seed}")
    report.append("")
    
    # 汇总表格
    report.append("-"*80)
    report.append(f"{'算法':<12} {'平均回报':<12} {'进球率':<12} {'最后100回报':<14} {'最后100进球率':<14} {'训练时间':<12}")
    report.append("-"*80)
    
    for algo_name, result in all_results.items():
        metrics = calculate_metrics(result['returns'], result['timesteps'], result['goals'])
        time_str = f"{result['training_time']/60:.1f}分钟"
        report.append(
            f"{algo_name:<12} "
            f"{metrics['mean_return']:<12.2f} "
            f"{metrics['goal_rate']*100:<12.1f}% "
            f"{metrics['last_100_mean']:<14.2f} "
            f"{metrics['last_100_goal_rate']*100:<14.1f}% "
            f"{time_str:<12}"
        )
    
    report.append("-"*80)
    report.append("")
    
    # 详细指标
    report.append("详细指标:")
    report.append("")
    
    for algo_name, result in all_results.items():
        report.append(f"【{algo_name}】")
        metrics = calculate_metrics(result['returns'], result['timesteps'], result['goals'])
        report.append(f"  总体平均回报: {metrics['mean_return']:.4f} ± {metrics['std_return']:.4f}")
        report.append(f"  总体进球率: {metrics['goal_rate']*100:.2f}%")
        report.append(f"  最后100回合平均回报: {metrics['last_100_mean']:.4f}")
        report.append(f"  最后100回合进球率: {metrics['last_100_goal_rate']*100:.2f}%")
        report.append(f"  最大回报: {metrics['max_return']:.2f}")
        report.append(f"  最小回报: {metrics['min_return']:.2f}")
        report.append(f"  平均时间步: {metrics['mean_timesteps']:.2f}")
        report.append(f"  进球平均时间步: {metrics['mean_timesteps_per_goal']:.2f}")
        report.append(f"  训练时间: {result['training_time']:.2f}秒 ({result['training_time']/60:.2f}分钟)")
        report.append("")
    
    report.append("="*80)
    
    # 保存报告
    report_text = "\n".join(report)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\nReport saved to: {save_path}")
    
    return report_text


def save_results_json(all_results, save_path, episodes, seed):
    """
    保存JSON格式结果
    """
    data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'episodes': episodes,
        'seed': seed,
        'environment': 'SoccerScoreGoal-v0',
        'algorithms': {}
    }
    
    for algo_name, result in all_results.items():
        metrics = calculate_metrics(result['returns'], result['timesteps'], result['goals'])
        data['algorithms'][algo_name] = {
            'metrics': metrics,
            'training_time': result['training_time'],
            'success': result['success'],
            'num_episodes': len(result['returns'])
        }
    
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"JSON results saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Compare Soccer environment algorithms')
    parser.add_argument('--episodes', type=int, default=20000, help='Number of training episodes')
    parser.add_argument('--seed', type=int, default=3279, help='Random seed')
    parser.add_argument('--save-dir', type=str, default='results/soccer_comparison', help='Save directory')
    parser.add_argument('--algorithms', type=str, nargs='+', default=None,
                       help='Algorithms to compare (default: all). Options: LIRL, PADDPG, PDQN, QPAMDP')
    parser.add_argument('--python-path', type=str, default=None, help='Python interpreter path')
    
    args = parser.parse_args()
    
    # 创建保存目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.save_dir, f'compare_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)
    
    print("="*70)
    print("Soccer环境算法对比实验")
    print("="*70)
    print(f"训练Episodes: {args.episodes}")
    print(f"随机种子: {args.seed}")
    print(f"保存目录: {save_dir}")
    print("="*70)
    
    # 选择要运行的算法
    if args.algorithms:
        algorithms_to_run = {k: v for k, v in ALGORITHMS.items() if k in args.algorithms}
    else:
        algorithms_to_run = ALGORITHMS
    
    print(f"将运行以下算法: {list(algorithms_to_run.keys())}")
    
    # 运行所有算法
    all_results = {}
    
    for algo_name, algo_config in algorithms_to_run.items():
        result = run_algorithm(
            algo_name, 
            algo_config, 
            args.episodes, 
            args.seed, 
            save_dir,
            args.python_path
        )
        all_results[algo_name] = result
        
        # 保存中间结果
        if len(result['returns']) > 0:
            np.save(
                os.path.join(save_dir, f'{algo_name}_returns.npy'),
                result['returns']
            )
            if len(result['goals']) > 0:
                np.save(
                    os.path.join(save_dir, f'{algo_name}_goals.npy'),
                    result['goals']
                )
    
    # 生成对比图
    print("\n生成对比图...")
    plot_comparison(all_results, os.path.join(save_dir, 'comparison.png'))
    plot_learning_curves_detail(all_results, os.path.join(save_dir, 'learning_curves.png'))
    plot_goal_rate_curves(all_results, os.path.join(save_dir, 'goal_rate_curves.png'))
    
    # 生成报告
    print("\n生成报告...")
    generate_report(all_results, os.path.join(save_dir, 'report.txt'), args.episodes, args.seed)
    save_results_json(all_results, os.path.join(save_dir, 'results.json'), args.episodes, args.seed)
    
    print("\n" + "="*70)
    print("对比实验完成!")
    print(f"结果保存在: {save_dir}")
    print("="*70)
    
    return all_results


if __name__ == '__main__':
    main()

