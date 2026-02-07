"""
算法对比实验脚本

对比 /home/one/LIRL-CPS-main/Protein_Crystallization_Screening/alg 目录下的算法：
1. LIRL (可微分投影)
2. PDQN (惩罚学习)
4. HPPO (惩罚学习)
6. CPO (约束策略优化)
7. LPPO (拉格朗日 PPO)

指标对比：
- 训练奖励曲线
- 约束违反率 (CVR)
- 收敛速度
- 最终性能
"""

import os
import sys
import json
import datetime
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Add algorithm path
sys.path.append(os.path.join(os.path.dirname(__file__), '../alg'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../env'))

import torch

# =======================
# CONFIGURATION
# =======================

# 统一的网络结构配置 (按相近参数量对比)
# 注意：LIRL 使用 Encoder-Decoder 架构，需要更大的隐藏层来达到相似的表达能力
UNIFIED_NETWORK_CONFIG = {
    # 通用网络参数 (HPPO/CPO/LPPO/PDQN)
    'hidden_dim': 256,          # 主隐藏层维度
    'hidden_dim1': 256,         # 第一隐藏层
    'hidden_dim2': 128,         # 第二隐藏层
    
    # LIRL 特定参数 - 保持原始配置以匹配其架构复杂性
    'latent_dim': 32,           # 潜在空间维度
    'lirl_hidden_dim': 512,     # LIRL 隐藏层 (Encoder-Decoder 需要更大)
    'num_layers': 3,            # 网络层数
    
    # PDQN 特定参数
    'q_hidden_dim1': 256,
    'q_hidden_dim2': 128,
    'param_hidden_dim1': 256,
    'param_hidden_dim2': 128,
}

COMPARE_CONFIG = {
    'num_episodes': 500,        # 每个算法训练的回合数
    'num_seeds': 10,             # 每个算法运行的种子数
    'seeds': [3047,294,714,1092,1386,2856,42,114514,2025,1993],    # 随机种子
    'batch_size_env': 2,
    'horizon': 25,
    'save_results': True,
    'plot_results': True,
}

# 快速测试配置
QUICK_TEST_CONFIG = {
    'num_episodes': 50,         # 少量回合用于测试
    'num_seeds': 1,             # 单个种子
    'seeds': [42],
    'batch_size_env': 2,
    'horizon': 25,
    'save_results': True,
    'plot_results': True,
}

# =======================
# ALGORITHM RUNNERS
# =======================

def run_lirl(config, seed):
    """运行 LIRL 算法"""
    from lirl import main as lirl_main, CONFIG as LIRL_CONFIG
    
    # 修改配置
    lirl_config = LIRL_CONFIG.copy()
    lirl_config['num_of_episodes'] = config['num_episodes']
    lirl_config['seed'] = seed
    lirl_config['batch_size_env'] = config['batch_size_env']
    lirl_config['horizon'] = config['horizon']
    lirl_config['save_models'] = False
    lirl_config['plot_training_curve'] = False
    lirl_config['print_interval'] = 50
    
    # 应用统一网络配置 - LIRL 使用其专用配置
    lirl_config['hidden_dim'] = UNIFIED_NETWORK_CONFIG['lirl_hidden_dim']  # LIRL 需要更大网络
    lirl_config['latent_dim'] = UNIFIED_NETWORK_CONFIG['latent_dim']
    lirl_config['num_layers'] = UNIFIED_NETWORK_CONFIG['num_layers']
    
    print(f"\n{'='*50}")
    print(f"Running LIRL with seed {seed}")
    print(f"Network: hidden={lirl_config['hidden_dim']}, latent={lirl_config['latent_dim']}, layers={lirl_config['num_layers']}")
    print(f"{'='*50}")
    
    # lirl.py returns: score_record, best_quality_record, {'policy': policy, 'q_net': q_net}
    scores, best_qualities, _ = lirl_main(lirl_config)
    
    return {
        'scores': scores,
        'best_qualities': best_qualities,
        'violations': 0,  # LIRL uses projection, so no violations
        'name': 'LIRL',
        'seed': seed
    }



def run_pdqn(config, seed):
    """运行 PDQN 算法"""
    import pdqn
    from pdqn import main as pdqn_main
    
    # PDQN 网络使用全局 CONFIG，需要直接修改
    pdqn.CONFIG['q_hidden_dim1'] = UNIFIED_NETWORK_CONFIG['q_hidden_dim1']
    pdqn.CONFIG['q_hidden_dim2'] = UNIFIED_NETWORK_CONFIG['q_hidden_dim2']
    pdqn.CONFIG['param_hidden_dim1'] = UNIFIED_NETWORK_CONFIG['param_hidden_dim1']
    pdqn.CONFIG['param_hidden_dim2'] = UNIFIED_NETWORK_CONFIG['param_hidden_dim2']
    
    pdqn_config = pdqn.CONFIG.copy()
    pdqn_config['num_of_episodes'] = config['num_episodes']
    pdqn_config['seed'] = seed
    pdqn_config['batch_size_env'] = config['batch_size_env']
    pdqn_config['horizon'] = config['horizon']
    pdqn_config['save_models'] = False
    pdqn_config['plot_training_curve'] = False
    pdqn_config['print_interval'] = 50
    
    print(f"\n{'='*50}")
    print(f"Running PDQN with seed {seed}")
    print(f"Network: q_hidden=[{pdqn_config['q_hidden_dim1']},{pdqn_config['q_hidden_dim2']}], param_hidden=[{pdqn_config['param_hidden_dim1']},{pdqn_config['param_hidden_dim2']}]")
    print(f"{'='*50}")
    
    # pdqn.py returns: score_record, best_quality_record, agent, total_violations
    scores, best_qualities, _, violations = pdqn_main(pdqn_config)
    
    return {
        'scores': scores,
        'best_qualities': best_qualities,
        'violations': violations,
        'name': 'PDQN',
        'seed': seed
    }





def run_hppo(config, seed):
    """运行 HPPO 算法"""
    import hppo
    from hppo import main as hppo_main
    
    # HPPO 网络使用全局 CONFIG，需要直接修改
    hppo.CONFIG['hidden_dim1'] = UNIFIED_NETWORK_CONFIG['hidden_dim1']
    hppo.CONFIG['hidden_dim2'] = UNIFIED_NETWORK_CONFIG['hidden_dim2']
    
    hppo_config = hppo.CONFIG.copy()
    hppo_config['num_of_episodes'] = config['num_episodes']
    hppo_config['seed'] = seed
    hppo_config['batch_size_env'] = config['batch_size_env']
    hppo_config['horizon'] = config['horizon']
    hppo_config['save_models'] = False
    hppo_config['plot_training_curve'] = False
    hppo_config['print_interval'] = 50
    
    print(f"\n{'='*50}")
    print(f"Running HPPO with seed {seed}")
    print(f"Network: hidden=[{hppo_config['hidden_dim1']},{hppo_config['hidden_dim2']}]")
    print(f"{'='*50}")
    
    # hppo.py returns: score_record, best_quality_record, agent, total_violations
    scores, best_qualities, _, violations = hppo_main(hppo_config)
    
    return {
        'scores': scores,
        'best_qualities': best_qualities,
        'violations': violations,
        'name': 'HPPO',
        'seed': seed
    }




def run_cpo(config, seed):
    """运行 CPO 算法"""
    from cpo import main as cpo_main, CONFIG as CPO_CONFIG
    
    cpo_config = CPO_CONFIG.copy()
    cpo_config['num_of_episodes'] = config['num_episodes']
    cpo_config['seed'] = seed
    cpo_config['batch_size_env'] = config['batch_size_env']
    cpo_config['horizon'] = config['horizon']
    cpo_config['save_models'] = False
    cpo_config['plot_training_curve'] = False
    cpo_config['print_interval'] = 50
    
    # 应用统一网络配置
    cpo_config['hidden_dim1'] = UNIFIED_NETWORK_CONFIG['hidden_dim1']
    cpo_config['hidden_dim2'] = UNIFIED_NETWORK_CONFIG['hidden_dim2']
    
    print(f"\n{'='*50}")
    print(f"Running CPO with seed {seed}")
    print(f"Network: hidden=[{cpo_config['hidden_dim1']},{cpo_config['hidden_dim2']}]")
    print(f"{'='*50}")
    
    # cpo.py returns: score_record, best_quality_record, cost_record, agent, total_violations
    scores, best_qualities, _, _, violations = cpo_main(cpo_config)
    
    return {
        'scores': scores,
        'best_qualities': best_qualities,
        'violations': violations,
        'name': 'CPO',
        'seed': seed
    }


def run_lppo(config, seed):
    """运行 LPPO 算法"""
    from lppo import main as lppo_main, CONFIG as LPPO_CONFIG
    
    lppo_config = LPPO_CONFIG.copy()
    lppo_config['num_of_episodes'] = config['num_episodes']
    lppo_config['seed'] = seed
    lppo_config['batch_size_env'] = config['batch_size_env']
    lppo_config['horizon'] = config['horizon']
    lppo_config['save_models'] = False
    lppo_config['plot_training_curve'] = False
    lppo_config['print_interval'] = 50
    
    # 应用统一网络配置
    lppo_config['hidden_dim1'] = UNIFIED_NETWORK_CONFIG['hidden_dim1']
    lppo_config['hidden_dim2'] = UNIFIED_NETWORK_CONFIG['hidden_dim2']
    
    print(f"\n{'='*50}")
    print(f"Running LPPO with seed {seed}")
    print(f"Network: hidden=[{lppo_config['hidden_dim1']},{lppo_config['hidden_dim2']}]")
    print(f"{'='*50}")
    
    # lppo.py returns: score_record, best_quality_record, lambda_record, agent, total_violations
    scores, best_qualities, _, _, violations = lppo_main(lppo_config)
    
    return {
        'scores': scores,
        'best_qualities': best_qualities,
        'violations': violations,
        'name': 'LPPO',
        'seed': seed
    }


# =======================
# PLOTTING FUNCTIONS
# =======================

def plot_comparison(all_results, save_dir):
    """绘制算法对比图"""
    
    # 整理数据
    algorithms = list(all_results.keys())
    colors = {
        'LIRL': '#1f77b4',
        'PDQN': '#ff7f0e',
        'HPPO': '#d62728',
        'CPO': '#8c564b',
        'LPPO': '#e377c2'
    }
    
    # 图1: 训练奖励曲线对比
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 子图1: 平均奖励曲线
    ax1 = axes[0, 0]
    for alg_name in algorithms:
        scores_list = [r['scores'] for r in all_results[alg_name]]
        # 对齐长度
        min_len = min(len(s) for s in scores_list)
        scores_array = np.array([s[:min_len] for s in scores_list])
        mean_scores = np.mean(scores_array, axis=0)
        std_scores = np.std(scores_array, axis=0)
        
        episodes = np.arange(len(mean_scores))
        ax1.plot(episodes, mean_scores, label=alg_name, color=colors.get(alg_name, 'gray'), linewidth=2)
        ax1.fill_between(episodes, mean_scores - std_scores, mean_scores + std_scores, 
                        alpha=0.2, color=colors.get(alg_name, 'gray'))
    
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Reward', fontsize=12)
    ax1.set_title('Training Reward Curves (Mean ± Std)', fontsize=14)
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 子图2: 平滑后的奖励曲线
    ax2 = axes[0, 1]
    window = 20
    for alg_name in algorithms:
        scores_list = [r['scores'] for r in all_results[alg_name]]
        min_len = min(len(s) for s in scores_list)
        scores_array = np.array([s[:min_len] for s in scores_list])
        mean_scores = np.mean(scores_array, axis=0)
        
        if len(mean_scores) > window:
            smoothed = np.convolve(mean_scores, np.ones(window)/window, mode='valid')
            ax2.plot(range(window-1, len(mean_scores)), smoothed, 
                    label=alg_name, color=colors.get(alg_name, 'gray'), linewidth=2)
    
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Reward (Smoothed)', fontsize=12)
    ax2.set_title(f'Smoothed Reward Curves (Window={window})', fontsize=14)
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 子图3: 最终性能柱状图
    ax3 = axes[1, 0]
    final_scores = []
    final_stds = []
    alg_names = []
    
    for alg_name in algorithms:
        scores_list = [r['scores'] for r in all_results[alg_name]]
        # 取最后50个episode的平均
        final_means = [np.mean(s[-50:]) for s in scores_list]
        final_scores.append(np.mean(final_means))
        final_stds.append(np.std(final_means))
        alg_names.append(alg_name)
    
    x = np.arange(len(alg_names))
    bars = ax3.bar(x, final_scores, yerr=final_stds, capsize=5,
                   color=[colors.get(n, 'gray') for n in alg_names], alpha=0.8)
    ax3.set_xticks(x)
    ax3.set_xticklabels(alg_names, rotation=45, ha='right')
    ax3.set_ylabel('Final Reward (Last 50 Eps)', fontsize=12)
    ax3.set_title('Final Performance Comparison', fontsize=14)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 在柱子上添加数值
    for bar, score in zip(bars, final_scores):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{score:.2f}', ha='center', va='bottom', fontsize=10)
    
    # 子图4: 约束违反率柱状图
    ax4 = axes[1, 1]
    cvr_scores = []
    cvr_stds = []
    cvr_names = []
    
    for alg_name in algorithms:
        results = all_results[alg_name]
        if 'violations' in results[0]:
            violations_list = [r.get('violations', 0) for r in results]
            # 计算 CVR
            total_steps = COMPARE_CONFIG['num_episodes'] * COMPARE_CONFIG['horizon'] * COMPARE_CONFIG['batch_size_env']
            cvr_list = [v / total_steps for v in violations_list]
            cvr_scores.append(np.mean(cvr_list))
            cvr_stds.append(np.std(cvr_list))
            cvr_names.append(alg_name)
        else:
            cvr_scores.append(0)
            cvr_stds.append(0)
            cvr_names.append(alg_name)
    
    x = np.arange(len(cvr_names))
    bars = ax4.bar(x, cvr_scores, yerr=cvr_stds, capsize=5,
                   color=[colors.get(n, 'gray') for n in cvr_names], alpha=0.8)
    ax4.set_xticks(x)
    ax4.set_xticklabels(cvr_names, rotation=45, ha='right')
    ax4.set_ylabel('Constraint Violation Rate', fontsize=12)
    ax4.set_title('Constraint Violation Rate Comparison', fontsize=14)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 在柱子上添加数值
    for bar, score in zip(bars, cvr_scores):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{score:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'algorithm_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved to: {os.path.join(save_dir, 'algorithm_comparison.png')}")
    
    # 图2: 收敛速度分析
    fig2, ax = plt.subplots(figsize=(10, 6))
    
    # 计算达到特定阈值的回合数
    threshold = 2.0  # 奖励阈值
    convergence_episodes = {}
    
    for alg_name in algorithms:
        scores_list = [r['scores'] for r in all_results[alg_name]]
        conv_eps = []
        for scores in scores_list:
            # 找到首次达到阈值的回合
            smoothed = np.convolve(scores, np.ones(10)/10, mode='valid') if len(scores) > 10 else scores
            reached = np.where(smoothed >= threshold)[0]
            if len(reached) > 0:
                conv_eps.append(reached[0] + 10)  # 补偿平滑窗口
            else:
                conv_eps.append(len(scores))  # 未达到则用最大回合数
        convergence_episodes[alg_name] = conv_eps
    
    x = np.arange(len(algorithms))
    conv_means = [np.mean(convergence_episodes[alg]) for alg in algorithms]
    conv_stds = [np.std(convergence_episodes[alg]) for alg in algorithms]
    
    bars = ax.bar(x, conv_means, yerr=conv_stds, capsize=5,
                  color=[colors.get(n, 'gray') for n in algorithms], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, rotation=45, ha='right')
    ax.set_ylabel(f'Episodes to Reach Reward ≥ {threshold}', fontsize=12)
    ax.set_title('Convergence Speed Comparison', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'convergence_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Convergence plot saved to: {os.path.join(save_dir, 'convergence_comparison.png')}")
    
    # 图3: Best Quality 对比
    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
    
    # 子图1: Best Quality 曲线
    ax3_1 = axes3[0]
    for alg_name in algorithms:
        if 'best_qualities' in all_results[alg_name][0]:
            bq_list = [r['best_qualities'] for r in all_results[alg_name]]
            min_len = min(len(bq) for bq in bq_list)
            bq_array = np.array([bq[:min_len] for bq in bq_list])
            mean_bq = np.mean(bq_array, axis=0)
            std_bq = np.std(bq_array, axis=0)
            
            episodes = np.arange(len(mean_bq))
            ax3_1.plot(episodes, mean_bq, label=alg_name, color=colors.get(alg_name, 'gray'), linewidth=2)
            ax3_1.fill_between(episodes, mean_bq - std_bq, mean_bq + std_bq, 
                              alpha=0.2, color=colors.get(alg_name, 'gray'))
    
    ax3_1.set_xlabel('Episode', fontsize=12)
    ax3_1.set_ylabel('Best Quality', fontsize=12)
    ax3_1.set_title('Best Quality Curves (Mean ± Std)', fontsize=14)
    ax3_1.legend(loc='lower right', fontsize=10)
    ax3_1.grid(True, alpha=0.3)
    
    # 子图2: 最终 Best Quality 柱状图
    ax3_2 = axes3[1]
    final_bq = []
    final_bq_stds = []
    bq_names = []
    
    for alg_name in algorithms:
        if 'best_qualities' in all_results[alg_name][0]:
            bq_list = [r['best_qualities'] for r in all_results[alg_name]]
            # 取最后一个 episode 的 best_quality (即整个 episode 的最优)
            final_bqs = [bq[-1] if len(bq) > 0 else 0 for bq in bq_list]
            final_bq.append(np.mean(final_bqs))
            final_bq_stds.append(np.std(final_bqs))
            bq_names.append(alg_name)
    
    if final_bq:
        x = np.arange(len(bq_names))
        bars = ax3_2.bar(x, final_bq, yerr=final_bq_stds, capsize=5,
                        color=[colors.get(n, 'gray') for n in bq_names], alpha=0.8)
        ax3_2.set_xticks(x)
        ax3_2.set_xticklabels(bq_names, rotation=45, ha='right')
        ax3_2.set_ylabel('Final Best Quality', fontsize=12)
        ax3_2.set_title('Final Best Quality Comparison', fontsize=14)
        ax3_2.grid(True, alpha=0.3, axis='y')
        
        # 在柱子上添加数值
        for bar, score in zip(bars, final_bq):
            ax3_2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                      f'{score:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'best_quality_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Best quality plot saved to: {os.path.join(save_dir, 'best_quality_comparison.png')}")


def create_summary_table(all_results, save_dir):
    """创建结果汇总表格"""
    
    summary = []
    total_steps = COMPARE_CONFIG['num_episodes'] * COMPARE_CONFIG['horizon'] * COMPARE_CONFIG['batch_size_env']
    
    for alg_name in all_results.keys():
        results = all_results[alg_name]
        
        # 计算指标
        final_scores = [np.mean(r['scores'][-50:]) for r in results]
        all_scores = [r['scores'] for r in results]
        
        # 约束违反
        if 'violations' in results[0]:
            violations = [r.get('violations', 0) for r in results]
            cvr = np.mean([v / total_steps for v in violations])
        else:
            cvr = 0.0
        
        # 投影率
        if 'projections' in results[0]:
            projections = [r.get('projections', 0) for r in results]
            proj_rate = np.mean([p / total_steps for p in projections])
        else:
            proj_rate = None
        
        # Best Quality
        if 'best_qualities' in results[0]:
            bq_list = [r['best_qualities'] for r in results]
            final_bqs = [bq[-1] if len(bq) > 0 else 0 for bq in bq_list]
            best_quality_mean = np.mean(final_bqs)
            best_quality_std = np.std(final_bqs)
            max_best_quality = max(max(bq) for bq in bq_list)
        else:
            best_quality_mean = None
            best_quality_std = None
            max_best_quality = None
        
        summary.append({
            'Algorithm': alg_name,
            'Final Reward (Mean)': np.mean(final_scores),
            'Final Reward (Std)': np.std(final_scores),
            'Best Quality (Mean)': best_quality_mean,
            'Best Quality (Std)': best_quality_std,
            'Max Best Quality': max_best_quality,
            'CVR': cvr,
            'Projection Rate': proj_rate if proj_rate is not None else 'N/A',
            'Max Reward': max(max(s) for s in all_scores),
        })
    
    # 保存为 JSON
    with open(os.path.join(save_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    # 打印表格
    print("\n" + "="*100)
    print("ALGORITHM COMPARISON SUMMARY")
    print("="*100)
    print(f"{'Algorithm':<12} {'Final Reward':<16} {'Best Quality':<18} {'Max BQ':<10} {'CVR':<12} {'Max Reward':<10}")
    print("-"*100)
    for s in summary:
        bq_mean = s.get('Best Quality (Mean)')
        bq_std = s.get('Best Quality (Std)')
        max_bq = s.get('Max Best Quality')
        
        bq_str = f"{bq_mean:.4f} ± {bq_std:.4f}" if bq_mean is not None else "N/A"
        max_bq_str = f"{max_bq:.4f}" if max_bq is not None else "N/A"
        
        print(f"{s['Algorithm']:<12} {s['Final Reward (Mean)']:.2f} ± {s['Final Reward (Std)']:.2f}  "
              f"{bq_str:<18} {max_bq_str:<10} {s['CVR']:.6f}   {s['Max Reward']:.2f}")
    print("="*100)
    
    return summary


# =======================
# MAIN
# =======================

def main(config=None):
    """主函数：运行所有算法并对比"""
    
    if config is None:
        config = COMPARE_CONFIG
    
    # 创建保存目录
    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(os.path.dirname(__file__), f"comparison_{now_str}")
    os.makedirs(save_dir, exist_ok=True)
    
    print("="*70)
    print("ALGORITHM COMPARISON EXPERIMENT")
    print("="*70)
    print(f"Episodes per algorithm: {config['num_episodes']}")
    print(f"Seeds: {config['seeds'][:config['num_seeds']]}")
    print(f"Save directory: {save_dir}")
    print("-"*70)
    print("Network Configuration:")
    print(f"  - PDQN/HPPO/CPO/LPPO: hidden=[{UNIFIED_NETWORK_CONFIG['hidden_dim1']}, {UNIFIED_NETWORK_CONFIG['hidden_dim2']}]")
    print(f"  - LIRL: hidden={UNIFIED_NETWORK_CONFIG['lirl_hidden_dim']}, latent={UNIFIED_NETWORK_CONFIG['latent_dim']}, layers={UNIFIED_NETWORK_CONFIG['num_layers']}")
    print("="*70)
    
    # 算法运行器
    runners = {
        'LIRL': run_lirl,
        'PDQN': run_pdqn,
        'HPPO': run_hppo,
        'CPO': run_cpo,
        'LPPO': run_lppo,
    }
    
    # 存储所有结果
    all_results = defaultdict(list)
    
    # 运行每个算法
    for alg_name, runner in runners.items():
        print(f"\n{'#'*70}")
        print(f"# Running {alg_name}")
        print(f"{'#'*70}")
        
        for i, seed in enumerate(config['seeds'][:config['num_seeds']]):
            try:
                result = runner(config, seed)
                all_results[alg_name].append(result)
                
                # 保存单次运行结果
                np.save(os.path.join(save_dir, f"{alg_name}_seed{seed}_scores.npy"), result['scores'])
                
            except Exception as e:
                print(f"Error running {alg_name} with seed {seed}: {e}")
                import traceback
                traceback.print_exc()
    
    # 绘制对比图
    if config['plot_results'] and len(all_results) > 0:
        plot_comparison(dict(all_results), save_dir)
    
    # 创建汇总表格
    if len(all_results) > 0:
        summary = create_summary_table(dict(all_results), save_dir)
    
    # 保存配置
    full_config = {
        'experiment_config': config,
        'unified_network_config': UNIFIED_NETWORK_CONFIG,
    }
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(full_config, f, indent=2)
    
    print(f"\nAll results saved to: {save_dir}")
    
    return all_results, save_dir


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Algorithm Comparison')
    parser.add_argument('--quick', action='store_true', help='Run quick test with fewer episodes')
    parser.add_argument('--algorithms', nargs='+', default=None, 
                       help='Specific algorithms to run (e.g., LIRL PDQN+Mask)')
    args = parser.parse_args()
    
    if args.quick:
        print("Running QUICK TEST mode...")
        config = QUICK_TEST_CONFIG
    else:
        config = COMPARE_CONFIG
    
    # 如果指定了特定算法，只运行这些算法
    if args.algorithms:
        # 创建一个临时的main函数来运行特定算法
        def main_selected(selected_algs):
            """运行选定的算法"""
            now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = os.path.join(os.path.dirname(__file__), f"comparison_{now_str}")
            os.makedirs(save_dir, exist_ok=True)
            
            all_runners = {
                'LIRL': run_lirl,
                'PDQN': run_pdqn,
                'HPPO': run_hppo,
                'CPO': run_cpo,
                'LPPO': run_lppo,
            }
            
            runners = {k: v for k, v in all_runners.items() if k in selected_algs}
            
            print("="*70)
            print(f"Running selected algorithms: {list(runners.keys())}")
            print("="*70)
            
            all_results = defaultdict(list)
            
            for alg_name, runner in runners.items():
                for seed in config['seeds'][:config['num_seeds']]:
                    try:
                        result = runner(config, seed)
                        all_results[alg_name].append(result)
                    except Exception as e:
                        print(f"Error running {alg_name}: {e}")
                        import traceback
                        traceback.print_exc()
            
            if len(all_results) > 0:
                plot_comparison(dict(all_results), save_dir)
                create_summary_table(dict(all_results), save_dir)
            
            return all_results, save_dir
        
        all_results, save_dir = main_selected(args.algorithms)
    else:
        # 运行所有算法
        all_results, save_dir = main(config)
