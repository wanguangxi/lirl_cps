"""
Protein Crystallization Experiment Visualization
Nature Journal Style - 1x3 Comparison Figure

Generates:
  1. Learning Curve - Episode rewards over training
  2. Final Best Quality Comparison - Bar chart
  3. Constraint Violation Rate Comparison - Bar chart
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ============================================================================
# NATURE JOURNAL STYLE CONFIGURATION
# ============================================================================

# Nature-style color palette (colorblind-friendly)
NATURE_COLORS = {
    'LIRL': '#0077BB',    # Strong blue
    'PDQN': '#EE7733',    # Orange
    'HPPO': '#CC3311',    # Red
    'CPO':  '#009988',    # Teal
    'LPPO': '#EE3377',    # Magenta
}

# Nature figure specifications
NATURE_STYLE = {
    'figure.figsize': (7, 2.5),  # 7 inches width for 1x3 layout
    'figure.dpi': 300,
    'font.family': 'Arial',
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'axes.titleweight': 'normal',
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'legend.fontsize': 7,
    'legend.frameon': False,
    'lines.linewidth': 1.2,
    'lines.markersize': 4,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
}


def load_experiment_data(data_dir):
    """Load all experiment data from directory"""
    
    # Load summary
    summary_path = os.path.join(data_dir, 'summary.json')
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    # Load config
    config_path = os.path.join(data_dir, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Get seeds from config
    seeds = config['experiment_config']['seeds']
    
    # Load score files for each algorithm
    algorithms = ['LIRL', 'PDQN', 'HPPO', 'CPO', 'LPPO']
    all_scores = {}
    
    for alg in algorithms:
        alg_scores = []
        for seed in seeds:
            score_file = os.path.join(data_dir, f'{alg}_seed{seed}_scores.npy')
            if os.path.exists(score_file):
                scores = np.load(score_file)
                alg_scores.append(scores)
        if alg_scores:
            all_scores[alg] = alg_scores
    
    return summary, config, all_scores


def smooth_curve(data, window=10):
    """Apply moving average smoothing"""
    if len(data) < window:
        return data
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode='valid')


def create_nature_figure(data_dir, output_dir=None):
    """Create 1x3 Nature-style comparison figure"""
    
    if output_dir is None:
        output_dir = data_dir
    
    # Apply Nature style
    plt.rcParams.update(NATURE_STYLE)
    
    # Load data
    summary, config, all_scores = load_experiment_data(data_dir)
    
    # Create figure with 1x3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.5))
    
    # Define algorithm order (LIRL first as proposed method)
    alg_order = ['LIRL', 'PDQN', 'HPPO', 'CPO', 'LPPO']
    algorithms = [a for a in alg_order if a in all_scores]
    
    # Extract data from summary
    summary_dict = {s['Algorithm']: s for s in summary}
    
    # ========================================================================
    # Learning Curve - Left
    # ========================================================================
    ax1 = axes[0]
    
    for alg in algorithms:
        scores_list = all_scores[alg]
        min_len = min(len(s) for s in scores_list)
        scores_array = np.array([s[:min_len] for s in scores_list])
        
        mean_scores = np.mean(scores_array, axis=0)
        std_scores = np.std(scores_array, axis=0)
        sem_scores = std_scores / np.sqrt(len(scores_list))  # Standard error
        
        # Smooth for cleaner visualization
        window = 15
        if len(mean_scores) > window:
            smoothed_mean = smooth_curve(mean_scores, window)
            smoothed_sem = smooth_curve(sem_scores, window)
            episodes = np.arange(window-1, len(mean_scores))
        else:
            smoothed_mean = mean_scores
            smoothed_sem = sem_scores
            episodes = np.arange(len(mean_scores))
        
        color = NATURE_COLORS.get(alg, '#888888')
        ax1.plot(episodes, smoothed_mean, label=alg, color=color, linewidth=1.2)
        ax1.fill_between(episodes, 
                         smoothed_mean - smoothed_sem, 
                         smoothed_mean + smoothed_sem,
                         alpha=0.2, color=color, linewidth=0)
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Learning curve', loc='center')
    legend = ax1.legend(loc='lower right', ncol=1, columnspacing=0.5, handlelength=1.2,
                        frameon=True, fancybox=False, edgecolor='#CCCCCC', 
                        framealpha=0.95, facecolor='white', labelspacing=0.3,
                        handletextpad=0.5, borderpad=0.4)
    legend.get_frame().set_linewidth(0.5)
    ax1.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    
    # ========================================================================
    # Final Best Quality Comparison - Middle
    # ========================================================================
    ax2 = axes[1]
    
    best_quality_means = []
    best_quality_stds = []
    bar_colors = []
    
    for alg in algorithms:
        if alg in summary_dict:
            bq_mean = summary_dict[alg].get('Best Quality (Mean)', 0)
            bq_std = summary_dict[alg].get('Best Quality (Std)', 0)
            best_quality_means.append(bq_mean)
            best_quality_stds.append(bq_std)
            bar_colors.append(NATURE_COLORS.get(alg, '#888888'))
    
    x_pos = np.arange(len(algorithms))
    bars = ax2.bar(x_pos, best_quality_means, 
                   yerr=best_quality_stds, 
                   capsize=3,
                   color=bar_colors, 
                   edgecolor='black',
                   linewidth=0.5,
                   alpha=0.85,
                   error_kw={'linewidth': 0.8, 'capthick': 0.8})
    
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(algorithms, rotation=0)
    ax2.set_ylabel('Best quality')
    ax2.set_title('Final best quality comparison', loc='center')
    ax2.set_ylim(bottom=0)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, best_quality_means)):
        height = bar.get_height()
        ax2.annotate(f'{val:.3f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height + best_quality_stds[i]),
                     ha='center', va='bottom',
                     fontsize=6, fontweight='normal')
    
    # ========================================================================
    # Constraint Violation Rate Comparison - Right
    # ========================================================================
    ax3 = axes[2]
    
    cvr_values = []
    cvr_colors = []
    
    for alg in algorithms:
        if alg in summary_dict:
            cvr = summary_dict[alg].get('CVR', 0)
            cvr_values.append(cvr * 100)  # Convert to percentage
            cvr_colors.append(NATURE_COLORS.get(alg, '#888888'))
    
    bars = ax3.bar(x_pos, cvr_values,
                   color=cvr_colors,
                   edgecolor='black',
                   linewidth=0.5,
                   alpha=0.85)
    
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(algorithms, rotation=0)
    ax3.set_ylabel('Constraint violation rate (%)')
    ax3.set_title('Constraint violation rate comparison', loc='center')
    ax3.set_ylim(bottom=0)
    
    # Add value labels on bars
    for bar, val in zip(bars, cvr_values):
        height = bar.get_height()
        if val > 0:
            ax3.annotate(f'{val:.1f}%',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         ha='center', va='bottom',
                         fontsize=6, fontweight='normal')
        else:
            ax3.annotate('0%',
                         xy=(bar.get_x() + bar.get_width() / 2, height + 0.5),
                         ha='center', va='bottom',
                         fontsize=6, fontweight='normal')
    
    # Highlight LIRL's zero violation with a star
    if cvr_values[0] == 0:
        ax3.annotate('★', xy=(0, 2), ha='center', va='bottom', 
                     fontsize=10, color=NATURE_COLORS['LIRL'])
    
    # ========================================================================
    # Final adjustments
    # ========================================================================
    plt.tight_layout(pad=0.8, w_pad=1.5)
    
    # Save figure
    output_path = os.path.join(output_dir, 'nature_comparison_figure.pdf')
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"PDF saved: {output_path}")
    
    output_path_png = os.path.join(output_dir, 'nature_comparison_figure.png')
    plt.savefig(output_path_png, format='png', dpi=300, bbox_inches='tight')
    print(f"PNG saved: {output_path_png}")
    
    plt.show()
    plt.close()
    
    return fig


def print_summary_table(data_dir):
    """Print formatted summary table"""
    summary_path = os.path.join(data_dir, 'summary.json')
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    print("\n" + "="*80)
    print("PROTEIN CRYSTALLIZATION EXPERIMENT RESULTS")
    print("="*80)
    print(f"{'Algorithm':<10} {'Final Reward':<18} {'Best Quality':<18} {'CVR':<12}")
    print("-"*80)
    
    for s in summary:
        reward_str = f"{s['Final Reward (Mean)']:.2f} ± {s['Final Reward (Std)']:.2f}"
        quality_str = f"{s['Best Quality (Mean)']:.4f} ± {s['Best Quality (Std)']:.4f}"
        cvr_str = f"{s['CVR']*100:.2f}%"
        print(f"{s['Algorithm']:<10} {reward_str:<18} {quality_str:<18} {cvr_str:<12}")
    
    print("="*80)


if __name__ == "__main__":
    # Data directory
    data_dir = r"C:\Users\wangu\Desktop\Major Revision\LIRL-CPS-main\lirl_cps\Results\Protein_crystalization\comparison_20260205_154257"
    
    # Print summary table
    print_summary_table(data_dir)
    
    # Create Nature-style figure
    create_nature_figure(data_dir)
