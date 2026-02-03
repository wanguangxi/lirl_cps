"""
EV Charging Station Algorithm Comparison Visualization
=======================================================
This script generates Nature journal-style comparison plots for EV charging algorithms.
It combines Pareto front analysis and performance heatmap in a single figure.

Output:
- combined_figure.png/pdf: Pareto front + Performance heatmap (1x2 layout)
"""

import json
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from matplotlib.gridspec import GridSpec
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE_DIR, 'result', 'algorithm_comparison_20260116_150929')
OUTPUT_DIR = BASE_DIR

SUMMARY_CSV = os.path.join(RESULT_DIR, 'comparison_summary.csv')
METRICS_CSV = os.path.join(RESULT_DIR, 'key_performance_metrics.csv')
RESULTS_JSON = os.path.join(RESULT_DIR, 'comparison_results.json')

ALGORITHMS = ['LIRL', 'PDQN', 'HPPO', 'LPPO', 'CPO']

# Nature journal style settings
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 8,
    'axes.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# Nature-style color palette (NPG)
COLORS = {
    'LIRL': '#E64B35',
    'PDQN': '#4DBBD5',
    'HPPO': '#00A087',
    'LPPO': '#3C5488',
    'CPO': '#F39B7F',
}

# ============================================================================
# Data Loading Functions
# ============================================================================

def load_pareto_data():
    """Load data for Pareto front plot from comparison_summary.csv."""
    df = pd.read_csv(SUMMARY_CSV)
    metrics = []
    
    reward_row = df[df['Metric'] == 'test_avg_reward']
    violations_row = df[df['Metric'] == 'test_avg_violations']
    reward_std_row = df[df['Metric'] == 'test_std_reward']
    violations_std_row = df[df['Metric'] == 'test_std_violations']
    
    for alg in ALGORITHMS:
        if not reward_row.empty and not violations_row.empty:
            metrics.append({
                'name': alg,
                'mean_reward': float(reward_row[alg].values[0]),
                'std_reward': float(reward_std_row[alg].values[0]) if not reward_std_row.empty else 0.0,
                'mean_violations': float(violations_row[alg].values[0]),
                'std_violations': float(violations_std_row[alg].values[0]) if not violations_std_row.empty else 0.0,
            })
    
    return metrics

def load_heatmap_data():
    """Load data for heatmap from key_performance_metrics.csv."""
    df = pd.read_csv(METRICS_CSV, encoding='utf-8')
    
    metrics_data = {}
    
    # Station Utilization
    station_row = df[df['Metric'].str.contains('Station Utilization', case=False, na=False)]
    if not station_row.empty:
        metrics_data['Station\nutilization (%)'] = [station_row[alg].values[0] for alg in ALGORITHMS]
    
    # Charging Success Rate
    success_row = df[df['Metric'].str.contains('Charging Success Rate', case=False, na=False)]
    if not success_row.empty:
        metrics_data['Charging\nsuccess (%)'] = [success_row[alg].values[0] for alg in ALGORITHMS]
    
    # Energy Delivered
    energy_row = df[df['Metric'].str.contains('Energy Delivered', case=False, na=False)]
    if not energy_row.empty:
        metrics_data['Energy\ndelivered (kWh)'] = [energy_row[alg].values[0] for alg in ALGORITHMS]
    
    # Violation Rate
    violation_row = df[df['Metric'].str.contains('Violation Rate', case=False, na=False)]
    if not violation_row.empty:
        metrics_data['Violation\nrate (%)'] = [violation_row[alg].values[0] for alg in ALGORITHMS]
    
    return metrics_data

# ============================================================================
# Pareto Front Functions
# ============================================================================

def find_pareto_front(points):
    """Find Pareto front points (reward: higher better, violations: lower better)."""
    if len(points) == 0:
        return []
    
    sorted_points = sorted(points, key=lambda x: x['reward'], reverse=True)
    pareto_front = []
    min_violations = float('inf')
    
    for point in sorted_points:
        if point['violations'] <= min_violations:
            pareto_front.append(point)
            min_violations = point['violations']
    
    return sorted(pareto_front, key=lambda x: x['violations'])

# ============================================================================
# Plotting Functions
# ============================================================================

def plot_pareto_front(ax, metrics):
    """Plot Pareto front scatter plot."""
    points = [{
        'name': m['name'],
        'reward': m['mean_reward'],
        'violations': m['mean_violations'],
        'std_reward': m.get('std_reward', 0),
        'std_violations': m.get('std_violations', 0),
    } for m in metrics]
    
    pareto_points = find_pareto_front(points)
    pareto_names = {p['name'] for p in pareto_points}
    points_sorted = sorted(points, key=lambda x: x['name'])
    
    # Plot all points (exclude LIRL from legend if it's Pareto optimal)
    for p in points_sorted:
        color = COLORS.get(p['name'], '#666666')
        is_pareto = p['name'] in pareto_names
        # Exclude Pareto optimal points from legend (will be annotated instead)
        label = None if is_pareto else p['name']
        ax.scatter(p['violations'], p['reward'], c=color, s=15, marker='s',
                  edgecolors='black', linewidths=0.8, zorder=10, label=label, alpha=0.9)
        
        if p['std_violations'] > 0 or p['std_reward'] > 0:
            ax.errorbar(p['violations'], p['reward'], 
                       xerr=p['std_violations'], yerr=p['std_reward'],
                       fmt='none', ecolor=color, capsize=3, capthick=1.0, 
                       alpha=0.6, zorder=5, elinewidth=1.0)
    
    # Plot Pareto front line and stars
    if len(pareto_points) > 1:
        pareto_x = [p['violations'] for p in pareto_points]
        pareto_y = [p['reward'] for p in pareto_points]
        ax.plot(pareto_x, pareto_y, 'k-', linewidth=2.0, alpha=0.8, 
               linestyle='--', dashes=(5, 3), label='Pareto front', zorder=8)
        
        for p in pareto_points:
            color = COLORS.get(p['name'], '#666666')
            ax.scatter(p['violations'], p['reward'], s=76, marker='*', 
                      c=color, edgecolors='black', linewidths=0.8, zorder=12)
    elif len(pareto_points) == 1:
        p = pareto_points[0]
        color = COLORS.get(p['name'], '#666666')
        ax.scatter(p['violations'], p['reward'], s=76, marker='*', 
                  c=color, edgecolors='black', linewidths=0.8, zorder=12)
    
    # Annotate Pareto optimal points
    for p in pareto_points:
        # LIRL label position lower
        offset = (5, -12) if p['name'] == 'LIRL' else (5, 5)
        ax.annotate(p['name'], (p['violations'], p['reward']), 
                   xytext=offset, textcoords='offset points', fontsize=6, alpha=0.8,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, 
                            edgecolor=COLORS.get(p['name'], '#666666'), linewidth=0.5))
    
    # Set labels
    ax.set_xlabel('Average violations', fontsize=8)
    ax.set_ylabel('Average reward', fontsize=8)
    ax.set_title('EV-charging: Pareto front', fontsize=8, pad=6)
    ax.tick_params(axis='both', labelsize=8)
    
    # Set axis limits
    all_violations = [p['violations'] for p in points]
    all_rewards = [p['reward'] for p in points]
    
    x_min = min(0, min(all_violations) - max(all_violations) * 0.05)
    x_max = max(all_violations) * 1.08
    y_min = min(all_rewards) - abs(max(all_rewards) - min(all_rewards)) * 0.08
    y_max = max(all_rewards) + abs(max(all_rewards) - min(all_rewards)) * 0.08
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # Scientific notation for y-axis
    ax.ticklabel_format(style='sci', axis='y', scilimits=(-2, 3), useMathText=True)
    ax.yaxis.get_offset_text().set_fontsize(7)
    
    ax.grid(False)
    
    # Legend
    ax.legend(loc='upper right', fontsize=7, frameon=True, 
             handletextpad=0.4, handlelength=1.2, columnspacing=0.8,
             ncol=1, labelspacing=0.3, borderpad=0.4,
             edgecolor='lightgray', facecolor='white', framealpha=0.8)

def plot_heatmap(ax, metrics_data):
    """Plot performance comparison heatmap."""
    # Create DataFrame
    heatmap_df = pd.DataFrame(metrics_data, index=ALGORITHMS)
    
    # Normalize data (0-1 scale)
    # Higher is better: Station Utilization, Charging Success, Energy Delivered
    # Lower is better: Violation Rate
    heatmap_normalized = pd.DataFrame()
    
    for col in heatmap_df.columns:
        col_data = heatmap_df[col].values
        col_min, col_max = col_data.min(), col_data.max()
        
        if 'Violation' in col:
            # Lower is better
            if col_max > col_min:
                heatmap_normalized[col] = 1 - (col_data - col_min) / (col_max - col_min)
            else:
                heatmap_normalized[col] = 1.0
        else:
            # Higher is better
            if col_max > col_min:
                heatmap_normalized[col] = (col_data - col_min) / (col_max - col_min)
            else:
                heatmap_normalized[col] = 0.5
    
    heatmap_normalized.index = ALGORITHMS
    
    # Format annotation text
    annot_text = heatmap_df.copy()
    for col in annot_text.columns:
        if 'Energy' in col:
            annot_text[col] = annot_text[col].apply(lambda x: f'{x:.0f}')
        else:
            annot_text[col] = annot_text[col].apply(lambda x: f'{x:.1f}')
    
    # Nature-style diverging colormap (red=bad, green=good)
    nature_diverging = ['#E64B35', '#F39B7F', '#FFFFFF', '#91D1C2', '#00A087']
    cmap_heatmap = mcolors.LinearSegmentedColormap.from_list('nature_div', nature_diverging)
    
    # Create heatmap
    sns.heatmap(heatmap_normalized, annot=annot_text, fmt='', 
               cmap=cmap_heatmap, vmin=0, vmax=1,
               cbar=True, cbar_kws={'shrink': 0.6, 'aspect': 11},
               ax=ax, linewidths=2, linecolor='white',
               annot_kws={'size': 7, 'family': 'Arial', 'weight': 'normal'})
    
    # Style colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=6, width=0.5, length=2)
    cbar.outline.set_linewidth(0.5)
    cbar.set_label('Score', fontsize=7)
    
    # Style axes
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('EV-charging: Performance comparison', fontsize=8, pad=6)
    ax.xaxis.set_ticks_position('bottom')
    ax.xaxis.set_label_position('bottom')
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=6, rotation=0, ha='center')
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=7, rotation=0)
    ax.tick_params(axis='both', which='both', length=0)

# ============================================================================
# Main Function
# ============================================================================

def main():
    """Main function to generate combined comparison figure."""
    print("Loading data...")
    pareto_data = load_pareto_data()
    heatmap_data = load_heatmap_data()
    
    print(f"Loaded {len(pareto_data)} algorithms for Pareto plot")
    print(f"Loaded {len(heatmap_data)} metrics for heatmap")
    
    # Print performance summary
    print("\n" + "="*80)
    print("ALGORITHM PERFORMANCE SUMMARY")
    print("="*80)
    print(f"\n{'Algorithm':<10} {'Avg Reward':>15} {'Avg Violations':>15}")
    print("-"*40)
    for m in sorted(pareto_data, key=lambda x: x['mean_reward'], reverse=True):
        print(f"{m['name']:<10} {m['mean_reward']:>15.2f} {m['mean_violations']:>15.2f}")
    print("="*80)
    
    # Create combined figure
    # fig_width_cm = 17.65
    # fig_height_cm = 5.72
    fig_width_inch = 7
    fig_height_inch = 1.72
    
    fig = plt.figure(figsize=(fig_width_inch, fig_height_inch))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1.3], wspace=0.35)
    
    # Plot Pareto front (left subplot)
    print("\nPlotting Pareto front...")
    ax1 = fig.add_subplot(gs[0])
    plot_pareto_front(ax1, pareto_data)
    
    # Plot heatmap (right subplot)
    print("Plotting heatmap...")
    ax2 = fig.add_subplot(gs[1])
    plot_heatmap(ax2, heatmap_data)
    
    # Adjust layout to fit within exact figure size
    fig.tight_layout()
    fig.subplots_adjust(left=0.08, right=0.92, top=0.88, bottom=0.22, wspace=0.35)
    
    # Save figures with exact size (no bbox_inches='tight')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, 'combined_figure.png')
    plt.savefig(output_path, dpi=300, facecolor='white')
    print(f"\nCombined figure saved to: {output_path}")
    
    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, format='pdf', facecolor='white')
    print(f"PDF saved to: {pdf_path}")
    
    plt.show()
    print("\nDone!")

if __name__ == "__main__":
    main()
