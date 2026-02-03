"""
Traffic Control Algorithm Comparison Visualization
==================================================
This script generates Nature journal-style comparison plots for traffic control algorithms.
It combines Pareto front analysis and performance heatmap in a single figure.

Author: Generated for Major Revision
Date: 2026-01-16
"""

import json
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib.gridspec import GridSpec

# ============================================================================
# Configuration
# ============================================================================

# Get script directory for relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# File paths (relative to script directory)
FIXED_DATA_PATH = os.path.join(SCRIPT_DIR, 'result', 'run_20260115_100648', 'summary.json')
ALGORITHM_DATA_PATH = os.path.join(SCRIPT_DIR, 'result', 'run_20260115_111112', 'summary.json')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'result')

# Algorithm name mapping
ALGORITHM_MAPPING = {'Lagrangian-PPO': 'LPPO'}

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
})

# Nature-style color palette (NPG - Nature Publishing Group)
NATURE_COLORS = {
    'red': '#E64B35',      # NPG red
    'blue': '#4DBBD5',     # NPG cyan/blue  
    'green': '#00A087',    # NPG green
    'purple': '#3C5488',   # NPG dark blue
    'orange': '#F39B7F',   # NPG salmon
    'yellow': '#8491B4',   # NPG lavender
    'gray': '#91D1C2',     # NPG mint
    'brown': '#DC0000',    # NPG bright red
}

# ============================================================================
# Data Loading Functions
# ============================================================================

def load_data():
    """Load evaluation data from JSON files."""
    with open(FIXED_DATA_PATH, 'r', encoding='utf-8') as f:
        data1 = json.load(f)
    
    with open(ALGORITHM_DATA_PATH, 'r', encoding='utf-8') as f:
        data2 = json.load(f)
    
    comparison = data1['comparison']
    fixed_data = comparison['Fixed green duration']
    evaluation = data2['evaluation_summary']
    
    return fixed_data, evaluation

def prepare_algorithm_data(fixed_data, evaluation):
    """Prepare algorithm names and performance metrics."""
    algorithms_original = list(evaluation.keys())
    algorithms_display = [ALGORITHM_MAPPING.get(alg, alg) for alg in algorithms_original]
    
    all_algorithms_original = ['Fixed'] + algorithms_original
    all_algorithms_display = ['Fixed'] + algorithms_display
    
    # Extract metrics
    throughput_values = [fixed_data['mean_throughput'] / 15]  # Divide by 15
    travel_time_values = [fixed_data['mean_travel_time']]
    violation_rate_values = [fixed_data.get('mean_violation_rate', 0.0)]
    
    for alg_orig in algorithms_original:
        throughput_values.append(evaluation[alg_orig]['mean_throughput'] / 15)
        travel_time_values.append(evaluation[alg_orig]['mean_travel_time'])
        violation_rate_values.append(evaluation[alg_orig].get('mean_violation_rate', 0.0))
    
    return (all_algorithms_original, all_algorithms_display, 
            throughput_values, travel_time_values, violation_rate_values)

# ============================================================================
# Pareto Front Calculation Functions
# ============================================================================

def find_pareto_front_2d(points):
    """
    Find 2D Pareto front points (throughput vs travel_time).
    
    Args:
        points: List of dicts with 'throughput' and 'travel_time' keys
        
    Returns:
        List of Pareto optimal points
    """
    if len(points) == 0:
        return []
    
    sorted_points = sorted(points, key=lambda x: (x['travel_time'], -x['throughput']))
    pareto_front = []
    max_throughput = float('-inf')
    
    for point in sorted_points:
        if point['throughput'] >= max_throughput:
            pareto_front.append(point)
            max_throughput = point['throughput']
    
    return sorted(pareto_front, key=lambda x: x['travel_time'])

def find_pareto_front_3d(points):
    """
    Find 3D Pareto front points (throughput, travel_time, violation_rate).
    
    Args:
        points: List of dicts with 'throughput', 'travel_time', and 'violation_rate' keys
        
    Returns:
        List of 3D Pareto optimal points
    """
    if len(points) == 0:
        return []
    
    pareto_front = []
    
    for i, point in enumerate(points):
        is_dominated = False
        for j, other_point in enumerate(points):
            if i == j:
                continue
            
            throughput_better = other_point['throughput'] >= point['throughput']
            travel_time_better = other_point['travel_time'] <= point['travel_time']
            violation_better = other_point['violation_rate'] <= point['violation_rate']
            
            if throughput_better and travel_time_better and violation_better:
                if (other_point['throughput'] > point['throughput'] or
                    other_point['travel_time'] < point['travel_time'] or
                    other_point['violation_rate'] < point['violation_rate']):
                    is_dominated = True
                    break
        
        if not is_dominated:
            pareto_front.append(point)
    
    return pareto_front

# ============================================================================
# Plotting Functions
# ============================================================================

def plot_pareto_front(ax, points, pareto_points_2d, pareto_points_3d):
    """Plot Pareto front scatter plot with violation rate color coding."""
    points_sorted = sorted(points, key=lambda x: x['name'])
    
    # Calculate violation rate range for color mapping
    all_violation_rates = [p['violation_rate'] for p in points]
    vmin = min(all_violation_rates)
    vmax = max(all_violation_rates) if max(all_violation_rates) > 0 else 1.0
    
    # Nature-style sequential colormap (light to dark blue-green)
    nature_colors_seq = ['#91D1C2', '#4DBBD5', '#00A087', '#3C5488', '#E64B35']
    cmap = mcolors.LinearSegmentedColormap.from_list('nature_seq', nature_colors_seq)
    
    # Plot points
    for p in points_sorted:
        if vmax > vmin:
            color_intensity = (p['violation_rate'] - vmin) / (vmax - vmin)
        else:
            color_intensity = 0.0
        
        point_color = cmap(color_intensity)
        is_3d_pareto = any(p3d['name'] == p['name'] for p3d in pareto_points_3d)
        marker = '*' if is_3d_pareto else 'o'
        size = 76 if is_3d_pareto else 38
        
        # Exclude LIRL from legend
        label = None if p['display_name'] == 'LIRL' else p['display_name']
        
        ax.scatter(p['travel_time'], p['throughput'], color=[point_color], s=size, 
                  marker=marker, edgecolors='#333333', linewidths=0.5, zorder=10, 
                  label=label, alpha=0.9)
    
    # Annotate 3D Pareto optimal points
    for p in pareto_points_3d:
        matching_point = next((p2 for p2 in points_sorted if p2['name'] == p['name']), None)
        if matching_point:
            # Adjust LIRL annotation position (move down)
            if matching_point['display_name'] == 'LIRL':
                xytext_offset = (5, -15)  # Move down
            else:
                xytext_offset = (5, 5)  # Default position
            
            ax.annotate(matching_point['display_name'], 
                       (matching_point['travel_time'], matching_point['throughput']), 
                       xytext=xytext_offset, textcoords='offset points', fontsize=6, alpha=0.8,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, 
                                edgecolor='#3C5488', linewidth=0.5))
    
    # Plot 2D Pareto front line
    if len(pareto_points_2d) > 1:
        pareto_x = [p['travel_time'] for p in pareto_points_2d]
        pareto_y = [p['throughput'] for p in pareto_points_2d]
        ax.plot(pareto_x, pareto_y, color='#3C5488', linewidth=1.5, alpha=0.7, 
               linestyle='--', dashes=(5, 3), label='Pareto front', zorder=8)
    
    # Set labels and title
    ax.set_xlabel('Travel time (s)', fontsize=8)
    ax.set_ylabel('Throughput (veh/h)', fontsize=8)
    ax.set_title('Traffic control: Pareto front', fontsize=8, pad=6)
    ax.tick_params(axis='both', labelsize=7)
    
    # Set axis limits
    all_travel_times = [p['travel_time'] for p in points]
    all_throughputs = [p['throughput'] for p in points]
    
    x_min = min(all_travel_times) - (max(all_travel_times) - min(all_travel_times)) * 0.05
    x_max = max(all_travel_times) + (max(all_travel_times) - min(all_travel_times)) * 0.05
    y_min = min(0, min(all_throughputs) - abs(max(all_throughputs) - min(all_throughputs)) * 0.05)
    y_max = max(all_throughputs) + abs(max(all_throughputs) - min(all_throughputs)) * 0.05
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='both')
    ax.set_axisbelow(True)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02, shrink=0.7)
    cbar.set_label('Violation rate (%)', fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    
    # Add legend
    ax.legend(loc='upper right', fontsize=6, frameon=True, 
             handletextpad=0.5, handlelength=1.2, columnspacing=0.8,
             ncol=1, labelspacing=0.5, borderpad=0.5,
             edgecolor='lightgray', facecolor='white', framealpha=0.9)
    
    return cmap, vmin, vmax

def plot_heatmap(ax, all_algorithms_display, throughput_values, 
                 travel_time_values, violation_rate_values):
    """Plot performance comparison heatmap."""
    # Prepare data
    metrics_data = {
        'Throughput\n(veh/h)': throughput_values,
        'Travel time \n (s)': travel_time_values,
        'Violation rate \n(%)': violation_rate_values
    }
    
    heatmap_df = pd.DataFrame(metrics_data, index=all_algorithms_display)
    
    # Normalize data (0-1 scale)
    # For "higher is better" metrics: (value - min) / (max - min)
    # For "lower is better" metrics: 1 - (value - min) / (max - min)
    heatmap_normalized_norm = pd.DataFrame()
    
    for col in heatmap_df.columns:
        col_data = heatmap_df[col].values
        if 'Travel time' in col or 'Violation' in col:
            # Lower is better
            col_min, col_max = col_data.min(), col_data.max()
            if col_max > col_min:
                heatmap_normalized_norm[col] = 1 - (col_data - col_min) / (col_max - col_min)
            else:
                heatmap_normalized_norm[col] = 0.5
        else:
            # Higher is better
            col_min, col_max = col_data.min(), col_data.max()
            if col_max > col_min:
                heatmap_normalized_norm[col] = (col_data - col_min) / (col_max - col_min)
            else:
                heatmap_normalized_norm[col] = 0.5
    
    heatmap_normalized_norm.index = all_algorithms_display
    
    # Format annotation text
    annot_text = heatmap_df.copy()
    for col in annot_text.columns:
        if 'Throughput' in col:
            annot_text[col] = annot_text[col].apply(lambda x: f'{x:.0f}')
        else:
            annot_text[col] = annot_text[col].apply(lambda x: f'{x:.1f}')
    
    # Nature-style diverging colormap (red=bad, blue/green=good)
    nature_diverging = ['#E64B35', '#F39B7F', '#FFFFFF', '#91D1C2', '#00A087']
    cmap_heatmap = mcolors.LinearSegmentedColormap.from_list('nature_div', nature_diverging)
    
    # Create heatmap
    sns.heatmap(heatmap_normalized_norm, annot=annot_text, fmt='', 
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
    ax.set_title('Traffic control: Performance comparison', fontsize=8, pad=6)
    ax.xaxis.set_ticks_position('bottom')
    ax.xaxis.set_label_position('bottom')
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=7, rotation=0, ha='center')
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=7, rotation=0)
    ax.tick_params(axis='both', which='both', length=0)

# ============================================================================
# Main Function
# ============================================================================

def main():
    """Main function to generate comparison figure."""
    # Load data
    print("Loading data...")
    fixed_data, evaluation = load_data()
    
    # Prepare algorithm data
    (all_algorithms_original, all_algorithms_display, 
     throughput_values, travel_time_values, violation_rate_values) = \
        prepare_algorithm_data(fixed_data, evaluation)
    
    print(f"Loaded {len(all_algorithms_display)} algorithms: {', '.join(all_algorithms_display)}")
    
    # Prepare points for Pareto analysis
    points = []
    for i, alg_orig in enumerate(all_algorithms_original):
        if alg_orig == 'Fixed':
            points.append({
                'name': alg_orig,
                'display_name': 'Fixed',
                'throughput': throughput_values[i],
                'travel_time': travel_time_values[i],
                'violation_rate': violation_rate_values[i],
            })
        else:
            points.append({
                'name': alg_orig,
                'display_name': all_algorithms_display[i],
                'throughput': throughput_values[i],
                'travel_time': travel_time_values[i],
                'violation_rate': violation_rate_values[i],
            })
    
    # Calculate Pareto fronts
    pareto_points_2d = find_pareto_front_2d(points)
    pareto_points_3d = find_pareto_front_3d(points)
    
    print(f"2D Pareto front: {len(pareto_points_2d)} points")
    print(f"3D Pareto front: {len(pareto_points_3d)} points")
    
    # Create figure with specified dimensions: 7 inches x 2.25 inches
    fig_width_inch = 7.0
    fig_height_inch = 1.72
    
    fig = plt.figure(figsize=(fig_width_inch, fig_height_inch))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1.3], wspace=0.35)
    
    # Plot Pareto front (left subplot)
    print("Plotting Pareto front...")
    ax1 = fig.add_subplot(gs[0])
    plot_pareto_front(ax1, points, pareto_points_2d, pareto_points_3d)
    
    # Plot heatmap (right subplot)
    print("Plotting heatmap...")
    ax2 = fig.add_subplot(gs[1])
    plot_heatmap(ax2, all_algorithms_display, throughput_values, 
                 travel_time_values, violation_rate_values)
    
    # Adjust layout for 7" x 1.72" dimensions (scaled from 2.25")
    fig.subplots_adjust(left=0.08, right=0.92, top=0.88, bottom=0.22)
    
    # Save figures with exact dimensions (no tight bounding box)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, 'combined_figure.png')
    plt.savefig(output_path, dpi=300, bbox_inches=None, facecolor='white', 
                pad_inches=0, format='png')
    print(f"\nCombined figure saved to: {output_path}")
    
    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches=None, facecolor='white', 
                pad_inches=0)
    print(f"PDF saved to: {pdf_path}")
    
    plt.show()

if __name__ == "__main__":
    main()
