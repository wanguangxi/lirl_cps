"""
Stationarity-Latency Pareto Front Visualization
================================================
This script generates Nature journal-style comparison plots for 
stationarity vs latency trade-off analysis.

Output:
- stationarity_latency_pareto.png/pdf: 1x3 layout showing projection accuracy, latency, and Pareto front
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import LogLocator, LogFormatterSciNotation, FuncFormatter

# ============================================================================
# Configuration
# ============================================================================

# Get script directory for relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# File paths (relative to script directory)
DATA_PATH = os.path.join(SCRIPT_DIR, 'inexact_projection_20260123_093508.json')
OUTPUT_DIR = SCRIPT_DIR

# Nature journal style settings
# Reference: Nature artwork guidelines
plt.rcParams.update({
    # Font settings (Nature requires Arial or Helvetica)
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 7,
    'axes.titlesize': 8,
    'axes.labelsize': 7,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 6,
    
    # Math text settings (use Arial for math as well)
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Arial',
    'mathtext.it': 'Arial:italic',
    'mathtext.bf': 'Arial:bold',
    'mathtext.sf': 'Arial',
    'mathtext.default': 'regular',
    
    # Line and tick settings
    'axes.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'xtick.minor.width': 0.3,
    'ytick.minor.width': 0.3,
    'lines.linewidth': 1.0,
    'lines.markersize': 4,
    
    # Spine settings
    'axes.spines.top': False,
    'axes.spines.right': False,
    
    # Figure settings
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.transparent': False,
    
    # PDF/PS font embedding
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    
    # Legend
    'legend.frameon': True,
    'legend.framealpha': 0.9,
    'legend.edgecolor': 'lightgray',
})

# Nature Publishing Group (NPG) color palette
# Reference: https://nanx.me/ggsci/reference/pal_npg.html
NPG_COLORS = {
    'red': '#E64B35',
    'blue': '#4DBBD5',
    'green': '#00A087',
    'purple': '#3C5488',
    'orange': '#F39B7F',
    'lavender': '#8491B4',
    'mint': '#91D1C2',
    'bright_red': '#DC0000',
    'brown': '#7E6148',
    'tan': '#B09C85',
}

# Color assignments for plots
COLORS = {
    'stationarity': NPG_COLORS['purple'],    # Dark blue for stationarity line
    'reference': '#999999',                   # Gray for O(ε) reference line
    'p50': NPG_COLORS['green'],              # Green for P50
    'p95': NPG_COLORS['orange'],             # Orange/Salmon for P95
    'p99': NPG_COLORS['red'],                # Red for P99
    'fill': NPG_COLORS['mint'],              # Mint for fill between P50-P99
    'pareto': NPG_COLORS['red'],             # Red for Pareto optimal points
    'dominated': NPG_COLORS['lavender'],     # Lavender for dominated points
}

# ============================================================================
# Data Loading Functions
# ============================================================================

def load_data():
    """Load data from JSON file."""
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def extract_metrics(data):
    """Extract metrics for plotting."""
    results = data['large']['results']
    
    metrics = []
    for r in results:
        metrics.append({
            'stationarity_mean': r['stationarity_mean'],
            'stationarity_std': r['stationarity_std'],
            'latency_mean': r['latency_mean'],
            'latency_p50': r['latency_p50'],
            'latency_p95': r['latency_p95'],
            'latency_p99': r['latency_p99'],
            'effective_eps': r['config']['effective_eps'],
            'qp_tol': r['config']['qp_tol'],
            'hungarian_topk': r['config']['hungarian_topk'],
        })
    
    # Sort by effective_eps
    metrics = sorted(metrics, key=lambda x: x['effective_eps'])
    return metrics

# ============================================================================
# Helper Functions
# ============================================================================

def format_log_ticks(ax, axis='both'):
    """
    Format log axis ticks uniformly with superscript notation.
    """
    def log_formatter(x, pos):
        if x <= 0:
            return ''
        exp = np.log10(x)
        if exp == int(exp):
            return r'$10^{%d}$' % int(exp)
        elif x >= 1:
            return f'{x:.0f}'
        elif x >= 0.1:
            return f'{x:.1f}'
        else:
            return f'{x:.2f}'
    
    formatter = FuncFormatter(log_formatter)
    
    if axis in ['x', 'both']:
        ax.xaxis.set_major_formatter(formatter)
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=6))
    if axis in ['y', 'both']:
        ax.yaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=6))

# ============================================================================
# Plotting Functions
# ============================================================================

def plot_projection_accuracy(ax, metrics):
    """Plot projection accuracy (stationarity vs epsilon)."""
    eps_vals = [m['effective_eps'] for m in metrics]
    stationarity_vals = [m['stationarity_mean'] for m in metrics]
    stationarity_stds = [m['stationarity_std'] for m in metrics]
    
    # Plot stationarity with error bars
    ax.errorbar(eps_vals, stationarity_vals, yerr=stationarity_stds,
                fmt='o-', color=COLORS['stationarity'], markersize=2,
                capsize=3, capthick=1, elinewidth=0.5, linewidth=1,
                label=r'$m = \|z - \Pi(z)\|/\eta$', zorder=1)
    
    # Plot O(ε) reference line
    eps_range = np.logspace(np.log10(min(eps_vals)), np.log10(max(eps_vals)), 100)
    # Scale reference line to pass through the plot area
    ref_scale = min(stationarity_vals) * 0.5 / min(eps_vals)
    ax.plot(eps_range, ref_scale * eps_range, '--', color=COLORS['reference'],
            linewidth=0.85, label=r'$\mathcal{O}(\varepsilon)$', zorder=1)
    
    # Set log scales
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Format log ticks uniformly
    format_log_ticks(ax, axis='both')
    
    # Set labels
    ax.set_xlabel(r'$\varepsilon$', fontsize=8, labelpad=2)
    ax.set_ylabel(r'Stationarity $m$', fontsize=8)
    ax.set_title('Stationarity proxy', fontsize=8, pad=4)
    ax.tick_params(axis='both', labelsize=6)
    
    # Set axis limits
    ax.set_xlim(min(eps_vals) * 0.5, max(eps_vals) * 2)
    y_min = min(min(stationarity_vals) - max(stationarity_stds), ref_scale * min(eps_vals)) * 0.3
    y_max = max(stationarity_vals) + max(stationarity_stds)
    ax.set_ylim(y_min, y_max * 3)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5, which='both')
    ax.set_axisbelow(True)
    
    # Legend
    ax.legend(loc='lower right', fontsize=6, frameon=True,
              edgecolor='lightgray', facecolor='white', framealpha=0.9)

def plot_latency(ax, metrics):
    """Plot end-to-end latency percentiles."""
    eps_vals = [m['effective_eps'] for m in metrics]
    p50_vals = [m['latency_p50'] for m in metrics]
    p95_vals = [m['latency_p95'] for m in metrics]
    p99_vals = [m['latency_p99'] for m in metrics]
    
    # Fill between P50 and P99
    ax.fill_between(eps_vals, p50_vals, p99_vals, alpha=0.25, 
                    color=COLORS['fill'], zorder=1)
    
    # Plot lines
    ax.plot(eps_vals, p50_vals, 'o-', color=COLORS['p50'], markersize=3,
            linewidth=1.0, label='P50', zorder=10)
    ax.plot(eps_vals, p95_vals, 's--', color=COLORS['p95'], markersize=3,
            linewidth=1.0, label='P95', zorder=10)
    ax.plot(eps_vals, p99_vals, '^:', color=COLORS['p99'], markersize=3,
            linewidth=1.0, label='P99', zorder=10)
    
    # Set log scale for x-axis only
    ax.set_xscale('log')
    
    # Format log ticks uniformly (x-axis only)
    format_log_ticks(ax, axis='x')
    
    # Set labels
    ax.set_xlabel(r'$\varepsilon$', fontsize=8, labelpad=2)
    ax.set_ylabel('Latency (ms)', fontsize=8)
    ax.set_title('End-to-end latency', fontsize=8, pad=4)
    ax.tick_params(axis='both', labelsize=6)
    
    # Set axis limits
    ax.set_xlim(min(eps_vals) * 0.5, max(eps_vals) * 2)
    ax.set_ylim(0, max(p99_vals) * 1.1)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5, which='both')
    ax.set_axisbelow(True)
    
    # Legend
    ax.legend(loc='upper right', fontsize=6, frameon=True, ncol=1,
              edgecolor='lightgray', facecolor='white', framealpha=0.9)

def find_pareto_front(metrics):
    """
    Find Pareto front points (stationarity: lower better, latency P99: lower better).
    """
    pareto_front = []
    
    for i, point in enumerate(metrics):
        is_dominated = False
        for j, other in enumerate(metrics):
            if i == j:
                continue
            # Other dominates if: lower stationarity AND lower latency P99
            if (other['stationarity_mean'] <= point['stationarity_mean'] and 
                other['latency_p99'] <= point['latency_p99']):
                if (other['stationarity_mean'] < point['stationarity_mean'] or 
                    other['latency_p99'] < point['latency_p99']):
                    is_dominated = True
                    break
        
        if not is_dominated:
            pareto_front.append(point)
    
    return sorted(pareto_front, key=lambda x: x['latency_p99'])

def plot_pareto_front(ax, metrics):
    """Plot stationarity vs latency P99 Pareto front."""
    pareto_points = find_pareto_front(metrics)
    pareto_latencies = {p['latency_p99'] for p in pareto_points}
    
    # Plot non-Pareto points (dominated)
    for m in metrics:
        if m['latency_p99'] not in pareto_latencies:
            ax.scatter(m['latency_p99'], m['stationarity_mean'], 
                      c=COLORS['dominated'], s=40, marker='o',
                      edgecolors='black', linewidths=0.5, alpha=0.7, zorder=5)
    
    # Plot Pareto points with stars
    for p in pareto_points:
        ax.scatter(p['latency_p99'], p['stationarity_mean'], 
                  c=COLORS['pareto'], s=80, marker='*',
                  edgecolors='black', linewidths=0.5, zorder=10)
    
    # Plot Pareto front line
    if len(pareto_points) > 1:
        pareto_x = [p['latency_p99'] for p in pareto_points]
        pareto_y = [p['stationarity_mean'] for p in pareto_points]
        ax.plot(pareto_x, pareto_y, color=COLORS['pareto'], linewidth=1.5, 
               linestyle='--', dashes=(5, 3), alpha=0.8, zorder=8)
    
    # Annotate Pareto points with ε values
    for p in pareto_points:
        eps = p['effective_eps']
        if eps < 1:
            eps_label = f"ε={eps:.2f}"
        else:
            eps_label = f"ε={eps:.1f}"
        ax.annotate(eps_label, (p['latency_p99'], p['stationarity_mean']),
                   xytext=(5, 5), textcoords='offset points', fontsize=5, alpha=0.8,
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7,
                            edgecolor='gray', linewidth=0.3))
    
    # Set labels
    ax.set_xlabel('Latency P99 (ms)', fontsize=8, labelpad=2)
    ax.set_ylabel(r'Stationarity $m$', fontsize=8)
    ax.set_title('Stationarity-latency Pareto', fontsize=8, pad=4)
    ax.tick_params(axis='both', labelsize=6)
    
    # Set axis limits
    all_latencies = [m['latency_p99'] for m in metrics]
    all_stationarities = [m['stationarity_mean'] for m in metrics]
    
    x_margin = (max(all_latencies) - min(all_latencies)) * 0.1
    y_margin = (max(all_stationarities) - min(all_stationarities)) * 0.15
    
    ax.set_xlim(min(all_latencies) - x_margin, max(all_latencies) + x_margin * 5)
    ax.set_ylim(min(all_stationarities) - y_margin, max(all_stationarities) + y_margin)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='*', color='w', markerfacecolor=COLORS['pareto'],
               markersize=8, markeredgecolor='black', markeredgewidth=0.5, label='Pareto optimal'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['dominated'],
               markersize=6, markeredgecolor='black', markeredgewidth=0.5, label='Dominated'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=5, frameon=True,
             edgecolor='lightgray', facecolor='white', framealpha=0.9)

# ============================================================================
# Main Function
# ============================================================================

def main():
    """Main function to generate comparison figure."""
    print("Loading data...")
    data = load_data()
    metrics = extract_metrics(data)
    
    print(f"Loaded {len(metrics)} configurations")
    
    # Print summary
    print("\n" + "="*70)
    print("CONFIGURATION SUMMARY")
    print("="*70)
    print(f"{'ε':>10} {'Stationarity':>15} {'P50(ms)':>10} {'P95(ms)':>10} {'P99(ms)':>10}")
    print("-"*55)
    for m in metrics:
        print(f"{m['effective_eps']:>10.3f} {m['stationarity_mean']:>15.4f} "
              f"{m['latency_p50']:>10.2f} {m['latency_p95']:>10.2f} {m['latency_p99']:>10.2f}")
    print("="*70)
    
    # Create figure
    fig_width_inch = 7.0
    fig_height_inch = 1.72
    
    fig = plt.figure(figsize=(fig_width_inch, fig_height_inch))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 1], wspace=0.40)
    
    # Plot projection accuracy (left subplot)
    print("\nPlotting projection accuracy...")
    ax1 = fig.add_subplot(gs[0])
    plot_projection_accuracy(ax1, metrics)
    
    # Plot latency (middle subplot)
    print("Plotting latency...")
    ax2 = fig.add_subplot(gs[1])
    plot_latency(ax2, metrics)
    
    # Plot Pareto front (right subplot)
    print("Plotting Pareto front...")
    ax3 = fig.add_subplot(gs[2])
    plot_pareto_front(ax3, metrics)
    
    # Adjust layout for Nature style (7" x 1.72")
    fig.subplots_adjust(left=0.06, right=0.98, top=0.85, bottom=0.20, wspace=0.35)
    
    # Save figures
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, 'stationarity_latency_pareto.png')
    plt.savefig(output_path, dpi=300, facecolor='white')
    print(f"\nFigure saved to: {output_path}")
    
    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, format='pdf', facecolor='white')
    print(f"PDF saved to: {pdf_path}")
    
    plt.show()
    print("\nDone!")

if __name__ == "__main__":
    main()
