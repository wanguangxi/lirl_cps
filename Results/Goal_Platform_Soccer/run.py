import json
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# Get script directory for relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 8

try:
    arial_font = fm.findfont(fm.FontProperties(family='Arial'))
    plt.rcParams['font.family'] = 'Arial'
except:
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

plt.rcParams.update({
    'axes.linewidth': 1.0,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.minor.width': 0.5,
    'ytick.minor.width': 0.5,
    'lines.linewidth': 1.5,
    'legend.frameon': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

goal_path = os.path.join(SCRIPT_DIR, "goal_comparison", "compare_20260110_082334", "results.json")
platform_path = os.path.join(SCRIPT_DIR, "platform_comparison", "compare_20260114_103334", "results.json")
soccer_path = os.path.join(SCRIPT_DIR, "soccer_comparison", "compare_20251227_111639", "results.json")

colors = {
    'LIRL': '#2E86AB',
    'PADDPG': '#A23B72',
    'PDQN': '#F18F01',
    'QPAMDP': '#C73E1D'
}

def load_data(file_path, metric_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    algorithms = []
    values = []
    for alg_name, alg_data in data['algorithms'].items():
        algorithms.append(alg_name)
        value = alg_data
        for key in metric_path:
            value = value[key]
        values.append(value)
    return algorithms, values

def reorder_data(algorithms, values, order):
    ordered_values = []
    for alg in order:
        if alg in algorithms:
            idx = algorithms.index(alg)
            ordered_values.append(values[idx])
        else:
            ordered_values.append(0)
    return ordered_values

def plot_comparison():
    goal_algs, goal_values = load_data(goal_path, ['aggregated_metrics', 'last_100_success_rate'])
    platform_algs, platform_values = load_data(platform_path, ['aggregated_metrics', 'last_100_mean'])
    soccer_algs, soccer_values = load_data(soccer_path, ['metrics', 'goal_rate'])
    
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 1.72))
    alg_order = ['LIRL', 'PADDPG', 'PDQN', 'QPAMDP']
    
    goal_values = reorder_data(goal_algs, goal_values, alg_order)
    platform_values = reorder_data(platform_algs, platform_values, alg_order)
    soccer_values = reorder_data(soccer_algs, soccer_values, alg_order)
    
    x = np.arange(len(alg_order))
    width = 0.6
    
    configs = [
        (axes[0], goal_values, 'Success rate', 'Robot soccer goal', True),
        (axes[1], platform_values, 'Reward', 'Platform', False),
        (axes[2], soccer_values, 'Goal rate', 'Half field offense', True)
    ]
    
    for ax, values, ylabel, title, is_percentage in configs:
        if is_percentage:
            values = [v * 100 for v in values]
            ylabel = ylabel + ' (%)'
        
        bars = ax.bar(x, values, width, color=[colors[alg] for alg in alg_order],
                     edgecolor='black', linewidth=0.5, alpha=0.8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(title, fontsize=8, pad=3)
        ax.set_xticks(x)
        ax.set_xticklabels(alg_order, fontsize=8)
        ax.set_ylim(0, max(values) * 1.15 if max(values) > 0 else 1)
        ax.set_facecolor('white')
        
        for bar, val in zip(bars, values):
            height = bar.get_height()
            if is_percentage:
                label_text = f'{val:.1f}%'
            else:
                label_text = f'{val:.3f}'
            ax.text(bar.get_x() + bar.get_width()/2., height, label_text,
                   ha='center', va='bottom', fontsize=7, fontweight='bold')
    
    plt.tight_layout(pad=1.5)
    output_path = os.path.join(SCRIPT_DIR, 'three_scenarios_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Figure saved to: {output_path}")
    plt.show()

if __name__ == "__main__":
    plot_comparison()
