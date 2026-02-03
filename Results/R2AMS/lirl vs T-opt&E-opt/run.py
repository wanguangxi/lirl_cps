import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 8
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['axes.spines.top'] = False    
plt.rcParams['axes.spines.right'] = False  
plt.rcParams['axes.spines.left'] = True
plt.rcParams['axes.spines.bottom'] = True
plt.rcParams['xtick.major.width'] = 0.8
plt.rcParams['ytick.major.width'] = 0.8
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'
plt.rcParams['lines.linewidth'] = 1.0
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['axes.titlesize'] = 9
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8

script_dir = os.path.dirname(os.path.abspath(__file__))

csv_path = os.path.join(script_dir, 'compare_reports', 'summary_metrics.csv')
df = pd.read_csv(csv_path)

df['scale_label'] = df['scale'].str.replace('scale_', '')

algorithms = ['cross-opt', 'energy-opt', 'time-opt']
algorithm_labels = ['LIRL', 'E-opt', 'T-opt']
colors = ['#4472C4', '#ED7D31', '#70AD47']
scale_labels = ['a', 'b', 'c', 'd']

fig, axes = plt.subplots(1, 4, figsize=(7.2, 1.5))
fig.patch.set_facecolor('white')

for scale_idx, scale in enumerate(scale_labels):
    ax = axes[scale_idx]
    
    data_to_plot = []
    labels = []
    
    for algo in algorithms:
        values = df[(df['scale_label'] == scale) & (df['mode'] == algo)]['score_mean_last100'].values
        if len(values) > 0:
            data_to_plot.append(values)
            labels.append(algorithm_labels[algorithms.index(algo)])
    
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                    showmeans=False,
                    widths=0.55,
                    boxprops=dict(linewidth=0.8),
                    medianprops=dict(color='white', linewidth=1.2),
                    whiskerprops=dict(linewidth=0.8, color='black'),
                    capprops=dict(linewidth=0.8, color='black'),
                    flierprops=dict(marker='o', markersize=3, markerfacecolor='gray', 
                                   markeredgecolor='black', markeredgewidth=0.5, alpha=0.7))
    
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(colors[i])
        patch.set_alpha(0.85)
        patch.set_edgecolor('black')
        patch.set_linewidth(0.8)
    
    if scale_idx == 0:
        ax.set_ylabel('Reward', fontsize=9)
    ax.set_title(f'Scale {scale.upper()}', fontsize=9, pad=6)
    
    ax.grid(False)
    
    ax.tick_params(axis='both', which='major', labelsize=8, width=0.8, length=3, direction='out')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    
    plt.setp(ax.get_xticklabels(), rotation=0, ha='center', fontsize=8)

plt.tight_layout(pad=1.0, w_pad=1.5)
output_path_scale = os.path.join(script_dir, 'compare_reports', 'boxplot_by_scale.png')
plt.savefig(output_path_scale, dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none', 
            transparent=False, format='png')
output_path_pdf = os.path.join(script_dir, 'compare_reports', 'boxplot_by_scale.pdf')
plt.savefig(output_path_pdf, dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none', format='pdf')
print(f"Boxplot saved to: {output_path_scale}")
print(f"PDF saved to: {output_path_pdf}")
plt.close()

print("Boxplot generation completed!")
