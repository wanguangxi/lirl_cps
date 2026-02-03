#!/usr/bin/env python3
"""
Create a clean Gantt chart with utilization rates under the line labels.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
from pathlib import Path
from matplotlib.patches import FancyBboxPatch

# Nature journal style settings
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 8,
    'axes.titlesize': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'axes.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 2,
    'ytick.major.size': 2,
    'lines.linewidth': 0.5,
    'axes.unicode_minus': False,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

def load_schedule_data(json_file_path):
    """Load schedule data from JSON file."""
    with open(json_file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def plot_gantt_on_ax(ax, schedule_data, title_prefix=""):
    """
    Plot a clean Gantt chart on the given axes.
    
    Args:
        ax: Matplotlib axes object to plot on
        schedule_data: Dictionary containing the schedule information
        title_prefix: Prefix for the title (e.g., "Default" or "Optimized")
    """
    
    # Extract operation schedules
    operations = schedule_data['operation_schedules']
    
    # Get the number of production lines
    line_end_times = schedule_data['line_end_times']
    num_lines = len(line_end_times)
    
    # Nature journal-style color palette (muted, colorblind-friendly)
    workpiece_colors = [
        '#4477AA',  # Blue
        '#EE6677',  # Red
        '#228833',  # Green
        '#CCBB44',  # Yellow
        '#66CCEE',  # Cyan
        '#AA3377',  # Purple
        '#BBBBBB',  # Gray
    ]
    
    # Y-axis positions for each line
    line_positions = list(range(num_lines))
    line_height = 0.7
    
    # Plot each workpiece with detailed operation breakdown
    for op in operations:
        workpiece_id = op['workpiece']
        line_id = op['line']
        start_time = op['start']
        op_times = op['op_times']
        op_energies = op['op_energies']
        
        # Choose color for workpiece
        base_color = workpiece_colors[workpiece_id % len(workpiece_colors)]
        
        # Calculate total duration and positions
        total_duration = sum(op_times)
        
        # Draw individual operations within workpiece - clean Nature style
        current_time = start_time
        for i, (op_time, op_energy) in enumerate(zip(op_times, op_energies)):
            # Simple rectangle for each operation - no rounded corners, no gradient
            op_rect = patches.Rectangle(
                (current_time, line_id - line_height/2),
                op_time,
                line_height,
                facecolor=base_color,
                edgecolor='black',
                linewidth=0.5,
                alpha=0.7,
                zorder=2
            )
            ax.add_patch(op_rect)
            
            # Add time and energy labels - Nature style
            center_x = current_time + op_time/2
            
            # Time label (above the operation)
            if op_time > 3:  # Only show if operation is large enough
                ax.text(
                    center_x,
                    line_id + 0.35,
                    f'{op_time:.1f}s',
                    ha='center', va='bottom',
                    fontsize=7,
                    color='black',
                    zorder=4
                )
            
            # Energy label (below the operation)
            if op_time > 3:
                ax.text(
                    center_x,
                    line_id - 0.35,
                    f'{op_energy:.1f}kJ',
                    ha='center', va='top',
                    fontsize=7,
                    color='black',
                    zorder=4
                )
            
            # Operation number in the center (if space allows)
            if op_time > 4:
                ax.text(
                    center_x,
                    line_id,
                    f'{i+1}',
                    ha='center', va='center',
                    fontsize=7,
                    color='white',
                    zorder=3
                )
            
            current_time += op_time
    
    # Customize chart appearance - adjust ylim to accommodate labels
    ax.set_ylim(-0.6, num_lines - 0.1)
    ax.set_yticks(line_positions)
    
    # Create y-axis labels with utilization rates
    line_labels = []
    for i in range(num_lines):
        utilization = schedule_data['line_utilization'][i]
        line_labels.append(f'Line {i}\n({utilization:.1%})')
    
    ax.set_yticklabels(line_labels)
    
    # Set x-axis
    max_time = max(line_end_times)
    ax.set_xlim(-2, max_time * 1.05)
    ax.set_xlabel('Time (s)')
    
    # Set y-axis label
    ax.set_ylabel('Production lines')
    
    # Clean Nature journal style grid
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, color='gray', axis='y')
    ax.set_axisbelow(True)
    ax.set_facecolor('white')
    
    # Nature style: only show left and bottom spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Clean Nature journal style title
    makespan = schedule_data['makespan']
    total_energy = schedule_data['total_energy']
    
    title = f'{title_prefix} | Makespan: {makespan:.1f}s | Energy: {total_energy:.1f}kJ'
    ax.set_title(title, pad=10)
    
    # Clean Nature style legend
    legend_elements = []
    workpiece_ids = sorted(set(op['workpiece'] for op in operations))
    for wp_id in workpiece_ids:
        color = workpiece_colors[wp_id % len(workpiece_colors)]
        legend_elements.append(
            patches.Patch(color=color, label=f'Workpiece {wp_id}', alpha=0.7, edgecolor='black', linewidth=0.5)
        )
    
    ax.legend(
        handles=legend_elements,
        loc='upper right',
        ncol=len(legend_elements),
        frameon=False,
        columnspacing=0.8
    )

def create_clean_gantt_chart(schedule_data, output_path=None):
    """
    Create a clean Gantt chart with utilization rates under line labels.
    
    Args:
        schedule_data: Dictionary containing the schedule information
        output_path: Path to save the chart (optional)
    """
    # Create figure - Nature style: 7 inches wide
    # Height reduced by 1/3
    fig, ax = plt.subplots(figsize=(7, 2.67))
    plot_gantt_on_ax(ax, schedule_data, "Default Schedule")
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show the plot
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Clean Gantt chart saved to: {output_path}")
    else:
        plt.show()
    
    return fig, ax

def main():
    """Main function to create two Gantt charts in 2x1 layout."""
    
    # Get script directory and build paths relative to it
    script_dir = Path(__file__).parent
    
    # Paths to the two JSON files
    json_file_path_1 = script_dir / 'result' / 'workpieces_5' / 'training_best_schedule_5.json'
    json_file_path_2 = script_dir / 'result' / 'workpieces_5_opt' / 'training_best_schedule_5.json'
    
    # Output path for the combined chart
    output_path = script_dir / 'gantt_comparison.png'
    
    try:
        # Load schedule data for both scenarios
        print(f"Loading schedule data from: {json_file_path_1}")
        schedule_data_1 = load_schedule_data(str(json_file_path_1))
        
        print(f"Loading schedule data from: {json_file_path_2}")
        schedule_data_2 = load_schedule_data(str(json_file_path_2))
        
        # Create figure with 2 rows, 1 column - Nature style: 7 inches wide
        # Height: each subplot reduced by 1/3, overall height reduced
        fig, axes = plt.subplots(2, 1, figsize=(7, 5))
        
        # Plot first Gantt chart (workpieces_5)
        print(f"Creating Gantt chart for workpieces_5...")
        plot_gantt_on_ax(axes[0], schedule_data_1, "Default schedule")
        
        # Plot second Gantt chart (workpieces_5_opt)
        print(f"Creating Gantt chart for workpieces_5_opt...")
        plot_gantt_on_ax(axes[1], schedule_data_2, "Optimized schedule")
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the combined chart
        plt.savefig(str(output_path), dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Combined Gantt chart saved to: {output_path}")
        
        print(f"Gantt charts successfully created!")
        print(f"Features:")
        print(f"- Two Gantt charts in 2x1 layout")
        print(f"- Utilization rates displayed under line labels")
        print(f"- Clean layout with minimal text")
        print(f"- Operation numbers shown on workpieces")
        print(f"- Simple legend and title")
        
    except Exception as e:
        print(f"Error creating Gantt charts: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()