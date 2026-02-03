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

def load_schedule_data(json_file_path):
    """Load schedule data from JSON file."""
    with open(json_file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_clean_gantt_chart(schedule_data, output_path=None):
    """
    Create a clean Gantt chart with utilization rates under line labels.
    
    Args:
        schedule_data: Dictionary containing the schedule information
        output_path: Path to save the chart (optional)
    """
    
    # Extract operation schedules
    operations = schedule_data['operation_schedules']
    
    # Get the number of production lines
    line_end_times = schedule_data['line_end_times']
    num_lines = len(line_end_times)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Define Nature journal-style color palette
    workpiece_colors = [
        '#1f77b4',  # Professional Blue
        '#ff7f0e',  # Warm Orange
        '#2ca02c',  # Natural Green
        '#d62728',  # Clear Red
        '#9467bd',  # Soft Purple
        '#8c564b',  # Earth Brown
        '#e377c2',  # Rose Pink
        '#7f7f7f',  # Neutral Gray
        '#bcbd22',  # Olive Green
        '#17becf'   # Cyan Blue
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
        
        # Draw main workpiece background with subtle shadow effect
        shadow_rect = FancyBboxPatch(
            (start_time + 0.3, line_id - line_height/2 - 0.015),
            total_duration,
            line_height,
            boxstyle="round,pad=0.01",
            facecolor='#f0f0f0',
            alpha=0.6,
            zorder=0
        )
        ax.add_patch(shadow_rect)
        
        # Draw main workpiece background with clean style
        main_rect = FancyBboxPatch(
            (start_time, line_id - line_height/2),
            total_duration,
            line_height,
            boxstyle="round,pad=0.01",
            facecolor=base_color,
            edgecolor='white',
            linewidth=0.5,  # Set to 0.5
            alpha=0.25,
            zorder=1
        )
        ax.add_patch(main_rect)
        
        # Draw individual operations within workpiece
        current_time = start_time
        for i, (op_time, op_energy) in enumerate(zip(op_times, op_energies)):
            # Create clean gradient effect for operations
            alpha_values = [0.8, 0.65, 0.5]
            alpha = alpha_values[i % len(alpha_values)]
            
            # Create clean rectangle for each operation
            op_rect = FancyBboxPatch(
                (current_time, line_id - line_height/2 + 0.04),
                op_time,
                line_height - 0.08,
                boxstyle="round,pad=0.005",
                facecolor=base_color,
                edgecolor='#333333',
                linewidth=0.5,  # Set to 0.5
                alpha=alpha,
                zorder=2
            )
            ax.add_patch(op_rect)
            
            # Add operation index number with clean style and Arial Bold (11pt)
            if op_time > 3:  # Only show numbers for operations that have enough space
                op_text = ax.text(
                    current_time + op_time/2,
                    line_id,
                    f'{i+1}',
                    ha='center', va='center',
                    fontsize=11, fontweight='bold', fontfamily='Arial',
                    color='white',
                    bbox=dict(boxstyle="circle,pad=0.15", facecolor='#2c3e50', edgecolor='white', linewidth=0.5, alpha=0.9),
                    zorder=4
                )
            
            # Add time and energy annotations with units (s, kJ) and Arial Bold (11pt)
            if op_time > 5:  # Show detailed info for longer operations
                # Time annotation (above) - Nature style with Arial Bold (11pt)
                ax.text(
                    current_time + op_time/2,
                    line_id + 0.25,
                    f'{op_time:.1f}s',
                    ha='center', va='center',
                    fontsize=11, fontweight='bold', fontfamily='Arial',
                    color='#2c3e50',
                    bbox=dict(boxstyle="round,pad=0.15", facecolor='#ecf0f1', alpha=0.95, edgecolor='#34495e', linewidth=0.5),
                    zorder=5
                )
                
                # Energy annotation (below) - Nature style with Arial Bold (11pt)
                ax.text(
                    current_time + op_time/2,
                    line_id - 0.25,
                    f'{op_energy:.1f}kJ',
                    ha='center', va='center',
                    fontsize=11, fontweight='bold', fontfamily='Arial',
                    color='#c0392b',
                    bbox=dict(boxstyle="round,pad=0.15", facecolor='#fadbd8', alpha=0.95, edgecolor='#e74c3c', linewidth=0.5),
                    zorder=5
                )
            elif op_time > 2:  # Show compact info for shorter operations
                # Compact annotation with clean style and Arial Bold (11pt)
                ax.text(
                    current_time + op_time/2,
                    line_id + 0.15,
                    f'{op_time:.1f}s\n{op_energy:.1f}kJ',
                    ha='center', va='center',
                    fontsize=11, fontweight='bold', fontfamily='Arial',
                    color='#2c3e50',
                    bbox=dict(boxstyle="round,pad=0.1", facecolor='#f8f9fa', alpha=0.9, edgecolor='#bdc3c7', linewidth=0.5),
                    zorder=5
                )
            
            current_time += op_time
        
        # Add clean workpiece label with Arial Bold (11pt) - positioned slightly lower
        workpiece_center_x = start_time + total_duration/2
        workpiece_center_y = line_id
        
        # Main workpiece ID with Nature journal style and Arial Bold (11pt) - moved down
        wp_text = ax.text(
            workpiece_center_x,
            workpiece_center_y - 0.45,  # Positioned below workpiece bar
            f'W{workpiece_id}',
            ha='center', va='center',
            fontsize=11, fontweight='bold', fontfamily='Arial',
            color='white',
            bbox=dict(
                boxstyle="round,pad=0.25", 
                facecolor=base_color, 
                edgecolor='white', 
                linewidth=0.5,  # Set to 0.5
                alpha=0.95
            ),
            zorder=3
        )
    
    # Customize chart appearance with Arial Bold font (11pt)
    ax.set_ylim(-0.8, num_lines - 0.2)
    ax.set_yticks(line_positions)
    
    # Create y-axis labels with utilization rates using Arial Bold (11pt)
    line_labels = []
    for i in range(num_lines):
        utilization = schedule_data['line_utilization'][i]
        line_labels.append(f'Line {i}\n({utilization:.1%})')
    
    ax.set_yticklabels(line_labels, fontsize=11, fontweight='bold', fontfamily='Arial')
    
    # Set x-axis with Arial Bold (12pt) Black
    max_time = max(line_end_times)
    ax.set_xlim(-2, max_time * 1.05)
    ax.set_xlabel('Time (s)', fontsize=12, fontweight='bold', fontfamily='Arial', color='black')
    
    # Set y-axis label with Arial Bold (12pt) Black
    ax.set_ylabel('Production Lines', fontsize=12, fontweight='bold', fontfamily='Arial', color='black')
    
    # Set x-axis tick labels to Arial Bold (11pt)
    for label in ax.get_xticklabels():
        label.set_fontfamily('Arial')
        label.set_fontweight('bold')
        label.set_fontsize(11)
    
    # Add clean grid with Nature journal style
    ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.5, color='#bdc3c7')
    ax.set_axisbelow(True)
    ax.set_facecolor('white')  # Pure white background
    
    # Add time markers (vertical lines only, no text above) with clean style
    time_intervals = np.arange(0, max_time + 1, 20)
    for t in time_intervals:
        if t > 0:  # Don't show 0
            ax.axvline(x=t, color='#95a5a6', alpha=0.5, linestyle='--', linewidth=0.5)
    
    # Clean Nature journal style title with Arial Bold (12pt) Black
    makespan = schedule_data['makespan']
    total_energy = schedule_data['total_energy']
    violations = schedule_data['violations']
    
    title = f'Default schedule | Makespan: {makespan:.1f} | Energy: {total_energy:.1f}'
    ax.set_title(title, fontsize=12, fontweight='bold', fontfamily='Arial', pad=15, color='black')
    
    # Create clean legend with Nature style and Arial Bold (11pt)
    legend_elements = []
    workpiece_ids = sorted(set(op['workpiece'] for op in operations))
    for wp_id in workpiece_ids:
        color = workpiece_colors[wp_id % len(workpiece_colors)]
        legend_elements.append(
            patches.Patch(color=color, label=f'Workpiece {wp_id}', alpha=0.8)
        )
    
    legend = ax.legend(
        handles=legend_elements,
        loc='upper right',
        ncol=len(legend_elements),  # Arrange all workpieces horizontally
        fontsize=11,
        frameon=True,
        fancybox=False,
        shadow=False,
        edgecolor='#bdc3c7',
        facecolor='white',
        framealpha=0.95,
        columnspacing=0.8  # Compact column spacing for upper right corner
    )
    
    # Set legend text to Arial Bold (11pt)
    for text in legend.get_texts():
        text.set_fontfamily('Arial')
        text.set_fontweight('bold')
        text.set_fontsize(11)
    
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
    """Main function to create clean Gantt chart."""
    
    # Get script directory and build paths relative to it
    script_dir = Path(__file__).parent
    json_file_path = script_dir / 'result' / 'workpieces_5_opt' / 'training_best_schedule_5.json'
    
    # Output path for the clean chart
    output_dir = json_file_path.parent
    output_path = output_dir / 'clean_gantt_with_utilization.png'
    
    try:
        # Load schedule data
        print(f"Loading schedule data from: {json_file_path}")
        schedule_data = load_schedule_data(str(json_file_path))
        
        # Create clean Gantt chart
        print(f"Creating clean Gantt chart with utilization rates...")
        fig, ax = create_clean_gantt_chart(schedule_data, str(output_path))
        
        print(f"Clean Gantt chart successfully created!")
        print(f"Features:")
        print(f"- Utilization rates displayed under line labels")
        print(f"- Clean layout with minimal text")
        print(f"- Operation numbers shown on workpieces")
        print(f"- Simple legend and title")
        
    except Exception as e:
        print(f"Error creating clean Gantt chart: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()