"""
Runtime Profiling Comparison Script
Compare NN inference, discrete mapping, continuous mapping, and total decision time
across three scenarios (CityFlow, EV-Charging, RMS) at small, medium, and large scales.

Style: Nature journal format
- Font: Arial 8pt
- Width: 7 inches
- All text in English
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Nature journal style settings
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 8,
    'axes.titlesize': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7,
    'figure.titlesize': 9,
    'axes.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'lines.linewidth': 1,
    'axes.unicode_minus': True,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

def load_cityflow_data(base_path):
    """Load CityFlow runtime data from summary.csv"""
    csv_path = base_path / "Cityflow_runtime_scaling_20260113_211415" / "summary.csv"
    df = pd.read_csv(csv_path, encoding='utf-8')
    
    # Extract timing data for small, medium, large scales
    data = {
        'small': {
            'nn_inference': df.iloc[0]['NNInference(ms)'],
            'discrete_mapping': df.iloc[0]['DiscreteMapping(ms)'],
            'continuous_mapping': df.iloc[0]['ContinuousMapping(ms)'],
            'total_decision': df.iloc[0]['TotalDecision(ms)']
        },
        'medium': {
            'nn_inference': df.iloc[1]['NNInference(ms)'],
            'discrete_mapping': df.iloc[1]['DiscreteMapping(ms)'],
            'continuous_mapping': df.iloc[1]['ContinuousMapping(ms)'],
            'total_decision': df.iloc[1]['TotalDecision(ms)']
        },
        'large': {
            'nn_inference': df.iloc[2]['NNInference(ms)'],
            'discrete_mapping': df.iloc[2]['DiscreteMapping(ms)'],
            'continuous_mapping': df.iloc[2]['ContinuousMapping(ms)'],
            'total_decision': df.iloc[2]['TotalDecision(ms)']
        }
    }
    return data

def load_evcharging_data(base_path):
    """Load EV-Charging runtime data from runtime_summary CSV"""
    csv_path = base_path / "EV-charging_runtime_scaling_exp_20251218_130354" / "runtime_summary_20251218_131057.csv"
    df = pd.read_csv(csv_path)
    
    # Column mapping: Policy Network = NN inference, Hungarian = discrete, QP = continuous
    data = {
        'small': {
            'nn_inference': df.iloc[0]['Policy Network (ms)'],
            'discrete_mapping': df.iloc[0]['Hungarian (ms)'],
            'continuous_mapping': df.iloc[0]['QP (ms)'],
            'total_decision': df.iloc[0]['Total (ms)']
        },
        'medium': {
            'nn_inference': df.iloc[1]['Policy Network (ms)'],
            'discrete_mapping': df.iloc[1]['Hungarian (ms)'],
            'continuous_mapping': df.iloc[1]['QP (ms)'],
            'total_decision': df.iloc[1]['Total (ms)']
        },
        'large': {
            'nn_inference': df.iloc[2]['Policy Network (ms)'],
            'discrete_mapping': df.iloc[2]['Hungarian (ms)'],
            'continuous_mapping': df.iloc[2]['QP (ms)'],
            'total_decision': df.iloc[2]['Total (ms)']
        }
    }
    return data

def load_rms_data(base_path):
    """Load RMS runtime data from results_summary.json"""
    json_path = base_path / "RMS_runtime_scaling_20251218_152208" / "results_summary.json"
    with open(json_path, 'r') as f:
        raw_data = json.load(f)
    
    experiments = raw_data['experiments']
    data = {}
    for scale in ['small', 'medium', 'large']:
        timing = experiments[scale]['timing']
        data[scale] = {
            'nn_inference': timing['network_forward_ms'],
            'discrete_mapping': timing['hungarian_ms'],
            'continuous_mapping': timing['qp_ms'],
            'total_decision': timing['total_decision_ms']
        }
    return data

def create_comparison_plot(cityflow_data, evcharging_data, rms_data, output_path):
    """Create a comparison plot for all scenarios and metrics - Nature style"""
    
    # Define metrics and their English labels
    metrics = ['nn_inference', 'discrete_mapping', 'continuous_mapping', 'total_decision']
    metric_labels = ['NN Inference', 'Discrete Mapping', 'Continuous Mapping', 'Total Decision']
    
    # Define scales
    scales = ['small', 'medium', 'large']
    scale_labels = ['Small', 'Medium', 'Large']
    
    # Define scenarios
    scenarios = ['CityFlow', 'EV-Charging', 'RMS']
    scenario_data = [cityflow_data, evcharging_data, rms_data]
    
    # Nature-style colors (muted, professional)
    colors = ['#E64B35', '#4DBBD5', '#00A087']  # Red, Cyan, Teal
    
    # Create figure: 7 inches wide, aspect ratio for 2x2 layout
    fig, axes = plt.subplots(2, 2, figsize=(7, 5))
    axes = axes.flatten()
    
    x = np.arange(len(scales))
    width = 0.25
    
    for idx, (metric, metric_label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx]
        
        for i, (scenario, data, color) in enumerate(zip(scenarios, scenario_data, colors)):
            values = [data[scale][metric] for scale in scales]
            offset = (i - 1) * width
            bars = ax.bar(x + offset, values, width, label=scenario, color=color, 
                         alpha=0.9, edgecolor='none', linewidth=0)
            
            # Add value labels on top of bars (smaller font)
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.annotate(f'{val:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 1),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=6)
        
        ax.set_xlabel('Scale', fontsize=8)
        ax.set_ylabel('Time (ms)', fontsize=8)
        ax.set_title(metric_label, fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(scale_labels, fontsize=8)
        
        # Only show legend in first subplot
        if idx == 0:
            ax.legend(loc='upper left', fontsize=7, frameon=False)
        
        # Clean axis style
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)
        ax.tick_params(width=0.5)
        
        # Set y-axis to start from 0
        ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved comparison plot to: {output_path}")
    print(f"Saved PDF version to: {output_path.with_suffix('.pdf')}")

def create_grouped_bar_table(cityflow_data, evcharging_data, rms_data, output_path):
    """Create a comprehensive table comparing all metrics"""
    
    metrics = ['nn_inference', 'discrete_mapping', 'continuous_mapping', 'total_decision']
    metric_labels = ['NN Inference (ms)', 'Discrete Mapping (ms)', 'Continuous Mapping (ms)', 'Total Decision (ms)']
    scales = ['small', 'medium', 'large']
    scale_labels = ['Small', 'Medium', 'Large']
    
    scenarios = ['CityFlow', 'EV-Charging', 'RMS']
    scenario_data = [cityflow_data, evcharging_data, rms_data]
    
    # Create DataFrame for the table
    rows = []
    for scenario, data in zip(scenarios, scenario_data):
        for scale, scale_label in zip(scales, scale_labels):
            row = {
                'Scenario': scenario,
                'Scale': scale_label,
                'NN Inference (ms)': data[scale]['nn_inference'],
                'Discrete Mapping (ms)': data[scale]['discrete_mapping'],
                'Continuous Mapping (ms)': data[scale]['continuous_mapping'],
                'Total Decision (ms)': data[scale]['total_decision']
            }
            rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Save to CSV
    csv_output = output_path.with_suffix('.csv')
    df.to_csv(csv_output, index=False, encoding='utf-8-sig')
    print(f"Saved comparison table to: {csv_output}")
    
    return df

def create_stacked_bar_plot(cityflow_data, evcharging_data, rms_data, output_path):
    """Create stacked bar chart showing composition of total decision time - Nature style"""
    
    scales = ['small', 'medium', 'large']
    scale_labels = ['Small', 'Medium', 'Large']
    scenarios = [r'$R^2$AMS', r'Traffic control', r'EV charging']
    scenario_data = [rms_data, cityflow_data, evcharging_data]
    
    # 7 inches wide, appropriate height for 1x3 layout
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.0))
    
    # Component colors (Nature-style muted palette)
    component_colors = ['#E64B35', '#4DBBD5', '#00A087']
    components = ['nn_inference', 'discrete_mapping', 'continuous_mapping']
    component_labels = ['NN inference', 'Discrete projection', 'Continuous projection']
    
    for ax, (scenario, data) in zip(axes, zip(scenarios, scenario_data)):
        x = np.arange(len(scales))
        bottom = np.zeros(len(scales))
        
        for comp, color, label in zip(components, component_colors, component_labels):
            values = np.array([data[scale][comp] for scale in scales])
            ax.bar(x, values, bottom=bottom, label=label, color=color, 
                   edgecolor='none', linewidth=0, width=0.6)
            bottom += values
        
        # Add total decision time labels on top of stacked bars
        total_values = [data[scale]['total_decision'] for scale in scales]
        for i, (xi, total) in enumerate(zip(x, total_values)):
            ax.annotate(f'{total:.2f}',
                       xy=(xi, bottom[i]),
                       xytext=(0, 2),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=6, fontweight='bold')
        
        # ax.set_xlabel('Scale', fontsize=8)
        ax.set_ylabel('Time (ms)', fontsize=8)
        ax.set_title(scenario, fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(scale_labels, fontsize=8)
        
        # Clean axis style
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)
        ax.tick_params(width=0.5)
    
    # Add legend to the last subplot
    axes[-1].legend(loc='upper left', fontsize=6, frameon=False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved stacked bar plot to: {output_path}")
    print(f"Saved PDF version to: {output_path.with_suffix('.pdf')}")

def main():
    # Get base path
    base_path = Path(__file__).parent
    
    print("=" * 60)
    print("Runtime Profiling Comparison (Nature Style)")
    print("=" * 60)
    
    # Load data from all three scenarios
    print("\nLoading data...")
    cityflow_data = load_cityflow_data(base_path)
    print("  [OK] CityFlow data loaded")
    
    evcharging_data = load_evcharging_data(base_path)
    print("  [OK] EV-Charging data loaded")
    
    rms_data = load_rms_data(base_path)
    print("  [OK] RMS data loaded")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Data Summary")
    print("=" * 60)
    
    scenarios = ['CityFlow', 'EV-Charging', 'RMS']
    all_data = [cityflow_data, evcharging_data, rms_data]
    
    for scenario, data in zip(scenarios, all_data):
        print(f"\n{scenario}:")
        print(f"  {'Scale':<10} {'NN Inference':<15} {'Discrete Map':<15} {'Continuous Map':<15} {'Total Decision':<15}")
        print("  " + "-" * 70)
        for scale, label in [('small', 'Small'), ('medium', 'Medium'), ('large', 'Large')]:
            d = data[scale]
            print(f"  {label:<10} {d['nn_inference']:<15.4f} {d['discrete_mapping']:<15.4f} "
                  f"{d['continuous_mapping']:<15.4f} {d['total_decision']:<15.4f}")
    
    # Generate plots
    print("\n" + "=" * 60)
    print("Generating Visualizations (Nature Style)")
    print("=" * 60)
    
    # Main comparison plot (4 subplots)
    output_comparison = base_path / "runtime_comparison_all_scenarios.png"
    create_comparison_plot(cityflow_data, evcharging_data, rms_data, output_comparison)
    
    # Stacked bar plot
    output_stacked = base_path / "runtime_composition_stacked.png"
    create_stacked_bar_plot(cityflow_data, evcharging_data, rms_data, output_stacked)
    
    # Create summary table
    output_table = base_path / "runtime_comparison_table"
    df = create_grouped_bar_table(cityflow_data, evcharging_data, rms_data, output_table)
    
    print("\n" + "=" * 60)
    print("Comparison Table")
    print("=" * 60)
    print(df.to_string(index=False))
    
    print("\n" + "=" * 60)
    print("All outputs generated successfully!")
    print("  - Figure width: 7 inches")
    print("  - Font: Arial 8pt")
    print("  - Format: PNG (300 DPI) + PDF")
    print("=" * 60)

if __name__ == "__main__":
    main()
