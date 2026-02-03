import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
import os

plt.style.use('seaborn-v0_8-darkgrid')
NATURE_COLORS = ['#3B4A6B', '#E07A5F', '#81B29A', '#F2CC8F', '#5F6F52', '#A44A3F']
sns.set_palette(NATURE_COLORS)

base_path = Path(os.path.dirname(os.path.abspath(__file__)))
generalization_path = base_path / "Generalization" / "generalization_results.csv"
machine_breakdown_path = base_path / "Machine_breakdown"
noise_level_path = base_path / "Noise_level"


def check_data_structure():
    """Check data structure and print column names"""
    print("Checking data structure...")
    
    if generalization_path.exists():
        gen_df = pd.read_csv(generalization_path)
        print(f"\nGeneralization data columns: {gen_df.columns.tolist()}")
        print(f"Generalization data first 5 rows:\n{gen_df.head()}")
    
    breakdown_folders = glob.glob(str(machine_breakdown_path / "ddpg_lirl_pi_multi_run_*_failure_*"))
    if breakdown_folders:
        sample_csv = glob.glob(str(Path(breakdown_folders[0]) / "ddpg_lirl_pi_all_episode_stats_*.csv"))
        if sample_csv:
            train_df = pd.read_csv(sample_csv[0])
            print(f"\nTraining data (Machine breakdown) columns: {train_df.columns.tolist()}")
            print(f"Training data first 5 rows:\n{train_df.head()}")
    
    noise_folders = glob.glob(str(noise_level_path / "ddpg_lirl_pi_multi_run_*_noise_*"))
    if noise_folders:
        sample_csv = glob.glob(str(Path(noise_folders[0]) / "ddpg_lirl_pi_all_episode_stats_*.csv"))
        if sample_csv:
            train_df = pd.read_csv(sample_csv[0])
            print(f"\nTraining data (Noise level) columns: {train_df.columns.tolist()}")
            print(f"Training data first 5 rows:\n{train_df.head()}")


def load_generalization_data():
    """Load generalization test results"""
    df = pd.read_csv(generalization_path)
    column_mapping = {
        'avg_makespan': 'makespan',
        'avg_energy': 'total_energy'
    }
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            df[new_col] = df[old_col]
    return df


def load_training_data(folder_path, pattern, config_type, config_value):
    """Load training data, taking only the last 100 data points for each seed"""
    all_data = []
    
    folders = glob.glob(str(folder_path / pattern))
    
    for folder in folders:
        csv_files = glob.glob(str(Path(folder) / "ddpg_lirl_pi_all_episode_stats_*.csv"))
        
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            
            if 'seed' in df.columns:
                grouped = df.groupby('seed')
                for seed, group in grouped:
                    last_100 = group.tail(100).copy()
                    last_100.loc[:, config_type] = config_value
                    last_100.loc[:, 'data_type'] = 'Training'
                    all_data.append(last_100[['seed', 'makespan', 'total_energy', config_type, 'data_type']])
            else:
                last_100 = df.tail(100).copy()
                last_100.loc[:, config_type] = config_value
                last_100.loc[:, 'data_type'] = 'Training'
                last_100.loc[:, 'seed'] = 0
                all_data.append(last_100[['seed', 'makespan', 'total_energy', config_type, 'data_type']])
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    else:
        return pd.DataFrame()


def prepare_data_for_comparison():
    """Prepare data for comparison"""
    check_data_structure()
    
    gen_data = load_generalization_data()
    print(f"\nGeneralization data actual columns: {gen_data.columns.tolist()}")
    
    failure_rates = [0.1, 0.3, 0.5]
    failure_data_list = []
    
    noise_col = 'nose_level'
    
    for fr in failure_rates:
        pattern = f"ddpg_lirl_pi_multi_run_*_failure_{fr}"
        train_data = load_training_data(machine_breakdown_path, pattern, 'failure_rate', fr)
        
        gen_subset = gen_data[(gen_data['failure_rate'] == fr) & (gen_data[noise_col] == 0)].copy()
        
        if not gen_subset.empty:
            gen_subset['data_type'] = 'Generalization'
            
            if not train_data.empty:
                common_cols = ['seed', 'makespan', 'total_energy', 'failure_rate', 'data_type']
                train_data = train_data[common_cols]
                gen_subset = gen_subset[common_cols]
                failure_data_list.append(pd.concat([train_data, gen_subset], ignore_index=True))
            else:
                gen_subset = gen_subset[['seed', 'makespan', 'total_energy', 'failure_rate', 'data_type']]
                failure_data_list.append(gen_subset)
    
    failure_comparison_data = pd.concat(failure_data_list, ignore_index=True) if failure_data_list else pd.DataFrame()
    
    noise_levels = [0.1, 0.3, 0.5]
    noise_data_list = []
    
    for nl in noise_levels:
        pattern = f"ddpg_lirl_pi_multi_run_*_noise_{nl}"
        train_data = load_training_data(noise_level_path, pattern, 'noise_level', nl)
        
        gen_subset = gen_data[(gen_data['failure_rate'] == 0) & (gen_data[noise_col] == nl)].copy()
        
        if not gen_subset.empty:
            gen_subset['data_type'] = 'Generalization'
            gen_subset['noise_level'] = nl
            
            if not train_data.empty:
                common_cols = ['seed', 'makespan', 'total_energy', 'noise_level', 'data_type']
                train_data = train_data[common_cols]
                gen_subset = gen_subset[common_cols]
                noise_data_list.append(pd.concat([train_data, gen_subset], ignore_index=True))
            else:
                gen_subset = gen_subset[['seed', 'makespan', 'total_energy', 'noise_level', 'data_type']]
                noise_data_list.append(gen_subset)
    
    noise_comparison_data = pd.concat(noise_data_list, ignore_index=True) if noise_data_list else pd.DataFrame()
    
    print(f"\nFailure rate data statistics:")
    if not failure_comparison_data.empty:
        print(failure_comparison_data.groupby(['failure_rate', 'data_type']).size())
    else:
        print("No data")
    
    print(f"\nNoise level data statistics:")
    if not noise_comparison_data.empty:
        print(noise_comparison_data.groupby(['noise_level', 'data_type']).size())
    else:
        print("No data")
    
    return failure_comparison_data, noise_comparison_data


def calculate_generalization_metrics(failure_data, noise_data):
    """Calculate generalization performance metrics (including weighted combination)"""
    failure_data = add_weighted_metric(failure_data)
    noise_data = add_weighted_metric(noise_data)
    metrics = {}

    def _aggregate(df, key_cols):
        grouped = df.groupby(key_cols)
        rows = []
        for keys, grp in grouped:
            stat = dict(zip(key_cols, keys if isinstance(keys, tuple) else [keys]))
            stat.update({
                'makespan_mean': grp['makespan'].mean(),
                'makespan_std': grp['makespan'].std(),
                'energy_mean': grp['total_energy'].mean(),
                'energy_std': grp['total_energy'].std(),
                'weighted_mean': grp['weighted_score'].mean(),
                'weighted_std': grp['weighted_score'].std(),
                'sample_size': len(grp)
            })
            rows.append(stat)
        return pd.DataFrame(rows)

    if not failure_data.empty:
        metrics['failure'] = _aggregate(failure_data, ['failure_rate', 'data_type'])
    if not noise_data.empty:
        metrics['noise'] = _aggregate(noise_data, ['noise_level', 'data_type'])
    return metrics


def add_weighted_metric(df: pd.DataFrame, w: float = 0.5) -> pd.DataFrame:
    """
    Add normalized weighted combination metric `weighted_score` to the dataframe.
    weighted_score = w * norm_makespan + (1-w) * norm_energy
    """
    if df.empty:
        return df
    df = df.copy()
    makespan_min, makespan_max = df['makespan'].min(), df['makespan'].max()
    energy_min, energy_max = df['total_energy'].min(), df['total_energy'].max()
    df['norm_makespan'] = (df['makespan'] - makespan_min) / (makespan_max - makespan_min + 1e-9)
    df['norm_energy'] = (df['total_energy'] - energy_min) / (energy_max - energy_min + 1e-9)
    df['weighted_score'] = w * df['norm_makespan'] + (1 - w) * df['norm_energy']
    return df


def plot_overall_comparison(failure_df: pd.DataFrame,
                            noise_df: pd.DataFrame,
                            metrics, w: float = 0.5):
    """
    1x4 overall comparison plot (Nature paper style):
      Noise: Makespan | Noise: Energy | Failure: Makespan | Failure: Energy
    """
    if failure_df.empty and noise_df.empty:
        return

    plt.style.use('default')
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['font.size'] = 8
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['xtick.major.width'] = 0.8
    plt.rcParams['ytick.major.width'] = 0.8
    plt.rcParams['xtick.direction'] = 'out'
    plt.rcParams['ytick.direction'] = 'out'
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 1.7))
    fig.patch.set_facecolor('white')

    palette = ['#3498DB', '#E74C3C']

    def _violin(ax, df, x_col, y_col, title, xlabel):
        if df.empty:
            ax.set_visible(False)
            return
        sns.violinplot(
            data=df, x=x_col, y=y_col, hue='data_type',
            split=True, inner='quart', palette=palette, ax=ax,
            linewidth=0.8
        )
        
        for collection in ax.collections:
            collection.set_alpha(0.6)
        
        for line in ax.lines:
            line.set_color('#5D6D7E')
            line.set_linewidth(0.8)
        
        ax.set_xlabel(xlabel, fontsize=8, fontfamily='Arial')
        ax.set_ylabel(y_col.replace('_', ' ').title(), fontsize=8, fontfamily='Arial')
        ax.set_title(title, fontsize=9, fontfamily='Arial', pad=4)
        if ax.get_legend():
            ax.get_legend().remove()
        ax.tick_params(axis='both', labelsize=7, width=0.8, length=3)
        ax.spines['left'].set_linewidth(0.8)
        ax.spines['bottom'].set_linewidth(0.8)

        cats = sorted(df[x_col].unique())
        for idx, cfg in enumerate(cats):
            tr = df[(df[x_col] == cfg) & (df['data_type'] == 'Training')][y_col]
            te = df[(df[x_col] == cfg) & (df['data_type'] == 'Generalization')][y_col]
            cov = coverage_ratio(tr, te)
            if np.isnan(cov):
                continue
            y_max = df[df[x_col] == cfg][y_col].max()
            y_pos = y_max * 1.02
            x_pos = idx - 0.3
            ax.text(x_pos, y_pos, f'{cov*100:.0f}%',
                    ha='left', va='bottom', fontsize=6, fontfamily='Arial', color='black')

    _violin(axes[0], noise_df, 'noise_level', 'makespan', 'Noise: Makespan', 'Noise level (μ)')
    _violin(axes[1], noise_df, 'noise_level', 'total_energy', 'Noise: Energy', 'Noise level (μ)')
    _violin(axes[2], failure_df, 'failure_rate', 'makespan', 'Failure: Makespan', 'Failure rate')
    _violin(axes[3], failure_df, 'failure_rate', 'total_energy', 'Failure: Energy', 'Failure rate')

    if not failure_df.empty:
        fail_levels = sorted(failure_df['failure_rate'].unique())
        pct_labels = [f'{int(lv*100)}%' for lv in fail_levels]
        for ax in axes[2:]:
            ax.set_xticks(range(len(fail_levels)))
            ax.set_xticklabels(pct_labels, fontfamily='Arial', fontsize=7)

    from matplotlib.patches import Patch
    custom_handles = [
        Patch(facecolor='#3498DB', alpha=0.6, edgecolor='#3498DB', label='Training'),
        Patch(facecolor='#E74C3C', alpha=0.6, edgecolor='#E74C3C', label='Generalization')
    ]
    
    plt.tight_layout(rect=[0, 0.12, 1, 1])
    
    fig.legend(handles=custom_handles, loc='lower center', ncol=2, fontsize=7,
              frameon=True, fancybox=False, edgecolor='#D1D5DB',
              facecolor='white', framealpha=1, prop={'family': 'Arial'},
              bbox_to_anchor=(0.5, 0.01))
    plt.savefig('overall_comparison.png', dpi=600, bbox_inches='tight', facecolor='white')
    plt.savefig('overall_comparison.pdf', dpi=600, bbox_inches='tight', facecolor='white', format='pdf')
    print("Saved: overall_comparison.png")
    print("Saved: overall_comparison.pdf")
    plt.show()
    plt.close(fig)


def coverage_ratio(train_s: pd.Series, test_s: pd.Series) -> float:
    """
    Generalization coverage ratio:
    Returns the proportion of Generalization samples that fall within the Training [min, max] range.
    Returns NaN for empty sets.
    """
    if train_s.empty or test_s.empty:
        return np.nan
    lo, hi = train_s.min(), train_s.max()
    return ((test_s >= lo) & (test_s <= hi)).mean()


def main():
    """Main function"""
    print("Starting robustness experiment data analysis...")
    
    print("Loading and preparing data...")
    failure_data, noise_data = prepare_data_for_comparison()
    
    print("\nCalculating generalization performance metrics...")
    metrics = calculate_generalization_metrics(failure_data, noise_data)
    
    print("Generating overall comparison plot...")
    plot_overall_comparison(failure_data, noise_data, metrics)
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
