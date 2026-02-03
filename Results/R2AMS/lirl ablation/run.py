"""
LIRL vs MASK Learning Curves Comparison - Nature Style
Compare learning curves between LIRL and MASK algorithms at different scales
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

class LIRLMaskComparison:
    def __init__(self, base_path=None):
        if base_path is None:
            base_path = os.path.dirname(os.path.abspath(__file__))
        self.base_path = Path(base_path)
        
        # Define scales and folder patterns
        self.scales = ['10_3', '20_3', '50_5', '100_5']
        self.scale_labels = {
            '10_3': 'Scale A',
            '20_3': 'Scale B',
            '50_5': 'Scale C',
            '100_5': 'Scale D'
        }
        
        # Algorithm configurations
        self.algorithms = {
            'LIRL': 'ddpg_lirl_pi_multi_run_',
            'Mask': 'ddpg_mask_multi_run_'
        }
        
        # Nature style colors
        self.colors = {
            'LIRL': '#E64B35',   # Red
            'Mask': '#4DBBD5'    # Cyan
        }
        
        self.data = {}
        
    def load_data(self):
        """Load score data from all folders"""
        print(f"Loading data from: {self.base_path}")
        
        for scale in self.scales:
            self.data[scale] = {}
            
            for algo_name, folder_prefix in self.algorithms.items():
                folder_name = f"{folder_prefix}{scale}"
                folder_path = self.base_path / folder_name
                
                if folder_path.exists():
                    # Find scores file
                    score_files = list(folder_path.glob('*_all_scores_*.npy'))
                    
                    if score_files:
                        scores_raw = np.load(score_files[0], allow_pickle=True)
                        scores = self._normalize_scores(scores_raw)
                        
                        self.data[scale][algo_name] = {
                            'scores': scores,
                            'mean': np.mean(scores, axis=0),
                            'std': np.std(scores, axis=0),
                            'num_runs': scores.shape[0],
                            'num_episodes': scores.shape[1]
                        }
                        
                        print(f"  ✓ {algo_name} @ {self.scale_labels[scale]}: "
                              f"{scores.shape[0]} runs, {scores.shape[1]} episodes")
                    else:
                        print(f"  ✗ No scores file found for {algo_name} @ {scale}")
                else:
                    print(f"  ✗ Folder not found: {folder_name}")
    
    def _normalize_scores(self, scores):
        """Normalize scores to 2D array [runs, episodes]"""
        arr = np.array(scores, dtype=object)
        if arr.dtype == object:
            runs = [np.asarray(r).astype(float).reshape(-1) for r in arr]
            if len(runs) == 0:
                return np.zeros((0, 0))
            min_len = min(len(r) for r in runs) if all(len(r) > 0 for r in runs) else 0
            if min_len == 0:
                return np.zeros((len(runs), 0))
            return np.stack([r[:min_len] for r in runs], axis=0)
        arr = np.asarray(scores)
        if arr.ndim == 1:
            return arr[np.newaxis, :]
        return arr
    
    def plot_comparison_1x4(self):
        """Plot 1x4 comparison figure - Nature style for paper"""
        # Nature paper style settings - All fonts use Arial
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['font.sans-serif'] = ['Arial']
        plt.rcParams['font.size'] = 8
        plt.rcParams['axes.linewidth'] = 0.8
        plt.rcParams['axes.spines.top'] = False
        plt.rcParams['axes.spines.right'] = False
        plt.rcParams['axes.labelsize'] = 9
        plt.rcParams['axes.titlesize'] = 9
        plt.rcParams['xtick.labelsize'] = 8
        plt.rcParams['ytick.labelsize'] = 8
        plt.rcParams['xtick.major.width'] = 0.8
        plt.rcParams['ytick.major.width'] = 0.8
        plt.rcParams['xtick.direction'] = 'out'
        plt.rcParams['ytick.direction'] = 'out'
        plt.rcParams['lines.linewidth'] = 1.0
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['legend.fontsize'] = 7
        
        # Create 1x4 figure (Nature double column width: 183mm = 7.2 inches)
        fig, axes = plt.subplots(1, 4, figsize=(7.2, 1.5))
        fig.patch.set_facecolor('white')
        
        for idx, scale in enumerate(self.scales):
            ax = axes[idx]
            
            if scale in self.data and self.data[scale]:
                for algo_name in ['LIRL', 'Mask']:
                    if algo_name in self.data[scale]:
                        data = self.data[scale][algo_name]
                        episodes = np.arange(data['num_episodes'])
                        mean = data['mean']
                        std = data['std']
                        color = self.colors[algo_name]
                        
                        # Plot mean line with Nature style
                        lw = 1.2 if algo_name == 'LIRL' else 1.0
                        ax.plot(episodes, mean, label=algo_name, color=color,
                               linewidth=lw, alpha=0.95)
                        
                        # Plot confidence interval (shaded area)
                        fill_alpha = 0.25 if algo_name == 'LIRL' else 0.15
                        ax.fill_between(episodes, mean - std, mean + std,
                                       color=color, alpha=fill_alpha, linewidth=0)
                
                ax.set_xlabel('Episode', fontsize=9, fontfamily='Arial')
                if idx == 0:
                    ax.set_ylabel('Reward', fontsize=9, fontfamily='Arial')
                ax.set_title(f'{self.scale_labels[scale]}', fontsize=9, fontfamily='Arial', pad=2)
                
                # Legend only in first subplot
                if idx == 0:
                    legend = ax.legend(fontsize=7, loc='lower right', frameon=True,
                             fancybox=False, edgecolor='#D1D5DB', framealpha=0.95,
                             handlelength=1.2, handletextpad=0.4, borderpad=0.3,
                             prop={'family': 'Arial'})
                
                ax.tick_params(axis='both', which='major', labelsize=8, width=0.8, length=3)
                # Set tick label font to Arial
                for label in ax.get_xticklabels() + ax.get_yticklabels():
                    label.set_fontfamily('Arial')
                
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_linewidth(0.8)
                ax.spines['bottom'].set_linewidth(0.8)
                ax.grid(False)
            else:
                ax.text(0.5, 0.5, f'No data',
                       ha='center', va='center', transform=ax.transAxes, fontsize=8, fontfamily='Arial')
                ax.set_title(f'{self.scale_labels[scale]}', fontsize=9, fontfamily='Arial', pad=2)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
        
        plt.tight_layout(pad=0.4, w_pad=1.0)
        
        # Save figures
        output_png = self.base_path / 'lirl_mask_comparison_1x4.png'
        output_pdf = self.base_path / 'lirl_mask_comparison_1x4.pdf'
        
        fig.savefig(output_png, dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none')
        fig.savefig(output_pdf, dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none', format='pdf')
        
        print(f"\nSaved: {output_png}")
        print(f"Saved: {output_pdf}")
        
        plt.show()
        plt.close(fig)
    
    def print_statistics(self):
        """Print statistical comparison"""
        print("\n" + "=" * 60)
        print("LIRL vs Mask Statistical Comparison")
        print("=" * 60)
        
        for scale in self.scales:
            if scale not in self.data:
                continue
                
            print(f"\nScale {self.scale_labels[scale]}:")
            print("-" * 40)
            
            for algo_name in ['LIRL', 'Mask']:
                if algo_name in self.data[scale]:
                    data = self.data[scale][algo_name]
                    scores = data['scores']
                    last_100 = scores[:, -100:] if scores.shape[1] >= 100 else scores
                    
                    final_mean = np.mean(last_100)
                    final_std = np.std(np.mean(last_100, axis=1))
                    
                    print(f"  {algo_name}:")
                    print(f"    Final Reward: {final_mean:.4f} ± {final_std:.4f}")
                    print(f"    Runs: {data['num_runs']}, Episodes: {data['num_episodes']}")
            
            # Statistical test
            if 'LIRL' in self.data[scale] and 'Mask' in self.data[scale]:
                lirl_final = np.mean(self.data[scale]['LIRL']['scores'][:, -100:], axis=1)
                mask_final = np.mean(self.data[scale]['Mask']['scores'][:, -100:], axis=1)
                
                t_stat, p_value = stats.ttest_ind(lirl_final, mask_final)
                print(f"\n  Statistical Test (t-test):")
                print(f"    t-statistic: {t_stat:.4f}")
                print(f"    p-value: {p_value:.4f}")
                if p_value < 0.05:
                    winner = "LIRL" if np.mean(lirl_final) > np.mean(mask_final) else "Mask"
                    print(f"    ✓ Significant difference (p < 0.05), {winner} performs better")
                else:
                    print(f"    ✗ No significant difference (p >= 0.05)")
    
    def run(self):
        """Run the full comparison analysis"""
        print("=" * 60)
        print("LIRL vs Mask Learning Curves Comparison")
        print("=" * 60)
        
        self.load_data()
        
        if not any(self.data.values()):
            print("\n⚠ No data loaded! Please check folder paths.")
            return
        
        self.print_statistics()
        
        print("\nGenerating 1x4 Nature style plot...")
        self.plot_comparison_1x4()
        
        print("\n✓ Analysis complete!")
        print("Generated files:")
        print("  - lirl_mask_comparison_1x4.png/pdf")


if __name__ == "__main__":
    analyzer = LIRLMaskComparison()
    analyzer.run()

