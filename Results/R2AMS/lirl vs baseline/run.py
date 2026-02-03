"""
Result Analysis Tool - Directly read multi-run training results of algorithms from Compare folder
"""
import json
import os
import argparse
from pathlib import Path
import re
from typing import Optional

# 尝试导入可选依赖
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

class CompareAnalyzer:
    def __init__(self, compare_dir: str = None):
        """
        Initialize result analyzer
        
        Args:
            compare_dir: Compare directory path, default is current directory
        """
        if compare_dir is None:
            compare_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.compare_dir = Path(compare_dir)
        print(f"Analysis directory: {self.compare_dir}")
        
        # Auto discover algorithm results
        self.algorithms = {}
        self.auto_discover_algorithms()
        
    def _normalize_scores(self, scores):
        """Normalize loaded scores into a 2D array [runs, episodes] by trimming to min length"""
        if not NUMPY_AVAILABLE:
            return scores
        try:
            arr = np.array(scores, dtype=object)
            if arr.dtype == object:
                runs = [np.asarray(r).astype(float).reshape(-1) for r in arr]
                if len(runs) == 0:
                    return np.zeros((0, 0))
                min_len = min(len(r) for r in runs) if all(len(r) > 0 for r in runs) else 0
                if min_len == 0:
                    return np.zeros((len(runs), 0))
                return np.stack([r[:min_len] for r in runs], axis=0)
            # If already numeric
            arr = np.asarray(scores)
            if arr.ndim == 1:
                return arr[np.newaxis, :]
            if arr.ndim >= 2:
                return arr[:, :arr.shape[1]]
            return arr
        except Exception as e:
            print(f"  - Score normalization failed: {e}")
            return np.array(scores, dtype=object)

    def auto_discover_algorithms(self):
        """Auto discover all algorithm results in Compare folder"""
        print("\nScanning algorithm results...")
        
        for item in self.compare_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                # Check if contains multi-run training results
                config_files = list(item.glob('config_*.json'))
                score_files = list(item.glob('*_all_scores_*.npy'))
                
                if not config_files:
                    print(f"Skip {item.name}: no config_*.json found")
                    continue
                if not score_files:
                    print(f"Skip {item.name}: no *_all_scores_*.npy found")
                    continue
                
                if NUMPY_AVAILABLE:
                    # Extract algorithm name
                    algo_name = self.extract_algorithm_name(item.name)
                    print(f"Found algorithm: {algo_name} -> {item.name}")
                    
                    try:
                        # Load data
                        config_file = config_files[0]
                        score_file = score_files[0]
                        
                        with open(config_file, 'r', encoding='utf-8') as f:
                            config = json.load(f)
                        
                        # Allow loading dtype=object arrays saved by numpy
                        scores_raw = np.load(score_file, allow_pickle=True)
                        scores = self._normalize_scores(scores_raw)
                        
                        self.algorithms[algo_name] = {
                            'path': item,
                            'config': config,
                            'scores': scores,
                            'config_file': config_file,
                            'score_file': score_file
                        }
                        
                        print(f"  - Config file: {config_file.name}")
                        print(f"  - Score file: {score_file.name}")
                        print(f"  - Number of runs: {len(scores)}")
                        print(f"  - Steps per run (trimmed to min length): {scores.shape[1] if scores.ndim == 2 else 0}")
                        
                    except Exception as e:
                        print(f"  - Loading failed ({item.name}): {e}")
                else:
                    print(f"Skip {item.name}: numpy unavailable")
        if not self.algorithms:
            print("No algorithm results found!")
            if not NUMPY_AVAILABLE:
                print("Note: numpy unavailable, cannot load .npy files")
        else:
            print(f"\nTotal found {len(self.algorithms)} algorithm results")
            
    def extract_algorithm_name(self, folder_name: str) -> str:
        """Extract algorithm name from folder name"""
        # Remove timestamp suffix
        name = re.sub(r'_multi_run_\d{8}_\d{6}$', '', folder_name)
        
        # Algorithm name mapping
        name_mapping = {
            'ddpg_lirl_pi': 'LIRL',
            'cpo': 'CPO',
            'hppo': 'HPPO',
            'hyar_vae': 'HyAR',
            "sac_lag": "SAC-Lag",
            "pdqn": "PDQN",
        }
        
        return name_mapping.get(name, name.upper())
    
    def plot_training_curves(self):
        """Plot training curves"""
        if not MATPLOTLIB_AVAILABLE or not NUMPY_AVAILABLE:
            print("matplotlib or numpy unavailable, skipping plotting")
            return

        print("\nPlotting training curves...")

        # Modern elegant color palette - inspired by Tailwind CSS and Material Design
        ELEGANT_PALETTE = [
            "#6366F1",  # Indigo - primary blue-purple
            "#F59E0B",  # Amber - warm orange
            "#10B981",  # Emerald - fresh green
            "#EC4899",  # Pink - vibrant pink
            "#14B8A6",  # Teal - cyan-green
            "#8B5CF6",  # Violet - purple
            "#F97316",  # Orange - pure orange
            "#06B6D4",  # Cyan - bright cyan
            "#84CC16",  # Lime - yellow-green
            "#A855F7",  # Purple - deep purple
        ]

        # Try to use a clean, elegant style
        try:
            import matplotlib as mpl
            from contextlib import contextmanager

            @contextmanager
            def _style_ctx():
                try:
                    with plt.style.context("seaborn-v0_8-whitegrid"):
                        yield
                except Exception:
                    yield

            with _style_ctx():
                plt.rcParams.update({
                    "figure.facecolor": "white",
                    "axes.facecolor": "white",
                    "savefig.facecolor": "white",
                    "axes.grid": True,
                    "grid.color": "#E5E7EB",
                    "grid.alpha": 0.8,
                    "grid.linewidth": 0.5,
                    "axes.edgecolor": "#D1D5DB",
                    "axes.linewidth": 1.2,
                    "axes.titleweight": "bold",
                    "axes.titlesize": 12,
                    "axes.labelsize": 12,
                    "axes.labelweight": "bold",
                    "xtick.labelsize": 12,
                    "ytick.labelsize": 12,
                    "legend.frameon": True,
                    "legend.facecolor": "white",
                    "legend.edgecolor": "#E5E7EB",
                    "legend.framealpha": 0.95,
                    "legend.borderpad": 0.8,
                    "legend.fontsize": 12,
                    "legend.title_fontsize": 12,
                    "font.family": "Arial",
                    "font.size": 12,
                    "font.weight": "bold",
                })

                def _beautify_axis(ax):
                    # Enhanced axis styling
                    ax.minorticks_on()
                    ax.grid(which="minor", axis="both", alpha=0.15, linewidth=0.3, color="#F3F4F6")
                    ax.grid(which="major", axis="both", alpha=0.3, linewidth=0.5, color="#E5E7EB")
                    
                    # Modern spine styling
                    for spine in ["top", "right"]:
                        ax.spines[spine].set_visible(False)
                    for spine in ["left", "bottom"]:
                        ax.spines[spine].set_color("#9CA3AF")
                        ax.spines[spine].set_linewidth(0.8)
                    
                    # Ensure axes use white background
                    ax.set_facecolor("white")

                    for tick in ax.get_xticklabels() + ax.get_yticklabels():
                        tick.set_fontweight('bold')
                        tick.set_fontsize(12)
                        tick.set_fontfamily('Arial')

                colors = ELEGANT_PALETTE
                color_map = {}
                color_idx = 0
                for algo_name in self.algorithms.keys():
                    if algo_name == 'LIRL':
                        color_map[algo_name] = "#EF4444"
                    else:
                        color_map[algo_name] = colors[color_idx % len(colors)]
                        color_idx += 1
                
                # Collect all final scores for determining x-axis range
                all_final_scores = []
                for algo_name, data in self.algorithms.items():
                    scores = data['scores']
                    if scores is not None and getattr(scores, "size", 0) > 0 and scores.ndim == 2 and scores.shape[1] > 0:
                        all_final_scores.extend(scores[:, -1])
                
                # Determine x-axis range for KDE
                if all_final_scores:
                    x_min = np.min(all_final_scores) - 0.02
                    x_max = np.max(all_final_scores) + 0.02
                    x_kde = np.linspace(x_min, x_max, 200)

                # ---------------------------------------
                # Figure 1: Algorithm Training Reward Comparison
                # ---------------------------------------
                fig_curve, ax_curve = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
                fig_curve.patch.set_facecolor("white")

                for algo_name, data in self.algorithms.items():
                    scores = data['scores']
                    # Skip if no data
                    if scores is None or getattr(scores, "size", 0) == 0:
                        continue
                    if scores.ndim != 2 or scores.shape[1] == 0:
                        continue

                    color = color_map.get(algo_name, colors[0])
                    linewidth = 1.0
                    alpha = 1.0 if algo_name == 'LIRL' else 0.9

                    mean_scores = np.mean(scores, axis=0)
                    std_scores = np.std(scores, axis=0)
                    episodes = np.arange(len(mean_scores))

                    ax_curve.plot(
                        episodes, mean_scores,
                        label=algo_name, color=color, linewidth=linewidth, alpha=alpha,
                        marker='', markersize=0,
                    )

                    ax_curve.fill_between(
                        episodes,
                        mean_scores - std_scores,
                        mean_scores + std_scores,
                        alpha=0.18, color=color, linewidth=0,
                        edgecolor='none'
                    )

                ax_curve.set_xlabel('Training Episode', fontweight='bold', fontsize=16, fontname='Arial')
                ax_curve.set_ylabel('Reward', fontweight='bold', fontsize=16, fontname='Arial')
                ax_curve.set_title('Algorithm Training Reward Comparison', fontweight='bold', fontsize=16, fontname='Arial', pad=12)

                legend_curve = ax_curve.legend(
                    title="Algorithms",
                    loc='lower right',
                    fontsize=10,
                    title_fontsize=10,
                    frameon=True,
                    fancybox=False,
                    shadow=False,
                    borderaxespad=0.5,
                    columnspacing=1.0,
                    handlelength=2.5,
                    edgecolor='#D1D5DB',
                    borderpad=0.6
                )
                legend_curve.get_frame().set_alpha(0.95)
                legend_curve.get_frame().set_linewidth(0.8)
                for text in legend_curve.get_texts():
                    text.set_fontweight('bold')
                    text.set_fontsize(10)
                    text.set_fontfamily('Arial')
                legend_title = legend_curve.get_title()
                if legend_title:
                    legend_title.set_fontweight('bold')
                    legend_title.set_fontsize(10)
                    legend_title.set_fontfamily('Arial')

                _beautify_axis(ax_curve)

                ax_curve.annotate('', xy=(0, 0), xytext=(0, -0.05),
                                  xycoords='axes fraction', textcoords='axes fraction',
                                  arrowprops=dict(arrowstyle='-', color='#E5E7EB', lw=2))

                training_save_path = self.compare_dir / 'algorithm_training_comparison.png'
                fig_curve.savefig(training_save_path, dpi=600, bbox_inches='tight', facecolor='white')
                print(f"Training curve plot saved to: {training_save_path}")

                plt.show()

                # ---------------------------------------
                # Figure 2: Final Reward Distribution
                # ---------------------------------------
                fig_dist, ax_dist = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
                fig_dist.patch.set_facecolor("white")

                for algo_name, data in self.algorithms.items():
                    scores = data['scores']
                    # Skip if no data
                    if scores is None or getattr(scores, "size", 0) == 0:
                        continue
                    if scores.ndim != 2 or scores.shape[1] == 0:
                        continue

                    color = color_map.get(algo_name, colors[0])

                    # Final score distribution using KDE
                    final_scores = scores[:, -1]
                    score_std = np.std(final_scores)
                    
                    # Use KDE for smooth distribution visualization
                    if SCIPY_AVAILABLE:
                        try:
                            # Calculate KDE
                            kde = stats.gaussian_kde(final_scores)
                            # Adjust bandwidth for better visualization
                            # Smaller bandwidth for concentrated data (like LIRL)
                            if score_std < 0.01:
                                kde.set_bandwidth(bw_method=kde.factor * 0.5)
                            
                            # Evaluate KDE on x grid
                            kde_values = kde(x_kde)
                            
                            # Plot KDE curve
                            ax_dist.plot(x_kde, kde_values,
                                         color=color,
                                         linewidth=1.0,
                                         alpha=0.9 if algo_name == 'LIRL' else 0.8,
                                         label=f"{algo_name} (σ={score_std:.4f})")
                            
                            # Fill under the curve for better visualization
                            ax_dist.fill_between(x_kde, 0, kde_values,
                                                 color=color,
                                                 alpha=0.3 if algo_name == 'LIRL' else 0.25)
                            
                        except Exception as e:
                            print(f"KDE failed for {algo_name}, falling back to histogram: {e}")
                            # Fallback to histogram with normalized frequency
                            ax_dist.hist(
                                final_scores, 
                                bins=10, 
                                density=False,
                                weights=np.ones_like(final_scores) / len(final_scores),
                                alpha=0.65 if algo_name != 'LIRL' else 0.85,
                                color=color, 
                                edgecolor='white',
                                linewidth=1.2,
                                label=f"{algo_name} (σ={score_std:.4f})"
                            )
                    else:
                        # If scipy not available, use normalized frequency histogram
                        ax_dist.hist(
                            final_scores, 
                            bins=10, 
                            density=False,
                            weights=np.ones_like(final_scores) / len(final_scores),  # Normalized frequency
                            alpha=0.65 if algo_name != 'LIRL' else 0.85,
                            color=color, 
                            edgecolor='white',
                            linewidth=1.2,
                            label=f"{algo_name} (σ={score_std:.4f})"
                        )

                ax_dist.set_ylim(bottom=0)

                if SCIPY_AVAILABLE:
                    ax_dist.set_ylabel('Probability Density (KDE)', fontweight='bold', fontsize=16, fontname='Arial')
                else:
                    ax_dist.set_ylabel('Normalized Frequency', fontweight='bold', fontsize=16, fontname='Arial')

                ax_dist.set_xlabel('Final Reward', fontweight='bold', fontsize=16, fontname='Arial')
                ax_dist.set_title('Final Reward Distribution',
                                   fontweight='bold', fontsize=16, fontname='Arial', pad=12)

                legend_dist = ax_dist.legend(
                    title="Algorithms",
                    loc='upper left',
                    fontsize=10,
                    title_fontsize=10,
                    frameon=True,
                    fancybox=False,
                    shadow=False,
                    borderaxespad=0.5,
                    edgecolor='#D1D5DB',
                    borderpad=0.6
                )
                legend_dist.get_frame().set_alpha(0.95)
                legend_dist.get_frame().set_linewidth(0.8)
                for text in legend_dist.get_texts():
                    text.set_fontweight('bold')
                    text.set_fontsize(10)
                    text.set_fontfamily('Arial')
                legend_title = legend_dist.get_title()
                if legend_title:
                    legend_title.set_fontweight('bold')
                    legend_title.set_fontsize(10)
                    legend_title.set_fontfamily('Arial')

                _beautify_axis(ax_dist)

                ax_dist.annotate('', xy=(0, 0), xytext=(0, -0.05),
                                 xycoords='axes fraction', textcoords='axes fraction',
                                 arrowprops=dict(arrowstyle='-', color='#E5E7EB', lw=2))

                distribution_save_path = self.compare_dir / 'final_Reward_distribution.png'
                fig_dist.savefig(distribution_save_path, dpi=600, bbox_inches='tight', facecolor='white')
                print(f"Final Reward distribution saved to: {distribution_save_path}")

                plt.show()

        except Exception as e:
            print(f"Plot styling failed, falling back to default style: {e}")
            # Fallback with improved colors (separate figures)
            fallback_colors = ELEGANT_PALETTE
            color_map = {}
            color_idx = 0
            for algo_name in self.algorithms.keys():
                if algo_name == 'LIRL':
                    color_map[algo_name] = "#EF4444"
                else:
                    color_map[algo_name] = fallback_colors[color_idx % len(fallback_colors)]
                    color_idx += 1

            # Collect all final scores to determine common range
            all_final_scores = []
            for algo_name, data in self.algorithms.items():
                scores = data['scores']
                if scores is None or getattr(scores, "size", 0) == 0:
                    continue
                if scores.ndim != 2 or scores.shape[1] == 0:
                    continue
                all_final_scores.extend(scores[:, -1])

            if SCIPY_AVAILABLE and all_final_scores:
                x_min = np.min(all_final_scores) - 0.02
                x_max = np.max(all_final_scores) + 0.02
                x_kde = np.linspace(x_min, x_max, 200)

            # Fallback training curve figure
            fig_curve, ax_curve = plt.subplots(figsize=(6.0, 5.6))
            fig_curve.patch.set_facecolor("white")
            ax_curve.set_facecolor("white")

            for algo_name, data in self.algorithms.items():
                scores = data['scores']
                if scores is None or getattr(scores, "size", 0) == 0:
                    continue
                if scores.ndim != 2 or scores.shape[1] == 0:
                    continue

                color = color_map.get(algo_name, fallback_colors[0])
                mean_scores = np.mean(scores, axis=0)
                std_scores = np.std(scores, axis=0)
                episodes = np.arange(len(mean_scores))

                ax_curve.plot(episodes, mean_scores, label=algo_name, color=color, linewidth=1.0)
                ax_curve.fill_between(episodes, mean_scores - std_scores, mean_scores + std_scores,
                                      alpha=0.2, color=color)

            ax_curve.set_xlabel('Episode', fontweight='bold', fontsize=16, fontname='Arial')
            ax_curve.set_ylabel('Score', fontweight='bold', fontsize=16, fontname='Arial')
            ax_curve.set_title('Algorithm Training Reward Comparison', fontweight='bold', fontsize=16, fontname='Arial')
            legend_curve = ax_curve.legend()
            if legend_curve:
                for text in legend_curve.get_texts():
                    text.set_fontweight('bold')
                    text.set_fontsize(10)
                    text.set_fontfamily('Arial')
                legend_title = legend_curve.get_title()
                if legend_title:
                    legend_title.set_fontweight('bold')
                    legend_title.set_fontsize(10)
                    legend_title.set_fontfamily('Arial')
            ax_curve.grid(True, alpha=0.3)

            fallback_training_path = self.compare_dir / 'algorithm_training_comparison.png'
            fig_curve.savefig(fallback_training_path, dpi=600, bbox_inches='tight', facecolor='white')
            print(f"Training curve plot saved to: {fallback_training_path}")

            plt.show()

            # Fallback distribution figure
            fig_dist, ax_dist = plt.subplots(figsize=(6.0, 5.6))
            fig_dist.patch.set_facecolor("white")
            ax_dist.set_facecolor("white")

            for algo_name, data in self.algorithms.items():
                scores = data['scores']
                if scores is None or getattr(scores, "size", 0) == 0:
                    continue
                if scores.ndim != 2 or scores.shape[1] == 0:
                    continue

                color = color_map.get(algo_name, fallback_colors[0])
                final_scores = scores[:, -1]

                if SCIPY_AVAILABLE and all_final_scores:
                    try:
                        kde = stats.gaussian_kde(final_scores)
                        kde_values = kde(x_kde)
                        ax_dist.plot(x_kde, kde_values, color=color, linewidth=1.0, label=algo_name)
                        ax_dist.fill_between(x_kde, 0, kde_values, color=color, alpha=0.3)
                    except Exception:
                        ax_dist.hist(final_scores, bins=15, alpha=0.7, label=algo_name,
                                     color=color, edgecolor='white', linewidth=0.5,
                                     density=False, weights=np.ones_like(final_scores)/len(final_scores))
                else:
                    ax_dist.hist(final_scores, bins=15, alpha=0.7, label=algo_name,
                                 color=color, edgecolor='white', linewidth=0.5,
                                 density=False, weights=np.ones_like(final_scores)/len(final_scores))

            ax_dist.set_xlabel('Final Score', fontweight='bold', fontsize=16, fontname='Arial')
            if SCIPY_AVAILABLE:
                ax_dist.set_ylabel('Probability Density (KDE)', fontweight='bold', fontsize=16, fontname='Arial')
            else:
                ax_dist.set_ylabel('Normalized Frequency', fontweight='bold', fontsize=16, fontname='Arial')
            ax_dist.set_title('Final Reward Distribution', fontweight='bold', fontsize=16, fontname='Arial')
            legend_dist = ax_dist.legend()
            if legend_dist:
                for text in legend_dist.get_texts():
                    text.set_fontweight('bold')
                    text.set_fontsize(10)
                    text.set_fontfamily('Arial')
                legend_title = legend_dist.get_title()
                if legend_title:
                    legend_title.set_fontweight('bold')
                    legend_title.set_fontsize(10)
                    legend_title.set_fontfamily('Arial')
            ax_dist.grid(True, alpha=0.3)

            fallback_distribution_path = self.compare_dir / 'final_Reward_distribution.png'
            fig_dist.savefig(fallback_distribution_path, dpi=600, bbox_inches='tight', facecolor='white')
            print(f"Final Reward distribution saved to: {fallback_distribution_path}")

            plt.show()
    
    def plot_convergence_speed(
        self,
        percentile: float = 0.95,
        smooth_window: int = 15,
        save_path: Optional[Path] = None,
    ):
        """Analyze and visualize how quickly each algorithm approaches its steady state."""
        if not MATPLOTLIB_AVAILABLE or not NUMPY_AVAILABLE:
            print("matplotlib or numpy unavailable, skipping convergence speed plot")
            return

        if not self.algorithms:
            print("No algorithm data available for convergence analysis")
            return

        # Sanitize user inputs
        try:
            percentile = float(percentile)
        except (TypeError, ValueError):
            percentile = 0.95
        if percentile <= 0 or percentile > 1:
            print(f"Invalid percentile={percentile:.3f}; falling back to 0.95")
            percentile = 0.95

        try:
            smooth_window = int(smooth_window)
        except (TypeError, ValueError):
            smooth_window = 15
        if smooth_window < 1:
            smooth_window = 1

        # Consistent palette with other figures
        palette = [
            "#6366F1",
            "#F59E0B",
            "#10B981",
            "#EC4899",
            "#14B8A6",
            "#8B5CF6",
            "#F97316",
            "#06B6D4",
            "#84CC16",
            "#A855F7",
        ]

        def _smooth_series(series: np.ndarray) -> np.ndarray:
            arr = np.asarray(series, dtype=float)
            if arr.ndim != 1 or arr.size == 0:
                return arr
            if smooth_window <= 1 or arr.size <= 1:
                return arr

            window = min(smooth_window, arr.size)
            kernel = np.ones(window, dtype=float) / window
            pad_left = window // 2
            pad_right = window - 1 - pad_left
            padded = np.pad(arr, (pad_left, pad_right), mode="edge")
            smoothed = np.convolve(padded, kernel, mode="valid")
            return smoothed.astype(float)

        color_map = {}
        color_idx = 0
        for algo_name in self.algorithms.keys():
            if algo_name == "LIRL":
                color_map[algo_name] = "#EF4444"
            else:
                color_map[algo_name] = palette[color_idx % len(palette)]
                color_idx += 1

        results = []

        print("\n" + "=" * 60)
        print(f"Convergence speed analysis (threshold: {percentile * 100:.1f}% of median gain)")
        print("=" * 60)

        for algo_name, data in self.algorithms.items():
            scores = data.get("scores")
            if scores is None or getattr(scores, "size", 0) == 0:
                print(f"{algo_name}: no valid score data; skipping")
                continue
            if scores.ndim != 2 or scores.shape[1] == 0:
                print(f"{algo_name}: score array has unexpected shape {scores.shape}; skipping")
                continue

            smoothed_runs = np.array([_smooth_series(run) for run in scores], dtype=float)

            median_initial = float(np.median(smoothed_runs[:, 0]))
            median_final = float(np.median(smoothed_runs[:, -1]))
            improvement = median_final - median_initial

            tail_window = max(1, min(smoothed_runs.shape[1], int(np.ceil(smoothed_runs.shape[1] * 0.05))))
            tail_segment = smoothed_runs[:, -tail_window:]
            tail_means = np.mean(tail_segment, axis=1)
            final_median_value = float(np.median(tail_means))
            final_mean_value = float(np.mean(tail_means))
            final_std_value = float(np.std(tail_means))

            if np.isclose(improvement, 0.0, atol=1e-8):
                threshold = median_final
            else:
                threshold = median_initial + percentile * improvement

            direction = 1 if improvement >= 0 else -1

            times_to_threshold = []
            for run in smoothed_runs:
                if direction >= 0:
                    reached = np.where(run >= threshold)[0]
                else:
                    reached = np.where(run <= threshold)[0]
                if reached.size > 0:
                    times_to_threshold.append(float(reached[0] + 1))
                else:
                    times_to_threshold.append(float(run.size))

            if not times_to_threshold:
                print(f"{algo_name}: unable to compute convergence episodes")
                continue

            times_arr = np.asarray(times_to_threshold, dtype=float)
            median_time = float(np.median(times_arr))
            q1 = float(np.percentile(times_arr, 25))
            q3 = float(np.percentile(times_arr, 75))

            spread_lower = median_time - q1
            spread_upper = q3 - median_time

            print(
                f"{algo_name:>12}: median episodes = {median_time:.1f}, IQR = [{q1:.1f}, {q3:.1f}]"
            )
            print(
                f"{'':>12}  final value (median ± std): {final_median_value:.4f} ± {final_std_value:.4f}"
            )

            results.append(
                {
                    "algo": algo_name,
                    "median": median_time,
                    "lower": max(spread_lower, 0.0),
                    "upper": max(spread_upper, 0.0),
                    "color": color_map.get(algo_name, palette[0]),
                    "final_median": final_median_value,
                    "final_mean": final_mean_value,
                    "final_std": final_std_value,
                }
            )

        if not results:
            print("No convergence statistics available for plotting")
            return

        print("\nFinal convergence values (averaged over last 5% episodes):")
        for item in sorted(results, key=lambda entry: entry["final_median"], reverse=True):
            print(
                f"  {item['algo']:>12}: median={item['final_median']:.4f}, "
                f"mean={item['final_mean']:.4f}, std={item['final_std']:.4f}"
            )

        results.sort(key=lambda item: item["median"])

        algo_labels = [item["algo"] for item in results]
        median_times = np.array([item["median"] for item in results], dtype=float)
        lower_errors = np.array([item["lower"] for item in results], dtype=float)
        upper_errors = np.array([item["upper"] for item in results], dtype=float)
        colors = [item["color"] for item in results]

        y_positions = np.arange(len(results))

        fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        bars = ax.barh(
            y_positions,
            median_times,
            color=colors,
            edgecolor="none",
            height=0.6,
        )

        if np.any(lower_errors) or np.any(upper_errors):
            for idx, (median_val, lower, upper, color) in enumerate(
                zip(median_times, lower_errors, upper_errors, colors)
            ):
                if lower <= 0 and upper <= 0:
                    continue
                ax.errorbar(
                    median_val,
                    idx,
                    xerr=np.array([[max(lower, 0.0)], [max(upper, 0.0)]]),
                    fmt="none",
                    ecolor=color,
                    elinewidth=1.8,
                    capsize=6,
                    capthick=1.8,
                )

        annotation_offset = max(median_times.max() * 0.015, 5.0)
        for bar, median_val in zip(bars, median_times):
            ax.text(
                median_val + annotation_offset,
                bar.get_y() + bar.get_height() / 2,
                f"{int(round(median_val))}",
                va="center",
                ha="left",
                fontweight="bold",
                fontsize=12,
                fontname="Arial",
                color="#111827",
            )

        ax.set_yticks(y_positions)
        ax.set_yticklabels(algo_labels, fontweight="bold", fontsize=12, fontname="Arial")
        ax.invert_yaxis()
        ax.set_xlabel(
            "Episodes to Reach Convergence Threshold",
            fontweight="bold",
            fontsize=16,
            fontname="Arial",
        )
        ax.set_title(
            f"Convergence Speed Comparison ({percentile * 100:.0f}% of Median Gain)",
            fontweight="bold",
            fontsize=16,
            fontname="Arial",
            pad=14,
        )

        ax.xaxis.grid(True, which="major", color="#E5E7EB", linewidth=0.8, alpha=0.7)
        ax.set_axisbelow(True)

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color("#D1D5DB")
            ax.spines[spine].set_linewidth(1.0)

        ax.tick_params(axis="x", labelsize=12)
        for tick in ax.get_xticklabels():
            tick.set_fontweight("bold")
            tick.set_fontname("Arial")

        if save_path is None:
            save_path = self.compare_dir / "convergence_speed_analysis.png"

        fig.savefig(save_path, dpi=600, bbox_inches="tight", facecolor="white")
        print(f"Convergence speed plot saved to: {save_path}")

        plt.show()
        plt.close(fig)

    
    
    def generate_report(self):
        """Generate analysis report"""
        print("\n" + "="*60)
        print("Generate Analysis Report")
        print("="*60)
        
        report_lines = []
        report_lines.append("Algorithm Comparison Analysis Report")
        report_lines.append("=" * 50)
        report_lines.append(f"Analysis directory: {self.compare_dir}")
        report_lines.append(f"Number of algorithms: {len(self.algorithms)}")
        report_lines.append("")
        
        if not NUMPY_AVAILABLE:
            report_lines.append("Note: numpy unavailable, some analysis functions were skipped")
            report_lines.append("")
        
        # Algorithm overview
        report_lines.append("Algorithm Overview:")
        for algo_name, data in self.algorithms.items():
            if NUMPY_AVAILABLE:
                scores = data['scores']
                runs = len(scores)
                episodes = len(scores[0]) if runs > 0 else 0
                report_lines.append(f"  - {algo_name}: {runs} runs, {episodes} episodes per run")
            else:
                report_lines.append(f"  - {algo_name}: data available")
        report_lines.append("")
        
        # Reward summary
        if NUMPY_AVAILABLE and self.algorithms:
            report_lines.append("Reward Summary:")
            perf_data = []
            for algo_name, data in self.algorithms.items():
                scores = data['scores']
                if scores.size > 0:
                    final_scores = scores[:, -1]
                    mean_final = np.mean(final_scores)
                    std_final = np.std(final_scores)
                    perf_data.append((mean_final, algo_name, std_final))
            
            # Sort by Reward
            perf_data.sort(reverse=True)
            for i, (mean, algo, std) in enumerate(perf_data, 1):
                report_lines.append(f"  {i}. {algo}: {mean:.4f} ± {std:.4f}")
            report_lines.append("")
        
        # Save report
        report_path = self.compare_dir / 'analysis_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"Report saved to: {report_path}")
        print("\nReport Content:")
        print('\n'.join(report_lines))
        
    def plot_combined_nature_style(self, percentile: float = 0.95, smooth_window: int = 15):
        """Plot all three figures in 1 row 3 columns, Nature paper style."""
        if not MATPLOTLIB_AVAILABLE or not NUMPY_AVAILABLE:
            print("matplotlib or numpy unavailable, skipping plotting")
            return
        
        if not self.algorithms:
            print("No algorithm data available for plotting")
            return
        
        print("\nPlotting combined figure (Nature style)...")
        
        # Nature论文风格设置 - 所有字体使用Arial
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
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['legend.fontsize'] = 7
        
        # Nature风格颜色（柔和但有区分度）
        NATURE_PALETTE = [
            "#E64B35",  # 红色 (LIRL)
            "#4DBBD5",  # 青色
            "#00A087",  # 绿色
            "#3C5488",  # 蓝色
            "#F39B7F",  # 橙色
            "#8491B4",  # 灰蓝色
            "#91D1C2",  # 薄荷绿
            "#DC0000",  # 深红
        ]
        
        # 颜色映射 - 包含所有可能的算法名
        color_map = {
            'LIRL': "#E64B35",      # 红色
            'SAC-Lag': "#4DBBD5",   # 青色
            'CPO': "#00A087",       # 绿色
            'H-PPO': "#3C5488",     # 蓝色
            'HPPO': "#3C5488",      # 蓝色 (别名)
            'HyAR': "#F39B7F",      # 橙色
            'PDQN': "#8491B4",      # 灰蓝色
        }
        # 为训练数据中的其他算法添加颜色
        color_idx = 0
        for algo_name in self.algorithms.keys():
            if algo_name not in color_map:
                color_map[algo_name] = NATURE_PALETTE[(color_idx + 6) % len(NATURE_PALETTE)]
                color_idx += 1
        
        # Nature论文标准尺寸
        # 双栏宽度：183mm = 7.2英寸 (18.3cm)
        # 高度：50mm = 2.0英寸 (5.08cm)
        fig = plt.figure(figsize=(7.2, 1.8))
        fig.patch.set_facecolor('white')
        # 使用显式边距确保图片尺寸精确，增加左边距以显示完整的Reward标签
        gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1, 1], wspace=0.32,
                              left=0.08, right=0.98, top=0.88, bottom=0.22)
        axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
        
        # ========== 子图1: Training Curves ==========
        ax1 = axes[0]
        
        for algo_name, data in self.algorithms.items():
            scores = data['scores']
            if scores is None or getattr(scores, "size", 0) == 0:
                continue
            if scores.ndim != 2 or scores.shape[1] == 0:
                continue
            
            color = color_map.get(algo_name, NATURE_PALETTE[0])
            mean_scores = np.mean(scores, axis=0)
            std_scores = np.std(scores, axis=0)
            episodes = np.arange(len(mean_scores))
            
            ax1.plot(episodes, mean_scores, label=algo_name, color=color, 
                    linewidth=1.0, alpha=0.9)
            ax1.fill_between(episodes, mean_scores - std_scores, mean_scores + std_scores,
                            alpha=0.2, color=color, linewidth=0)
        
        ax1.set_xlabel('Episode', fontsize=9, fontfamily='Arial')
        ax1.set_ylabel('Reward', fontsize=9, fontfamily='Arial')
        ax1.set_title('Training curves', fontsize=9, fontfamily='Arial', pad=2)
        # Legend放在图a内部下方，2行显示（每行3个）
        ax1.legend(fontsize=6, loc='lower right', frameon=True, fancybox=False,
                  edgecolor='#D1D5DB', framealpha=0.95, ncol=3, 
                  handlelength=1.2, handletextpad=0.3, columnspacing=0.6,
                  borderpad=0.3, labelspacing=0.2, prop={'family': 'Arial'})
        ax1.tick_params(axis='both', which='major', labelsize=8, width=0.8, length=3)
        for label in ax1.get_xticklabels() + ax1.get_yticklabels():
            label.set_fontfamily('Arial')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_linewidth(0.8)
        ax1.spines['bottom'].set_linewidth(0.8)
        ax1.grid(False)
        
        # ========== 子图2: Final Reward Distribution (KDE曲线) ==========
        ax2 = axes[1]
        
        # 收集所有最终分数确定x轴范围
        all_final_scores = []
        for algo_name, data in self.algorithms.items():
            scores = data['scores']
            if scores is not None and getattr(scores, "size", 0) > 0 and scores.ndim == 2 and scores.shape[1] > 0:
                all_final_scores.extend(scores[:, -1])
        
        if all_final_scores:
            # 扩大x轴范围以显示完整的分布曲线
            data_range = np.max(all_final_scores) - np.min(all_final_scores)
            padding = max(data_range * 0.15, 0.05)
            x_min = np.min(all_final_scores) - padding
            x_max = np.max(all_final_scores) + padding
            x_kde = np.linspace(x_min, x_max, 300)
            
            for algo_name, data in self.algorithms.items():
                scores = data['scores']
                if scores is None or getattr(scores, "size", 0) == 0:
                    continue
                if scores.ndim != 2 or scores.shape[1] == 0:
                    continue
                
                color = color_map.get(algo_name, NATURE_PALETTE[0])
                final_scores = scores[:, -1]
                
                # 使用scipy的KDE绘制核密度曲线
                if SCIPY_AVAILABLE:
                    try:
                        kde = stats.gaussian_kde(final_scores)
                        kde_values = kde(x_kde)
                        
                        # 绘制KDE分布曲线
                        line_width = 0.6 if algo_name == 'LIRL' else 0.5
                        ax2.plot(x_kde, kde_values, color=color, linewidth=line_width, 
                                alpha=1.0, label=algo_name, zorder=3)
                        
                        # 填充曲线下方区域
                        fill_alpha = 0.35 if algo_name == 'LIRL' else 0.15
                        ax2.fill_between(x_kde, 0, kde_values, color=color, alpha=fill_alpha, zorder=2)
                    except Exception as e:
                        print(f"KDE failed for {algo_name}: {e}")
                        # 回退到直方图
                        ax2.hist(final_scores, bins=15, density=True, alpha=0.6,
                                color=color, edgecolor='white', linewidth=0.5, label=algo_name)
                else:
                    # 如果scipy不可用，使用直方图
                    ax2.hist(final_scores, bins=15, density=True, alpha=0.6,
                            color=color, edgecolor='white', linewidth=0.5, label=algo_name)
        
        ax2.set_xlabel('Final reward', fontsize=9, fontfamily='Arial')
        ax2.set_ylabel('Density', fontsize=9, fontfamily='Arial')
        ax2.set_title('Reward distribution', fontsize=9, fontfamily='Arial', pad=2)
        ax2.legend(fontsize=6, loc='upper left', frameon=True, fancybox=False,
                  edgecolor='#D1D5DB', framealpha=0.9, prop={'family': 'Arial'})
        ax2.tick_params(axis='both', which='major', labelsize=8, width=0.8, length=3)
        for label in ax2.get_xticklabels() + ax2.get_yticklabels():
            label.set_fontfamily('Arial')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_linewidth(0.8)
        ax2.spines['bottom'].set_linewidth(0.8)
        ax2.grid(False)
        ax2.set_ylim(bottom=0)
        
        # ========== 子图3: Computation Time Comparison (Stacked) ==========
        ax3 = axes[2]
        
        # 加载计算时间数据
        computation_time_file = self.compare_dir / 'computation_time_results_20260123_110712.json'
        
        if computation_time_file.exists():
            with open(computation_time_file, 'r', encoding='utf-8') as f:
                time_data = json.load(f)
            
            time_results = time_data.get('results', {})
            
            # 准备数据
            algo_names = []
            network_times = []
            postprocess_times = []
            total_times = []
            
            # 按照total_ms排序
            sorted_algos = sorted(time_results.items(), key=lambda x: x[1].get('total_ms', 0))
            
            for algo_name, timing in sorted_algos:
                algo_names.append(algo_name)
                network_times.append(timing.get('network_ms', 0))
                postprocess_times.append(timing.get('postprocess_ms', 0))
                total_times.append(timing.get('total_ms', 0))
            
            y_positions = np.arange(len(algo_names))
            bar_height = 0.6
            
            # 堆叠条形图颜色
            color_network = '#4DBBD5'      # 青色 - NN Inference
            color_postprocess = '#E64B35'  # 红色 - Post-processing
            
            # 绘制堆叠水平条形图
            bars1 = ax3.barh(y_positions, network_times, color=color_network, 
                            edgecolor='none', height=bar_height, alpha=0.85,
                            label='NN inference')
            bars2 = ax3.barh(y_positions, postprocess_times, left=network_times,
                            color=color_postprocess, edgecolor='none', height=bar_height, 
                            alpha=0.85, label='Post-processing')
            
            # 在条形图右侧添加总时间标签
            for idx, total_val in enumerate(total_times):
                ax3.text(total_val + 0.02, y_positions[idx],
                        f'{total_val:.2f}', va='center', ha='left',
                        fontsize=7, fontfamily='Arial')
            
            ax3.set_yticks(y_positions)
            ax3.set_yticklabels(algo_names, fontsize=8, fontfamily='Arial')
            ax3.invert_yaxis()
            ax3.set_xlabel('Time (ms)', fontsize=9, fontfamily='Arial')
            ax3.set_title('Computation time', fontsize=9, fontfamily='Arial', pad=2)
            ax3.tick_params(axis='both', which='major', labelsize=8, width=0.8, length=3)
            for label in ax3.get_xticklabels() + ax3.get_yticklabels():
                label.set_fontfamily('Arial')
            ax3.spines['top'].set_visible(False)
            ax3.spines['right'].set_visible(False)
            ax3.spines['left'].set_linewidth(0.8)
            ax3.spines['bottom'].set_linewidth(0.8)
            ax3.grid(False)
            # 调整x轴范围以显示标签
            ax3.set_xlim(0, max(total_times) * 1.25)
            
            # 添加图例
            ax3.legend(fontsize=6, loc='upper right', frameon=True, fancybox=False,
                      edgecolor='#D1D5DB', framealpha=0.9, prop={'family': 'Arial'},
                      handlelength=1.0, handletextpad=0.3)
        else:
            print(f"Warning: Computation time file not found: {computation_time_file}")
            ax3.text(0.5, 0.5, 'No computation time data', ha='center', va='center',
                    transform=ax3.transAxes, fontsize=9, fontfamily='Arial')
        
        
        # 不使用tight_layout，使用gridspec的显式边距
        # plt.tight_layout(pad=0.5, w_pad=1.5)
        
        # 保存为PNG和PDF - 不使用bbox_inches='tight'以保持精确尺寸
        output_path_png = self.compare_dir / 'algorithm_comparison_combined.png'
        output_path_pdf = self.compare_dir / 'algorithm_comparison_combined.pdf'
        
        # 保存时不裁剪，保持7.2英寸(18.3cm)宽度
        fig.savefig(output_path_png, dpi=600, facecolor='white', edgecolor='none')
        fig.savefig(output_path_pdf, dpi=600, facecolor='white', edgecolor='none', format='pdf')
        
        print(f"Combined figure saved to: {output_path_png}")
        print(f"PDF saved to: {output_path_pdf}")
        
        plt.show()
        plt.close(fig)

    def run_analysis(self, percentile: float = 0.95, smooth_window: int = 100):
        """Run complete analysis with default plots."""
        if not self.algorithms:
            print("No algorithm results to analyze!")
            return
        
        # 使用合并的Nature风格图表
        self.plot_combined_nature_style(percentile=percentile, smooth_window=smooth_window)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Analyze algorithm comparison results")
    parser.add_argument('--dir', type=str, help='Compare directory path')
    parser.add_argument('--plot', action='store_true', help='Only generate training and reward plots')
    parser.add_argument('--convergence', action='store_true', help='Only generate convergence speed analysis')
    parser.add_argument('--combined', action='store_true', help='Generate combined 1x3 Nature style figure')
    parser.add_argument('--report', action='store_true', help='Only generate report')
    parser.add_argument('--percentile', type=float, default=0.95,
                        help='Percentile of median performance gain used as convergence threshold (0-1].')
    parser.add_argument('--smooth-window', type=int, default=5,
                        help='Window size for moving-average smoothing before detecting convergence.')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = CompareAnalyzer(args.dir)
    
    if not analyzer.algorithms:
        print("No algorithm results found, exiting...")
        return
    
    # Execute specified function
    if args.plot:
        analyzer.plot_training_curves()
    elif args.convergence:
        analyzer.plot_convergence_speed(percentile=args.percentile, smooth_window=args.smooth_window)
    elif args.combined:
        analyzer.plot_combined_nature_style(percentile=args.percentile, smooth_window=args.smooth_window)
    elif args.report:
        analyzer.generate_report()
    else:
        analyzer.run_analysis(percentile=args.percentile, smooth_window=args.smooth_window)

if __name__ == "__main__":
    main()
