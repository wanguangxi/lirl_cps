"""
Replot 1x3 publication figure for Protein Crystallization Screening.

Uses saved *_scores.npy (learning curves) and summary.json (bar charts)
from a comparison_YYYYMMDD_HHMMSS/ result folder.

Panels:
  1. Learning curve (mean ± std across seeds)
  2. Final best quality comparison
  3. Constraint violation rate (%)
"""

import argparse
import json
import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np

ALGORITHMS = ["LIRL", "PDQN", "HPPO", "CPO", "LPPO"]
COLORS = {
    "LIRL": "#1f77b4",
    "PDQN": "#ff7f0e",
    "HPPO": "#d62728",
    "CPO": "#8c564b",
    "LPPO": "#e377c2",
}


def setup_style():
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def is_lfs_pointer(path):
    with open(path, "rb") as f:
        return f.read(64).startswith(b"version https://git-lfs")


def load_score_array(path):
    if is_lfs_pointer(path):
        raise ValueError(f"LFS pointer (run `git lfs pull` first): {path}")
    raw = np.load(path, allow_pickle=True)
    arr = np.asarray(raw, dtype=object) if isinstance(raw, np.ndarray) and raw.dtype == object else np.asarray(raw)
    if arr.dtype == object:
        arr = np.concatenate([np.asarray(x).astype(float).reshape(-1) for x in arr])
    else:
        arr = arr.astype(float).reshape(-1)
    return arr


def load_scores(data_dir):
    scores_by_alg = {}
    for alg in ALGORITHMS:
        pattern = os.path.join(data_dir, f"{alg}_seed*_scores.npy")
        files = sorted(glob(pattern))
        if not files:
            raise FileNotFoundError(f"No score files found for {alg}: {pattern}")
        scores_by_alg[alg] = [load_score_array(f) for f in files]
    return scores_by_alg


def load_summary(data_dir):
    summary_path = os.path.join(data_dir, "summary.json")
    with open(summary_path, encoding="utf-8") as f:
        rows = json.load(f)
    return {row["Algorithm"]: row for row in rows}


def plot_learning_curve(ax, scores_by_alg):
    for alg in ALGORITHMS:
        scores_list = scores_by_alg[alg]
        min_len = min(len(s) for s in scores_list)
        arr = np.array([s[:min_len] for s in scores_list])
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        episodes = np.arange(min_len)
        color = COLORS[alg]
        ax.plot(episodes, mean, label=alg, color=color, linewidth=1.0)
        ax.fill_between(episodes, mean - std, mean + std, color=color, alpha=0.2, linewidth=0)

    ax.axhline(0, color="#cccccc", linestyle="--", linewidth=0.8, zorder=0)
    ax.set_xlabel("Episode", fontsize=8)
    ax.set_ylabel("Reward", fontsize=8)
    ax.set_title("Learning curve", fontsize=8, pad=4)
    ax.legend(loc="lower right", frameon=True, edgecolor="#D1D5DB", framealpha=0.95)
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(False)


def plot_best_quality(ax, summary):
    means, stds, labels = [], [], []
    for alg in ALGORITHMS:
        row = summary[alg]
        means.append(row["Best Quality (Mean)"])
        stds.append(row["Best Quality (Std)"])
        labels.append(alg)

    x = np.arange(len(labels))
    bars = ax.bar(
        x,
        means,
        yerr=stds,
        capsize=3,
        color=[COLORS[a] for a in labels],
        edgecolor="black",
        linewidth=0.5,
        alpha=0.85,
        width=0.6,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Best quality", fontsize=8)
    ax.set_title("Final best quality comparison", fontsize=8, pad=4)
    ax.set_ylim(0, max(means) * 1.25 if max(means) > 0 else 1)
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(False, axis="y")

    for bar, val in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=6,
        )


def plot_cvr(ax, summary):
    cvr_pct = [summary[alg]["CVR"] * 100 for alg in ALGORITHMS]

    x = np.arange(len(ALGORITHMS))
    bars = ax.bar(
        x,
        cvr_pct,
        color=[COLORS[a] for a in ALGORITHMS],
        edgecolor="black",
        linewidth=0.5,
        alpha=0.85,
        width=0.6,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(ALGORITHMS, fontsize=7)
    ax.set_ylabel("Constraint violation rate (%)", fontsize=8)
    ax.set_title("Constraint violation rate comparison", fontsize=8, pad=4)
    ax.set_ylim(0, max(cvr_pct) * 1.15 if max(cvr_pct) > 0 else 1)
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(False, axis="y")

    for bar, val in zip(bars, cvr_pct):
        label = f"{val:.1f}%" if val > 0 else "0%"
        y = bar.get_height() if val > 0 else 0.5
        ax.text(bar.get_x() + bar.get_width() / 2, y, label, ha="center", va="bottom", fontsize=6)


def plot_combined_figure(data_dir, output_path=None, bars_only=False):
    setup_style()
    summary = load_summary(data_dir)
    scores_by_alg = None
    if not bars_only:
        try:
            scores_by_alg = load_scores(data_dir)
        except (ValueError, OSError) as exc:
            print(f"Warning: {exc}")
            print("Continuing without learning curve. Use --bars-only to suppress this warning.")
            bars_only = True

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 1.72))
    fig.patch.set_facecolor("white")

    if scores_by_alg is not None:
        plot_learning_curve(axes[0], scores_by_alg)
    else:
        axes[0].text(
            0.5,
            0.5,
            "Learning curve unavailable\n(run git lfs pull for *_scores.npy)",
            ha="center",
            va="center",
            transform=axes[0].transAxes,
            fontsize=7,
        )
        axes[0].set_title("Learning curve", fontsize=8, pad=4)
        axes[0].set_xlabel("Episode", fontsize=8)
        axes[0].set_ylabel("Reward", fontsize=8)
    plot_best_quality(axes[1], summary)
    plot_cvr(axes[2], summary)

    fig.subplots_adjust(left=0.07, right=0.98, top=0.82, bottom=0.22, wspace=0.38)

    if output_path is None:
        output_path = os.path.join(data_dir, "combined_figure_b.png")
    fig.savefig(output_path, dpi=300, facecolor="white")
    pdf_path = os.path.splitext(output_path)[0] + ".pdf"
    fig.savefig(pdf_path, format="pdf", facecolor="white")
    plt.close(fig)

    print(f"Saved: {output_path}")
    print(f"Saved: {pdf_path}")
    return output_path


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_dir = os.path.join(script_dir, "comparison_20260205_154257")

    parser = argparse.ArgumentParser(description="Replot 1x3 combined figure from saved results")
    parser.add_argument(
        "--data-dir",
        default=default_dir,
        help="Path to comparison_YYYYMMDD_HHMMSS folder",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output PNG path (default: <data-dir>/combined_figure_b.png)",
    )
    parser.add_argument(
        "--bars-only",
        action="store_true",
        help="Skip learning curve (use when *_scores.npy are unavailable)",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")

    plot_combined_figure(args.data_dir, args.output, bars_only=args.bars_only)


if __name__ == "__main__":
    main()
