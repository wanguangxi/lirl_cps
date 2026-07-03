# LIRL-CPS

**Logic-Informed Reinforcement Learning for Safe Hybrid Cyber-Physical Systems**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)

Official implementation accompanying:

> **Logic-informed reinforcement learning enables safe decision-making in hybrid cyber–physical systems**  
> Guangxi Wan, Peng Zeng, Xiaoting Dong, *et al.*, *Nature Communications* (2026).

This repository provides the **LIRL** (Logic-Informed Reinforcement Learning) framework, simulation environments, baseline implementations, pre-computed experimental outputs, and figure-reproduction scripts used in the paper.

**Repository:** [https://github.com/wanguangxi/lirl_cps](https://github.com/wanguangxi/lirl_cps)

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Reproducibility Protocol](#reproducibility-protocol)
- [Quick Start: Reproduce Paper Figures](#quick-start-reproduce-paper-figures)
- [Training from Scratch](#training-from-scratch)
- [Figure-to-Script Mapping](#figure-to-script-mapping)
- [Data Availability](#data-availability)
- [License and Third-Party Code](#license-and-third-party-code)
- [Citation](#citation)
- [Contact](#contact)
- [中文说明](#中文说明)

---

## Overview

LIRL is a neuro-symbolic reinforcement learning framework that embeds **quantifier-free linear real arithmetic (QF_LRA)** domain specifications into a learning-compatible **logic-to-manifold projection operator**. A neural policy outputs an unconstrained continuous intent in a smooth latent space; a non-learnable projector maps this intent onto the state-dependent feasible manifold, guaranteeing **step-wise constraint satisfaction by construction**.

This repository covers five benchmark domains from the paper:

| Domain | Code directory | Paper name |
|--------|----------------|------------|
| Robotic reducer-assembly manufacturing | `RMS/` | **R2AMS** |
| Urban traffic signal control | `CityFlow/` | Traffic control |
| EV charging station scheduling | `EV-Charging/` | EV charging |
| Crystallization-inspired screening | `Results/Protein_Crystallization_Screening/` | Synthetic crystallization benchmark |
| Parameterized hybrid control | `MP-DQN/` | Goal / Platform / Soccer |

The `Results/` directory contains **pre-computed experimental outputs** and **publication-style visualization scripts** (`run.py`) that regenerate the main paper figures from saved JSON/CSV/NPY data.

---

## Repository Structure

```
LIRL-CPS/
├── README.md                 # This file
├── requirements.txt          # Consolidated Python dependencies
├── LICENSE                   # MIT license (LIRL code)
├── CITATION.cff              # Citation metadata
│
├── RMS/                      # R2AMS: manufacturing scheduling (code name: RMS)
│   ├── env/                  # Environment, energy model, robot kinematics
│   ├── algs/                 # LIRL, HPPO, CPO, PDQN, HyAR, SAC-Lag
│   └── exp/                  # Constraint-change & runtime-scaling experiments
│
├── EV-Charging/              # EV charging station environment & algorithms
│   ├── env/                  # Gymnasium environment
│   ├── alg/                  # LIRL, PDQN, HPPO, LPPO, CPO
│   └── exp/                  # Multi-algorithm comparison
│
├── CityFlow/                 # Traffic signal control (CityFlow simulator + RL)
│   ├── algs/                 # LIRL, PDQN, HPPO, LPPO, CPO
│   ├── env/                  # Multi-intersection wrapper
│   └── exp/                  # Algorithm comparison, constraint edit, scaling
│
├── MP-DQN/                   # Parameterized-action baselines (Goal/Platform/Soccer)
│
└── Results/                  # Figure reproduction & archived experiment outputs
    ├── R2AMS/                # Fig. 3 and related R2AMS analyses
    ├── CityFlow/             # Traffic control figures
    ├── EV-Charging/          # EV charging figures
    ├── Protein_Crystallization_Screening/
    ├── Goal_Platform_Soccer/ # Hybrid benchmark comparison
    ├── Constraint-edit/      # Zero-shot constraint transfer (Fig. 5)
    ├── Runtime_profiling/    # Inference latency breakdown
    ├── stationarity_latency_Pareto/  # Projection optimality & Pareto analysis
    └── Elevator/             # Real-factory Gantt visualization (see Data Availability)
```

> **Naming note:** The manufacturing benchmark is referred to as **R2AMS** in the paper and **RMS** in the source code (`RMS/`). They refer to the same system.

Each subfolder under `Results/` typically includes a `README.txt` with script-specific details.

---

## System Requirements

| Component | Specification |
|-----------|---------------|
| **OS** | Linux (recommended), Windows 10+, or WSL2 |
| **Python** | 3.10 (tested) |
| **RAM** | ≥ 16 GB recommended for large-scale R2AMS (100 jobs × 5 robots) |
| **GPU** | Optional; CUDA-capable GPU accelerates training. CPU-only reproduction of figures is supported. |
| **Compiler** | Required only for CityFlow C++ extension (CMake + C++17). Use Linux/WSL2 or Docker (`CityFlow/Dockerfile`). |
| **Font** | Arial (or Helvetica/DejaVu Sans fallback) for publication-style figures |

**Estimated runtimes (approximate, single GPU):**

| Task | Duration |
|------|----------|
| Reproduce all paper figures from `Results/` | < 5 min |
| R2AMS LIRL training (100 jobs × 5 robots, 10 seeds × 1000 episodes) | 12–24 h |
| EV-Charging 5-algorithm comparison (1000 episodes each) | 4–8 h |
| CityFlow algorithm comparison | 6–12 h |
| Protein crystallization comparison | 2–4 h |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/wanguangxi/lirl_cps.git
cd lirl_cps
```

If the repository uses Git LFS for large `.npy` / model files:

```bash
git lfs install
git lfs pull
```

### 2. Create a virtual environment

```bash
conda create -n lirl-cps python=3.10 -y
conda activate lirl-cps
```

### 3. Install PyTorch

Install the appropriate PyTorch build for your system from [pytorch.org](https://pytorch.org/). Example (CUDA 11.8):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Domain-specific setup

**R2AMS (`RMS/`):**

```bash
pip install roboticstoolbox-python spatialmath-python
```

**CityFlow (optional, for training from scratch):**

```bash
cd CityFlow
pip install -e .
# Or use Docker — see CityFlow/README.rst
cd ..
```

**MP-DQN / hybrid benchmarks (optional):**

```bash
pip install -e git+https://github.com/cycraig/gym-platform#egg=gym_platform
pip install -e git+https://github.com/cycraig/gym-goal#egg=gym_goal
pip install -e git+https://github.com/cycraig/gym-soccer#egg=gym_soccer
```

See `MP-DQN/README.md` for details.

---

## Reproducibility Protocol

The experimental protocol follows the **Methods** section of the paper:

| Parameter | Value |
|-----------|-------|
| Independent random seeds | **N = 10** per algorithm and configuration |
| Default seeds (R2AMS LIRL) | `[3047, 294, 714, 1092, 1386, 2856, 42, 114514, 2025, 1993]` |
| Reported metrics | Mean ± **95% confidence interval** over seeds |
| Tabular results | Mean over **final 100 evaluation episodes** |
| Statistical test | **Welch's t-test** (two-tailed), significance threshold **p < 0.05** |
| Baseline fairness | Non-safe RL baselines augmented with **external shielding filters** where applicable |

All figure scripts in `Results/` read from **bundled result files** (JSON, CSV, NPY) produced under this protocol. Re-running training with the same seeds should yield statistically consistent results; minor numerical differences may arise from hardware or PyTorch version differences.

---

## Quick Start: Reproduce Paper Figures

From the repository root, run the visualization scripts below. Each script reads pre-computed data and writes publication-ready PNG/PDF files (Arial 8 pt, 300–600 DPI).

```bash
# --- R2AMS (Fig. 3) ---
python Results/R2AMS/lirl\ vs\ T-opt\&E-opt/run.py          # LIRL vs E-opt/T-opt boxplots
python Results/R2AMS/lirl\ vs\ baseline/run.py              # LIRL vs RL baselines (training curves)
python Results/R2AMS/lirl\ ablation/run.py                   # LIRL vs external shielding (ablation)
python Results/R2AMS/lirl\ robust/run.py                     # Robustness under breakdown/noise

# --- Traffic control ---
python Results/CityFlow/run.py

# --- EV charging ---
python Results/EV-Charging/run.py

# --- Crystallization-inspired benchmark ---
python Results/Protein_Crystallization_Screening/exp/plot_combined_figure.py \
    --data-dir Results/Protein_Crystallization_Screening/exp/comparison_20260205_154257

# --- Hybrid control benchmarks (Goal / Platform / Soccer) ---
python Results/Goal_Platform_Soccer/run.py

# --- Zero-shot constraint transfer ---
python Results/Constraint-edit/run.py

# --- Runtime profiling & projection analysis ---
python Results/Runtime_profiling/run.py
python Results/stationarity_latency_Pareto/run.py

# --- Real-factory Gantt chart (requires factory schedule JSON) ---
python Results/Elevator/plot_clean_gantt.py
```

**Windows (PowerShell):**

```powershell
python "Results\R2AMS\lirl vs T-opt&E-opt\run.py"
python "Results\R2AMS\lirl vs baseline\run.py"
python Results\CityFlow\run.py
python Results\EV-Charging\run.py
python Results\Goal_Platform_Soccer\run.py
python Results\Constraint-edit\run.py
python Results\Runtime_profiling\run.py
python Results\stationarity_latency_Pareto\run.py
```

---

## Training from Scratch

### R2AMS (robotic manufacturing)

```bash
cd RMS/algs
python lirl.py --episodes 1000                    # 10-seed multi-run (default)
python lirl.py --single-run --episodes 1000 --jobs 50 --robots 5   # quick validation
```

Other baselines: `cpo_policy.py`, `hppo_policy.py`, `pdqn_policy.py`, `hyar_policy.py`, `sac_lag_policy.py`.

Experiments:

```bash
python RMS/exp/lirl_runtime_scaling.py      # Runtime scaling across problem sizes
python RMS/exp/lirl_change_constraints.py   # Constraint-change transfer learning
```

> Update any hard-coded absolute paths (e.g., `pretrained_model_path`) before running on your machine.

### EV charging

```bash
# Single algorithm
python EV-Charging/alg/lirl.py --episodes 200 --stations 5 --power 150 --arrival-rate 0.75

# Five-algorithm comparison
python EV-Charging/exp/compare_algorithm.py --episodes 1000 --test-episodes 10
```

### Traffic control (CityFlow)

```bash
python CityFlow/exp/algorithm_compare.py      # Multi-algorithm comparison
python CityFlow/exp/constraint_edit.py        # Constraint-change experiment
python CityFlow/exp/run_scale_time.py         # Runtime scaling
```

### Crystallization-inspired benchmark

```bash
python Results/Protein_Crystallization_Screening/exp/compare.py
```

### Hybrid control (MP-DQN)

```bash
cd MP-DQN
python run_platform_paddpg.py    # Example; see MP-DQN/README.md for all domains
```

---

## Figure-to-Script Mapping

| Paper content | Script | Output |
|---------------|--------|--------|
| **Fig. 3a** — R2AMS cross-scale optimization (LIRL vs E-opt/T-opt) | `Results/R2AMS/lirl vs T-opt&E-opt/run.py` | `compare_reports/boxplot_by_scale.pdf` |
| **Fig. 3b** — Learning efficiency & reward distributions | `Results/R2AMS/lirl vs baseline/run.py` | `algorithm_comparison_combined.pdf` |
| **Fig. 3c** — LIRL vs external shielding ablation | `Results/R2AMS/lirl ablation/run.py` | `lirl_mask_comparison_1x4.pdf` |
| **Fig. 3d** — Robustness (breakdown / noise) | `Results/R2AMS/lirl robust/run.py` | `overall_comparison.pdf` |
| **Fig. 4** — Traffic control Pareto & heatmap | `Results/CityFlow/run.py` | `result/combined_figure.pdf` |
| **Fig. 5** — Zero-shot constraint transfer | `Results/Constraint-edit/run.py` | `three_scenarios_comparison.pdf` |
| **Fig. 6** — EV charging comparison | `Results/EV-Charging/run.py` | `combined_figure.pdf` |
| **Fig. 7** — Crystallization screening | `Results/Protein_Crystallization_Screening/exp/plot_combined_figure.py` | `combined_figure.pdf` |
| **Fig. 8** — Goal / Platform / Soccer | `Results/Goal_Platform_Soccer/run.py` | `three_scenarios_comparison.png` |
| **Extended Data** — Runtime profiling | `Results/Runtime_profiling/run.py` | `runtime_comparison_all_scenarios.pdf` |
| **Extended Data** — Projection optimality / Pareto | `Results/stationarity_latency_Pareto/run.py` | `stationarity_latency_pareto.pdf` |
| **Extended Data** — Real-factory Gantt | `Results/Elevator/plot_clean_gantt.py` | `clean_gantt_with_utilization.png` |

> Fig. 1–2 are schematic/overview illustrations prepared separately from this codebase.

---

## Data Availability

| Data type | Access |
|-----------|--------|
| Simulation environments (R2AMS, traffic, EV charging, crystallization) | **Procedurally generated** — reproducible via the provided training scripts |
| Pre-computed experiment outputs | Bundled under `Results/` subdirectories |
| Industrial elevator door-header factory data | Available from the corresponding author upon reasonable request, subject to **non-disclosure agreements** |
| Source Data (paper) | Provided as supplementary/source data files with the manuscript |

---

## License and Third-Party Code

| Component | License |
|-----------|---------|
| LIRL framework and experiment code (this repo, excluding submodules) | [MIT](LICENSE) |
| `CityFlow/` | See [CityFlow/LICENSE.txt](CityFlow/LICENSE.txt) |
| `MP-DQN/` | See [MP-DQN/LICENSE.md](MP-DQN/LICENSE.md) |

---

## Citation

If you use this code, please cite:

```bibtex
@article{wan2026lirl,
  title   = {Logic-informed reinforcement learning enables safe decision-making in hybrid cyber--physical systems},
  author  = {Wan, Guangxi and Zeng, Peng and Dong, Xiaoting and others},
  journal = {Nature Communications},
  year    = {2026},
  note    = {Code available at \url{https://github.com/wanguangxi/lirl_cps}}
}
```

---

## Contact

- **Guangxi Wan** — wanguangxi@sia.cn  
- **Corresponding author: Peng Zeng** — zp@sia.cn  

For questions about reproduction, please open a [GitHub Issue](https://github.com/wanguangxi/lirl_cps/issues) or contact the authors directly.

---

## 中文说明

本仓库为 *Nature Communications* 论文 **「Logic-informed reinforcement learning enables safe decision-making in hybrid cyber–physical systems」** 的官方配套代码，包含 LIRL 算法、仿真环境、基线方法及论文图表复现脚本。

### 快速复现图表

1. 安装 Python 3.10 及依赖：`pip install -r requirements.txt`
2. 在仓库根目录运行 `Results/` 下对应脚本（详见上方 [Quick Start](#quick-start-reproduce-paper-figures)）
3. 所有图表脚本均读取已保存的实验结果（JSON/CSV/NPY），无需重新训练即可生成论文级 PDF/PNG

### 从头训练

| 任务 | 入口脚本 |
|------|----------|
| 工业机器人调度（R2AMS） | `RMS/algs/lirl.py`（代码目录名为 RMS，论文中称为 R2AMS） |
| 交通信号控制 | `CityFlow/exp/algorithm_compare.py` |
| 电动汽车充电 | `EV-Charging/exp/compare_algorithm.py` |
| 结晶筛选基准 | `Results/Protein_Crystallization_Screening/exp/compare.py` |

### 复现说明

- **开源协议**：LIRL 代码采用 [MIT](LICENSE) 协议
- **随机种子**：所有实验默认 **10 个独立随机种子**，报告均值 ± 95% 置信区间
- **工厂数据**：电梯门楣装配产线的工业数据需联系通讯作者（zp@sia.cn）并签署保密协议

如有问题，请提交 [GitHub Issue](https://github.com/wanguangxi/lirl_cps/issues) 或联系：wanguangxi@sia.cn
