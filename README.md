# LIRL-CPS

**Logic-Informed Reinforcement Learning for Safe Hybrid Cyber-Physical Systems**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)

Official implementation accompanying:

> **Logic-informed reinforcement learning enables safe decision-making in hybrid cyber–physical systems**  
> Guangxi Wan, Peng Zeng, Xiaoting Dong, *et al.*, *Nature Communications* (2026).

**Repository:** [https://github.com/wanguangxi/lirl_cps](https://github.com/wanguangxi/lirl_cps)

This repository provides the **LIRL** (Logic-Informed Reinforcement Learning) framework, simulation environments, baseline implementations, **bundled demo datasets**, and figure-reproduction scripts used in the paper.

---

## Table of Contents

- [Required Content](#required-content)
- [1. System Requirements](#1-system-requirements)
- [2. Installation Guide](#2-installation-guide)
- [3. Demo](#3-demo)
- [4. Instructions for Use](#4-instructions-for-use)
- [5. Reproduction Instructions](#5-reproduction-instructions)
- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Data Availability](#data-availability)
- [License and Third-Party Code](#license-and-third-party-code)
- [Citation](#citation)
- [Contact](#contact)

---

## Required Content

This submission includes all items required by the Nature Research Code and Software checklist:

| Item | Location |
|------|----------|
| **Source code** | Full LIRL framework and baselines under `RMS/`, `EV-Charging/`, `CityFlow/`, `MP-DQN/`, and `Results/Protein_Crystallization_Screening/` |
| **Demo dataset** | Pre-computed experiment outputs bundled under `Results/` (JSON, CSV, NPY, XLSX) — no external download required |
| **README** | This file |

LIRL is a neuro-symbolic reinforcement learning framework that embeds **quantifier-free linear real arithmetic (QF_LRA)** domain specifications into a learning-compatible **logic-to-manifold projection operator**. A neural policy outputs an unconstrained continuous intent; a non-learnable projector maps this intent onto the state-dependent feasible manifold, guaranteeing **step-wise constraint satisfaction by construction**.

---

## 1. System Requirements

### Operating systems

| OS | Version tested |
|----|----------------|
| **Linux** (recommended) | Ubuntu 20.04 / 22.04 |
| **Windows** | Windows 10 / 11 |
| **WSL2** | Ubuntu 22.04 (recommended for CityFlow C++ build on Windows) |

### Software dependencies (with version numbers)

| Package | Minimum version | Notes |
|---------|-----------------|-------|
| Python | **3.10** | Required |
| PyTorch | **≥ 1.7.0** | Tested with 2.0+ (CPU and CUDA builds) |
| NumPy | ≥ 1.19.0 | |
| SciPy | ≥ 1.5.0 | |
| pandas | ≥ 1.0.0 | |
| matplotlib | ≥ 3.3.0 | |
| seaborn | ≥ 0.10.0 | |
| Gymnasium | ≥ 0.26.0 | |
| roboticstoolbox-python | ≥ 1.0.0 | Required for R2AMS (`RMS/`) |
| spatialmath-python | ≥ 1.0.0 | Required for R2AMS (`RMS/`) |
| Git LFS | **≥ 3.0** | Required to download bundled `.npy` / model files |
| CMake | ≥ 3.10 | Required only for CityFlow C++ extension |
| C++ compiler | C++17 | Required only for CityFlow (g++ / MSVC) |

All Python dependencies are listed in [`requirements.txt`](requirements.txt). Install PyTorch separately from [pytorch.org](https://pytorch.org/) for your CPU/CUDA configuration.

**Optional dependencies** (only needed for specific domains):

| Package | Domain |
|---------|--------|
| CityFlow (local install) | Traffic signal control training |
| gym-platform, gym-goal, gym-soccer | Hybrid control benchmarks (`MP-DQN/`) |

### Versions tested

The software has been tested on the following configurations:

| Component | Tested version |
|-----------|----------------|
| Python | 3.10.x |
| PyTorch | 2.0+ (CUDA 11.8) and CPU-only builds |
| NumPy | 1.24.x |
| Gymnasium | 0.29.x |
| OS | Ubuntu 22.04, Windows 10/11 |

Minor numerical differences may arise when using other PyTorch or CUDA versions; figure reproduction from bundled data is not affected.

### Hardware

| Component | Requirement |
|-----------|-------------|
| **CPU** | Any modern x86_64 processor (demo and figure reproduction run on CPU) |
| **RAM** | ≥ 8 GB for demo; ≥ 16 GB recommended for full-scale R2AMS training (100 jobs × 5 robots) |
| **GPU** | **Optional.** CUDA-capable GPU accelerates training; not required for demo or figure reproduction |
| **Disk** | ≥ 5 GB free space (including Git LFS objects) |
| **Non-standard hardware** | **None required** |

> **Font note:** Publication-style figures use Arial (or Helvetica / DejaVu Sans as fallback).

---

## 2. Installation Guide

### Instructions

Run the following from a terminal at the repository root.

**Step 1 — Clone the repository (with Git LFS):**

```bash
git lfs install
git clone https://github.com/wanguangxi/lirl_cps.git
cd lirl_cps
git lfs pull
```

**Step 2 — Create a virtual environment:**

```bash
conda create -n lirl-cps python=3.10 -y
conda activate lirl-cps
```

**Step 3 — Install PyTorch** (choose CPU or CUDA build from [pytorch.org](https://pytorch.org/)):

```bash
# Example: CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Example: CPU only
pip install torch
```

**Step 4 — Install core dependencies:**

```bash
pip install -r requirements.txt
```

**Step 5 — Domain-specific setup (as needed):**

```bash
# R2AMS — already included in requirements.txt; verify:
pip install roboticstoolbox-python spatialmath-python

# CityFlow — only if training traffic control from scratch:
cd CityFlow && pip install -e . && cd ..

# MP-DQN hybrid benchmarks — optional:
pip install -e git+https://github.com/cycraig/gym-platform#egg=gym_platform
pip install -e git+https://github.com/cycraig/gym-goal#egg=gym_goal
pip install -e git+https://github.com/cycraig/gym-soccer#egg=gym_soccer
```

**Step 6 — Verify installation:**

```bash
python -c "import torch, numpy, gymnasium; print('PyTorch', torch.__version__, '| CUDA:', torch.cuda.is_available())"
```

### Typical install time

On a normal desktop computer with broadband internet:

| Step | Estimated time |
|------|----------------|
| Clone repository + Git LFS pull | 10–30 min (depends on network; ~2.6 GB LFS data) |
| Create conda environment + pip install | 5–10 min |
| CityFlow C++ build (optional) | 10–20 min |
| **Total (demo / figure reproduction only)** | **~15–40 min** |
| **Total (including CityFlow build)** | **~25–60 min** |

---

## 3. Demo

The bundled demo uses **pre-computed results** under `Results/` — no training or external data download is required. This is the recommended first step for reviewers and new users.

### Demo instructions

From the repository root, after completing [Installation](#2-installation-guide):

**Linux / macOS:**

```bash
python Results/EV-Charging/run.py
```

**Windows (PowerShell):**

```powershell
python Results\EV-Charging\run.py
```

This script reads the bundled EV-charging comparison data in `Results/EV-Charging/result/` and generates a publication-style figure.

### Expected output

| Output file | Location |
|-------------|----------|
| `combined_figure.png` | `Results/EV-Charging/` |
| `combined_figure.pdf` | `Results/EV-Charging/` |

The figure contains a **Pareto front analysis** and **performance heatmap** comparing LIRL against PDQN, HPPO, LPPO, and CPO on the EV charging benchmark (corresponds to **Fig. 6** in the paper).

Console output ends without error; matplotlib may show a brief font warning if Arial is not installed (figures still render correctly with fallback fonts).

### Expected demo runtime

| Task | Runtime (normal desktop, CPU) |
|------|-------------------------------|
| Single demo script above | **< 30 seconds** |
| All figure-reproduction scripts in `Results/` | **< 5 minutes** |

### Additional quick demos

```bash
# Traffic control (Fig. 4)
python Results/CityFlow/run.py

# R2AMS boxplot (Fig. 3a)
python "Results/R2AMS/lirl vs T-opt&E-opt/run.py"    # Windows
python Results/R2AMS/lirl\ vs\ T-opt\&E-opt/run.py    # Linux
```

---

## 4. Instructions for Use

This section describes how to run LIRL on **your own problem instances** in each benchmark domain. All environments are **procedurally generated** from configurable parameters — no external dataset files are needed for simulation-based experiments.

### R2AMS — robotic manufacturing scheduling

Code directory: `RMS/` (paper name: **R2AMS**).

**Quick validation (single seed, reduced scale):**

```bash
cd RMS/algs
python lirl.py --single-run --episodes 100 --jobs 50 --robots 5
```

**Full-scale training (10 seeds, default):**

```bash
python lirl.py --episodes 1000
```

Key CLI parameters: `--jobs`, `--robots`, `--episodes`, `--single-run`, `--seeds`.

Other algorithms: `cpo_policy.py`, `hppo_policy.py`, `pdqn_policy.py`, `hyar_policy.py`, `sac_lag_policy.py`.

### EV charging station scheduling

```bash
# Train LIRL on a custom configuration
python EV-Charging/alg/lirl.py --episodes 200 --stations 5 --power 150 --arrival-rate 0.75

# Compare five algorithms
python EV-Charging/exp/compare_algorithm.py --episodes 1000 --test-episodes 10
```

Key CLI parameters: `--stations`, `--power`, `--arrival-rate`, `--episodes`.

### Traffic signal control (CityFlow)

Requires CityFlow C++ extension (see [Installation](#2-installation-guide)).

```bash
python CityFlow/exp/algorithm_compare.py
python CityFlow/exp/constraint_edit.py
python CityFlow/exp/run_scale_time.py
```

### Crystallization-inspired screening benchmark

```bash
python Results/Protein_Crystallization_Screening/exp/compare.py
python Results/Protein_Crystallization_Screening/exp/plot_combined_figure.py \
    --data-dir Results/Protein_Crystallization_Screening/exp/comparison_20260205_154257
```

### Hybrid control (Goal / Platform / Soccer)

```bash
cd MP-DQN
python run_platform_paddpg.py    # See MP-DQN/README.md for all domains
```

### Adapting to custom CPS domains

To apply LIRL to a new hybrid CPS problem:

1. Define QF_LRA domain constraints in a Gymnasium-compatible environment (see `RMS/env/` or `EV-Charging/env/ev.py` as templates).
2. Implement the logic-to-manifold projection operator for your constraint set (see `RMS/algs/lirl.py` and domain-specific `alg/lirl.py` files).
3. Train with the provided DDPG-LIRL loop; use `--single-run` with a small episode count to validate the pipeline before full multi-seed runs.

> **Note:** Update any hard-coded absolute paths (e.g., `pretrained_model_path` in `RMS/exp/lirl_change_constraints.py`) before running on your machine.

---

## 5. Reproduction Instructions

This section provides instructions to reproduce **all quantitative results and figures** reported in the manuscript.

### Experimental protocol

| Parameter | Value |
|-----------|-------|
| Independent random seeds | **N = 10** per algorithm and configuration |
| Default seeds (R2AMS LIRL) | `[3047, 294, 714, 1092, 1386, 2856, 42, 114514, 2025, 1993]` |
| Reported metrics | Mean ± **95% confidence interval** over seeds |
| Tabular results | Mean over **final 100 evaluation episodes** |
| Statistical test | **Welch's t-test** (two-tailed), significance threshold **p < 0.05** |
| Baseline fairness | Non-safe RL baselines augmented with **external shielding filters** where applicable |

All figure scripts in `Results/` read from bundled result files produced under this protocol. Re-running training with the same seeds should yield statistically consistent results.

### Reproduce all paper figures

From the repository root (after installation):

```bash
# --- R2AMS (Fig. 3) ---
python Results/R2AMS/lirl\ vs\ T-opt\&E-opt/run.py
python Results/R2AMS/lirl\ vs\ baseline/run.py
python Results/R2AMS/lirl\ ablation/run.py
python Results/R2AMS/lirl\ robust/run.py

# --- Traffic control (Fig. 4) ---
python Results/CityFlow/run.py

# --- Zero-shot constraint transfer (Fig. 5) ---
python Results/Constraint-edit/run.py

# --- EV charging (Fig. 6) ---
python Results/EV-Charging/run.py

# --- Crystallization screening (Fig. 7) ---
python Results/Protein_Crystallization_Screening/exp/plot_combined_figure.py \
    --data-dir Results/Protein_Crystallization_Screening/exp/comparison_20260205_154257

# --- Hybrid control (Fig. 8) ---
python Results/Goal_Platform_Soccer/run.py

# --- Extended Data ---
python Results/Runtime_profiling/run.py
python Results/stationarity_latency_Pareto/run.py
python Results/Elevator/plot_clean_gantt.py
```

**Windows (PowerShell):**

```powershell
python "Results\R2AMS\lirl vs T-opt&E-opt\run.py"
python "Results\R2AMS\lirl vs baseline\run.py"
python "Results\R2AMS\lirl ablation\run.py"
python "Results\R2AMS\lirl robust\run.py"
python Results\CityFlow\run.py
python Results\Constraint-edit\run.py
python Results\EV-Charging\run.py
python Results\Goal_Platform_Soccer\run.py
python Results\Runtime_profiling\run.py
python Results\stationarity_latency_Pareto\run.py
```

**Estimated runtime:** < 5 min (CPU, all figure scripts).

### Figure-to-script mapping

| Paper content | Script | Output |
|---------------|--------|--------|
| **Fig. 3a** — R2AMS cross-scale optimization | `Results/R2AMS/lirl vs T-opt&E-opt/run.py` | `compare_reports/boxplot_by_scale.pdf` |
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

### Re-run training from scratch (approximate runtimes, single GPU)

| Experiment | Command | Duration |
|------------|---------|----------|
| R2AMS LIRL (100 jobs × 5 robots, 10 seeds × 1000 ep.) | `RMS/algs/lirl.py --episodes 1000` | 12–24 h |
| EV-Charging 5-algorithm comparison | `EV-Charging/exp/compare_algorithm.py --episodes 1000` | 4–8 h |
| CityFlow algorithm comparison | `CityFlow/exp/algorithm_compare.py` | 6–12 h |
| Crystallization comparison | `Results/Protein_Crystallization_Screening/exp/compare.py` | 2–4 h |
| R2AMS constraint-change transfer | `RMS/exp/lirl_change_constraints.py` | 4–8 h |
| R2AMS runtime scaling | `RMS/exp/lirl_runtime_scaling.py` | 2–6 h |

---

## Overview

This repository covers five benchmark domains from the paper:

| Domain | Code directory | Paper name |
|--------|----------------|------------|
| Robotic reducer-assembly manufacturing | `RMS/` | **R2AMS** |
| Urban traffic signal control | `CityFlow/` | Traffic control |
| EV charging station scheduling | `EV-Charging/` | EV charging |
| Crystallization-inspired screening | `Results/Protein_Crystallization_Screening/` | Synthetic crystallization benchmark |
| Parameterized hybrid control | `MP-DQN/` | Goal / Platform / Soccer |

---

## Repository Structure

```
LIRL-CPS/
├── README.md                 # This file
├── requirements.txt          # Consolidated Python dependencies
├── LICENSE                   # MIT license (LIRL code)
├── CITATION.cff              # Citation metadata
│
├── RMS/                      # R2AMS: manufacturing scheduling
│   ├── env/                  # Environment, energy model, robot kinematics
│   ├── algs/                 # LIRL, HPPO, CPO, PDQN, HyAR, SAC-Lag
│   └── exp/                  # Constraint-change & runtime-scaling experiments
│
├── EV-Charging/              # EV charging environment & algorithms
│   ├── env/                  # Gymnasium environment
│   ├── alg/                  # LIRL, PDQN, HPPO, LPPO, CPO
│   └── exp/                  # Multi-algorithm comparison
│
├── CityFlow/                 # Traffic signal control simulator + RL
│   ├── algs/                 # LIRL, PDQN, HPPO, LPPO, CPO
│   ├── env/                  # Multi-intersection wrapper
│   └── exp/                  # Algorithm comparison, constraint edit, scaling
│
├── MP-DQN/                   # Parameterized-action baselines
│
└── Results/                  # Demo datasets + figure reproduction scripts
    ├── R2AMS/                # Fig. 3 analyses
    ├── CityFlow/             # Fig. 4
    ├── EV-Charging/          # Fig. 6
    ├── Protein_Crystallization_Screening/  # Fig. 7
    ├── Goal_Platform_Soccer/ # Fig. 8
    ├── Constraint-edit/      # Fig. 5
    ├── Runtime_profiling/    # Extended Data
    ├── stationarity_latency_Pareto/
    └── Elevator/             # Real-factory Gantt (see Data Availability)
```

> **Naming note:** **R2AMS** (paper) = **RMS** (code). Each `Results/` subfolder includes a `README.txt` with script-specific details.

---

## Data Availability

| Data type | Access |
|-----------|--------|
| Simulation environments | **Procedurally generated** — reproducible via training scripts |
| Demo / pre-computed outputs | Bundled under `Results/` (included in repository via Git LFS) |
| Industrial elevator factory data | Available from corresponding author upon request (NDA required) |
| Source Data (paper) | Provided as supplementary files with the manuscript |

---

## License and Third-Party Code

| Component | License |
|-----------|---------|
| LIRL framework and experiment code | [MIT](LICENSE) |
| `CityFlow/` | [CityFlow/LICENSE.txt](CityFlow/LICENSE.txt) |
| `MP-DQN/` | [MP-DQN/LICENSE.md](MP-DQN/LICENSE.md) |

---

## Citation

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

For questions about installation or reproduction, please open a [GitHub Issue](https://github.com/wanguangxi/lirl_cps/issues) or contact the authors directly.
