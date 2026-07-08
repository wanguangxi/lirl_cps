# LIRL-CPS

**Logic-Informed Reinforcement Learning for Safe Hybrid Cyber-Physical Systems**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)

Official implementation accompanying:

> **Logic-informed reinforcement learning enables safe decision-making in hybrid cyber–physical systems**  
> Guangxi Wan, Peng Zeng, Xiaoting Dong, *et al.*, *Nature Communications* (2026).

**Repository:** [https://github.com/wanguangxi/lirl_cps](https://github.com/wanguangxi/lirl_cps)

LIRL is a neuro-symbolic reinforcement learning framework that embeds **quantifier-free linear real arithmetic (QF_LRA)** domain specifications into a learning-compatible **logic-to-manifold projection operator**. A neural policy outputs an unconstrained continuous intent; a non-learnable projector maps this intent onto the state-dependent feasible manifold, guaranteeing **step-wise constraint satisfaction by construction**.

A complete, detailed description of the code's functionality (algorithms and pseudocode) is provided in the **Methods** section of the manuscript.

---

## Table of Contents

- [Required Content](#required-content)
- [1. System Requirements](#1-system-requirements)
- [2. Installation Guide](#2-installation-guide)
- [3. Demo](#3-demo)
- [4. Instructions for Use](#4-instructions-for-use)
- [5. Reproduction Instructions](#5-reproduction-instructions)
- [Additional Information](#additional-information)
- [Repository Structure](#repository-structure)
- [Data Availability](#data-availability)
- [Citation](#citation)
- [Contact](#contact)

---

## Required Content

All items below are available at [https://github.com/wanguangxi/lirl_cps](https://github.com/wanguangxi/lirl_cps) (Git LFS required for large artifacts).

| Required item | Provided | Location |
|---------------|----------|----------|
| **Source code** (with version details) | Yes | `RMS/`, `EV-Charging/`, `CityFlow/`, `MP-DQN/`, `Results/Protein_Crystallization_Screening/`; release tag recommended for publication (e.g. `v1.0.0`) |
| **Demo dataset** (simulated) | Yes | Pre-computed outputs under `Results/` (JSON, CSV, NPY, XLSX) — no external download required |
| **README** (this file) | Yes | Repository root |

> **Reviewer note:** Prior to submission, we recommend that an unfamiliar colleague clone the repository, follow Sections 2–3, and confirm that the demo runs successfully on a normal desktop computer.

---

## 1. System Requirements

### 1.1 Operating systems (including version numbers)

| OS | Version tested |
|----|----------------|
| **Linux** (recommended) | Ubuntu 20.04, Ubuntu 22.04 |
| **Windows** | Windows 10, Windows 11 |
| **WSL2** | Ubuntu 22.04 (recommended for CityFlow C++ build on Windows) |

### 1.2 Software dependencies (including version numbers)

| Package | Minimum version | Required for |
|---------|-----------------|--------------|
| **Python** | **3.10** | All domains |
| **PyTorch** | **≥ 1.7.0** (tested with 2.0+) | All domains |
| NumPy | ≥ 1.19.0 | All domains |
| SciPy | ≥ 1.5.0 | All domains |
| pandas | ≥ 1.0.0 | All domains |
| matplotlib | ≥ 3.3.0 | All domains |
| seaborn | ≥ 0.10.0 | All domains |
| Gymnasium | ≥ 0.26.0 | All domains |
| roboticstoolbox-python | ≥ 1.0.0 | R2AMS (`RMS/`) |
| spatialmath-python | ≥ 1.0.0 | R2AMS (`RMS/`) |
| Git LFS | ≥ 3.0 | Downloading bundled `.npy` / model files |
| CMake | ≥ 3.10 | CityFlow C++ extension only |
| C++ compiler (C++17) | g++ / MSVC | CityFlow C++ extension only |

Full Python dependency list: [`requirements.txt`](requirements.txt).  
Install PyTorch separately from [pytorch.org](https://pytorch.org/) for your CPU or CUDA build.

**Optional dependencies** (domain-specific):

| Package | Domain |
|---------|--------|
| CityFlow (local `pip install -e .`) | Traffic signal control training |
| gym-platform, gym-goal, gym-soccer | Hybrid control benchmarks (`MP-DQN/`) |

### 1.3 Versions the software has been tested on

| Component | Tested version |
|-----------|----------------|
| Python | 3.10.x |
| PyTorch | 2.0+ (CUDA 11.8) and CPU-only builds |
| NumPy | 1.24.x |
| SciPy | 1.10.x |
| pandas | 2.0.x |
| matplotlib | 3.7.x |
| Gymnasium | 0.29.x |
| OS | Ubuntu 22.04; Windows 10 / 11 |

Minor numerical differences may arise with other PyTorch or CUDA versions; figure reproduction from bundled data is not affected.

### 1.4 Non-standard hardware

**None required.**

| Component | Specification |
|-----------|---------------|
| CPU | Any modern x86_64 processor |
| RAM | ≥ 8 GB (demo); ≥ 16 GB recommended for full-scale R2AMS training |
| GPU | Optional — CUDA-capable GPU accelerates training; not required for demo |
| Disk | ≥ 5 GB free space (including Git LFS objects) |

Publication-style figures use Arial (Helvetica / DejaVu Sans fallback if Arial is unavailable).

---

## 2. Installation Guide

### 2.1 Instructions

Run from a terminal at the repository root.

**Step 1 — Clone with Git LFS:**

```bash
git lfs install
git clone https://github.com/wanguangxi/lirl_cps.git
cd lirl_cps
git lfs pull
```

**Step 2 — Create environment:**

```bash
conda create -n lirl-cps python=3.10 -y
conda activate lirl-cps
```

**Step 3 — Install PyTorch** ([pytorch.org](https://pytorch.org/)):

```bash
# CUDA 11.8 example
pip install torch --index-url https://download.pytorch.org/whl/cu118

# CPU-only example
pip install torch
```

**Step 4 — Install dependencies:**

```bash
pip install -r requirements.txt
```

**Step 5 — Optional domain setup:**

```bash
# CityFlow (traffic control training only)
cd CityFlow && pip install -e . && cd ..

# MP-DQN hybrid benchmarks
pip install -e git+https://github.com/cycraig/gym-platform#egg=gym_platform
pip install -e git+https://github.com/cycraig/gym-goal#egg=gym_goal
pip install -e git+https://github.com/cycraig/gym-soccer#egg=gym_soccer
```

**Step 6 — Verify:**

```bash
python -c "import torch, numpy, gymnasium; print('PyTorch', torch.__version__, '| CUDA:', torch.cuda.is_available())"
```

### 2.2 Typical install time on a normal desktop computer

| Task | Estimated time |
|------|----------------|
| Clone + Git LFS pull (~2.6 GB) | 10–30 min |
| Conda environment + `pip install` | 5–10 min |
| CityFlow C++ build (optional) | 10–20 min |
| **Total (demo / figure reproduction)** | **~15–40 min** |
| **Total (including CityFlow build)** | **~25–60 min** |

Times assume broadband internet and a current desktop (4+ CPU cores, ≥ 8 GB RAM).

---

## 3. Demo

The demo uses the **bundled simulated dataset** in `Results/EV-Charging/result/` — no training or external download is required.

### 3.1 Instructions to run on data

From the repository root, after completing [Section 2](#2-installation-guide):

**Linux / macOS:**

```bash
python Results/EV-Charging/run.py
```

**Windows (PowerShell):**

```powershell
python Results\EV-Charging\run.py
```

The script reads pre-computed comparison results and generates a publication-style figure comparing LIRL, PDQN, HPPO, LPPO, and CPO on the EV charging benchmark.

### 3.2 Expected output

| File | Location | Description |
|------|----------|-------------|
| `combined_figure.png` | `Results/EV-Charging/` | Pareto front + performance heatmap (PNG, 300 DPI) |
| `combined_figure.pdf` | `Results/EV-Charging/` | Same figure (PDF) |

Corresponds to **Fig. 6** in the manuscript. The script completes without error; a matplotlib font warning may appear if Arial is not installed (fallback fonts are used).

### 3.3 Expected run time on a normal desktop computer

| Task | Runtime (CPU) |
|------|---------------|
| Demo command above | **< 30 seconds** |
| All figure-reproduction scripts (`Results/`) | **< 5 minutes** |

---

## 4. Instructions for Use

This section describes how to run the software on **your own data / problem instances**. All benchmark environments are **procedurally generated simulators** — configure parameters via CLI flags; no external dataset files are required.

### 4.1 R2AMS — robotic manufacturing scheduling

Code directory: `RMS/` (paper name: **R2AMS**).

```bash
cd RMS/algs

# Quick validation (single seed, reduced scale)
python lirl.py --single-run --episodes 100 --jobs 50 --robots 5

# Full-scale training (10 seeds, default)
python lirl.py --episodes 1000
```

Key parameters: `--jobs`, `--robots`, `--episodes`, `--single-run`, `--seeds`.  
Other algorithms: `cpo_policy.py`, `hppo_policy.py`, `pdqn_policy.py`, `hyar_policy.py`, `sac_lag_policy.py`.

### 4.2 EV charging station scheduling

```bash
python EV-Charging/alg/lirl.py --episodes 200 --stations 5 --power 150 --arrival-rate 0.75
python EV-Charging/exp/compare_algorithm.py --episodes 1000 --test-episodes 10
```

Key parameters: `--stations`, `--power`, `--arrival-rate`, `--episodes`.

### 4.3 Traffic signal control (CityFlow)

Requires CityFlow C++ extension ([Section 2](#2-installation-guide)).

```bash
python CityFlow/exp/algorithm_compare.py
python CityFlow/exp/constraint_edit.py
python CityFlow/exp/run_scale_time.py
```

### 4.4 Crystallization-inspired screening

```bash
python Results/Protein_Crystallization_Screening/exp/compare.py
python Results/Protein_Crystallization_Screening/exp/plot_combined_figure.py \
    --data-dir Results/Protein_Crystallization_Screening/exp/comparison_20260205_154257
```

### 4.5 Hybrid control (Goal / Platform / Soccer)

```bash
cd MP-DQN
python run_platform_paddpg.py    # See MP-DQN/README.md for all domains
```

### 4.6 Adapting to a new CPS domain

1. Define QF_LRA constraints in a Gymnasium-compatible environment (`RMS/env/` or `EV-Charging/env/ev.py` as templates).
2. Implement the logic-to-manifold projection operator (`RMS/algs/lirl.py` and domain `alg/lirl.py` files).
3. Train with the DDPG-LIRL loop; validate with `--single-run` before full multi-seed runs.

> Update hard-coded absolute paths (e.g., `pretrained_model_path` in `RMS/exp/lirl_change_constraints.py`) before running on your machine.

---

## 5. Reproduction Instructions

Instructions to reproduce **all quantitative results and figures** in the manuscript.

### 5.1 Experimental protocol

| Parameter | Value |
|-----------|-------|
| Independent random seeds | **N = 10** per algorithm and configuration |
| Default seeds (R2AMS LIRL) | `[3047, 294, 714, 1092, 1386, 2856, 42, 114514, 2025, 1993]` |
| Reported metrics | Mean ± **95% confidence interval** over seeds |
| Tabular results | Mean over **final 100 evaluation episodes** |
| Statistical test | **Welch's t-test** (two-tailed), *p* < 0.05 |
| Baseline fairness | Non-safe RL baselines augmented with **external shielding filters** where applicable |

### 5.2 Reproduce all paper figures

From the repository root:

```bash
python Results/R2AMS/lirl\ vs\ T-opt\&E-opt/run.py
python Results/R2AMS/lirl\ vs\ baseline/run.py
python Results/R2AMS/lirl\ ablation/run.py
python Results/R2AMS/lirl\ robust/run.py
python Results/CityFlow/run.py
python Results/Constraint-edit/run.py
python Results/EV-Charging/run.py
python Results/Protein_Crystallization_Screening/exp/plot_combined_figure.py \
    --data-dir Results/Protein_Crystallization_Screening/exp/comparison_20260205_154257
python Results/Goal_Platform_Soccer/run.py
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

**Runtime:** < 5 min (CPU, all figure scripts).

### 5.3 Figure-to-script mapping

| Paper content | Script | Output |
|---------------|--------|--------|
| Fig. 3a — R2AMS cross-scale optimization | `Results/R2AMS/lirl vs T-opt&E-opt/run.py` | `compare_reports/boxplot_by_scale.pdf` |
| Fig. 3b — Learning efficiency | `Results/R2AMS/lirl vs baseline/run.py` | `algorithm_comparison_combined.pdf` |
| Fig. 3c — LIRL vs external shielding | `Results/R2AMS/lirl ablation/run.py` | `lirl_mask_comparison_1x4.pdf` |
| Fig. 3d — Robustness | `Results/R2AMS/lirl robust/run.py` | `overall_comparison.pdf` |
| Fig. 4 — Traffic control | `Results/CityFlow/run.py` | `result/combined_figure.pdf` |
| Fig. 5 — Constraint transfer | `Results/Constraint-edit/run.py` | `three_scenarios_comparison.pdf` |
| Fig. 6 — EV charging | `Results/EV-Charging/run.py` | `combined_figure.pdf` |
| Fig. 7 — Crystallization screening | `Results/Protein_Crystallization_Screening/exp/plot_combined_figure.py` | `combined_figure.pdf` |
| Fig. 8 — Goal / Platform / Soccer | `Results/Goal_Platform_Soccer/run.py` | `three_scenarios_comparison.png` |
| Extended Data — Runtime profiling | `Results/Runtime_profiling/run.py` | `runtime_comparison_all_scenarios.pdf` |
| Extended Data — Projection Pareto | `Results/stationarity_latency_Pareto/run.py` | `stationarity_latency_pareto.pdf` |
| Extended Data — Factory Gantt | `Results/Elevator/plot_clean_gantt.py` | `clean_gantt_with_utilization.png` |

> Fig. 1–2 are schematic illustrations prepared separately from this codebase.

### 5.4 Re-run training from scratch (single GPU)

| Experiment | Command | Duration |
|------------|---------|----------|
| R2AMS LIRL (100 jobs × 5 robots, 10 seeds) | `RMS/algs/lirl.py --episodes 1000` | 12–24 h |
| EV-Charging 5-algorithm comparison | `EV-Charging/exp/compare_algorithm.py --episodes 1000` | 4–8 h |
| CityFlow algorithm comparison | `CityFlow/exp/algorithm_compare.py` | 6–12 h |
| Crystallization comparison | `Results/Protein_Crystallization_Screening/exp/compare.py` | 2–4 h |
| R2AMS constraint-change transfer | `RMS/exp/lirl_change_constraints.py` | 4–8 h |
| R2AMS runtime scaling | `RMS/exp/lirl_runtime_scaling.py` | 2–6 h |

---

## Additional Information

As required by the Nature Portfolio **Code and Software Submission Checklist**:

| Item | Details |
|------|---------|
| **License** | [MIT License](LICENSE) — [Open Source Initiative](https://opensource.org/licenses/MIT) approved |
| **Open-source repository** | [https://github.com/wanguangxi/lirl_cps](https://github.com/wanguangxi/lirl_cps) |
| **Release / DOI** | GitHub release tag (e.g. `v1.0.0`) recommended; archive on [Zenodo](https://zenodo.org/) for a citable DOI upon publication |
| **Code functionality description** | Main text **Methods** section (algorithms, projection operator, training protocol) |
| **Third-party licenses** | `CityFlow/` — [LICENSE.txt](CityFlow/LICENSE.txt); `MP-DQN/` — [LICENSE.md](MP-DQN/LICENSE.md) |

**Benchmark domains covered:**

| Domain | Code directory | Paper name |
|--------|----------------|------------|
| Robotic reducer-assembly manufacturing | `RMS/` | R2AMS |
| Urban traffic signal control | `CityFlow/` | Traffic control |
| EV charging station scheduling | `EV-Charging/` | EV charging |
| Crystallization-inspired screening | `Results/Protein_Crystallization_Screening/` | Synthetic crystallization benchmark |
| Parameterized hybrid control | `MP-DQN/` | Goal / Platform / Soccer |

---

## Repository Structure

```
LIRL-CPS/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── LICENSE                   # MIT license
├── CITATION.cff              # Citation metadata
├── RMS/                      # R2AMS manufacturing scheduling
├── EV-Charging/              # EV charging environment & algorithms
├── CityFlow/                 # Traffic signal control
├── MP-DQN/                   # Parameterized-action baselines
└── Results/                  # Demo datasets + figure scripts
```

> **R2AMS** (paper) = **RMS** (code). Each `Results/` subfolder includes a `README.txt` with script-specific details.

---

## Data Availability

| Data type | Access |
|-----------|--------|
| Simulation environments | Procedurally generated — reproducible via training scripts |
| Demo / pre-computed outputs | Bundled under `Results/` (Git LFS) |
| Industrial factory data | Available from corresponding author upon request (NDA) |
| Source Data (paper) | Supplementary files with the manuscript |

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

For installation or reproduction questions, open a [GitHub Issue](https://github.com/wanguangxi/lirl_cps/issues) or contact the authors directly.
