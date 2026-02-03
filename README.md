# LIRL-CPS (Major Revision)

This repository focuses on **cross-domain optimization for large-scale cyber-physical systems (CPS)**. It unifies **Logic-Informed Reinforcement Learning (LIRL)** and multiple baselines (HPPO / CPO / PDQN / SAC-Lag / HyAR, etc.) across several task domains for reproducible experiments and comparisons.

- **RMS**: manufacturing / robot task scheduling (with energy model and constraints)
- **EV-Charging**: EV charging station scheduling with constraints (Gymnasium environment)
- **CityFlow**: traffic signal control (CityFlow simulator + algorithm comparison)
- **MP-DQN**: parameterized action space baselines (Goal / Platform / Soccer)
- **Results**: Nature-style visualization scripts, comparison plots, exported models/curves/stats

Contact: `wanguangxi@sia.cn`

## Repository Layout

- `RMS/`: manufacturing scheduling (core code)
  - `RMS/env/`: environment + energy model (depends on Robotics Toolbox)
  - `RMS/algs/`: training/testing scripts (`lirl.py`, `hyar_policy.py`, `hppo_policy.py`, `cpo_policy.py`, `pdqn_policy.py`, `sac_lag_policy.py`)
  - `RMS/exp/`: experiment scripts (constraint change, runtime scaling)
- `EV-Charging/`: EV charging environment, algorithms, and comparisons
  - `EV-Charging/env/ev.py`: `EVChargingEnv` (Gymnasium)
  - `EV-Charging/alg/`: algorithm implementations (LIRL / PDQN / HPPO / LPPO / CPO)
  - `EV-Charging/exp/`: comparison/ablation scripts (e.g. `compare_algorithm.py`)
- `CityFlow/`: CityFlow simulator and related algorithms (includes C++ extension)
- `MP-DQN/`: parameterized action space baseline (upstream repo kept)
- `Results/`: generated plots + JSON/CSV stats + visualization scripts (each subfolder usually contains a `README.txt`)

## Environment Setup

### Python

Recommended: **Python 3.10** (Windows / Linux). Example with Conda:

```powershell
conda create -n lirl-cps python=3.10 -y
conda activate lirl-cps
```

### Core Dependencies

```powershell
pip install numpy scipy matplotlib pandas seaborn
```

Install PyTorch (CPU or CUDA build) using the command from `pytorch.org`.

### RMS Extra Dependencies (Required)

`RMS/env/energy_model.py` depends on Robotics Toolbox:

```powershell
pip install roboticstoolbox-python spatialmath-python
```

> If you only run EV-Charging, you do not need this set.

### EV-Charging Dependencies

```powershell
pip install -r EV-Charging/requirements.txt
```

### CityFlow (Optional, compiled)

`CityFlow/` is a simulator subproject with a C++ extension. On Windows you may need a proper build toolchain (CMake + compiler). We recommend Linux/WSL2 or using Docker (see `CityFlow/Dockerfile`):

- Source install: see `CityFlow/README.rst`

### MP-DQN (Optional)

`MP-DQN/` depends on external Gym environments (`gym-platform/gym-goal/gym-soccer`). See `MP-DQN/README.md`.

## Quick Start

### RMS (Manufacturing / Robot Scheduling)

Go to the algorithms directory:

```powershell
Set-Location RMS\algs
```

- **LIRL (DDPG-LIRL) training** (multi-run enabled by default; output dir includes timestamp):

```powershell
python lirl.py --episodes 1000
```

- **Single run training**:

```powershell
python lirl.py --single-run --episodes 1000 --jobs 50 --robots 5
```

- **Evaluate an existing model**:

```powershell
python lirl.py --test-only --model-path ".\ddpg_lirl_pi_multi_run_YYYYMMDD_HHMMSS\run_1_seed_3047\ddpg_lirl_pi_mu_0_YYYYMMDD_HHMMSS.pth"
```

> `HyAR` imports `ddpg_lirl_pi`: this repo provides a compatibility shim `RMS/algs/ddpg_lirl_pi.py`. If you move/delete it, `hyar_policy.py` may raise `ModuleNotFoundError`.

### RMS Experiments

- **Runtime scaling (different problem sizes)**:

```powershell
python RMS\exp\lirl_runtime_scaling.py
```

Outputs are saved under `RMS/exp/lirl_runtime_scaling_<timestamp>/`.

- **Constraint change (3-phase experiment)**:

```powershell
python RMS\exp\lirl_change_constraints.py
```

Note: in this script, `CONFIG['pretrained_model_path']` may be an absolute path from the author's machine. Update it to your local model directory before running.

### EV-Charging (EV Charging Station)

- **Train a single algorithm (LIRL)**:

```powershell
python EV-Charging\alg\lirl.py --episodes 200 --stations 5 --power 150 --arrival-rate 0.75
```

- **5-algorithm comparison (train + test + save results)**:

```powershell
python EV-Charging\exp\compare_algorithm.py --episodes 1000 --test-episodes 10
```

The script creates `algorithm_comparison_<timestamp>/`, containing `comparison_summary.csv` / `comparison_results.json` / training curves / exported models, etc.

### Results (Figure reproduction)

`run.py` under `Results/` typically renders a specific experimental output into a Nature-style comparison figure:

- EV-Charging visualization:

```powershell
python Results\EV-Charging\run.py
```

- CityFlow (traffic control) visualization:

```powershell
python Results\CityFlow\run.py
```

For more scenarios, see each folder’s `README.txt` (e.g. `Results/Constraint-edit/README.txt`, `Results/Goal_Platform_Soccer/README.txt`).

## Notes / FAQ

- **Windows path length**: use a short path (e.g. `C:\proj\LIRL-CPS`) to avoid file I/O issues when saving/loading models.
- **Training is slow**: reduce `--episodes`, reduce `--seeds`, or use `--single-run` to validate the pipeline first.
- **Robotics Toolbox errors**: ensure `roboticstoolbox-python` and `spatialmath-python` are installed. If prompted for optional packages (e.g. `rtb-data`), install them as instructed.
- **GPU/CPU**: scripts usually auto-detect CUDA. To force CPU, set environment variables accordingly (or disable CUDA in code).

## Licenses

- `CityFlow/`: see `CityFlow/LICENSE.txt`
- `MP-DQN/`: see `MP-DQN/LICENSE.md`
