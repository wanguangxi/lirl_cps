========================================
LIRL vs T-opt & E-opt Experimental Results Comparison
========================================

Project Description
-------------------
This project compares the performance of three scheduling algorithms:
- LIRL (cross-opt): Learning-based Inverse Reinforcement Learning algorithm
- E-opt (energy-opt): Energy optimization algorithm
- T-opt (time-opt): Time optimization algorithm

Project Structure
-----------------
.
├── run.py                  # Main script for generating boxplots
├── result/                 # Experimental results data directory
│   ├── cross-opt/         # LIRL algorithm results (JSON, PNG, NPY formats)
│   ├── energy-opt/        # E-opt algorithm results
│   └── time-opt/          # T-opt algorithm results
└── compare_reports/        # Comparison reports output directory
    ├── summary_metrics.csv              # Summary metrics data
    ├── boxplot_by_scale.png            # Boxplots grouped by scale (PNG format)
    ├── boxplot_by_scale.pdf            # Boxplots grouped by scale (PDF format)
    ├── scale_*_scores_swarm.png        # Swarm plots for each scale
    └── scale_*_weighted_combination.png # Weighted combination plots for each scale

Usage
-----
1. Ensure the following Python packages are installed:
   - pandas
   - matplotlib
   - numpy

2. Run the main script:
   python run.py

3. The script will:
   - Read data from compare_reports/summary_metrics.csv
   - Generate boxplots grouped by scale (a, b, c, d)
   - Use Nature journal-style figure format
   - Output high-resolution images (600 DPI) to the compare_reports/ directory

Data Description
----------------
- summary_metrics.csv contains the following fields:
  * mode: Algorithm type (cross-opt/energy-opt/time-opt)
  * scale: Experiment scale (scale_a/scale_b/scale_c/scale_d)
  * config: Configuration parameters
  * score_mean_last100: Mean reward over the last 100 rounds
  * score_std_last100: Standard deviation over the last 100 rounds
  * total_energy: Total energy consumption
  * makespan: Maximum completion time

- Each algorithm subdirectory in result/ contains:
  * JSON files: Experimental configurations and detailed results
  * PNG files: Visualization charts
  * NPY files: Data in NumPy array format

Output Description
------------------
Generated boxplot features:
- Nature journal-style design
- High-resolution output (600 DPI), suitable for academic publication
- Both PNG and PDF formats generated
- Figure size approximately 183mm x 56mm (double-column width)

Chart content:
- X-axis: Algorithm type (LIRL, E-opt, T-opt)
- Y-axis: Reward (reward value)
- Four subplots correspond to different scales (a, b, c, d)

Notes
-----
- Ensure compare_reports/summary_metrics.csv exists before running
- Output directory compare_reports/ will be created automatically if it doesn't exist
- Charts use Arial font; ensure the font is installed on your system
- All charts have white backgrounds, suitable for printing and publication

Version Information
-------------------
Created: 2024
Purpose: Major Revision experimental results comparison
