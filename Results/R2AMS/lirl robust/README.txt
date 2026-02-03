=============================================================================
                    Robustness Analysis Script
=============================================================================

OVERVIEW
--------
This script analyzes robustness experiment data for LIRL (Learning from 
Inverse Reinforcement Learning) models. It compares training performance 
against generalization performance under different failure rates and noise 
levels, generating publication-quality comparison plots.

REQUIREMENTS
------------
- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- pathlib (standard library)
- glob (standard library)
- os (standard library)

DIRECTORY STRUCTURE
-------------------
The script expects the following directory structure:

    .
    ├── run.py
    ├── Generalization/
    │   └── generalization_results.csv
    ├── Machine_breakdown/
    │   └── ddpg_lirl_pi_multi_run_*_failure_*/
    │       └── ddpg_lirl_pi_all_episode_stats_*.csv
    └── Noise_level/
        └── ddpg_lirl_pi_multi_run_*_noise_*/
            └── ddpg_lirl_pi_all_episode_stats_*.csv

DATA FORMAT
-----------
1. Generalization Data (generalization_results.csv):
   - Columns should include: 'seed', 'failure_rate', 'nose_level' (note: typo 
     in original data), 'makespan' (or 'avg_makespan'), 'total_energy' 
     (or 'avg_energy')

2. Training Data:
   - CSV files named: ddpg_lirl_pi_all_episode_stats_*.csv
   - Should contain columns: 'seed', 'makespan', 'total_energy'
   - The script uses the last 100 data points from each seed for analysis

FUNCTIONS
---------
- check_data_structure(): 
  Examines and prints the structure of loaded data files

- load_generalization_data(): 
  Loads generalization test results from CSV file

- load_training_data(folder_path, pattern, config_type, config_value): 
  Loads training data, extracting the last 100 data points per seed

- prepare_data_for_comparison(): 
  Prepares comparison datasets for failure rates and noise levels

- calculate_generalization_metrics(failure_data, noise_data): 
  Computes statistical metrics including means, standard deviations, and 
  weighted scores

- add_weighted_metric(df, w=0.5): 
  Adds a normalized weighted combination metric (weighted_score) combining 
  makespan and energy consumption

- plot_overall_comparison(failure_df, noise_df, metrics, w=0.5): 
  Generates a 1x4 comparison plot showing makespan and energy for both 
  noise levels and failure rates, in Nature journal style

- coverage_ratio(train_s, test_s): 
  Calculates the proportion of generalization samples within the training 
  data range

USAGE
-----
Simply run the script:

    python run.py

The script will:
1. Check and display data structure
2. Load generalization and training data
3. Prepare comparison datasets
4. Calculate generalization metrics
5. Generate overall comparison plots (PNG and PDF formats)

OUTPUT
------
The script generates:
- overall_comparison.png: High-resolution (600 DPI) comparison plot
- overall_comparison.pdf: PDF version of the comparison plot

The plot contains four panels:
1. Noise: Makespan - Makespan comparison across noise levels
2. Noise: Energy - Energy consumption comparison across noise levels
3. Failure: Makespan - Makespan comparison across failure rates
4. Failure: Energy - Energy consumption comparison across failure rates

Each panel shows:
- Split violin plots comparing Training vs Generalization performance
- Coverage ratios (percentage of generalization samples within training range)
- Different configurations (failure rates: 10%, 30%, 50%; noise levels: 0.1, 0.3, 0.5)

CONFIGURATION
-------------
Key parameters that can be modified in the code:

- failure_rates: List of failure rates to analyze (default: [0.1, 0.3, 0.5])
- noise_levels: List of noise levels to analyze (default: [0.1, 0.3, 0.5])
- weighted_score weight (w): Weight for weighted metric combination (default: 0.5)
- Plot dimensions: Adjustable in plot_overall_comparison() function

NOTES
-----
- The script assumes a specific column name 'nose_level' (likely a typo for 
  'noise_level') in the generalization data
- Training data uses the last 100 episodes from each seed for analysis
- The plot style follows Nature journal publication standards with Arial font
- Coverage ratio annotations show what percentage of generalization samples 
  fall within the training data range

CONTACT
-------
For questions or issues, please refer to the project documentation or 
contact the development team.

=============================================================================

