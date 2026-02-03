================================================================================
EV Charging Station Algorithm Comparison - Visualization Script
================================================================================

This directory contains a Python script for generating Nature journal-style
visualization figures from algorithm comparison results.

SCRIPT OVERVIEW
================================================================================

run.py
  - Generates a combined 1x4 subplot figure:
    - Pareto front plot (performance vs safety trade-off)
    - Station Utilization bar chart
    - Charging Success Rate bar chart
    - Energy Delivered bar chart
  - Output: combined_figure.png and combined_figure.pdf

REQUIREMENTS
================================================================================

Python packages required:
- pandas
- numpy
- matplotlib

Install dependencies:
    pip install pandas numpy matplotlib

FILE STRUCTURE
================================================================================

Expected directory structure:
    .
    ├── run.py
    ├── README.txt
    └── result/
        └── algorithm_comparison_20260115_105751/
            ├── comparison_summary.csv
            ├── comparison_results.json
            └── key_performance_metrics.csv

DATA FILES
================================================================================

1. comparison_summary.csv
   - Contains test results summary
   - Required metrics: test_avg_reward, test_avg_violations, 
     test_std_reward, test_std_violations

2. key_performance_metrics.csv
   - Contains performance metrics for each algorithm
   - Required metrics: Station Utilization, Charging Success Rate, Energy Delivered

3. comparison_results.json (optional)
   - Contains detailed training and test results

USAGE
================================================================================

Run the script:
    python run.py

Output:
    - combined_figure.png (PNG, 300 DPI)
    - combined_figure.pdf (PDF for publication)

FIGURE SPECIFICATIONS
================================================================================

Style: Nature Journal Publication Style
- Font: Arial, 8pt
- Width: Double-column (18 cm)
- Colors: Nature Publishing Group (NPG) color palette
- Markers: Square for data points, star for Pareto optimal points
- Grid: None
- Background: White

ALGORITHMS
================================================================================

Supported algorithms:
- LIRL (Learning Imitation Reinforcement Learning)
- PDQN (Parameterized Deep Q-Network)
- HPPO (Hybrid Proximal Policy Optimization)
- LPPO (Lagrangian Proximal Policy Optimization)
- CPO (Constrained Policy Optimization)

CONFIGURATION
================================================================================

File paths are configured using relative paths based on script location.
To use different data files, modify the paths in the Configuration section:

    RESULT_DIR = os.path.join(BASE_DIR, 'result', 'algorithm_comparison_20260115_105751')

================================================================================
