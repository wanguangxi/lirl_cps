================================================================================
Traffic Control Algorithm Comparison Visualization
================================================================================

DESCRIPTION
-----------
This program generates Nature journal-style comparison plots for traffic 
control algorithms. It combines Pareto front analysis and performance heatmap 
in a single figure with two subplots arranged horizontally.

The visualization includes:
1. Left subplot: Pareto front scatter plot showing the trade-off between 
   throughput and travel time, with violation rate represented by color intensity
2. Right subplot: Performance heatmap comparing all algorithms across three 
   metrics (throughput, travel time, violation rate)

FEATURES
--------
- Nature journal style formatting (Arial font, 8pt, double-column width)
- Pareto front analysis (2D and 3D)
- Performance heatmap with normalized scores
- Color-coded violation rates
- Publication-ready output (PNG and PDF formats)

REQUIREMENTS
------------
Python 3.6 or higher

Required Python packages:
- matplotlib >= 3.0
- numpy >= 1.18
- pandas >= 1.0
- seaborn >= 0.10

INSTALLATION
------------
Install required packages using pip:

    pip install matplotlib numpy pandas seaborn

Or using conda:

    conda install matplotlib numpy pandas seaborn

FILE STRUCTURE
--------------
CityFlow/
├── run.py                          # Main visualization script
├── README.txt                      # This file
└── result/
    ├── run_20260115_100648/
    │   └── summary.json            # Fixed green time baseline data
    ├── run_20260115_111112/
    │   └── summary.json            # Algorithm evaluation data
    ├── combined_figure.png         # Output: Combined visualization (PNG)
    └── combined_figure.pdf        # Output: Combined visualization (PDF)

USAGE
-----
1. Ensure the required JSON data files are in the correct directories:
   - result/run_20260115_100648/summary.json (Fixed baseline)
   - result/run_20260115_111112/summary.json (Algorithm evaluations)

2. Run the script:

    python run.py

3. The output figures will be saved to:
   - result/combined_figure.png
   - result/combined_figure.pdf

CONFIGURATION
-------------
You can modify the following settings in run.py:

1. File paths (lines 20-23):
   - BASE_DIR: Base directory path
   - FIXED_DATA_PATH: Path to fixed baseline data
   - ALGORITHM_DATA_PATH: Path to algorithm evaluation data
   - OUTPUT_DIR: Output directory for figures

2. Algorithm name mapping (line 26):
   - ALGORITHM_MAPPING: Dictionary to map algorithm names for display

3. Figure dimensions (lines 100-102):
   - fig_width_cm: Figure width in centimeters (default: 18cm for Nature double-column)
   - fig_height_inch: Figure height (calculated automatically)

4. Subplot spacing (line 105):
   - width_ratios: Width ratio between left and right subplots
   - wspace: Horizontal spacing between subplots

DATA FORMAT
-----------
The program expects JSON files with the following structure:

For fixed baseline (run_20260115_100648/summary.json):
{
    "comparison": {
        "Fixed green duration": {
            "mean_throughput": <float>,
            "mean_travel_time": <float>,
            "mean_violation_rate": <float> (optional)
        }
    }
}

For algorithm evaluations (run_20260115_111112/summary.json):
{
    "evaluation_summary": {
        "<algorithm_name>": {
            "mean_throughput": <float>,
            "mean_travel_time": <float>,
            "mean_violation_rate": <float> (optional),
            "std_throughput": <float> (optional),
            "std_travel_time": <float> (optional)
        },
        ...
    }
}

ALGORITHMS
----------
The program compares the following algorithms:
- Fixed: Fixed green time baseline
- LIRL: Learning-based traffic control
- PDQN: Policy Decomposition Q-Network
- HPPO: Hierarchical Proximal Policy Optimization
- LPPO: Lagrangian Proximal Policy Optimization (Lagrangian-PPO)
- CPO: Constrained Policy Optimization

METRICS
-------
1. Throughput (veh/h): Mean vehicle throughput per hour (divided by 15)
   - Higher is better
   
2. Travel Time (s): Mean vehicle travel time in seconds
   - Lower is better
   
3. Violation Rate (%): Mean constraint violation rate percentage
   - Lower is better

PARETO FRONT ANALYSIS
---------------------
The program calculates two types of Pareto fronts:

1. 2D Pareto Front: Optimal trade-off between throughput and travel time
   - Shown as a dashed line connecting Pareto optimal points
   
2. 3D Pareto Front: Optimal points considering throughput, travel time, 
   and violation rate simultaneously
   - Marked with star (*) markers
   - Annotated with algorithm names

VISUALIZATION STYLE
-------------------
- Font: Arial, 8pt (Nature journal standard)
- Figure width: 18cm (Nature double-column width)
- Color scheme: Nature Publishing Group (NPG) color palette
- Resolution: 300 DPI for publication quality
- Output formats: PNG and PDF

COLOR CODING
------------
Pareto Front Plot:
- Point colors represent violation rate (sequential colormap)
- Light colors = low violation rate
- Dark colors = high violation rate
- Star markers = 3D Pareto optimal points

Heatmap:
- Red = Poor performance (low score)
- White = Medium performance
- Green = Good performance (high score)
- Scores are normalized to 0-1 range for each metric

TROUBLESHOOTING
---------------
1. FileNotFoundError: Check that JSON files exist in the specified paths
2. KeyError: Verify JSON structure matches expected format
3. ImportError: Install missing packages using pip or conda
4. Empty figure: Check that data files contain valid evaluation results

AUTHOR
------
Generated for Major Revision
Date: 2026-01-16

LICENSE
-------
This code is provided for research purposes.

================================================================================


