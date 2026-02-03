================================================================================
LIRL vs MASK Learning Curves Comparison Tool
================================================================================

OVERVIEW
--------
This tool compares learning curves between LIRL (Learning from Imperfect 
Reward Learning) and MASK algorithms implemented with DDPG (Deep Deterministic 
Policy Gradient) across different experimental scales. It generates publication-
quality figures in Nature journal style format.

FEATURES
--------
- Compares LIRL and MASK algorithms across 4 different scales (10_3, 20_3, 50_5, 100_5)
- Loads experimental results from multiple runs (typically 10 runs per configuration)
- Generates statistical comparisons including t-tests
- Creates publication-quality 1x4 figure layout (suitable for Nature-style papers)
- Outputs both PNG (600 DPI) and PDF formats
- Uses Nature journal style formatting (Arial font, clean aesthetics)

REQUIREMENTS
------------
Python 3.x with the following packages:
- numpy
- matplotlib
- scipy
- pathlib (built-in)

Install dependencies:
    pip install numpy matplotlib scipy

DATA STRUCTURE
--------------
The script expects data directories organized as follows:

    <base_directory>/
    ├── ddpg_lirl_pi_multi_run_10_3/
    │   ├── *_all_scores_*.npy
    │   ├── *_all_actions_*.npy
    │   └── config_*.json
    ├── ddpg_lirl_pi_multi_run_20_3/
    ├── ddpg_lirl_pi_multi_run_50_5/
    ├── ddpg_lirl_pi_multi_run_100_5/
    ├── ddpg_mask_multi_run_10_3/
    ├── ddpg_mask_multi_run_20_3/
    ├── ddpg_mask_multi_run_50_5/
    └── ddpg_mask_multi_run_100_5/

Each directory should contain:
- *_all_scores_*.npy: NumPy array containing scores from all runs
  Format: Array of arrays, where each sub-array contains episode scores from one run
- *_all_actions_*.npy: (Optional) Action data
- config_*.json: (Optional) Configuration file with experiment parameters

SCALE CONFIGURATIONS
--------------------
The script compares results across 4 scales:
- Scale A (10_3): Typically 10 jobs, 3 robots
- Scale B (20_3): Typically 20 jobs, 3 robots
- Scale C (50_5): Typically 50 jobs, 5 robots
- Scale D (100_5): Typically 100 jobs, 5 robots

USAGE
-----
1. Basic usage (analyzes data in the same directory as the script):
   
   python lirl_mask_comparison.py

2. Custom base path:
   
   from lirl_mask_comparison import LIRLMaskComparison
   analyzer = LIRLMaskComparison(base_path="/path/to/your/data")
   analyzer.run()

3. Generate plot only (after loading data):
   
   analyzer = LIRLMaskComparison()
   analyzer.load_data()
   analyzer.plot_comparison_1x4()

4. Print statistics only:
   
   analyzer = LIRLMaskComparison()
   analyzer.load_data()
   analyzer.print_statistics()

OUTPUT FILES
------------
The script generates the following output files:

1. lirl_mask_comparison_1x4.png
   - High-resolution PNG (600 DPI)
   - 1x4 layout showing learning curves for all 4 scales
   - Suitable for presentations and reports

2. lirl_mask_comparison_1x4.pdf
   - Vector PDF format (600 DPI)
   - Same 1x4 layout
   - Best for publications and print media

FIGURE DETAILS
--------------
The generated figure:
- Layout: 1 row x 4 columns (one subplot per scale)
- Figure size: 7.2 x 1.5 inches (Nature double-column width)
- Colors: 
  * LIRL: Red (#E64B35)
  * MASK: Cyan (#4DBBD5)
- Style: Nature journal formatting with Arial font
- Features:
  * Mean learning curves for each algorithm
  * Shaded regions showing ±1 standard deviation
  * Legend in first subplot only
  * Clean, minimal design with no grid

STATISTICAL ANALYSIS
--------------------
The script performs the following statistical analyses:

1. Final Performance Metrics:
   - Mean and standard deviation of final 100 episodes
   - Calculated separately for each algorithm and scale

2. Statistical Tests:
   - Independent two-sample t-test between LIRL and MASK
   - Tests significance of performance difference (p < 0.05)
   - Reports t-statistic and p-value
   - Identifies the better-performing algorithm when significant

Console output includes:
- Data loading status
- Summary statistics for each scale
- Statistical test results
- File save confirmations

TROUBLESHOOTING
---------------
Problem: "No data loaded! Please check folder paths."
Solution: Ensure data directories follow the naming convention:
         - ddpg_lirl_pi_multi_run_<scale>/
         - ddpg_mask_multi_run_<scale>/
         Where <scale> is one of: 10_3, 20_3, 50_5, 100_5

Problem: "No scores file found"
Solution: Ensure each directory contains a file matching pattern:
         *_all_scores_*.npy

Problem: Figure displays but plt.show() fails
Solution: For headless environments, comment out plt.show() line (line 180)
         The files will still be saved correctly.

Problem: Font issues (Arial not available)
Solution: The script will fall back to default fonts. Arial is recommended
         for Nature-style publications but not strictly required.

CODE STRUCTURE
--------------
The LIRLMaskComparison class contains:

Methods:
- __init__(base_path): Initialize with optional custom data path
- load_data(): Load all score files from data directories
- _normalize_scores(scores): Normalize score arrays to consistent format
- plot_comparison_1x4(): Generate 1x4 comparison figure
- print_statistics(): Print statistical analysis to console
- run(): Main execution method (loads data, prints stats, generates plot)

Key Attributes:
- base_path: Path to data directory
- scales: List of scale identifiers ['10_3', '20_3', '50_5', '100_5']
- algorithms: Dictionary mapping algorithm names to folder prefixes
- colors: Color scheme for plotting
- data: Loaded and processed experimental data

NOTES
-----
- The script automatically detects score files using glob pattern matching
- Scores are normalized to handle variable-length run data
- Statistical analysis uses the final 100 episodes for performance comparison
- All file paths use pathlib for cross-platform compatibility
- The figure uses matplotlib's tight_layout for optimal spacing

AUTHOR & VERSION
----------------
For ablation study comparison between LIRL and MASK methods using DDPG.
Last updated: 2025

================================================================================
