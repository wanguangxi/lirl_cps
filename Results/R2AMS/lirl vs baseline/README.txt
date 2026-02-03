================================================================================
Algorithm Comparison Result Analyzer
================================================================================

DESCRIPTION
-----------
This tool analyzes multi-run training results from algorithm comparison 
experiments. It automatically discovers algorithm result folders and generates 
comprehensive visualizations including training curves, reward distributions, 
and convergence speed analysis in Nature journal publication style.

FEATURES
--------
- Automatic discovery of algorithm result folders
- Training curve visualization with mean and standard deviation
- Final reward distribution analysis using KDE (Kernel Density Estimation)
- Convergence speed analysis
- Combined Nature-style figure (1 row × 3 columns)
- Individual plots for training curves and distributions
- Statistical analysis and comparison reports

SUPPORTED ALGORITHMS
--------------------
- LIRL (ddpg_lirl_pi)
- CPO (cpo)
- HPPO (hppo)
- HyAR (hyar_vae)
- SAC-Lag (sac_lag)
- PDQN (pdqn)

REQUIREMENTS
------------
Required:
- Python 3.6+
- numpy
- matplotlib

Optional (for enhanced features):
- scipy (for KDE visualization and statistical tests)

INSTALLATION
------------
Install required packages:
    pip install numpy matplotlib scipy

USAGE
-----
1. Place this script in the directory containing algorithm result folders.
   Each algorithm folder should contain:
   - config_*.json (configuration file)
   - *_all_scores_*.npy (score data file)

2. Run the script with one of the following options:

   Default (generates combined Nature-style figure):
       python compare_analyzer.py

   Generate only training and reward plots:
       python compare_analyzer.py --plot

   Generate only convergence speed analysis:
       python compare_analyzer.py --convergence

   Generate combined Nature-style figure (1×3):
       python compare_analyzer.py --combined

   Generate analysis report only:
       python compare_analyzer.py --report

   Specify custom directory:
       python compare_analyzer.py --dir /path/to/results

   Advanced options:
       python compare_analyzer.py --percentile 0.95 --smooth-window 15
       python compare_analyzer.py --combined --percentile 0.9 --smooth-window 20

COMMAND LINE ARGUMENTS
----------------------
--dir PATH            : Specify the directory containing algorithm results
                       (default: current directory)

--plot                : Generate only training curves and reward distribution plots

--convergence         : Generate only convergence speed analysis plot

--combined            : Generate combined Nature-style figure (1 row × 3 columns)
                       This is the default behavior when no specific option is given

--report              : Generate text analysis report only

--percentile FLOAT    : Percentile of median performance gain used as convergence
                       threshold (default: 0.95, range: 0-1)

--smooth-window INT   : Window size for moving-average smoothing before detecting
                       convergence (default: 5, recommended: 5-20)

OUTPUT FILES
------------
When using default or --combined option:
- algorithm_comparison_combined.png  : Combined figure (PNG, 600 DPI)
- algorithm_comparison_combined.pdf  : Combined figure (PDF, 600 DPI)

When using --plot option:
- algorithm_training_comparison.png  : Training curves comparison
- final_Reward_distribution.png     : Final reward distribution

When using --convergence option:
- convergence_speed_analysis.png    : Convergence speed comparison

When using --report option:
- analysis_report.txt               : Text analysis report

FIGURE LAYOUT (Combined Mode)
------------------------------
The combined figure consists of three subplots in one row:

1. Training Curves (left)
   - Shows mean reward ± standard deviation over training episodes
   - Includes shaded confidence regions

2. Reward Distribution (center)
   - KDE (Kernel Density Estimation) curves for final reward distribution
   - Falls back to histograms if scipy is not available

3. Convergence Speed (right)
   - Horizontal bar chart showing episodes to reach convergence threshold
   - Error bars represent interquartile range (IQR)

All figures are generated in Nature journal publication style with:
- Arial font family
- Appropriate sizing (7.2 inches width for combined figure)
- High resolution (600 DPI)
- Clean, professional appearance

ALGORITHM NAME MAPPING
----------------------
Folder names are automatically mapped to display names:
- ddpg_lirl_pi_multi_run_* -> LIRL
- cpo_multi_run_*          -> CPO
- hppo_multi_run_*         -> HPPO
- hyar_vae_multi_run_*     -> HyAR
- sac_lag_multi_run_*      -> SAC-Lag
- pdqn_multi_run_*         -> PDQN

EXAMPLES
--------
Example 1: Analyze results in current directory
    python compare_analyzer.py

Example 2: Analyze results in specific directory
    python compare_analyzer.py --dir /path/to/results

Example 3: Generate only convergence speed analysis with custom parameters
    python compare_analyzer.py --convergence --percentile 0.9 --smooth-window 20

Example 4: Generate combined figure with custom convergence parameters
    python compare_analyzer.py --combined --percentile 0.95 --smooth-window 15

Example 5: Generate all individual plots
    python compare_analyzer.py --plot
    python compare_analyzer.py --convergence
    python compare_analyzer.py --report

NOTES
-----
- The script automatically trims score arrays to the minimum length across all
  runs for fair comparison
- Convergence threshold is calculated as: initial + percentile × (final - initial)
- Figures are saved in the same directory as the script (or specified --dir)
- All figures use high-resolution settings suitable for publication
- The script handles missing optional dependencies gracefully with fallbacks

TROUBLESHOOTING
---------------
If no algorithms are found:
- Ensure each algorithm folder contains config_*.json and *_all_scores_*.npy
- Check that the directory path is correct
- Verify numpy is installed (required for loading .npy files)

If plotting fails:
- Ensure matplotlib is installed
- Check that score data files are not corrupted
- Verify sufficient disk space for output files

If KDE visualization fails:
- Install scipy for KDE support: pip install scipy
- The script will automatically fall back to histograms if scipy is unavailable

For additional help, check the script source code comments.

================================================================================
Version: 1.0
Last Updated: 2025
================================================================================
