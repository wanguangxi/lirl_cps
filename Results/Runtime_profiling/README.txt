================================================================================
Runtime Profiling Comparison Script
================================================================================

DESCRIPTION
-----------
This script performs runtime profiling analysis comparing neural network 
inference, discrete mapping, continuous mapping, and total decision time
across three scenarios at different scales:

1. Traffic Control (CityFlow) - Traffic signal control system
2. EV-Charging - Electric vehicle charging management
3. R²AMS - Robot warehouse scheduling system

Each scenario is evaluated at three scales:
- Small scale
- Medium scale  
- Large scale

REQUIREMENTS
------------
- Python 3.6+
- pandas
- numpy
- matplotlib
- pathlib (standard library)

INSTALLATION
------------
Install required packages:
    pip install pandas numpy matplotlib

DATA FILES
----------
The script requires the following data directories and files:

1. Cityflow_runtime_scaling_20260113_211415/
   - summary.csv
     * Contains timing data for CityFlow at different scales
     * Columns: NNInference(ms), DiscreteMapping(ms), ContinuousMapping(ms), TotalDecision(ms)

2. EV-charging_runtime_scaling_exp_20251218_130354/
   - runtime_summary_20251218_131057.csv
     * Contains timing data for EV-Charging at different scales
     * Columns: Policy Network (ms), Hungarian (ms), QP (ms), Total (ms)

3. RMS_runtime_scaling_20251218_152208/
   - results_summary.json
     * Contains timing data for RMS at different scales
     * Structure: experiments[scale]['timing'] with keys:
       - network_forward_ms (NN inference)
       - hungarian_ms (discrete mapping)
       - qp_ms (continuous mapping)
       - total_decision_ms (total decision time)

USAGE
-----
Run the script from the command line:
    python run.py

OUTPUT
------
The script generates the following files:

1. runtime_comparison_all_scenarios.png
   - 2x2 subplot layout comparing all four metrics
   - Each subplot shows one metric across all three scenarios
   - High-resolution PNG (300 DPI)
   - Also generates PDF version

2. runtime_composition_stacked.png
   - 1x3 subplot layout showing stacked bar charts
   - Each subplot represents one scenario
   - Shows composition of total decision time (NN inference, discrete mapping, 
     continuous mapping)
   - Subplot order: R²AMS, Traffic Control, EV-Charging
   - Legend appears only in the last subplot
   - High-resolution PNG (300 DPI)
   - Also generates PDF version

3. runtime_comparison_table.csv
   - Comprehensive table with all metrics for all scenarios and scales
   - Columns: Scenario, Scale, NN Inference (ms), Discrete Mapping (ms), 
     Continuous Mapping (ms), Total Decision (ms)

FIGURE SPECIFICATIONS
---------------------
- Figure width: 7 inches (Nature double column format)
- Font: Arial, 8pt
- Style: Nature journal format
- Colors: Nature-style muted palette
  * Red (#E64B35) - NN Inference
  * Cyan (#4DBBD5) - Discrete Mapping
  * Teal (#00A087) - Continuous Mapping

METRICS EXPLAINED
-----------------
1. NN Inference (ms)
   - Time for neural network forward pass
   - Policy network inference time

2. Discrete Mapping (ms)
   - Time for discrete assignment/matching
   - Uses Hungarian algorithm for assignment problems

3. Continuous Mapping (ms)
   - Time for continuous optimization
   - Uses Quadratic Programming (QP) solver

4. Total Decision (ms)
   - Total decision-making time
   - Sum of all three components

SCENARIOS
---------
1. Traffic Control (CityFlow)
   - Traffic signal control at intersections
   - Scales: 3x5, 5x10, 10x10 intersections

2. EV-Charging
   - Electric vehicle charging station management
   - Scales: 5, 50, 100 charging stations

3. R²AMS
   - Robot warehouse scheduling system
   - Scales: Small, Medium, Large configurations

NOTES
-----
- All paths are relative to the script location
- The script automatically loads data from the specified subdirectories
- Output files are saved in the same directory as the script
- Stacked bar chart shows the relative contribution of each component
- Total decision time is labeled on top of each stacked bar

AUTHOR
------
Generated for runtime profiling analysis

DATE
----
2026-01-18

================================================================================

