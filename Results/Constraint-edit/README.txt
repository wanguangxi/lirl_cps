================================================================================
Multi-Scenario Performance Comparison Under Constraint Changes
================================================================================

DESCRIPTION
-----------
This script generates a Nature journal-style figure comparing performance 
metrics across three scenarios under constraint changes:

1. R2AMS - Robot Warehouse Scheduling
2. Traffic Control (CityFlow) - Traffic Signal Control
3. EV Charging - Electric Vehicle Charging

Each scenario is evaluated in three phases:
- Phase 1: Original Environment
- Phase 2: Constrained (No Retrain)
- Phase 3: Transfer Learning

REQUIREMENTS
------------
- Python 3.6+
- matplotlib
- numpy
- pathlib (standard library)

INSTALLATION
------------
Install required packages:
    pip install matplotlib numpy

DATA FILES
----------
The script requires the following data files in the same directory:

1. R2AMS_constraint_change_results_20260115_171909.json
   - Contains R2AMS makespan and energy consumption data

2. EV_Charging_20260118_104918/constraint_results_20260118_105536.csv
   - Contains EV charging success rate and energy output data

3. CityFlow_run_20260116_211144/results.json
   - Contains traffic control throughput and travel time data

USAGE
-----
Run the script from the command line:
    python run.py

OUTPUT
------
The script generates two files:

1. three_scenarios_comparison.png
   - High-resolution PNG image (300 DPI)
   - Figure width: 7 inches (Nature double column format)
   - Font: Arial, 8pt

2. three_scenarios_comparison.pdf
   - PDF version for publication use
   - Vector format for high-quality printing

FIGURE STRUCTURE
----------------
The figure consists of 3 rows x 3 columns:

Row 1: R2AMS
  - Column 1: Makespan (s)
  - Column 2: Energy (×10³ J)
  - Column 3: Violation Rate (all 0%)

Row 2: Traffic Control
  - Column 1: Throughput (×10³ veh/h)
  - Column 2: Travel Time (s)
  - Column 3: Violation Rate (all 0%)

Row 3: EV Charging
  - Column 1: Success Rate (%)
  - Column 2: Energy Output (MWh)
  - Column 3: Violation Rate (all 0%)

Each bar chart shows three phases:
- Blue: Original
- Red: Constrained (No Retrain)
- Green: Transfer Learning

NOTES
-----
- All paths are relative to the script location
- The figure follows Nature journal style guidelines
- Constraint violation rates are 0% across all scenarios and phases
- Best performing phase is indicated with a gold border

AUTHOR
------
Generated for constraint change experiment analysis

DATE
----
2026-01-18

================================================================================

