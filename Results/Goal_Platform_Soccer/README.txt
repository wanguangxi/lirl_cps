Three Scenarios Comparison Script
==================================

DESCRIPTION
-----------
This script generates a 1x3 comparison figure for three different scenarios:
- Robot Soccer Goal: compares last_100_success_rate
- Platform: compares last_100_mean (reward)
- Half Field Offense: compares goal_rate

The figure follows Nature journal style guidelines with Arial font at 8pt.

REQUIREMENTS
------------
- Python 3.x
- matplotlib
- numpy

DATA FILES
----------
The script expects the following JSON files in relative paths:
- goal_comparison/compare_20260110_082334/results.json
- platform_comparison/compare_20260114_103334/results.json
- soccer_comparison/compare_20251227_111639/results.json

USAGE
-----
Run the script from the project root directory:
    python run.py

OUTPUT
------
- three_scenarios_comparison.png: 1x3 comparison figure (Nature double column width: 183mm = 7.2 inches)
- Figure saved at 300 DPI resolution

ALGORITHMS COMPARED
-------------------
- LIRL
- PADDPG
- PDQN
- QPAMDP

FIGURE SPECIFICATIONS
---------------------
- Width: 7.2 inches (183mm, Nature double column width)
- Height: 2.5 inches
- Font: Arial, 8pt
- DPI: 300
- Format: PNG

