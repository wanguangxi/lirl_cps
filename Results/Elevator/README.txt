================================================================================
                    Gantt Chart Visualization Tool
================================================================================

DESCRIPTION
-----------
This tool generates clean, publication-quality Gantt charts from production 
line scheduling data stored in JSON format. The charts visualize workpiece 
operations across production lines with utilization rates, operation times, 
and energy consumption information.

FEATURES
--------
- Clean, Nature journal-style visualization
- Utilization rates displayed for each production line
- Operation breakdown with time (s) and energy (kJ) annotations
- Professional color palette for workpiece identification
- High-resolution output (300 DPI) suitable for publications
- Arial font family with bold styling for clear readability

REQUIREMENTS
------------
- Python 3.7 or higher
- matplotlib
- numpy

To install dependencies:
    pip install matplotlib numpy

USAGE
-----
1. Prepare your schedule data file in JSON format. The file should contain:
   - operation_schedules: List of operation entries with workpiece IDs, 
     line assignments, start times, operation times, and energies
   - line_end_times: End times for each production line
   - line_utilization: Utilization rates for each line (0-1)
   - makespan: Total schedule makespan
   - total_energy: Total energy consumption
   - violations: Number of constraint violations

2. Modify the script to point to your JSON file:
   Edit plot_clean_gantt.py, line 281:
   
   json_file_path = os.path.join('result', 'workpieces_5_opt', 
                                  'training_best_schedule_5.json')
   
   Change the path to match your data file location.

3. Run the script:
   python plot_clean_gantt.py

4. The output chart will be saved in the same directory as the input JSON file
   with the name: clean_gantt_with_utilization.png

FILE STRUCTURE
--------------
plot_clean_gantt.py          - Main script for generating Gantt charts
result/                       - Directory containing schedule data and outputs
  workpieces_5/              - Example data directory
  workpieces_5_opt/          - Example data directory

OUTPUT FORMAT
-------------
The generated chart includes:
- Production lines on the y-axis with utilization percentages
- Time scale on the x-axis
- Workpieces displayed as colored bars with operation segments
- Operation numbers, times, and energy values as annotations
- Legend identifying all workpieces
- Title showing makespan and total energy

CUSTOMIZATION
-------------
You can customize the chart by modifying:
- Colors: Change the workpiece_colors list (lines 38-49)
- Figure size: Adjust figsize parameter (line 35)
- Font sizes: Modify fontsize parameters throughout the code
- Output resolution: Change dpi parameter (line 270)

NOTES
-----
- The script uses relative paths. Ensure you run it from the project root
  directory or adjust paths accordingly.
- Input JSON files must follow the expected structure format.
- Output images are saved as PNG files with white background.

================================================================================
                            Version Information
================================================================================
Last Updated: 2026
Python Version: 3.7+

================================================================================

