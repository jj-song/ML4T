# Fall 2019 Project 6: Manual Strategy

## This readme contains directions to properly execute the 3 python files required to produce the
## statistics and the graphs required in this project.
## This readme assumes the user has properly logged into the buffet server and into the directory
## jsong350/ML4T_2019Fall/manual_strategy

### Part 1: Technical Indicators, run the following command:
PYTHONPATH=../:. python indicators.py

### Part 2: Theoretically Optimal Strategy, run the following command:
PYTHONPATH=../:. python TheoreticallyOptimalStrategy.py

### Part 3: Manual Rule-Based Trader, run the following command:
PYTHONPATH=../:. python ManualStrategy.py
*Note: In order to produce the proper "Manual Strategy vs Benchmark (Out-of-Sample)" graph, you will need to change value of "sd" and "ed" to dt.datetime(2010,1,1) and dt.datetime(2011,12,31) respectively in the main() function. By default, it will use the in-sample dates.

