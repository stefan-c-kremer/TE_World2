# TE-Simulations

This folder contains utility functions to generation a large amount of configuration files, run simulations and summarize the results of simulations.

This is the code to:
- Generate TE configuration files to runs simulations.
- Run the simulations.
- Graph the results of the simulations.

## Requirements
- pip2
- PyYAML

## To Run

### Create Trials

```
python3 CreateTrials.py
```

The files will be subsequently created after running this command.

### Run Simulations

```
python3 ./RunSimulations.py
```

The simulations will be run (all 3072 of them).

### Analyze Results

```
python3 ResultsAnalyzer.py
```

The files (graphs, images, excel summaries) will be created and written to the `TE-Simulations/output` folder.

## Analysis Notes

### TE Results Overview Plots

In the results overview plots the colours correspond with the following scenarios:

- **Teal:** TE persistence (both autonomous and non-autonomous)
- **Orange:** Autonomous TE persistence
- **Red:** Non-autonomous TE persistence
- **Yellow:** TE extinction (both autonomous and non-autonomous)
- **Purple:** Host extinction

![Example results overview plot](example-results-overview-plot.png)