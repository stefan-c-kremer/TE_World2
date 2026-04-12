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

### Create Trace Graphs

```
python3 CreateTraceGraphs.py trace-1.csv trace-2.csv
```

This will generate graphs for the specified trace files, suffixed with '-1' and '-2'. You can specify an unlimited number of CSV files.

### Create Trials

```
python3 CreateTrials.py
```

The files will be subsequently created after running this command.

### Run Simulations

```
python3 ./RunSimulations.py
```

After running this, a validation prompt will verify that you wanted to run this script. Alternatively, you can provide `-s` to skip this message.
Additionally `-f` will run the script in 'fast mode'.

### Analyze Results

```
python3 ResultsAnalyzer.py
```

The files (graphs, images, excel summaries) will be created and written to the `TE-Simulations/output` folder.

### Delete Simulation Results

```
./delete-simulation-files.sh
```

This deletes all CSV and state files associated with a simulation. **Do not run it** if you do not want to delete all your results. The actions in this script are irreversable.

## Analysis Notes

### TE Results Overview Plots

In the results overview plots the colours correspond with the following scenarios:

- **Teal:** TE persistence (both autonomous and non-autonomous)
- **Orange:** Autonomous TE persistence
- **Red:** Non-autonomous TE persistence
- **Yellow:** TE extinction (both autonomous and non-autonomous)
- **Purple:** Host extinction

![Example results overview plot](example-results-overview-plot.png)