# TE_World2

A TE-ecology agent-based model (ABM) that simulates the accumulation of transposable elements (TEs).

This code was used to generate the results reported in Kremer et al. [[1](https://www.researchgate.net/publication/341491913_Transposable_element_persistence_via_potential_genome-level_ecosystem_engineering)]. This paper describes the model in more detail, as well as the rationale for using each parameter.

## Set-up

The code runs with Python 3.10.
It is worth considering using [pyenv](https://github.com/pyenv/pyenv) to seamlessly switch to a valid Python version on your local computer.

### Running on Compute Canada

To run on Compute Canada, you will need to load the following Python3 module.

```
module load python/3.10.13
```

Production simulations are intended to run as independent SLURM array tasks on
Nibi. Each task runs one experiment with one CPU, rather than starting several
Python processes inside a single large allocation. This lets SLURM distribute
experiments across nodes and isolates failures and memory growth.

Two simulation backends are available. `TESim.py` is the reference
implementation. `TESimCompact.py` preserves its model and random-number call
order while storing coordinates in NumPy arrays and using a compact binary
checkpoint. Production array jobs use the compact backend by default. NumPy
must be installed in the selected Python module or virtual environment.

For a local compact run, use the same arguments as the reference simulator:

```
python3 /path/to/TEWorldCodeV2/TESimCompact.py 4 experiment-name --seed 12345
```

Seeded equivalence tests compare every scientific trace field between the two
backends; only the wall-clock trace column is allowed to differ. On a local
three-generation sample using an actual 5,000-gene, 300-host experiment file,
the compact backend ran in 0.84 seconds versus 8.63 seconds for the reference
backend. These timings are validation measurements, not Nibi performance
estimates.

For a dedicated environment, create it once on the Nibi login node:

```
module load python/3.10.13
python3 -m venv .venv-nibi
.venv-nibi/bin/python -m pip install -r requirements-compact.txt
```

From `TEWorldCodeV2/TE-Simulations`, submit a new replicate number with:

```
./submit-nibi-array.sh 4
```

The submission script discovers the number of experiment directories and
creates the matching array automatically. It defaults to 64 concurrent tasks,
8 GB per task, a two-day time limit, account `def-skremer`, and the
`python/3.10.13` module. These can be adjusted without editing the scripts:

```
MAX_CONCURRENT=32 MEMORY_PER_TASK=16G WALL_TIME=3-00:00 \
PYTHON_MODULE=python/3.10.13 SLURM_ACCOUNT=def-skremer \
./submit-nibi-array.sh 4
```

If NumPy is installed in a virtual environment, set `PYTHON_EXECUTABLE` to its
Python executable when submitting; SLURM exports that setting to each task:

```
PYTHON_EXECUTABLE="$PWD/../../.venv-nibi/bin/python" \
./submit-nibi-array.sh 4
```

Set `SIMULATION_BACKEND=reference` to run the original backend through the
same array machinery. This is useful for small validation runs, but is not
recommended for the full production grid.

Use a new run number for corrected production simulations. If an array task is
resubmitted, it skips experiments whose provenance says they completed and
resumes incomplete experiments from their latest checkpoint. SLURM job and
array identifiers, cluster, node, CPU, memory, account, partition, and submit
context are recorded in each provenance file.

See the Alliance documentation for [Nibi](https://docs.alliancecan.ca/wiki/Nibi),
[running jobs](https://docs.alliancecan.ca/wiki/Running_jobs), and
[Python environments](https://docs.alliancecan.ca/wiki/Python).

## Running Individual Experiments

1. You can run a pre-created experiment by changing into the `TE-Experiments` or `Paper-Experiments` sub-folders.

```
cd Paper-Experiments/XHExp001
```

2. Then, run the `TESim.py` script within that directory.
This ensures that the simulator uses the configurations associated with *that* experiment's `parameters.py` file.

```
python3 ../../TEWorldCodeV2/TESim.py
```

Each time this command is run, it will re-run the same experiment with a different randomly generated seed.
Unless, the `seed` field has been pre-configured for that experiment's `parameters.py` folder.

A trace file will be created for the associated experiment.
These CSV files store relevant data about TE accumulation.

### Reproducibility and provenance

Every simulation writes a `provenance-<run>-<iteration>.json` file next to its
trace. The record includes the concrete initial random seed, the exact
`parameters.py` source and checksum, simulator and utility checksums, Git commit,
Python/runtime information, checkpoint origin, and completion status.

When `parameters.seed` is `None`, the simulator generates a concrete integer
seed and records it before initializing the population. To replay a run, use the
recorded seed with the same parameter source and code revision:

```
python3 ../../TEWorldCodeV2/TESim.py 1 experiment-name --seed 123456789
```

The biological and statistical trace columns will be identical. The `time`
column measures wall-clock performance and is therefore not deterministic.

## Creating a New Experiment

1. Copy the default [`parameters.py`](TEWorldCodeV2/parameters.py) file.
2. Reference the [original TE model paper](https://link.springer.com/article/10.1186/s12864-020-6763-1) to understand how to interpret each parameter.
3. Change the parameter values of your copied `parameters.py` file accordingly.
4. To test the experiment, run [`TESim.py`](TEWorldCodeV2/TESim.py) in the location of your `parameters.py` file.

## Bulk Experiment Simulation

For the purposes of running hundreds of experiments at a time, several scripts were implemented in the [TE-Simulations](TEWorldCodeV2/TE-Simulations) folder.
These scripts were used to create, run and analyze the experiments in the [TE-Experiments](TE-Experiments) directory.
The code for generating the configuration files in the [Paper-Experiments](Paper-Experiments) directory is not included in this repository.

Please read the [associated documention](TEWorldCodeV2/TE-Simulations/README.md) for more details.

## Alternative Model

Noah Zeidenberg created a revised version of Dr. Kremer's original model, which includes a variety of other features and computational enhancements.
These changes can be referenced on [his website.](https://noahzeidenberg.github.io/research_papers/te_agents2/TE_Agents_site_index.html)

Noah's revised model is located in the [TE-Agents2](TE-Agents2) directory, and the document can be referenced [here](TE-Agents2/README.md).

## References

[[1](https://www.researchgate.net/publication/341491913_Transposable_element_persistence_via_potential_genome-level_ecosystem_engineering)] Kremer SC, Linquist S, Saylor B, Elliot TA, Gregory TR, Cottenie K. 2020. Transposable element persistence via 
potential genome-level ecosystem engineering. Under review.
