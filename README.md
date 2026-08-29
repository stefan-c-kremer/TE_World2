# TE_World2

TE_World2 is an agent-based model of transposable-element (TE) accumulation
and persistence in host genomes. It supports autonomous and non-autonomous TE
lineages, host selection, TE death and excision, insertion effects, biased
insertion locations, checkpointing, and complete generation-by-generation
trace output.

The model was developed for the experiments reported by Kremer et al. in
*Transposable element persistence via potential genome-level ecosystem
engineering*. See [References](#references) for the original paper and its
follow-up analysis.

## Scientific status

The simulator now preserves TE autonomy and length when a TE is copied and
correctly handles collisions during initial gene placement. Historical result
files may have been generated with earlier code or experiment configurations;
do not silently combine them with corrected runs. Use a new run number for all
new production simulations and retain their provenance records.

The full trace is intentionally preserved because it is difficult to predict
which statistics future analyses and figures will require.

## Simulation backends

Two backends implement the same biological model:

- `TEWorldCodeV2/TESim.py` is the reference implementation and correctness
  oracle.
- `TEWorldCodeV2/TESimCompact.py` is the production backend. It stores
  coordinates in capacity-managed NumPy arrays, shares gene objects during
  cloning, and writes compact binary checkpoints.

The compact backend retains the reference backend's random-number call order.
Automated seeded tests require every scientific trace field to match exactly;
only the wall-clock `time` column may differ. Tests cover both TE lineages, TE
death, retrotransposition, excision, both gene-collision modes, and checkpoint
resume.

Local validation with an actual 5,000-gene, 300-host experiment configuration
gave the following results:

| Validation workload | Reference | Compact | Difference |
| --- | ---: | ---: | ---: |
| Three generations | 8.63 s | 0.84 s | 10.3× faster |
| Initialization plus one generation | 5.24 s | 0.49 s | 10.7× faster |
| Peak memory for the one-generation run | 969 MiB | 175 MiB | 5.5× lower |

These are local validation measurements, not Nibi performance estimates.

## Installation

The reference backend requires Python 3.10 or later and uses the standard
library. The compact backend additionally requires the NumPy version recorded
in [`requirements-compact.txt`](requirements-compact.txt).

To create a local environment from the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-compact.txt
```

## Running one experiment

Run the simulator from the experiment directory so that it imports that
directory's `parameters.py` file. For example:

```bash
cd TE-Experiments/IS-HHHHHHHHH-EXP
../../.venv/bin/python ../../TEWorldCodeV2/TESimCompact.py \
  4 IS-HHHHHHHHH-EXP --seed 123456789
```

The positional arguments are the run number and experiment name. A concrete
seed is strongly recommended. If neither `--seed` nor `parameters.seed` is
provided, the simulator generates a concrete seed and records it before
initializing the population.

For a small reference-backend comparison, substitute `TESim.py` for
`TESimCompact.py`:

```bash
python3 ../../TEWorldCodeV2/TESim.py \
  4 IS-HHHHHHHHH-EXP --seed 123456789
```

Each execution creates:

- `trace-<run>-<iteration>.csv`, containing the complete trace;
- `state-<run>-<iteration>-<generation>.gz`, containing checkpoints; and
- `provenance-<run>-<iteration>.json`, describing the run and its inputs.

Existing output files are not overwritten. The iteration number increases
when the same run number is started again.

## Reproducibility and provenance

Every provenance record contains:

- the concrete initial random seed and how it was selected;
- the exact `parameters.py` source and SHA-256 checksum;
- the Git commit and checksums of the launcher, simulation engine, backend,
  and utilities;
- the backend, checkpoint format, Python version, and NumPy version when
  applicable;
- the working directory and command-line arguments;
- checkpoint origin and completion status; and
- hostname and SLURM allocation context when run under SLURM.

To replay a completed run, check out the recorded Git revision, restore the
recorded parameter source, and use the recorded replay command. Scientific
trace columns should be identical; wall-clock timing is not deterministic.

To resume a checkpoint explicitly:

```bash
../../.venv/bin/python ../../TEWorldCodeV2/TESimCompact.py \
  4 IS-HHHHHHHHH-EXP \
  --state state-004-001-0000050.gz
```

The compact backend can resume both legacy reference checkpoints and compact
binary checkpoints. Compact binary checkpoints must be resumed with
`TESimCompact.py`.

## Running the tests

From the repository root:

```bash
python3 -m unittest discover -s tests -v
```

The reference-versus-compact equivalence tests are the release gate for the
compact backend.

## Running on Nibi with SLURM

Production runs are submitted as SLURM arrays on the Alliance Nibi cluster.
Each array task runs one experiment with one CPU. This distributes independent
experiments across nodes and isolates failures and memory growth.

Run `diskusage_report` before staging a run. Simulation outputs are written
inside the experiment directories, so the repository's filesystem must have
adequate space and file quota. Scratch is appropriate for pilots and temporary
production output but is not archival storage; preserve completed results in
an appropriate backed-up location according to Alliance storage policy.

### Prepare the environment

Create the environment once on a Nibi login node, from the repository root:

```bash
module load python/3.10.13
virtualenv --no-download .venv-nibi
.venv-nibi/bin/python -m pip install --no-index -r requirements-compact.txt
```

Then enter the submission directory and export the absolute environment path:

```bash
cd TEWorldCodeV2/TE-Simulations
export PYTHON_EXECUTABLE="$PWD/../../.venv-nibi/bin/python"
```

### Inspect one array task

The task runner maps sorted experiment directories to stable array indices.
Inspect a task without starting a simulation:

```bash
"$PYTHON_EXECUTABLE" RunNibiArrayTask.py \
  --run 4 --index 0 --dry-run
```

The current experiment tree contains 768 tasks, but the submission script
discovers the count dynamically.

### Run a pilot array

Before a complete production launch, run several representative tasks and
measure elapsed time and peak memory:

```bash
mkdir -p logs
sbatch \
  --account=def-skremer_cpu \
  --time=02:00:00 \
  --mem=8G \
  --array=0,109,219,329,438,548,658,767%4 \
  --output=logs/%A_%a.out \
  --error=logs/%A_%a.err \
  nibi-array-job.sh 4
```

Review the pilot logs, provenance files, and SLURM accounting before choosing
the full-run memory and wall-time requests.

### Submit the complete array

```bash
./submit-nibi-array.sh 4
```

The defaults are 64 concurrent tasks, 8 GiB per task, a two-day time limit,
account `def-skremer_cpu`, module `python/3.10.13`, and the compact backend. Override
them without editing the scripts:

```bash
MAX_CONCURRENT=32 \
MEMORY_PER_TASK=16G \
WALL_TIME=3-00:00 \
PYTHON_MODULE=python/3.10.13 \
SLURM_ACCOUNT=def-skremer_cpu \
SIMULATION_BACKEND=compact \
./submit-nibi-array.sh 4
```

Set `SIMULATION_BACKEND=reference` only for small validation runs.

When an array is resubmitted with the same run number, completed experiments
are skipped and incomplete experiments resume from their latest checkpoint.
For corrected production simulations, start with a run number not used by the
historical results.

Alliance documentation:

- [Nibi](https://docs.alliancecan.ca/wiki/Nibi)
- [Running jobs](https://docs.alliancecan.ca/wiki/Running_jobs)
- [Python environments](https://docs.alliancecan.ca/wiki/Python)

## Creating a new experiment

1. Create a directory containing a copy of
   [`TEWorldCodeV2/parameters.py`](TEWorldCodeV2/parameters.py).
2. Change the copied parameter values for the intended hypothesis.
3. Run a small seeded compact simulation from that directory.
4. Run the same seed with the reference backend and compare scientific trace
   columns when introducing new model behavior.
5. Assign new production run numbers and retain the resulting provenance.

The parameter meanings and biological rationale are described in the original
paper.

## Experiment-generation and analysis scripts

The [`TEWorldCodeV2/TE-Simulations`](TEWorldCodeV2/TE-Simulations) directory
contains scripts used to create, launch, and analyze the experiments in
[`TE-Experiments`](TE-Experiments). Additional notes are in its
[`README.md`](TEWorldCodeV2/TE-Simulations/README.md).

The code that originally generated all configurations in
[`Paper-Experiments`](Paper-Experiments) is not included in this repository.

## TE-Agents2

[`TE-Agents2`](TE-Agents2) is a historical experimental rewrite by Noah
Zeidenberg. It contains additional ideas and performance work, but it is not
the production backend and has not passed the trace-equivalence tests described
above. Use `TESimCompact.py` for corrected production simulations.

The associated project documentation is available on
[Noah Zeidenberg's website](https://noahzeidenberg.github.io/research_papers/te_agents2/TE_Agents_site_index.html).

## References

1. *Transposable element persistence via potential genome-level ecosystem
   engineering*. *BMC Genomics* (2020).
   [https://doi.org/10.1186/s12864-020-6763-1](https://doi.org/10.1186/s12864-020-6763-1)
2. *Long-term TE persistence even without beneficial insertion*.
   *BMC Genomics* (2021).
   [https://doi.org/10.1186/s12864-021-07568-4](https://doi.org/10.1186/s12864-021-07568-4)
