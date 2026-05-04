#!/bin/bash

#SBATCH --time=4-00:00 # D-HH:MM
#SBATCH --account=def-skremer
#SBATCH --mem=16G
#SBATCH --cpus-per-task=48
module load ipython-kernel/3.10

# Re-runs simulation #1
python3 RunSimulations.py -s -r 1