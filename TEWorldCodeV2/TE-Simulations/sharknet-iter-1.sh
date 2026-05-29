#!/bin/bash

#SBATCH --time=4-00:00 # D-HH:MM
#SBATCH --account=def-skremer
#SBATCH --mem=64G
#SBATCH --cpus-per-task=48
module load ipython-kernel/3.10

# Re-runs simulation #1 only for experiments that have less than 300 generations
python3 RunSimulations.py -s -r 1 -g 300