#!/bin/bash

#SBATCH --time=8-00:00 # D-HH:MM
#SBATCH --account=def-skremer
#SBATCH --mem=64G
#SBATCH --cpus-per-task=48
module load ipython-kernel/3.10

# Re-runs simulation #1 only for experiments that have less than 300 generations
python3 RunSimulations.py -s -r 3 -g 300