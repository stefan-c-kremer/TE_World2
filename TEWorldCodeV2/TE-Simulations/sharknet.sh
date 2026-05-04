#!/bin/bash

#SBATCH --time=4-00:00 # D-HH:MM
#SBATCH --account=def-skremer
#SBATCH --mem=16G
#SBATCH --cpus-per-task=48
module load ipython-kernel/3.10

# Runs simulation #1 when `-r 1` is specified
python3 RunSimulations.py -s -r 1