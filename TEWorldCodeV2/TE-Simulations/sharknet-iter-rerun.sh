#!/bin/bash

#SBATCH --time=1-18:00 # D-HH:MM
#SBATCH --account=def-skremer
#SBATCH --mem=16G
#SBATCH --cpus-per-task=48
module load ipython-kernel/3.10

python3 RunSimulations.py -s -i 1