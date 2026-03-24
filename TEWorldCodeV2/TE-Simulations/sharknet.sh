#!/bin/bash

#SBATCH --time=3-00:00 # D-HH:MM
#SBATCH --account=def-skremer
#SBATCH --mem=16G
#SBATCH --cpus-per-task=48
module load ipython-kernel/3.10

python3 TE_World2/TEWorldCodeV2/TE-Simulations/RunSimulations.py