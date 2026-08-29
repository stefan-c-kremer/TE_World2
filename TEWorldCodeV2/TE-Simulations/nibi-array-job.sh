#!/bin/bash
#SBATCH --job-name=te-world2
#SBATCH --cpus-per-task=1

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 RUN_NUMBER" >&2
  exit 2
fi

run_number=$1
script_directory=${TE_SIMULATION_SCRIPT_DIR:-${SLURM_SUBMIT_DIR:-}}
if [[ ! -f $script_directory/RunNibiArrayTask.py ]]; then
  echo "Cannot find RunNibiArrayTask.py under $script_directory" >&2
  exit 1
fi
python_module=${PYTHON_MODULE:-python/3.10.13}
standard_environment_module=${STANDARD_ENV_MODULE:-StdEnv/2023}
simulation_backend=${SIMULATION_BACKEND:-compact}
python_executable=${PYTHON_EXECUTABLE:-python3}

module --force purge
module load "$standard_environment_module"
module load "$python_module"

if [[ $simulation_backend == compact ]]; then
  "$python_executable" -c 'import numpy' || {
    echo "The compact backend requires NumPy in the selected Python environment." >&2
    exit 1
  }
fi

"$python_executable" "$script_directory/RunNibiArrayTask.py" \
  --run "$run_number" \
  --index "$SLURM_ARRAY_TASK_ID" \
  --backend "$simulation_backend" \
  --resume-latest
