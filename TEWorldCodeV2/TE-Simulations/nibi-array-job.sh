#!/bin/bash
#SBATCH --job-name=te-world2
#SBATCH --cpus-per-task=1

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 RUN_NUMBER" >&2
  exit 2
fi

run_number=$1
script_directory=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
python_module=${PYTHON_MODULE:-python/3.10.13}

module --force purge
module load "$python_module"

python3 "$script_directory/RunNibiArrayTask.py" \
  --run "$run_number" \
  --index "$SLURM_ARRAY_TASK_ID" \
  --resume-latest
