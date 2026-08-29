#!/bin/bash
set -euo pipefail

if [[ $# -ne 1 || ! $1 =~ ^[0-9]+$ ]]; then
  echo "Usage: $0 RUN_NUMBER" >&2
  exit 2
fi

run_number=$1
script_directory=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
experiment_root=$(cd "$script_directory/../.." && pwd)/TE-Experiments
log_directory="$script_directory/logs"
max_concurrent=${MAX_CONCURRENT:-64}
slurm_account=${SLURM_ACCOUNT:-def-skremer_cpu}
wall_time=${WALL_TIME:-2-00:00}
memory=${MEMORY_PER_TASK:-8G}

experiment_count=$(python3 "$script_directory/RunNibiArrayTask.py" \
  --experiment-root "$experiment_root" \
  --count)

if (( experiment_count == 0 )); then
  echo "No experiment parameter files found under $experiment_root" >&2
  exit 1
fi

mkdir -p "$log_directory"
last_index=$((experiment_count - 1))

echo "Submitting run $run_number as $experiment_count independent Nibi tasks."
echo "At most $max_concurrent tasks will run concurrently."

sbatch \
  --account="$slurm_account" \
  --time="$wall_time" \
  --mem="$memory" \
  --array="0-${last_index}%${max_concurrent}" \
  --output="$log_directory/%A_%a.out" \
  --error="$log_directory/%A_%a.err" \
  --chdir="$script_directory" \
  --export="ALL,TE_SIMULATION_SCRIPT_DIR=$script_directory" \
  "$script_directory/nibi-array-job.sh" "$run_number"
