#!/bin/bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 MANIFEST.csv" >&2
  exit 2
fi

script_directory=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
manifest=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
log_directory=${LOG_DIRECTORY:-$(dirname "$manifest")/logs}
max_concurrent=${MAX_CONCURRENT:-64}
slurm_account=${SLURM_ACCOUNT:-def-skremer_cpu}
wall_time=${WALL_TIME:-00:30:00}
memory=${MEMORY_PER_TASK:-4G}
signal_seconds=${CHECKPOINT_SIGNAL_SECONDS:-300}

if [[ ! -f $manifest ]]; then
  echo "Manifest does not exist: $manifest" >&2
  exit 1
fi

pending_indices=$(python3 "$script_directory/RunNibiArrayTask.py" \
  --manifest "$manifest" --pending-indices)
if [[ -z $pending_indices ]]; then
  echo "Every manifest task is already complete."
  exit 0
fi

array_indices=${ARRAY_INDICES:-$pending_indices}
mkdir -p "$log_directory"

echo "Submitting manifest $manifest."
echo "Array indices: $array_indices; maximum concurrency: $max_concurrent."
echo "Resources per task: $wall_time, $memory, one CPU."

submission_id=$(sbatch --parsable \
  --account="$slurm_account" \
  --time="$wall_time" \
  --mem="$memory" \
  --signal="B:USR1@${signal_seconds}" \
  --array="${array_indices}%${max_concurrent}" \
  --output="$log_directory/%A_%a.out" \
  --error="$log_directory/%A_%a.err" \
  --chdir="$script_directory" \
  --export="ALL,TE_SIMULATION_SCRIPT_DIR=$script_directory,TE_SIMULATION_MANIFEST=$manifest" \
  "$script_directory/nibi-manifest-job.sh")
submission_id=${submission_id%%;*}
echo "Submitted batch job $submission_id"

git_commit=$(git -C "$script_directory" rev-parse HEAD 2>/dev/null || true)
python3 "$script_directory/TrackNibiResources.py" record-submission \
  --job-id "$submission_id" \
  --manifest "$manifest" \
  --array-indices "$array_indices" \
  --max-concurrent "$max_concurrent" \
  --account "$slurm_account" \
  --wall-time "$wall_time" \
  --memory "$memory" \
  --backend "${SIMULATION_BACKEND:-compact}" \
  --git-commit "$git_commit" >/dev/null
