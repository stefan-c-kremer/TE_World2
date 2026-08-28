"""Run one TE World experiment selected by a SLURM array index."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


COMPLETED_STATUSES = {"maximum_generations", "host_extinction", "te_extinction"}
STATE_PATTERN = re.compile(r"state-(\d{3})-(\d{3})-(\d{7})\.gz$")


def experiment_directories(root: Path) -> list[Path]:
    return sorted(path.parent for path in root.glob("*/parameters.py"))


def completed_provenance(experiment: Path, run: int) -> Path | None:
    for path in sorted(experiment.glob(f"provenance-{run:03d}-*.json"), reverse=True):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if record.get("status") in COMPLETED_STATUSES:
            return path
    return None


def latest_checkpoint(experiment: Path, run: int) -> Path | None:
    checkpoints = []
    for path in experiment.glob(f"state-{run:03d}-*-*.gz"):
        match = STATE_PATTERN.fullmatch(path.name)
        if match and int(match.group(1)) == run:
            iteration = int(match.group(2))
            generation = int(match.group(3))
            checkpoints.append((generation, iteration, path))
    return max(checkpoints, default=(None, None, None))[2]


def parse_arguments(args=None):
    script_directory = Path(__file__).resolve().parent
    default_root = script_directory.parents[1] / "TE-Experiments"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=int)
    parser.add_argument("--index", type=int)
    parser.add_argument("--experiment-root", type=Path, default=default_root)
    parser.add_argument("--resume-latest", action="store_true")
    parser.add_argument(
        "--backend",
        choices=("compact", "reference"),
        default="compact",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--count", action="store_true")
    return parser.parse_args(args)


def main(args=None) -> int:
    options = parse_arguments(args)
    experiments = experiment_directories(options.experiment_root.resolve())
    if options.count:
        print(len(experiments))
        return 0

    if options.run is None:
        raise SystemExit("--run is required")

    index = options.index
    if index is None:
        value = os.environ.get("SLURM_ARRAY_TASK_ID")
        if value is None:
            raise SystemExit("--index or SLURM_ARRAY_TASK_ID is required")
        index = int(value)

    if not 0 <= index < len(experiments):
        raise SystemExit(
            f"Experiment index {index} is outside 0..{len(experiments) - 1}"
        )

    experiment = experiments[index]
    completed = completed_provenance(experiment, options.run)
    if completed:
        print(f"Skipping completed experiment {experiment.name}: {completed.name}")
        return 0

    simulator_name = "TESimCompact.py" if options.backend == "compact" else "TESim.py"
    simulator = Path(__file__).resolve().parent.parent / simulator_name
    command = [
        sys.executable,
        str(simulator),
        str(options.run),
        experiment.name,
    ]

    checkpoint = None
    if options.resume_latest:
        checkpoint = latest_checkpoint(experiment, options.run)
        if checkpoint:
            command.extend(["--state", str(checkpoint)])

    print(
        f"Array index {index}/{len(experiments) - 1}: {experiment.name}; "
        f"checkpoint={checkpoint.name if checkpoint else 'none'}",
        flush=True,
    )
    print("Command:", " ".join(command), flush=True)
    if options.dry_run:
        return 0

    return subprocess.run(command, cwd=experiment, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
