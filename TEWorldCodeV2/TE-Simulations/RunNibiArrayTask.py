"""Run one TE World experiment selected by a SLURM array index."""

from __future__ import annotations

import argparse
import csv
import hashlib
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


def manifest_tasks(path: Path) -> list[dict[str, object]]:
    tasks = []
    with path.open(newline="", encoding="utf-8") as source:
        for expected_index, row in enumerate(csv.DictReader(source)):
            index = int(row["task_index"])
            if index != expected_index:
                raise ValueError(
                    f"Manifest index {index} is not the expected {expected_index}"
                )
            experiment = (path.parent / row["experiment_directory"]).resolve()
            if not (experiment / "parameters.py").is_file():
                raise FileNotFoundError(f"Missing parameters.py under {experiment}")
            tasks.append(
                {
                    "index": index,
                    "condition": row["condition_code"],
                    "replicate": row["replicate"],
                    "run": int(row["run"]),
                    "seed": int(row["seed"]),
                    "experiment": experiment,
                }
            )
    return tasks


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--resume-latest", action="store_true")
    parser.add_argument(
        "--backend",
        choices=("compact", "reference"),
        default="compact",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--count", action="store_true")
    parser.add_argument("--pending-indices", action="store_true")
    return parser.parse_args(args)


def main(args=None) -> int:
    options = parse_arguments(args)
    tasks = manifest_tasks(options.manifest.resolve()) if options.manifest else None
    experiments = (
        [task["experiment"] for task in tasks]
        if tasks is not None
        else experiment_directories(options.experiment_root.resolve())
    )
    if options.count:
        print(len(experiments))
        return 0

    if options.pending_indices:
        if tasks is None:
            raise SystemExit("--pending-indices requires --manifest")
        pending = [
            str(task["index"])
            for task in tasks
            if completed_provenance(task["experiment"], task["run"]) is None
        ]
        print(",".join(pending))
        return 0

    if options.run is None and tasks is None:
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

    task = tasks[index] if tasks is not None else None
    experiment = experiments[index]
    run = task["run"] if task is not None else options.run
    completed = completed_provenance(experiment, run)
    if completed:
        print(f"Skipping completed experiment {experiment.name}: {completed.name}")
        return 0

    simulator_name = "TESimCompact.py" if options.backend == "compact" else "TESim.py"
    simulator = Path(__file__).resolve().parent.parent / simulator_name
    command = [
        sys.executable,
        str(simulator),
        str(run),
        experiment.name,
    ]

    checkpoint = None
    if options.resume_latest:
        checkpoint = latest_checkpoint(experiment, run)
        if checkpoint:
            command.extend(["--state", str(checkpoint)])
    if checkpoint is None and task is not None:
        command.extend(["--seed", str(task["seed"])])

    print(
        f"Array index {index}/{len(experiments) - 1}: {experiment.name}; "
        f"replicate={task['replicate'] if task else f'R{run:02d}'}; "
        f"checkpoint={checkpoint.name if checkpoint else 'none'}",
        flush=True,
    )
    print("Command:", " ".join(command), flush=True)
    if options.dry_run:
        return 0

    environment = os.environ.copy()
    if task is not None:
        manifest_path = options.manifest.resolve()
        environment.update(
            {
                "TE_STUDY_MANIFEST": str(manifest_path),
                "TE_STUDY_MANIFEST_SHA256": sha256(manifest_path),
                "TE_STUDY_TASK_INDEX": str(task["index"]),
                "TE_STUDY_CONDITION": str(task["condition"]),
                "TE_STUDY_REPLICATE": str(task["replicate"]),
            }
        )
    return subprocess.run(
        command,
        cwd=experiment,
        env=environment,
        check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
