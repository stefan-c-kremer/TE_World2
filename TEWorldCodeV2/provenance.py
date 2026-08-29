"""Provenance helpers for reproducible TE World simulations."""

from __future__ import annotations

import datetime as _datetime
import hashlib
import json
import os
import platform
import secrets
import shlex
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 3

SLURM_ENVIRONMENT_KEYS = (
    "SLURM_JOB_ID",
    "SLURM_ARRAY_JOB_ID",
    "SLURM_ARRAY_TASK_ID",
    "SLURM_JOB_NAME",
    "SLURM_CLUSTER_NAME",
    "SLURM_JOB_ACCOUNT",
    "SLURM_JOB_PARTITION",
    "SLURM_JOB_NODELIST",
    "SLURMD_NODENAME",
    "SLURM_CPUS_PER_TASK",
    "SLURM_MEM_PER_NODE",
    "SLURM_SUBMIT_DIR",
    "SLURM_SUBMIT_HOST",
)

STUDY_ENVIRONMENT_KEYS = (
    "TE_STUDY_MANIFEST",
    "TE_STUDY_MANIFEST_SHA256",
    "TE_STUDY_TASK_INDEX",
    "TE_STUDY_CONDITION",
    "TE_STUDY_REPLICATE",
)


def resolve_seed(requested_seed: Any = None) -> int:
    """Return the requested seed or generate and return a concrete seed."""
    if requested_seed is None:
        return secrets.randbits(64)
    if isinstance(requested_seed, bool) or not isinstance(requested_seed, int):
        raise TypeError("Simulation seed must be an integer or None")
    return requested_seed


def _utc_now() -> str:
    return _datetime.datetime.now(_datetime.timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_record(path: Path, include_source: bool = False) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
    }
    if include_source:
        record["source"] = path.read_text(encoding="utf-8")
    return record


def _git_commit(code_directory: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(code_directory), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def slurm_context() -> dict[str, str] | None:
    """Return the scheduler context when running inside a SLURM allocation."""
    context = {
        key: os.environ[key]
        for key in SLURM_ENVIRONMENT_KEYS
        if key in os.environ
    }
    return context or None


def study_context() -> dict[str, str] | None:
    """Return manifest identity supplied by a study task runner."""
    context = {
        key: os.environ[key]
        for key in STUDY_ENVIRONMENT_KEYS
        if key in os.environ
    }
    return context or None


def build_provenance(
    *,
    experiment_name: str,
    run: int,
    iteration: int,
    initial_seed: int,
    seed_source: str,
    parameter_file: str,
    simulator_file: str,
    backend_name: str,
    backend_file: str,
    engine_file: str,
    backend_runtime: dict[str, Any],
    checkpoint_format: str,
    utility_file: str,
    resumed_from: str | None,
) -> dict[str, Any]:
    """Build a self-contained description of the inputs to a simulation."""
    parameter_path = Path(parameter_file)
    simulator_path = Path(simulator_file)
    backend_path = Path(backend_file)
    engine_path = Path(engine_file)
    utility_path = Path(utility_file)
    code_directory = simulator_path.parent

    resumed_record = None
    if resumed_from:
        resumed_path = Path(resumed_from)
        resumed_record = _source_record(resumed_path) if resumed_path.exists() else {
            "path": str(resumed_path.resolve()),
            "sha256": None,
        }

    replay_arguments = [
        sys.executable,
        str(simulator_path.resolve()),
        str(run),
        experiment_name,
    ]
    if resumed_from:
        replay_arguments.extend(["--state", str(Path(resumed_from).resolve())])
    else:
        replay_arguments.extend(["--seed", str(initial_seed)])

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "started_at_utc": _utc_now(),
        "experiment_name": experiment_name,
        "run": run,
        "iteration": iteration,
        "initial_seed": initial_seed,
        "seed_source": seed_source,
        "random_generator": "python.random.MersenneTwister",
        "simulation_backend": backend_name,
        "checkpoint_format": checkpoint_format,
        "backend_runtime": backend_runtime,
        "study": study_context(),
        "resumed_from": resumed_record,
        "code": {
            "git_commit": _git_commit(code_directory),
            "simulator": _source_record(simulator_path),
            "backend": _source_record(backend_path),
            "engine": _source_record(engine_path),
            "utilities": _source_record(utility_path),
        },
        # Parameter files contain callables that cannot be faithfully converted
        # to JSON values. Recording their exact source preserves the real input.
        "parameters": _source_record(parameter_path, include_source=True),
        "runtime": {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "hostname": socket.gethostname(),
            "cpu_count": os.cpu_count(),
            "executable": sys.executable,
            "argv": list(sys.argv),
            "working_directory": os.getcwd(),
            "slurm": slurm_context(),
        },
        "outputs": {
            "trace": f"trace-{run:03d}-{iteration:03d}.csv",
            "provenance": f"provenance-{run:03d}-{iteration:03d}.json",
        },
        "determinism": {
            "replay_command": shlex.join(replay_arguments),
            "note": (
                "Scientific results are reproducible with the recorded seed, "
                "parameter source, and code hashes. Wall-clock timing fields "
                "are not deterministic."
            ),
        },
    }


def finalize_provenance(
    record: dict[str, Any],
    *,
    status: str,
    final_generation: int,
    final_population_size: int,
) -> None:
    record["status"] = status
    record["completed_at_utc"] = _utc_now()
    record["final_generation"] = final_generation
    record["final_population_size"] = final_population_size


def write_provenance(path: str, record: dict[str, Any]) -> None:
    """Atomically write a provenance record."""
    destination = Path(path)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
