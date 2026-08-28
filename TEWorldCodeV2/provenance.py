"""Provenance helpers for reproducible TE World simulations."""

from __future__ import annotations

import datetime as _datetime
import hashlib
import json
import os
import platform
import secrets
import subprocess
import sys
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1


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


def build_provenance(
    *,
    experiment_name: str,
    run: int,
    iteration: int,
    initial_seed: int,
    seed_source: str,
    parameter_file: str,
    simulator_file: str,
    utility_file: str,
    resumed_from: str | None,
) -> dict[str, Any]:
    """Build a self-contained description of the inputs to a simulation."""
    parameter_path = Path(parameter_file)
    simulator_path = Path(simulator_file)
    utility_path = Path(utility_file)
    code_directory = simulator_path.parent

    resumed_record = None
    if resumed_from:
        resumed_path = Path(resumed_from)
        resumed_record = _source_record(resumed_path) if resumed_path.exists() else {
            "path": str(resumed_path.resolve()),
            "sha256": None,
        }

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
        "resumed_from": resumed_record,
        "code": {
            "git_commit": _git_commit(code_directory),
            "simulator": _source_record(simulator_path),
            "utilities": _source_record(utility_path),
        },
        # Parameter files contain callables that cannot be faithfully converted
        # to JSON values. Recording their exact source preserves the real input.
        "parameters": _source_record(parameter_path, include_source=True),
        "runtime": {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "executable": sys.executable,
            "argv": list(sys.argv),
            "working_directory": os.getcwd(),
        },
        "outputs": {
            "trace": f"trace-{run:03d}-{iteration:03d}.csv",
            "provenance": f"provenance-{run:03d}-{iteration:03d}.json",
        },
        "determinism": {
            "replay_command": (
                f"{sys.executable} {simulator_path.resolve()} "
                f"{run} {experiment_name} --seed {initial_seed}"
            ),
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
