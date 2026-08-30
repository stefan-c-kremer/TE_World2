#!/usr/bin/env python3
"""Record Nibi submissions and collect per-attempt simulation resources."""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import re
import subprocess
from pathlib import Path


SUBMISSION_FIELDS = (
    "job_id", "submitted_at_utc", "manifest", "manifest_sha256",
    "array_indices", "max_concurrent", "account", "wall_time", "memory",
    "cpus_per_task", "backend", "dependency", "git_commit",
)
ATTEMPT_FIELDS = (
    "job_id", "task_index", "condition_code", "replicate", "run", "seed",
    "te_progeny", "te_death_rate", "insertion_bias",
    "corrected_mutation_rate", "nc_bp", "mutation_effect",
    "carrying_capacity", "interaction", "requested_wall_time",
    "requested_memory", "requested_cpus", "max_concurrent", "account",
    "scheduler_state", "exit_code", "elapsed_seconds", "max_rss_bytes",
    "max_vmsize_bytes", "partition", "node_list", "failure_class",
    "scientific_status", "final_generation", "final_population_size",
    "last_trace_generation", "last_live_tes", "last_median_genome_size",
    "latest_checkpoint", "resumed_from", "backend", "git_commit",
    "provenance_file", "stdout_file", "stderr_file", "collected_at_utc",
)
CODE_FIELDS = (
    "te_progeny", "te_death_rate", "insertion_bias",
    "corrected_mutation_rate", "nc_bp", "mutation_effect",
    "carrying_capacity",
)


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def write_csv(path: Path, fields, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)
    temporary.replace(path)


def upsert(path: Path, fields, row, key_fields) -> None:
    rows = read_csv(path)
    key = tuple(str(row.get(field, "")) for field in key_fields)
    replacement = {field: str(row.get(field, "")) for field in fields}
    for index, existing in enumerate(rows):
        if tuple(existing.get(field, "") for field in key_fields) == key:
            rows[index] = replacement
            break
    else:
        rows.append(replacement)
    rows.sort(key=lambda item: tuple(item.get(field, "") for field in key_fields))
    write_csv(path, fields, rows)


def parse_indices(specification: str) -> list[int]:
    specification = specification.split("%", 1)[0]
    result = set()
    for part in specification.split(","):
        if not part:
            continue
        if "-" in part:
            start, end = map(int, part.split("-", 1))
            result.update(range(start, end + 1))
        else:
            result.add(int(part))
    return sorted(result)


def parse_bytes(value: str) -> int | str:
    if not value:
        return ""
    match = re.fullmatch(r"([0-9.]+)([KMGTPE]?)", value.strip(), re.IGNORECASE)
    if not match:
        return ""
    powers = {"": 0, "K": 1, "M": 2, "G": 3, "T": 4, "P": 5, "E": 6}
    return round(float(match.group(1)) * 1024 ** powers[match.group(2).upper()])


def classify_failure(stderr: str, scheduler_state: str, scientific_status: str) -> str:
    lowered = stderr.lower()
    if "oom_kill" in lowered or "out of memory" in lowered:
        return "out_of_memory"
    if "due to time limit" in lowered or scheduler_state.startswith("TIMEOUT"):
        return "time_limit"
    if "user defined signal" in lowered:
        return "signal_routing_failure"
    if scientific_status == "checkpointed":
        return "checkpointed"
    if scientific_status in {"maximum_generations", "host_extinction", "te_extinction"}:
        return "none"
    if stderr.strip():
        return "other"
    return "unknown"


def accounting(job_ids: list[str]) -> dict[tuple[str, str], dict[str, str]]:
    if not job_ids:
        return {}
    extended_fields = (
        "JobID", "ArrayJobID", "ArrayTaskID", "State", "ExitCode",
        "ElapsedRaw", "MaxRSS", "MaxVMSize", "ReqMem", "TimelimitRaw",
        "Partition", "NodeList",
    )
    legacy_fields = tuple(
        field for field in extended_fields
        if field not in {"ArrayJobID", "ArrayTaskID"}
    )
    result = None
    fields = extended_fields
    last_error = None
    for candidate_fields in (extended_fields, legacy_fields):
        command = [
            "sacct", "-j", ",".join(job_ids), "-n", "-P",
            "--format=" + ",".join(candidate_fields),
        ]
        try:
            result = subprocess.run(
                command, check=True, capture_output=True, text=True
            )
            fields = candidate_fields
            break
        except (OSError, subprocess.CalledProcessError) as error:
            last_error = error
    if result is None:
        print(
            "Warning: sacct unavailable; collecting file-based data only: "
            f"{last_error}"
        )
        return {}

    records: dict[tuple[str, str], dict[str, str]] = {}
    for values in csv.reader(result.stdout.splitlines(), delimiter="|"):
        values = values[:len(fields)]
        if len(values) != len(fields):
            continue
        row = dict(zip(fields, values))
        array_job = row.get("ArrayJobID", "")
        task = row.get("ArrayTaskID", "")
        if not array_job or not task:
            match = re.fullmatch(r"(\d+)_(\d+)(?:\..+)?", row["JobID"])
            if match:
                array_job, task = match.groups()
        if not array_job or not task or task in {"4294967294", "N/A"}:
            continue
        key = (array_job, task)
        current = records.setdefault(key, {})
        is_step = "." in row["JobID"]
        if not is_step:
            current.update({
                "scheduler_state": row["State"].split("+", 1)[0],
                "exit_code": row["ExitCode"],
                "elapsed_seconds": row["ElapsedRaw"],
                "partition": row["Partition"],
                "node_list": row["NodeList"],
            })
        rss = parse_bytes(row["MaxRSS"])
        vm = parse_bytes(row["MaxVMSize"])
        if rss != "" and int(rss) > int(current.get("max_rss_bytes") or 0):
            current["max_rss_bytes"] = str(rss)
        if vm != "" and int(vm) > int(current.get("max_vmsize_bytes") or 0):
            current["max_vmsize_bytes"] = str(vm)
    return records


def matching_provenance(directory: Path, run: int, job_id: str, task: int):
    matches = []
    for path in directory.glob(f"provenance-{run:03d}-*.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        slurm = record.get("runtime", {}).get("slurm") or {}
        if (
            slurm.get("SLURM_ARRAY_JOB_ID") == job_id
            and slurm.get("SLURM_ARRAY_TASK_ID") == str(task)
        ):
            matches.append((path, record))
    return max(matches, default=(None, {}), key=lambda item: item[0].name if item[0] else "")


def trace_metrics(directory: Path, provenance_record: dict) -> dict[str, str]:
    trace_name = provenance_record.get("outputs", {}).get("trace")
    if not trace_name:
        return {}
    path = directory / trace_name
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source, skipinitialspace=True))
    if not rows:
        return {}
    last = {key.strip(): value for key, value in rows[-1].items()}
    return {
        "last_trace_generation": last.get("GEN", last.get("GENERATION", "")),
        "last_live_tes": last.get("LTETOTAL", ""),
        "last_median_genome_size": last.get("GSIZE050", ""),
    }


def collect(manifest: Path, submission_file: Path, output: Path, job_ids: list[str]) -> None:
    manifest_rows = {int(row["task_index"]): row for row in read_csv(manifest)}
    ledger = {
        (row["job_id"], row["task_index"]): row
        for row in read_csv(output)
    }
    submissions = [
        row for row in read_csv(submission_file)
        if not job_ids or row["job_id"] in job_ids
    ]
    accounting_rows = accounting([row["job_id"] for row in submissions])
    for submission in submissions:
        job_id = submission["job_id"]
        for task_index in parse_indices(submission["array_indices"]):
            task = manifest_rows.get(task_index)
            if task is None:
                continue
            directory = manifest.parent / task["experiment_directory"]
            stdout = manifest.parent / "logs" / f"{job_id}_{task_index}.out"
            stderr = manifest.parent / "logs" / f"{job_id}_{task_index}.err"
            provenance_path, provenance_record = matching_provenance(
                directory, int(task["run"]), job_id, task_index
            )
            if not stdout.exists() and not stderr.exists() and provenance_path is None:
                continue
            scheduler = accounting_rows.get((job_id, str(task_index)), {})
            stderr_text = stderr.read_text(encoding="utf-8", errors="replace") if stderr.exists() else ""
            scientific_status = provenance_record.get("status", "")
            bits, interaction = task["condition_code"].split("-")
            row = {
                "job_id": job_id,
                "task_index": task_index,
                "condition_code": task["condition_code"],
                "replicate": task["replicate"],
                "run": task["run"],
                "seed": task["seed"],
                **dict(zip(CODE_FIELDS, bits)),
                "interaction": interaction,
                "requested_wall_time": submission["wall_time"],
                "requested_memory": submission["memory"],
                "requested_cpus": submission["cpus_per_task"],
                "max_concurrent": submission["max_concurrent"],
                "account": submission["account"],
                **scheduler,
                "scientific_status": scientific_status,
                "final_generation": provenance_record.get("final_generation", ""),
                "final_population_size": provenance_record.get("final_population_size", ""),
                "latest_checkpoint": max(
                    (path.name for path in directory.glob(f"state-{int(task['run']):03d}-*.gz")),
                    default="",
                ),
                "resumed_from": (provenance_record.get("resumed_from") or {}).get("path", ""),
                "backend": provenance_record.get("simulation_backend", submission["backend"]),
                "git_commit": provenance_record.get("code", {}).get("git_commit", submission["git_commit"]),
                "provenance_file": str(provenance_path or ""),
                "stdout_file": str(stdout if stdout.exists() else ""),
                "stderr_file": str(stderr if stderr.exists() else ""),
                "collected_at_utc": utc_now(),
            }
            row.update(trace_metrics(directory, provenance_record))
            row["failure_class"] = classify_failure(
                stderr_text, row.get("scheduler_state", ""), scientific_status
            )
            normalized = {field: str(row.get(field, "")) for field in ATTEMPT_FIELDS}
            ledger[(normalized["job_id"], normalized["task_index"])] = normalized
    rows = sorted(ledger.values(), key=lambda item: (item["job_id"], int(item["task_index"])))
    write_csv(output, ATTEMPT_FIELDS, rows)


def parse_arguments(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    record = subparsers.add_parser("record-submission")
    record.add_argument("--job-id", required=True)
    record.add_argument("--manifest", type=Path, required=True)
    record.add_argument("--array-indices", required=True)
    record.add_argument("--max-concurrent", required=True)
    record.add_argument("--account", required=True)
    record.add_argument("--wall-time", required=True)
    record.add_argument("--memory", required=True)
    record.add_argument("--cpus-per-task", default="1")
    record.add_argument("--backend", default="compact")
    record.add_argument("--dependency", default="")
    record.add_argument("--git-commit", default="")
    record.add_argument("--output", type=Path)
    collector = subparsers.add_parser("collect")
    collector.add_argument("--manifest", type=Path, required=True)
    collector.add_argument("--submissions", type=Path)
    collector.add_argument("--output", type=Path)
    collector.add_argument("--job-id", action="append", default=[])
    return parser.parse_args(args)


def main(args=None) -> int:
    options = parse_arguments(args)
    manifest = options.manifest.resolve()
    if options.command == "record-submission":
        output = (options.output or manifest.parent / "submission-ledger.csv").resolve()
        row = {
            "job_id": options.job_id,
            "submitted_at_utc": utc_now(),
            "manifest": str(manifest),
            "manifest_sha256": sha256(manifest),
            "array_indices": options.array_indices,
            "max_concurrent": options.max_concurrent,
            "account": options.account,
            "wall_time": options.wall_time,
            "memory": options.memory,
            "cpus_per_task": options.cpus_per_task,
            "backend": options.backend,
            "dependency": options.dependency,
            "git_commit": options.git_commit,
        }
        upsert(output, SUBMISSION_FIELDS, row, ("job_id",))
        print(output)
        return 0
    submissions = (options.submissions or manifest.parent / "submission-ledger.csv").resolve()
    output = (options.output or manifest.parent / "resource-attempts.csv").resolve()
    collect(manifest, submissions, output, options.job_id)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
