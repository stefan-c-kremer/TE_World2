#!/usr/bin/env python3
"""Generate the corrected autonomous/non-autonomous experiment grid."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import re
import textwrap
from pathlib import Path


CONDITION_PATTERN = re.compile(r"^[HL]{7}-(?:HH|HL|LH|LL|Z)$")
PARAMETER_ORDER = (
    "TE_progeny",
    "TE_death_rate",
    "Insertion_bias",
    "Corrected_mutation_rate",
    "NC_BP",
    "Mutation_effect",
    "Carrying_capacity",
)
INTERACTIONS = {
    "HH": (0.07, 3),
    "HL": (0.07, 1),
    "LH": (0.02, 3),
    "LL": (0.02, 1),
    "Z": (0.0, 0),
}


def deterministic_seed(master_seed: str, condition: str, replicate: int) -> int:
    identity = f"{master_seed}\0{condition}\0R{replicate:02d}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(identity).digest()[:8], "big")


def parameter_source(bits: str, interaction: str) -> str:
    if len(bits) != 7 or set(bits) - {"H", "L"}:
        raise ValueError(f"Invalid seven-variable code: {bits!r}")
    if interaction not in INTERACTIONS:
        raise ValueError(f"Invalid interaction code: {interaction!r}")

    high = [letter == "H" for letter in bits]
    kidnapping_coefficient, initial_non_autonomous = INTERACTIONS[interaction]

    te_progeny = (
        "ProbabilityTable(0.00, 0, 0.55, 1, 0.30, 2, 0.15, 3)"
        if high[0]
        else "ProbabilityTable(0.15, 0, 0.55, 1, 0.30, 2)"
    )
    te_death_rate = 0.0005 if high[1] else 0.005
    te_distribution = "Triangle(pmax=0, pzero=3.0/3.0)" if high[2] else "Flat()"
    gene_distribution = "Triangle(pzero=1.0/3.0, pmax=1)" if high[2] else "Flat()"
    initial_genes = 5000 if high[3] else 500
    host_mutation_rate = 0.3 if high[3] else 0.03
    junk_bp = 14_000_000 if high[4] else 1_400_000
    mutation_effect = 0.1 if high[5] else 0.01
    carrying_capacity = 300 if high[6] else 30
    condition = f"{bits}-{interaction}"

    return textwrap.dedent(
        f'''\
        """Generated parameters for condition {condition}.

        Code order: TE_progeny, TE_death_rate, Insertion_bias,
        Corrected_mutation_rate, NC_BP, Mutation_effect, Carrying_capacity;
        suffix: Kidnapping_frequency/Initial_NAut_TEs, or Z for none.
        """

        from TEUtil import *

        seed = None
        saved = None

        output = {{
            "SPLAT": False,
            "SPLAT FITNESS": False,
            "INITIALIZATION": False,
            "GENERATION": True,
            "HOST EXTINCTION": True,
            "TE EXTINCTION": True,
            "TRIAL NO": True,
            "GENE INIT": False,
            "TE INIT": False,
            "BULK SIM": True,
            "CHECKPOINT": True,
        }}

        Gene_length = 1000
        TE_length = lambda autonomous: 6000 if autonomous else 300
        Append_gene = True
        Initial_Aut_TEs = 1
        Initial_NAut_TEs = {initial_non_autonomous}
        TE_excision_rate = 0.0
        Host_start_fitness = 1.0
        Host_reproduction_rate = 1
        Maximum_generations = 1500
        Terminate_no_TEs = True
        save_frequency = 50

        TE_progeny = {te_progeny}
        TE_death_rate = {te_death_rate!r}
        TE_Insertion_Distribution = {te_distribution}
        Gene_Insertion_Distribution = {gene_distribution}
        Initial_genes = {initial_genes}
        Host_mutation_rate = {host_mutation_rate!r}
        Junk_BP = {junk_bp}
        Host_mutation = ProbabilityTable(
            0.40, lambda fit: 0.0,
            0.30, lambda fit: fit - random.random() * {mutation_effect!r},
            0.15, lambda fit: fit,
            0.15, lambda fit: fit + random.random() * {mutation_effect!r},
        )
        Insertion_effect = ProbabilityTable(
            0.30, lambda fit: 0.0,
            0.20, lambda fit: fit - random.random() * {mutation_effect!r},
            0.30, lambda fit: fit,
            0.20, lambda fit: fit + random.random() * {mutation_effect!r},
        )
        Carrying_capacity = {carrying_capacity}
        Host_survival_rate = lambda propfit: min(Carrying_capacity * propfit, 0.95)
        Kidnapping_frequency = lambda live_aut, live_naut: (
            1 - 1 / (1 + {kidnapping_coefficient!r} * live_naut)
        )
        '''
    )


def condition_codes(interactions: list[str]) -> list[str]:
    return [
        f"{''.join(bits)}-{interaction}"
        for bits in itertools.product("HL", repeat=7)
        for interaction in interactions
    ]


def write_new_or_identical(path: Path, content: str) -> None:
    if path.exists() and path.read_text(encoding="utf-8") != content:
        raise FileExistsError(f"Refusing to replace differing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def generate(
    output_root: Path,
    interactions: list[str],
    replicates: int,
    master_seed: str,
    manifest_name: str,
) -> tuple[Path, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    codes = condition_codes(interactions)
    for code in codes:
        bits, interaction = code.split("-")
        if not CONDITION_PATTERN.fullmatch(code):
            raise AssertionError(code)
        write_new_or_identical(
            output_root / code / "parameters.py",
            parameter_source(bits, interaction),
        )

    manifest_path = output_root / manifest_name
    rows = []
    for condition in codes:
        for replicate in range(1, replicates + 1):
            rows.append(
                {
                    "task_index": len(rows),
                    "condition_code": condition,
                    "replicate": f"R{replicate:02d}",
                    "run": replicate,
                    "seed": deterministic_seed(master_seed, condition, replicate),
                    "experiment_directory": condition,
                }
            )

    fieldnames = list(rows[0])
    from io import StringIO

    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    write_new_or_identical(manifest_path, buffer.getvalue())

    metadata_path = manifest_path.with_suffix(".json")
    metadata = {
        "schema_version": 1,
        "condition_code_format": "[HL]{7}-(HH|HL|LH|LL|Z)",
        "parameter_order": list(PARAMETER_ORDER),
        "interaction_codes": {
            code: {
                "kidnapping_coefficient": coefficient,
                "initial_non_autonomous_tes": count,
            }
            for code, (coefficient, count) in INTERACTIONS.items()
            if code in interactions
        },
        "replicates": replicates,
        "master_seed": master_seed,
        "seed_derivation": "first 64 bits of SHA-256(master_seed\\0condition\\0replicate)",
        "condition_count": len(codes),
        "simulation_count": len(rows),
        "manifest": manifest_path.name,
    }
    write_new_or_identical(
        metadata_path,
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
    )
    return manifest_path, metadata_path


def parse_arguments(args=None):
    script_directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=script_directory.parents[1] / "TE-Interaction-Experiments",
    )
    parser.add_argument(
        "--interaction",
        action="append",
        choices=tuple(INTERACTIONS),
        help="Interaction suffix to generate; repeat as needed (default: all)",
    )
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--master-seed", required=True)
    parser.add_argument("--manifest-name", default="manifest.csv")
    return parser.parse_args(args)


def main(args=None) -> int:
    options = parse_arguments(args)
    if options.replicates < 1:
        raise SystemExit("--replicates must be positive")
    interactions = options.interaction or list(INTERACTIONS)
    manifest, metadata = generate(
        options.output_root.resolve(),
        interactions,
        options.replicates,
        options.master_seed,
        options.manifest_name,
    )
    print(f"Generated {manifest}")
    print(f"Metadata: {metadata}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
