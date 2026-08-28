import csv
import json
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CODE_DIRECTORY = REPOSITORY_ROOT / "TEWorldCodeV2"
REFERENCE_SIMULATOR = CODE_DIRECTORY / "TESim.py"
COMPACT_SIMULATOR = CODE_DIRECTORY / "TESimCompact.py"


PARAMETERS = """
from TEUtil import *

output = {{}}
TE_Insertion_Distribution = Triangle(pmax=0, pzero=2.0 / 3.0)
Gene_Insertion_Distribution = Triangle(pzero=1.0 / 3.0, pmax=1)
Gene_length = 40
TE_length = lambda autonomous: 60 if autonomous else 30
TE_death_rate = 0.15
TE_excision_rate = {excision_rate}
TE_progeny = ProbabilityTable(0.30, 0, 0.50, 1, 0.20, 2)
Initial_genes = 8
Append_gene = {append_gene}
Junk_BP = 5000
Host_start_fitness = 1.0
Host_mutation_rate = 0.05
Host_mutation = ProbabilityTable(1.0, lambda fit: fit)
Insertion_effect = ProbabilityTable(1.0, lambda fit: fit)
Carrying_capacity = 8
Host_reproduction_rate = 1
Host_survival_rate = lambda propfit: min(Carrying_capacity * propfit, 0.95)
Initial_Aut_TEs = 3
Initial_NAut_TEs = 2
Maximum_generations = 7
Terminate_no_TEs = False
seed = None
save_frequency = 1
saved = None
Kidnapping_frequency = lambda live_aut, live_naut: 1 - 1 / (1 + 0.07 * live_naut)
"""


def write_parameters(directory: Path, *, excision_rate=0.0, append_gene=True):
    (directory / "parameters.py").write_text(
        textwrap.dedent(PARAMETERS).format(
            excision_rate=excision_rate,
            append_gene=append_gene,
        ),
        encoding="utf-8",
    )


def run_simulator(simulator: Path, directory: Path, *extra_arguments):
    return subprocess.run(
        [
            sys.executable,
            str(simulator),
            "9",
            "compact-equivalence-test",
            *map(str, extra_arguments),
        ],
        cwd=directory,
        check=True,
        capture_output=True,
        text=True,
    )


def scientific_trace(path: Path):
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.reader(source, skipinitialspace=True))
    return [row[1:] for row in rows]


class CompactBackendTests(unittest.TestCase):
    def assert_backends_match(self, *, excision_rate, append_gene, seed):
        with tempfile.TemporaryDirectory() as reference_name, tempfile.TemporaryDirectory() as compact_name:
            reference_directory = Path(reference_name)
            compact_directory = Path(compact_name)
            write_parameters(
                reference_directory,
                excision_rate=excision_rate,
                append_gene=append_gene,
            )
            write_parameters(
                compact_directory,
                excision_rate=excision_rate,
                append_gene=append_gene,
            )
            run_simulator(REFERENCE_SIMULATOR, reference_directory, "--seed", seed)
            run_simulator(COMPACT_SIMULATOR, compact_directory, "--seed", seed)
            self.assertEqual(
                scientific_trace(reference_directory / "trace-009-001.csv"),
                scientific_trace(compact_directory / "trace-009-001.csv"),
            )

    def test_retrotransposon_trace_matches_reference(self):
        self.assert_backends_match(
            excision_rate=0.0,
            append_gene=True,
            seed=20260828,
        )

    def test_excision_and_gene_retry_trace_matches_reference(self):
        self.assert_backends_match(
            excision_rate=0.35,
            append_gene=False,
            seed=8675309,
        )

    def test_compact_checkpoint_resume_is_identical(self):
        with tempfile.TemporaryDirectory() as original_name, tempfile.TemporaryDirectory() as resumed_name:
            original = Path(original_name)
            resumed = Path(resumed_name)
            write_parameters(original)
            write_parameters(resumed)
            run_simulator(COMPACT_SIMULATOR, original, "--seed", 314159265)

            checkpoint_name = "state-009-001-0000003.gz"
            shutil.copy2(original / checkpoint_name, resumed / checkpoint_name)
            run_simulator(
                COMPACT_SIMULATOR,
                resumed,
                "--state",
                checkpoint_name,
            )

            original_rows = scientific_trace(original / "trace-009-001.csv")
            resumed_rows = scientific_trace(resumed / "trace-009-001.csv")
            generation_index = 0
            generation_four_onward = [
                original_rows[0],
                *[
                    row
                    for row in original_rows[1:]
                    if int(row[generation_index]) >= 4
                ],
            ]
            self.assertEqual(generation_four_onward, [resumed_rows[0], *resumed_rows[2:]])

            effect_fields = {
                "TEDEATH",
                "COLLISIO",
                "TOTAL_JU",
                "LETHAL_J",
                "DELETE_J",
                "NEUTRA_J",
                "BENEFI_J",
            }
            state_indices = [
                index
                for index, heading in enumerate(original_rows[0])
                if heading not in effect_fields
            ]
            original_generation_three = next(
                row for row in original_rows[1:] if int(row[generation_index]) == 3
            )
            resumed_generation_three = resumed_rows[1]
            self.assertEqual(
                [original_generation_three[index] for index in state_indices],
                [resumed_generation_three[index] for index in state_indices],
            )

            provenance = json.loads(
                (resumed / "provenance-009-001.json").read_text(encoding="utf-8")
            )
            self.assertEqual(provenance["simulation_backend"], "compact_numpy_v1")
            self.assertEqual(provenance["checkpoint_format"], "pickle_gzip_v1")
            self.assertEqual(provenance["initial_seed"], 314159265)

    def test_compact_backend_can_resume_reference_checkpoint(self):
        with tempfile.TemporaryDirectory() as reference_name, tempfile.TemporaryDirectory() as resumed_name:
            reference_directory = Path(reference_name)
            resumed_directory = Path(resumed_name)
            write_parameters(reference_directory)
            write_parameters(resumed_directory)
            run_simulator(
                REFERENCE_SIMULATOR,
                reference_directory,
                "--seed",
                271828182,
            )

            checkpoint_name = "state-009-001-0000003.gz"
            shutil.copy2(
                reference_directory / checkpoint_name,
                resumed_directory / checkpoint_name,
            )
            run_simulator(
                COMPACT_SIMULATOR,
                resumed_directory,
                "--state",
                checkpoint_name,
            )

            reference_rows = scientific_trace(
                reference_directory / "trace-009-001.csv"
            )
            resumed_rows = scientific_trace(resumed_directory / "trace-009-001.csv")
            reference_generation_four_onward = [
                reference_rows[0],
                *[row for row in reference_rows[1:] if int(row[0]) >= 4],
            ]
            self.assertEqual(
                reference_generation_four_onward,
                [resumed_rows[0], *resumed_rows[2:]],
            )


if __name__ == "__main__":
    unittest.main()
