import csv
import gzip
import hashlib
import json
import shutil
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest import mock
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SIMULATOR = REPOSITORY_ROOT / "TEWorldCodeV2" / "TESim.py"
CODE_DIRECTORY = SIMULATOR.parent


PARAMETERS = """
from TEUtil import *

output = {}
TE_Insertion_Distribution = Triangle(pmax=0, pzero=2.0 / 3.0)
Gene_Insertion_Distribution = Triangle(pzero=1.0 / 3.0, pmax=1)
Gene_length = 40
TE_length = lambda autonomous: 60 if autonomous else 30
TE_death_rate = 0.15
TE_excision_rate = 0.0
TE_progeny = ProbabilityTable(0.30, 0, 0.50, 1, 0.20, 2)
Initial_genes = 4
Append_gene = True
Junk_BP = 5000
Host_start_fitness = 1.0
Host_mutation_rate = 0.05
Host_mutation = ProbabilityTable(1.0, lambda fit: fit)
Insertion_effect = ProbabilityTable(1.0, lambda fit: fit)
Carrying_capacity = 8
Host_reproduction_rate = 1
Host_survival_rate = lambda propfit: min(Carrying_capacity * propfit, 0.95)
Initial_Aut_TEs = 2
Initial_NAut_TEs = 2
Maximum_generations = 4
Terminate_no_TEs = False
seed = None
save_frequency = 50
saved = None
Kidnapping_frequency = lambda live_aut, live_naut: 1 - 1 / (1 + 0.07 * live_naut)
"""


def write_parameters(directory: Path, saved=None) -> None:
    source = textwrap.dedent(PARAMETERS)
    if saved is not None:
        source = source.replace("saved = None", f"saved = {saved!r}")
    (directory / "parameters.py").write_text(source, encoding="utf-8")


def run_simulation(directory: Path, seed=None) -> subprocess.CompletedProcess:
    command = [sys.executable, str(SIMULATOR), "7", "reproducibility-test"]
    if seed is not None:
        command.extend(["--seed", str(seed)])
    return subprocess.run(
        command,
        cwd=directory,
        check=True,
        capture_output=True,
        text=True,
    )


def scientific_trace(path: Path):
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.reader(source, skipinitialspace=True))
    # The first column is wall-clock performance data and is intentionally not
    # deterministic. Every biological/statistical field must be identical.
    return [row[1:] for row in rows]


class ReproducibilityTests(unittest.TestCase):
    def test_slurm_environment_is_recorded_when_present(self):
        sys.path.insert(0, str(CODE_DIRECTORY))
        try:
            import provenance
        finally:
            sys.path.pop(0)

        with mock.patch.dict(
            os.environ,
            {
                "SLURM_JOB_ID": "12345",
                "SLURM_ARRAY_JOB_ID": "12300",
                "SLURM_ARRAY_TASK_ID": "17",
                "SLURM_CLUSTER_NAME": "nibi",
                "SLURM_CPUS_PER_TASK": "1",
            },
            clear=False,
        ):
            self.assertEqual(
                provenance.slurm_context(),
                {
                    "SLURM_JOB_ID": "12345",
                    "SLURM_ARRAY_JOB_ID": "12300",
                    "SLURM_ARRAY_TASK_ID": "17",
                    "SLURM_CLUSTER_NAME": "nibi",
                    "SLURM_CPUS_PER_TASK": "1",
                },
            )

    def test_generated_seed_can_replay_identical_scientific_trace(self):
        with tempfile.TemporaryDirectory() as first_name, tempfile.TemporaryDirectory() as replay_name:
            first = Path(first_name)
            replay = Path(replay_name)
            write_parameters(first)
            write_parameters(replay)

            run_simulation(first)
            provenance = json.loads(
                (first / "provenance-007-001.json").read_text(encoding="utf-8")
            )
            seed = provenance["initial_seed"]
            self.assertIsInstance(seed, int)
            self.assertEqual(provenance["seed_source"], "generated")
            self.assertEqual(provenance["status"], "maximum_generations")
            expected_commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=REPOSITORY_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            self.assertEqual(provenance["code"]["git_commit"], expected_commit)
            self.assertIn("Initial_Aut_TEs = 2", provenance["parameters"]["source"])

            run_simulation(replay, seed=seed)

            self.assertEqual(
                scientific_trace(first / "trace-007-001.csv"),
                scientific_trace(replay / "trace-007-001.csv"),
            )

            replay_provenance = json.loads(
                (replay / "provenance-007-001.json").read_text(encoding="utf-8")
            )
            self.assertEqual(replay_provenance["initial_seed"], seed)
            self.assertEqual(replay_provenance["seed_source"], "command_line")
            self.assertEqual(
                replay_provenance["parameters"]["sha256"],
                provenance["parameters"]["sha256"],
            )

    def test_seeded_scientific_trace_matches_reference_baseline(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            write_parameters(directory)
            run_simulation(directory, seed=123456789)
            rows = scientific_trace(directory / "trace-007-001.csv")
            payload = "\n".join(",".join(row) for row in rows).encode("utf-8")
            self.assertEqual(
                hashlib.sha256(payload).hexdigest(),
                "f00e61e0123e4c116166822bd1d9c781f61672d2ae3d92b0fd99f26e5193c974",
            )

    def test_checkpoint_records_initial_seed(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            write_parameters(directory)
            run_simulation(directory, seed=8675309)

            state_path = directory / "state-007-001-0000000.gz"
            with gzip.open(state_path, "rt", encoding="utf-8") as state_file:
                state = state_file.read()
            self.assertIn("self.initial_seed = 8675309;", state)

    def test_resume_preserves_seed_and_random_state(self):
        with tempfile.TemporaryDirectory() as original_name, tempfile.TemporaryDirectory() as resumed_name:
            original = Path(original_name)
            resumed = Path(resumed_name)
            write_parameters(original)
            run_simulation(original, seed=314159265)

            state_name = "state-007-001-0000000.gz"
            shutil.copy2(original / state_name, resumed / state_name)
            write_parameters(resumed, saved=state_name)
            run_simulation(resumed)

            resumed_provenance = json.loads(
                (resumed / "provenance-007-001.json").read_text(encoding="utf-8")
            )
            self.assertEqual(resumed_provenance["initial_seed"], 314159265)
            self.assertEqual(resumed_provenance["seed_source"], "checkpoint")
            self.assertEqual(
                scientific_trace(original / "trace-007-001.csv"),
                scientific_trace(resumed / "trace-007-001.csv"),
            )

    def test_te_copy_preserves_autonomy_and_length(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            write_parameters(directory)
            program = textwrap.dedent(
                f"""
                import sys
                sys.path.insert(0, {str(CODE_DIRECTORY)!r})
                import TESim
                original = TESim.SelectiveInsertTE(123, True, False)
                original.chromosome = None
                copied = original.copy()
                print(original.start, original.dead, original.autonomous, original.length)
                print(copied.start, copied.dead, copied.autonomous, copied.length)
                """
            )
            result = subprocess.run(
                [sys.executable, "-c", program],
                cwd=directory,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                result.stdout.splitlines(),
                ["123 True False 30", "123 True False 30"],
            )

    def test_incremental_live_te_counts_match_element_state(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            write_parameters(directory)
            program = textwrap.dedent(
                f"""
                import sys
                sys.path.insert(0, {str(CODE_DIRECTORY)!r})
                import TESim
                experiment = TESim.Experiment(seed=24680)
                for generation in range(5):
                    for individual in experiment.pop.individual:
                        chromosome = individual.chromosome[0]
                        autonomous = sum(
                            1 for te in chromosome.TEs(live=True, dead=False)
                            if te.autonomous
                        )
                        non_autonomous = sum(
                            1 for te in chromosome.TEs(live=True, dead=False)
                            if not te.autonomous
                        )
                        assert chromosome._live_autonomous_tes == autonomous
                        assert chromosome._live_non_autonomous_tes == non_autonomous
                        assert all(
                            first.start <= second.start
                            for first, second in zip(
                                chromosome.elements, chromosome.elements[1:]
                            )
                        )
                    if generation < 4:
                        experiment.pop.generation()
                """
            )
            subprocess.run(
                [sys.executable, "-c", program],
                cwd=directory,
                check=True,
                capture_output=True,
                text=True,
            )

    def test_gene_collision_append_mode_preserves_sorted_non_overlapping_genes(self):
        self._assert_gene_collision_mode(
            append_gene=True,
            samples=[0.1, 0.1],
            expected_starts="500 540",
        )

    def test_gene_collision_retry_mode_does_not_modify_rejected_attempt(self):
        self._assert_gene_collision_mode(
            append_gene=False,
            samples=[0.1, 0.1, 0.2],
            expected_starts="500 1008",
        )

    def test_interval_lookup_preserves_earliest_overlapping_element(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            write_parameters(directory)
            program = textwrap.dedent(
                f"""
                import sys
                sys.path.insert(0, {str(CODE_DIRECTORY)!r})
                import TESim

                first = TESim.Element(length=40, start=100)
                second = TESim.Element(length=30, start=110)
                chromosome = TESim.Chromosome(elements=[first, second])
                assert chromosome[115] is first
                assert chromosome[139] is first
                assert chromosome[140] is TESim.JUNK
                assert chromosome[99] is TESim.JUNK
                """
            )
            subprocess.run(
                [sys.executable, "-c", program],
                cwd=directory,
                check=True,
                capture_output=True,
                text=True,
            )

    def _assert_gene_collision_mode(self, append_gene, samples, expected_starts):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            write_parameters(directory)
            program = textwrap.dedent(
                f"""
                import sys
                sys.path.insert(0, {str(CODE_DIRECTORY)!r})
                import TESim

                class FixedDistribution:
                    def __init__(self, values):
                        self.values = iter(values)
                    def sample(self):
                        return next(self.values)

                TESim.parameters.Append_gene = {append_gene!r}
                TESim.parameters.Gene_Insertion_Distribution = FixedDistribution({samples!r})
                chromosome = TESim.TestChromosome2(length=5000)
                chromosome.initial_aut_tes = 0
                chromosome.initial_naut_tes = 0
                chromosome.add_elements(genes=2)
                genes = chromosome.genes()
                assert all(a.start <= b.start for a, b in zip(genes, genes[1:]))
                assert all(a.end <= b.start for a, b in zip(genes, genes[1:]))
                print(" ".join(str(gene.start) for gene in genes))
                print(chromosome.length)
                """
            )
            result = subprocess.run(
                [sys.executable, "-c", program],
                cwd=directory,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.stdout.splitlines(), [expected_starts, "5080"])


if __name__ == "__main__":
    unittest.main()
