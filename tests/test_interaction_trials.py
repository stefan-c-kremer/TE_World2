import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = (
    REPOSITORY_ROOT
    / "TEWorldCodeV2"
    / "TE-Simulations"
    / "CreateInteractionTrials.py"
)
spec = importlib.util.spec_from_file_location("interaction_trials", GENERATOR_PATH)
interaction_trials = importlib.util.module_from_spec(spec)
spec.loader.exec_module(interaction_trials)


class InteractionTrialTests(unittest.TestCase):
    def test_z_phase_contains_128_conditions_and_three_replicates(self):
        with tempfile.TemporaryDirectory() as directory_name:
            root = Path(directory_name)
            manifest, metadata = interaction_trials.generate(
                root, ["Z"], 3, "test-master-seed", "manifest-z-r3.csv"
            )

            parameter_files = sorted(root.glob("*/parameters.py"))
            self.assertEqual(len(parameter_files), 128)
            with manifest.open(newline="", encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(len(rows), 384)
            self.assertEqual([row["task_index"] for row in rows], [str(i) for i in range(384)])
            self.assertEqual(len({row["seed"] for row in rows}), 384)
            self.assertEqual({row["replicate"] for row in rows}, {"R01", "R02", "R03"})
            self.assertTrue(all(row["condition_code"].endswith("-Z") for row in rows))

            record = json.loads(metadata.read_text(encoding="utf-8"))
            self.assertEqual(record["condition_count"], 128)
            self.assertEqual(record["simulation_count"], 384)

    def test_parameter_order_and_z_control_values(self):
        source = interaction_trials.parameter_source("HLHLHLH", "Z")
        self.assertIn("TE_progeny = ProbabilityTable(0.00", source)
        self.assertIn("TE_death_rate = 0.005", source)
        self.assertIn("TE_Insertion_Distribution = Triangle", source)
        self.assertIn("Initial_genes = 500", source)
        self.assertIn("Junk_BP = 14000000", source)
        self.assertIn("random.random() * 0.01", source)
        self.assertIn("Carrying_capacity = 300", source)
        self.assertIn("Initial_NAut_TEs = 0", source)
        self.assertIn("1 + 0.0 * live_naut", source)

    def test_seed_derivation_is_stable_and_identity_specific(self):
        first = interaction_trials.deterministic_seed("master", "HHHHHHH-Z", 1)
        self.assertEqual(
            first,
            interaction_trials.deterministic_seed("master", "HHHHHHH-Z", 1),
        )
        self.assertNotEqual(
            first,
            interaction_trials.deterministic_seed("master", "HHHHHHH-Z", 2),
        )
        self.assertNotEqual(
            first,
            interaction_trials.deterministic_seed("master", "LHHHHHH-Z", 1),
        )


if __name__ == "__main__":
    unittest.main()
