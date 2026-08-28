import importlib.util
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = (
    REPOSITORY_ROOT
    / "TEWorldCodeV2"
    / "TE-Simulations"
    / "RunNibiArrayTask.py"
)

spec = importlib.util.spec_from_file_location("nibi_runner", RUNNER_PATH)
nibi_runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nibi_runner)


class NibiRunnerTests(unittest.TestCase):
    def test_experiments_are_mapped_to_stable_sorted_indices(self):
        with tempfile.TemporaryDirectory() as directory_name:
            root = Path(directory_name)
            for name in ["IS-Z-EXP", "IS-A-EXP", "not-an-experiment"]:
                directory = root / name
                directory.mkdir()
                if name != "not-an-experiment":
                    (directory / "parameters.py").write_text("", encoding="utf-8")

            self.assertEqual(
                [path.name for path in nibi_runner.experiment_directories(root)],
                ["IS-A-EXP", "IS-Z-EXP"],
            )

    def test_latest_checkpoint_is_selected_by_generation_then_iteration(self):
        with tempfile.TemporaryDirectory() as directory_name:
            experiment = Path(directory_name)
            for name in [
                "state-004-003-0000050.gz",
                "state-004-001-0000100.gz",
                "state-004-002-0000100.gz",
                "state-005-009-0009999.gz",
            ]:
                (experiment / name).touch()

            self.assertEqual(
                nibi_runner.latest_checkpoint(experiment, 4).name,
                "state-004-002-0000100.gz",
            )

    def test_completed_provenance_prevents_duplicate_task(self):
        with tempfile.TemporaryDirectory() as directory_name:
            root = Path(directory_name)
            experiment = root / "IS-A-EXP"
            experiment.mkdir()
            (experiment / "parameters.py").write_text("", encoding="utf-8")
            (experiment / "provenance-004-001.json").write_text(
                json.dumps({"status": "maximum_generations"}),
                encoding="utf-8",
            )

            output = StringIO()
            with redirect_stdout(output):
                return_code = nibi_runner.main(
                    [
                        "--run",
                        "4",
                        "--index",
                        "0",
                        "--experiment-root",
                        str(root),
                        "--dry-run",
                    ]
                )

            self.assertEqual(return_code, 0)
            self.assertIn("Skipping completed experiment", output.getvalue())

    def test_compact_backend_is_default_and_reference_is_selectable(self):
        with tempfile.TemporaryDirectory() as directory_name:
            root = Path(directory_name)
            experiment = root / "IS-A-EXP"
            experiment.mkdir()
            (experiment / "parameters.py").write_text("", encoding="utf-8")

            for extra_arguments, expected_launcher in [
                ([], "TESimCompact.py"),
                (["--backend", "reference"], "TESim.py"),
            ]:
                output = StringIO()
                with redirect_stdout(output):
                    return_code = nibi_runner.main(
                        [
                            "--run",
                            "4",
                            "--index",
                            "0",
                            "--experiment-root",
                            str(root),
                            "--dry-run",
                            *extra_arguments,
                        ]
                    )
                self.assertEqual(return_code, 0)
                self.assertIn(expected_launcher, output.getvalue())


if __name__ == "__main__":
    unittest.main()
