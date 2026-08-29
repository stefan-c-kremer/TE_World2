import importlib.util
import csv
import json
import signal
import tempfile
import unittest
from unittest import mock
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
    @unittest.skipUnless(hasattr(signal, "SIGUSR1"), "requires SIGUSR1")
    def test_checkpoint_signal_is_forwarded_to_simulator_child(self):
        process = mock.Mock()
        process.poll.return_value = None
        registered = {}

        def register(signum, handler):
            if callable(handler):
                registered[signum] = handler

        def wait():
            registered[signal.SIGUSR1](signal.SIGUSR1, None)
            return 75

        process.wait.side_effect = wait
        with (
            mock.patch.object(nibi_runner.subprocess, "Popen", return_value=process),
            mock.patch.object(nibi_runner.signal, "getsignal", return_value=signal.SIG_DFL),
            mock.patch.object(nibi_runner.signal, "signal", side_effect=register),
        ):
            status = nibi_runner.run_with_checkpoint_forwarding(
                ["simulator"], cwd=Path("/tmp"), env={}
            )

        self.assertEqual(status, 75)
        process.send_signal.assert_called_once_with(signal.SIGUSR1)

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

    def test_manifest_selects_replicate_and_seed(self):
        with tempfile.TemporaryDirectory() as directory_name:
            root = Path(directory_name)
            experiment = root / "HHHHHHH-Z"
            experiment.mkdir()
            (experiment / "parameters.py").write_text("", encoding="utf-8")
            manifest = root / "manifest.csv"
            with manifest.open("w", newline="", encoding="utf-8") as destination:
                writer = csv.DictWriter(
                    destination,
                    fieldnames=[
                        "task_index",
                        "condition_code",
                        "replicate",
                        "run",
                        "seed",
                        "experiment_directory",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "task_index": 0,
                        "condition_code": "HHHHHHH-Z",
                        "replicate": "R02",
                        "run": 2,
                        "seed": 123456789,
                        "experiment_directory": "HHHHHHH-Z",
                    }
                )

            output = StringIO()
            with redirect_stdout(output):
                return_code = nibi_runner.main(
                    ["--manifest", str(manifest), "--index", "0", "--dry-run"]
                )

            self.assertEqual(return_code, 0)
            self.assertIn("replicate=R02", output.getvalue())
            self.assertIn("--seed 123456789", output.getvalue())

    def test_manifest_pending_indices_exclude_only_completed_runs(self):
        with tempfile.TemporaryDirectory() as directory_name:
            root = Path(directory_name)
            experiment = root / "HHHHHHH-Z"
            experiment.mkdir()
            (experiment / "parameters.py").write_text("", encoding="utf-8")
            manifest = root / "manifest.csv"
            manifest.write_text(
                "task_index,condition_code,replicate,run,seed,experiment_directory\n"
                "0,HHHHHHH-Z,R01,1,10,HHHHHHH-Z\n"
                "1,HHHHHHH-Z,R02,2,20,HHHHHHH-Z\n",
                encoding="utf-8",
            )
            (experiment / "provenance-001-001.json").write_text(
                json.dumps({"status": "maximum_generations"}), encoding="utf-8"
            )
            (experiment / "provenance-002-001.json").write_text(
                json.dumps({"status": "checkpointed"}), encoding="utf-8"
            )

            output = StringIO()
            with redirect_stdout(output):
                return_code = nibi_runner.main(
                    ["--manifest", str(manifest), "--pending-indices"]
                )

            self.assertEqual(return_code, 0)
            self.assertEqual(output.getvalue().strip(), "1")


if __name__ == "__main__":
    unittest.main()
