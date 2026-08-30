import importlib.util
import tempfile
import unittest
import subprocess
from unittest import mock
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TRACKER_PATH = (
    REPOSITORY_ROOT
    / "TEWorldCodeV2"
    / "TE-Simulations"
    / "TrackNibiResources.py"
)
spec = importlib.util.spec_from_file_location("resource_tracker", TRACKER_PATH)
resource_tracker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(resource_tracker)


class ResourceTrackingTests(unittest.TestCase):
    def test_array_index_ranges_and_throttle_are_parsed(self):
        self.assertEqual(
            resource_tracker.parse_indices("1,3-5,9%4"),
            [1, 3, 4, 5, 9],
        )

    def test_slurm_memory_units_are_binary(self):
        self.assertEqual(resource_tracker.parse_bytes("851296K"), 851296 * 1024)
        self.assertEqual(resource_tracker.parse_bytes("4G"), 4 * 1024**3)
        self.assertEqual(resource_tracker.parse_bytes(""), "")

    def test_failure_classification_prefers_specific_causes(self):
        self.assertEqual(
            resource_tracker.classify_failure(
                "Detected 1 oom_kill event", "OUT_OF_MEMORY", "running"
            ),
            "out_of_memory",
        )
        self.assertEqual(
            resource_tracker.classify_failure(
                "CANCELLED DUE TO TIME LIMIT", "TIMEOUT", "running"
            ),
            "time_limit",
        )
        self.assertEqual(
            resource_tracker.classify_failure("", "COMPLETED", "host_extinction"),
            "none",
        )

    def test_submission_records_are_idempotently_updated(self):
        with tempfile.TemporaryDirectory() as directory_name:
            path = Path(directory_name) / "submissions.csv"
            first = {field: "" for field in resource_tracker.SUBMISSION_FIELDS}
            first.update({"job_id": "123", "memory": "4G"})
            resource_tracker.upsert(
                path,
                resource_tracker.SUBMISSION_FIELDS,
                first,
                ("job_id",),
            )
            second = dict(first, memory="16G")
            resource_tracker.upsert(
                path,
                resource_tracker.SUBMISSION_FIELDS,
                second,
                ("job_id",),
            )
            rows = resource_tracker.read_csv(path)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["memory"], "16G")

    def test_accounting_falls_back_when_array_fields_are_unavailable(self):
        legacy_output = (
            "20793821_7|COMPLETED|0:0|42|1024K|2048K|4G|1800|base|c123\n"
            "20793821_7.batch|COMPLETED|0:0|42|3072K|4096K|4G|1800|base|c123\n"
        )
        with mock.patch.object(
            resource_tracker.subprocess,
            "run",
            side_effect=[
                subprocess.CalledProcessError(1, ["sacct"]),
                mock.Mock(stdout=legacy_output),
            ],
        ) as run:
            records = resource_tracker.accounting(["20793821"])

        self.assertEqual(run.call_count, 2)
        self.assertEqual(records[("20793821", "7")]["scheduler_state"], "COMPLETED")
        self.assertEqual(records[("20793821", "7")]["elapsed_seconds"], "42")
        self.assertEqual(
            records[("20793821", "7")]["max_rss_bytes"],
            str(3072 * 1024),
        )


if __name__ == "__main__":
    unittest.main()
