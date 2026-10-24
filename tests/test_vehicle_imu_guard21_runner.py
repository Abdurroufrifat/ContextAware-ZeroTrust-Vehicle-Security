import importlib.util
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "imu_guard21_runner",
    Path(__file__).resolve().parents[1] / "scripts/run_vehicle_imu_guard_stage21.py",
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class RunnerTests(unittest.TestCase):
    def test_detection_loss_blocks_low_false_alarm_candidate(self):
        s = runner.screen(0.1, 0.03, 0.04, 0.8, 0.9, 0.08)
        self.assertFalse(s["passed"])
        self.assertFalse(s["eligible_for_test"])

    def test_primary_cap_is_fixed_and_not_deployable(self):
        self.assertEqual(runner.PRIMARY, "context_bounded")
        self.assertEqual(runner.PRIMARY_CAP, 1.0)

    def test_wrong_source_rejected_without_output(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            (p / "bad.zip").write_bytes(b"bad source")
            with self.assertRaisesRegex(ValueError, "hash"):
                runner.run(p / "bad.zip", p / "result")
            self.assertFalse((p / "result").exists())


if __name__ == "__main__":
    unittest.main()
