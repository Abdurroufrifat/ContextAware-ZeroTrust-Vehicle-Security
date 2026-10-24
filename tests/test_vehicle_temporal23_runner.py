import importlib.util
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "temporal23_runner",
    Path(__file__).resolve().parents[1] / "scripts/evaluate_vehicle_temporal_stage23.py",
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class RunnerTests(unittest.TestCase):
    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "already exist"):
                runner.run("missing.zip", "missing21", temp)

    def test_wrong_source_produces_no_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "bad.zip").write_bytes(b"bad source")
            with self.assertRaisesRegex(ValueError, "hash"):
                runner.run(root / "bad.zip", "missing21", root / "result")
            self.assertFalse((root / "result").exists())


if __name__ == "__main__":
    unittest.main()
