import importlib.util
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "context22_runner",
    Path(__file__).resolve().parents[1] / "scripts/audit_vehicle_context_stage22.py",
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class RunnerTests(unittest.TestCase):
    def test_stage21_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "file_hashes.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "manifest hash"):
                runner.verify_stage21(root)

    def test_existing_output_rejected_before_input_access(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "already exist"):
                runner.run("missing.zip", "missing21", temp)

    def test_wrong_source_creates_no_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "bad.zip").write_bytes(b"bad source")
            with self.assertRaisesRegex(ValueError, "hash"):
                runner.run(root / "bad.zip", "missing21", root / "output")
            self.assertFalse((root / "output").exists())


if __name__ == "__main__":
    unittest.main()
