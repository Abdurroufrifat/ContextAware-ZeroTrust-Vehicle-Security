import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "scripts"))
spec = importlib.util.spec_from_file_location(
    "sumo_runner", BASE / "scripts/run_vehicle_sumo_pilot.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((BASE / "configs/vehicle_sumo_stage3.json").read_text())

    def test_default_protocol_valid(self):
        runner.validate_config(self.config, "smoke")
        runner.validate_config(self.config, "pilot")

    def test_duplicate_seeds_and_invalid_windows_rejected(self):
        for change in ("duplicate", "empty", "bool", "window", "nan", "step", "warmup"):
            cfg = copy.deepcopy(self.config)
            if change == "duplicate":
                cfg["smoke"]["seeds"] = [11, 11]
            if change == "empty":
                cfg["smoke"]["seeds"] = []
            if change == "bool":
                cfg["smoke"]["seeds"] = [True]
            if change == "window":
                cfg["window_fraction"] = [0.5, 0.5]
            if change == "nan":
                cfg["smoke"]["duration_s"] = float("nan")
            if change == "step":
                cfg["smoke"]["step_s"] = 0
            if change == "warmup":
                cfg["warmup_s"] = 10
            with self.subTest(change=change), self.assertRaises(ValueError):
                runner.validate_config(cfg, "smoke")

    def test_imported_sources_must_match_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "src/coact_cps/types.py"
            source.parent.mkdir(parents=True)
            source.write_text("example")
            runner.check_source_root(root, {"src/coact_cps/types.py": source})
            with self.assertRaises(ValueError):
                runner.check_source_root(
                    root, {"src/coact_cps/types.py": source.with_name("other.py")}
                )

    def test_latest_symlinks_rejected_without_changing_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.json"
            target.write_text("keep")
            latest = root / "latest.json"
            try:
                latest.symlink_to(target)
            except OSError:
                self.skipTest("symlink creation unavailable")
            with self.assertRaises(ValueError):
                runner.write_latest(latest, "replace", root)
            self.assertEqual(target.read_text(), "keep")

    def test_latest_atomic_write_and_outside_root_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            latest = root / "latest.json"
            runner.write_latest(latest, "ok", root)
            self.assertEqual(latest.read_text(), "ok")
            with self.assertRaises(ValueError):
                runner.write_latest(root.parent / "outside.json", "bad", root)


if __name__ == "__main__":
    unittest.main()
