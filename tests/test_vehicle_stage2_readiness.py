from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts/audit_vehicle_stage2_readiness.py"


def load_audit():
    spec = importlib.util.spec_from_file_location("vehicle_readiness", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReadinessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "CO-ACT-CPS"
        (self.root / "data/raw").mkdir(parents=True)

    def tearDown(self):
        self.temp.cleanup()

    def write_csv(self, name, columns, rows):
        path = self.root / "data/raw" / name
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(columns)
            writer.writerows(rows)
        return path

    def test_industrial_headers_do_not_establish_vehicle_context(self):
        audit = load_audit()
        path = self.write_csv(
            "hai.csv", ["timestamp", "P1_FCV01D", "attack"], [["2026-01-01", 1, 0]]
        )
        result = audit.inspect_file(path, self.root)
        self.assertEqual(result["vehicle_domains"], [])
        self.assertFalse(result["full_flat_schema_candidate"])

    def test_can_only_capture_cannot_establish_multisource_readiness(self):
        audit = load_audit()
        path = self.write_csv(
            "can.csv",
            ["timestamp", "can_id", "can_dlc", "can_data"],
            [[1, "0x123", 8, "0011223344556677"]],
        )
        result = audit.inspect_file(path, self.root)
        self.assertEqual(result["vehicle_domains"], ["can"])
        self.assertFalse(result["full_flat_schema_candidate"])

    def test_complete_schema_is_still_only_a_candidate(self):
        audit = load_audit()
        row = {name: 0 for name in audit.REQUIRED_FLAT_FIELDS}
        row.update(
            drive_id="drive-1",
            frame_id="f-1",
            vehicle_id="v-1",
            timestamp_ms=1000,
            subject_id="s-1",
            credential_valid=True,
            revoked=False,
            attestation_ok=True,
        )
        path = self.write_csv("vehicle.csv", list(row), [list(row.values())])
        result = audit.inspect_file(path, self.root)
        self.assertTrue(result["full_flat_schema_candidate"])
        self.assertEqual(result["quality_scope"], "bounded schema/sample check only")
        self.assertEqual(result["sample_issues"], [])

    def test_bad_scores_and_duplicate_frames_are_reported(self):
        audit = load_audit()
        row = {name: 0 for name in audit.REQUIRED_FLAT_FIELDS}
        row.update(
            drive_id="drive-1",
            frame_id="f-1",
            vehicle_id="v-1",
            timestamp_ms=1000,
            subject_id="s-1",
            credential_valid=True,
            revoked=False,
            attestation_ok=True,
            can_anomaly_score=2.0,
        )
        path = self.write_csv("bad.csv", list(row), [list(row.values()), list(row.values())])
        issues = audit.inspect_file(path, self.root)["sample_issues"]
        self.assertIn("out_of_range:can_anomaly_score", issues)
        self.assertIn("duplicate_drive_vehicle_frame", issues)

    def test_headerless_and_broken_json_are_not_accepted_as_schema(self):
        audit = load_audit()
        path = self.root / "data/raw/headerless.csv"
        path.write_text("1,2,3,4\n5,6,7,8\n")
        self.assertEqual(
            audit.inspect_file(path, self.root)["schema_status"], "headerless_or_ambiguous"
        )
        path = self.root / "data/raw/broken.json"
        path.write_text("{")
        self.assertEqual(audit.inspect_file(path, self.root)["schema_status"], "unreadable_schema")

    def test_jsonl_nested_signal_schema_is_detected_without_exporting_values(self):
        audit = load_audit()
        path = self.root / "data/raw/telemetry.jsonl"
        path.write_text(
            json.dumps(
                {
                    "timestamp_ms": 1000,
                    "gnss": {"speed_mps": 10},
                    "imu": {"acceleration_mps2": 1, "yaw_rate_rps": 0.1},
                }
            )
            + "\n"
        )
        result = audit.inspect_file(path, self.root)
        self.assertEqual(result["vehicle_domains"], ["gnss", "imu"])
        self.assertNotIn("sample_values", result)

    def test_split_source_files_are_not_silently_joined(self):
        audit = load_audit()
        self.write_csv("gnss.csv", ["timestamp_ms", "gnss_speed_mps"], [[1000, 10]])
        self.write_csv(
            "imu.csv", ["timestamp_ms", "imu_acceleration_mps2", "imu_yaw_rate_rps"], [[1000, 0, 0]]
        )
        report = audit.audit_data(self.root, max_files=100)
        self.assertEqual(report["full_flat_schema_candidates"], [])
        self.assertEqual(
            report["decision"], "PARTIAL_VEHICLE_SOURCES_REQUIRE_MAPPING_AND_ALIGNMENT"
        )

    def test_scan_limit_is_reported(self):
        audit = load_audit()
        for i in range(3):
            self.write_csv(f"{i}.csv", ["timestamp", "P1"], [[0, 1]])
        report = audit.audit_data(self.root, max_files=1)
        self.assertFalse(report["scan_complete"])
        self.assertEqual(len(report["files"]), 1)

    def test_freeze_preserves_matching_source_and_reports_drift(self):
        audit = load_audit()
        source = self.root / "scripts/example.py"
        source.parent.mkdir()
        source.write_bytes(b"print(1)\r\n")
        metrics_dir = self.root / "artifacts/vehicle_zero_trust_stage1"
        metrics_dir.mkdir(parents=True)
        metrics = {
            "source_sha256": {
                "D:\\CO-ACT-CPS\\scripts\\example.py": hashlib.sha256(
                    source.read_bytes()
                ).hexdigest()
            }
        }
        (metrics_dir / "metrics.json").write_text(json.dumps(metrics))
        output = self.root / "artifacts/audit-run"
        output.mkdir()
        result = audit.freeze_stage1(self.root, output)
        self.assertTrue(result["all_recorded_sources_match"])
        self.assertEqual(
            (output / "stage1_evidence/source_snapshot/scripts/example.py").read_bytes(),
            source.read_bytes(),
        )
        source.write_text("print(2)")
        output2 = self.root / "artifacts/audit-run2"
        output2.mkdir()
        result = audit.freeze_stage1(self.root, output2)
        self.assertFalse(result["all_recorded_sources_match"])

    def test_manifest_paths_cannot_escape_project_root(self):
        audit = load_audit()
        for name in (
            "../secret.txt",
            "D:\\other_project\\secret.txt",
            "src/../../secret.txt",
            "/etc/passwd",
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                audit.source_path(self.root, name)

    def test_missing_data_root_is_explicit(self):
        audit = load_audit()
        report = audit.audit_data(self.root / "missing", max_files=100)
        self.assertEqual(report["decision"], "DATA_DIRECTORY_MISSING")
        self.assertFalse(report["scan_complete"])

    def test_data_link_cannot_scan_another_project(self):
        audit = load_audit()
        linked = self.root / "linked-project"
        linked.mkdir()
        try:
            os.symlink(self.root / "data", linked / "data", target_is_directory=True)
        except OSError:
            self.skipTest("symlink creation unavailable")
        report = audit.audit_data(linked, max_files=100)
        self.assertEqual(report["decision"], "DATA_PATH_OUTSIDE_PROJECT")
        self.assertEqual(report["files"], [])

    def test_linked_metrics_outside_root_are_not_read(self):
        audit = load_audit()
        outside = self.root.parent / "outside.json"
        outside.write_text("{}")
        metrics_dir = self.root / "artifacts/vehicle_zero_trust_stage1"
        metrics_dir.mkdir(parents=True)
        try:
            os.symlink(outside, metrics_dir / "metrics.json")
        except OSError:
            self.skipTest("symlink creation unavailable")
        output = self.root / "artifacts/new-run"
        output.mkdir()
        result = audit.freeze_stage1(self.root, output)
        self.assertEqual(result["error"], "stage1_metrics_path_outside_project")

    def test_native_absolute_source_inside_root_is_supported(self):
        audit = load_audit()
        source = self.root / "scripts/example.py"
        source.parent.mkdir()
        source.write_text("print(1)")
        self.assertEqual(audit.source_path(self.root, str(source)), source.resolve())

    def test_invalid_stage1_metrics_still_return_an_explicit_audit_status(self):
        audit = load_audit()
        metrics_dir = self.root / "artifacts/vehicle_zero_trust_stage1"
        metrics_dir.mkdir(parents=True)
        output = self.root / "artifacts/new-run"
        output.mkdir()
        for text in ("{", "[]", '{"source_sha256":[]}'):
            (metrics_dir / "metrics.json").write_text(text)
            with self.subTest(text=text):
                result = audit.freeze_stage1(self.root, output)
                self.assertEqual(result["error"], "invalid_stage1_metrics")


if __name__ == "__main__":
    unittest.main()
