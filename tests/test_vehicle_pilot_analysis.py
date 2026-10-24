"""Regression cases: inconsistent evidence must never produce a success report."""

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from pilot_evidence_fixture import fixture

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analyze_vehicle_sumo_pilot.py"


def load():
    if not SCRIPT.exists():
        raise AssertionError("analysis implementation does not exist yet")
    spec = importlib.util.spec_from_file_location("pilot_analysis", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def example(m):
    labels, rows = [], []
    for seed in [11, 23]:
        for index, (scenario, attack) in enumerate(m.SCENARIOS):
            for frame in [2, 3]:
                label = dict(
                    seed=seed,
                    scenario=scenario,
                    frame=frame,
                    episode_id=f"seed-{seed}-episode-{index:02d}",
                    time_s=frame * 0.2,
                    active=frame == 3,
                    is_attack=attack,
                )
                labels.append(label)
                for policy in m.POLICIES:
                    row = dict(
                        label,
                        policy=policy,
                        any_grant=True,
                        full_grant=True,
                        outcome="ALLOW",
                        mode="stream",
                        risk=0.1,
                        latency_ms=0.5,
                        display_mutated=False,
                        display_accepted=None,
                    )
                    rows.append(row)
    return labels, rows


class AnalysisTests(unittest.TestCase):
    def test_complete_file_evidence_passes(self):
        m = load()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _, _ = fixture(root, m)
            metrics, rows, payloads = m.verify_evidence(root, path)
            self.assertEqual(len(rows), metrics["policy_request_rows"])
            self.assertIn("input/raw/labels.csv", payloads)

    def test_changed_source_stops_analysis(self):
        m = load()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _, _ = fixture(root, m)
            (root / m.SOURCES[0]).write_text("changed source")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                m.verify_evidence(root, path)

    def test_changed_raw_data_stops_analysis(self):
        m = load()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, raw, _ = fixture(root, m)
            (raw / "labels.csv").write_text("changed labels")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                m.verify_evidence(root, path)

    def test_blank_or_null_declared_hash_cannot_disable_verification(self):
        m = load()
        for field in ("data_sha256", "source_sha256"):
            for bad in ("", None, False, 0):
                with self.subTest(field=field, bad=bad), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    path, _, _ = fixture(root, m)
                    metrics = json.loads(path.read_text())
                    key = next(iter(metrics[field]))
                    metrics[field][key] = bad
                    path.write_text(json.dumps(metrics))
                    with self.assertRaises(ValueError):
                        m.verify_evidence(root, path)

    def test_external_latest_symlink_keeps_target(self):
        m = load()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target"
            target.write_bytes(b"keep")
            latest = root / "latest"
            try:
                latest.symlink_to(target)
            except OSError:
                self.skipTest("symlinks unavailable")
            with self.assertRaises(ValueError):
                m.latest_copy(root, latest, b"changed")
            self.assertEqual(target.read_bytes(), b"keep")

    def test_complete_paired_rows_pass(self):
        m = load()
        labels, rows = example(m)
        m.check_pairing(rows, labels, [11, 23])

    def test_duplicate_rows_are_not_independent_samples(self):
        m = load()
        labels, rows = example(m)
        with self.assertRaises(ValueError):
            m.check_pairing(rows + [rows[0]], labels, [11, 23])

    def test_missing_policy_row_is_rejected(self):
        m = load()
        labels, rows = example(m)
        with self.assertRaises(ValueError):
            m.check_pairing(rows[1:], labels, [11, 23])

    def test_changed_attack_label_is_rejected(self):
        m = load()
        labels, rows = example(m)
        rows[0]["is_attack"] = True
        with self.assertRaises(ValueError):
            m.check_pairing(rows, labels, [11, 23])

    def test_duplicate_labels_are_rejected(self):
        m = load()
        labels, rows = example(m)
        with self.assertRaises(ValueError):
            m.check_pairing(rows, labels + [labels[0]], [11, 23])

    def test_nonfinite_csv_value_is_rejected(self):
        m = load()
        for bad in ["nan", "inf", "-inf"]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                m.number(bad)
        self.assertEqual(m.number("0.5"), 0.5)

    def test_boolean_strings_are_strict(self):
        m = load()
        self.assertFalse(m.boolean("False"))
        with self.assertRaises(ValueError):
            m.boolean("yes")

    def test_seed_bootstrap_preserves_cluster_unit(self):
        m = load()
        a = m.seed_interval([0, 1], "example", 1000)
        self.assertEqual(a["mean"], 0.5)
        self.assertEqual(a["traffic_seed_clusters"], 2)
        self.assertEqual(a, m.seed_interval([0, 1], "example", 1000))
        self.assertEqual(a["percentile_95"], [0, 1])

    def test_paired_difference_uses_matching_seed(self):
        m = load()
        result = m.paired_interval({11: 1.0, 23: 0.0}, {23: 0.0, 11: 1.0}, "paired", 100)
        self.assertEqual(result["mean"], 0)
        self.assertEqual(result["percentile_95"], [0, 0])
        with self.assertRaises(ValueError):
            m.paired_interval({11: 1.0}, {23: 1.0}, "bad", 100)

    def test_scenario_failure_is_retained(self):
        m = load()
        labels, rows = example(m)
        groups = m.group_rates(rows)
        rates = groups[("full", "coordinated_speed_spoof", "active_any_grant")]
        self.assertEqual(rates, {11: 1.0, 23: 1.0})

    def test_changed_aggregate_cannot_pass(self):
        m = load()
        labels, rows = example(m)
        expected = m.aggregate(rows)
        m.check_aggregate(rows, expected)
        changed = copy.deepcopy(expected)
        changed["full"]["requests"] += 1
        with self.assertRaises(ValueError):
            m.check_aggregate(rows, changed)

    def test_outside_project_path_is_rejected(self):
        m = load()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with self.assertRaises(ValueError):
                m.inside(root, root / "../other/data")
            self.assertEqual(m.inside(root, root / "data"), root / "data")


if __name__ == "__main__":
    unittest.main()
