import importlib.util
import sys
import unittest
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
FILE = BASE / "scripts/run_vehicle_stage5_study.py"


def module():
    if not FILE.exists():
        raise AssertionError("study runner does not exist")
    spec = importlib.util.spec_from_file_location("stage5_runner", FILE)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


class RunnerTests(unittest.TestCase):
    def test_smoke_has_no_fabricated_interval(self):
        m = module()
        stats = {(11, "stage1", "normal", "active"): [4, 4, 4, 0, 0]}
        summary, deltas, seed_rows = m.summarize_stats(stats, "free_flow")
        self.assertIsNone(summary[0]["percentile_95"])
        self.assertEqual(summary[0]["traffic_seed_clusters"], 1)
        self.assertEqual(deltas, [])
        self.assertTrue(seed_rows)

    def test_seed_and_phase_counts_preserve_drop_recovery_cost(self):
        m = module()
        stats = {}
        row = dict(
            seed=11,
            policy="temporal_degraded",
            scenario="benign_imu_dropout",
            phase="after",
            any_grant=False,
            full_grant=False,
            display_mutated=False,
            display_accepted=None,
        )
        m.update_stats(stats, row)
        m.update_stats(stats, row)
        summary, _, _ = m.summarize_stats(stats, "dense")
        self.assertEqual(summary[0]["mean"], 0)
        self.assertEqual(summary[0]["requests"], 2)

    def test_signed_jitter_offsets_all_sources_without_labels(self):
        m = module()
        o = m.core.make_observations(
            [
                dict(
                    frame=0,
                    time_s=0.2,
                    x_m=0.0,
                    y_m=0.0,
                    speed_mps=0.0,
                    acceleration_mps2=0.0,
                    angle_deg=0.0,
                )
            ],
            11,
        )[0]
        h = m.StudyHarness(None)
        p = h.packets(o, 1, "f1", 1000, {"source_delays_ms": {s: 10 for s in m.core.SOURCES}})
        for packet in p.values():
            body = m.analysis.json.loads(packet)["body"]
            self.assertEqual(body["timestamp_ms"], 990)
            self.assertNotIn("scenario", body)


if __name__ == "__main__":
    unittest.main()
