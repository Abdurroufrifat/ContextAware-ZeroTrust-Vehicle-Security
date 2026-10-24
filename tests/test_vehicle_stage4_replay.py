import importlib.util
import sys
import unittest
from pathlib import Path

FILE = Path(__file__).resolve().parents[1] / "scripts/run_vehicle_stage4_replay.py"


def load():
    if not FILE.exists():
        raise AssertionError("replay runner does not exist")
    spec = importlib.util.spec_from_file_location("stage4_replay", FILE)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


class ReplayTests(unittest.TestCase):
    def test_paired_seed_comparison_reports_candidate_minus_stage1(self):
        m = load()
        rows = []
        for seed in [11, 23]:
            for policy, grant in [("stage1", True), ("temporal_degraded", False)]:
                rows.append(
                    dict(
                        seed=seed,
                        policy=policy,
                        scenario="coordinated_speed_spoof",
                        active=True,
                        is_attack=True,
                        any_grant=grant,
                        full_grant=grant,
                        display_mutated=False,
                        display_accepted=None,
                    )
                )
        seed_rows, deltas = m.seed_comparisons(rows)
        selected = [d for d in deltas if d["measure"] == "active_any_grant"][0]
        self.assertEqual(selected["mean"], -1.0)
        self.assertEqual(selected["traffic_seed_clusters"], 2)
        self.assertEqual(len(seed_rows), 8)

    def test_startup_bias_is_applied_before_enrollment(self):
        m = load()
        o = {
            "frame": 2,
            "timestamp_ms": 1000600,
            "signals": {
                "gnss": {"speed_mps": 10.0},
                "wheel": {"speed_mps": 10.0, "acceleration_mps2": 0.0},
                "imu": {"acceleration_mps2": 0.0},
            },
        }
        changed, transport, active = m.stress_observation(
            o, "startup_speed_bias", 0.6, 12.25, 22.75
        )
        self.assertTrue(active)
        self.assertEqual(transport, {})
        self.assertEqual(changed["signals"]["wheel"]["speed_mps"], 18.0)
        self.assertEqual(o["signals"]["wheel"]["speed_mps"], 10.0)

    def test_joint_motion_forgery_is_kinematically_coherent(self):
        m = load()
        o = {
            "frame": 2,
            "timestamp_ms": 1000600,
            "signals": {
                "gnss": {"speed_mps": 10.0},
                "wheel": {"speed_mps": 10.0, "acceleration_mps2": 0.0},
                "imu": {"acceleration_mps2": 0.0},
            },
        }
        changed, _, active = m.stress_observation(o, "coherent_motion_forgery", 14.25, 12.25, 22.75)
        self.assertTrue(active)
        self.assertEqual(changed["signals"]["gnss"]["speed_mps"], 20.0)
        self.assertEqual(changed["signals"]["imu"]["acceleration_mps2"], 5.0)
        self.assertEqual(changed["signals"]["wheel"]["acceleration_mps2"], 5.0)

    def test_archive_duplicate_and_traversal_names_rejected(self):
        m = load()
        for names in [["a", "a"], ["../x"], ["/x"], ["a\\x"]]:
            with self.subTest(names=names), self.assertRaises(ValueError):
                m.check_archive_names(names)

    def test_policy_outputs_do_not_depend_on_scenario_labels(self):
        m = load()
        o = sys.modules["vehicle_sumo_pilot_core"].make_observations(
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
        a = m.TrialHarness(m.TemporalConfig()).step(o, {}, True, "attack")["full"]
        b = m.TrialHarness(m.TemporalConfig()).step(o, {}, False, "normal")["full"]
        for field in ("outcome", "risk", "mode", "any_grant", "full_grant", "display_accepted"):
            self.assertEqual(a[field], b[field])


if __name__ == "__main__":
    unittest.main()
