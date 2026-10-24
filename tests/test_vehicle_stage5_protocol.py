import copy
import importlib.util
import json
import unittest
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
SCRIPT = BASE / "scripts/vehicle_stage5_protocol.py"


def module():
    if not SCRIPT.exists():
        raise AssertionError("Stage 5 protocol does not exist")
    spec = importlib.util.spec_from_file_location("study_protocol", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def observation():
    return dict(
        frame=10,
        timestamp_ms=1002000,
        signals=dict(
            imu=dict(acceleration_mps2=0.0, yaw_rate_rps=0.0),
            wheel=dict(speed_mps=10.0, acceleration_mps2=0.0, yaw_rate_rps=0.0),
            gnss=dict(speed_mps=10.0),
        ),
    )


class ProtocolTests(unittest.TestCase):
    def test_fixed_config_accepted(self):
        m = module()
        m.validate_config(json.loads((BASE / "configs/vehicle_stage5.json").read_text()))

    def test_duplicate_or_overlapping_seed_rejected(self):
        m = module()
        cfg = json.loads((BASE / "configs/vehicle_stage5.json").read_text())
        for bad in [True, 11, 201, 131, 101]:
            changed = copy.deepcopy(cfg)
            changed["profiles"][0]["seeds"][1] = bad
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                m.validate_config(changed)

    def test_invalid_profile_cannot_escape_or_disable_simulation(self):
        m = module()
        cfg = json.loads((BASE / "configs/vehicle_stage5.json").read_text())
        for key, bad in [
            ("name", "../outside"),
            ("step_s", 0),
            ("duration_s", float("nan")),
            ("background", -1),
        ]:
            changed = copy.deepcopy(cfg)
            changed["profiles"][0][key] = bad
            with self.assertRaises(ValueError):
                m.validate_config(changed)

    def test_bias_is_persistent_without_changing_base_or_labels(self):
        m = module()
        o = observation()
        before = copy.deepcopy(o)
        changed, transport, active = m.fault(
            o, "benign_imu_bias_positive", 2, 10, 20, "free_flow", 101
        )
        self.assertTrue(active)
        self.assertEqual(changed["signals"]["imu"]["acceleration_mps2"], 0.06)
        self.assertEqual(before, o)
        self.assertEqual(transport, {})
        self.assertEqual(set(changed), set(o))

    def test_dropout_label_matches_two_second_fault(self):
        m = module()
        for t, expected in [(9.9, False), (10, True), (11.9, True), (12, False), (15, False)]:
            _, transport, active = m.fault(
                observation(), "benign_imu_dropout", t, 10, 20, "dense", 131
            )
            self.assertEqual(active, expected)
            self.assertEqual(bool(transport), expected)

    def test_jitter_repeatable_bounded_and_separate_from_signals(self):
        m = module()
        o = observation()
        a = m.fault(o, "benign_timestamp_delay_jitter", 15, 10, 20, "dense", 131)
        self.assertEqual(a, m.fault(o, "benign_timestamp_delay_jitter", 15, 10, 20, "dense", 131))
        self.assertEqual(a[0], o)
        self.assertTrue(
            all(type(v) is int and 0 <= v <= 20 for v in a[1]["source_delays_ms"].values())
        )


if __name__ == "__main__":
    unittest.main()
