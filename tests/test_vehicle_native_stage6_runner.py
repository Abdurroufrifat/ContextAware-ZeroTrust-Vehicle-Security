import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))


def modules():
    import run_vehicle_native_stage6 as runner
    import vehicle_native_scenarios as scenarios

    return scenarios, runner


class NativeRunnerTests(unittest.TestCase):
    def test_calibration_edit_is_rejected_by_detached_freeze_receipt(self):
        _, r = modules()
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            c = root / r.CALIBRATION
            c.parent.mkdir(parents=True)
            cfg = root / r.CONFIG
            cfg.parent.mkdir(parents=True)
            cfg.write_text("{}")
            data = root / r.DATA
            data.parent.mkdir(parents=True)
            data.write_text("{}")
            frozen = {
                "thresholds": {"chi": 1},
                "source_sha256": r.fingerprints(root, []),
                "config_sha256": r.sha(cfg.read_bytes()),
                "manifest_sha256": r.sha(data.read_bytes()),
            }
            c.write_text(json.dumps(frozen))
            receipt = root / r.RECEIPT
            receipt.write_text(json.dumps({"calibration_sha256": r.sha(c.read_bytes())}))
            frozen["thresholds"]["chi"] = 1e12
            c.write_text(json.dumps(frozen))
            with self.assertRaises(ValueError):
                r.check_frozen(root)

    def frames(self):
        from vehicle_native_data import NativeFrame

        return [
            NativeFrame(
                "trip",
                i,
                1000 + i * 200,
                800 + i * 200,
                1000 + i * 200,
                995 + i * 200,
                998 + i * 200,
                10.0,
                10.0,
                0.0,
                10,
            )
            for i in range(90)
        ]

    def test_split_validation_rejects_route_overlap(self):
        _, r = modules()
        m = {
            "trips": [
                {"id": str(i), "route_id": str(i), "phase": p, "parts": [str(i)]}
                for i, p in enumerate(["calibration"] * 4 + ["validation"] * 2 + ["test"] * 5)
            ]
        }
        r.validate_split(m)
        m["trips"][-1]["route_id"] = "0"
        with self.assertRaises(ValueError):
            r.validate_split(m)

    def test_fixed_seventeen_cases_and_coherent_translation(self):
        s, _ = modules()
        self.assertEqual(len(s.CASES), 17)
        f = self.frames()
        a = s.perturb(f, "startup_common_bias", None)
        self.assertTrue(
            all(
                x.gnss_speed - y.gnss_speed == 8
                and x.wheel_speed - y.wheel_speed == 8
                and x.imu_accel == y.imu_accel
                for x, y in zip(a, f, strict=True)
            )
        )

    def test_coherent_acceleration_uses_native_dt_and_holds_offset(self):
        s, _ = modules()
        f = self.frames()
        a = s.perturb(f, "coherent_hold", None)
        k = next(i for i, x in enumerate(a) if x.imu_accel != f[i].imu_accel)
        self.assertAlmostEqual(a[k].wheel_speed - f[k].wheel_speed, 0.4 * 0.2)
        self.assertGreater(a[-1].wheel_speed - f[-1].wheel_speed, 0)
        self.assertEqual(a[-1].imu_accel, 0)

    def test_adaptive_attack_is_generated_once_as_shared_stream(self):
        s, _ = modules()
        from coact_cps.vehicle.native_detectors import ModelConfig

        f = self.frames()
        a = s.perturb(f, "observer_aware", ModelConfig())
        self.assertEqual(a, s.perturb(f, "observer_aware", ModelConfig()))
        self.assertTrue(any(x.gnss_speed != y.gnss_speed for x, y in zip(a, f, strict=True)))
        self.assertTrue(all(set(x.__dict__) == set(f[0].__dict__) for x in a))

    def test_weighted_quantile_does_not_overweight_long_trip(self):
        _, r = modules()
        self.assertEqual(r.drive_quantile([[1] * 100, [10]], 0.5), 1)
        self.assertEqual(r.drive_quantile([[1] * 100, [10]], 0.51), 10)

    def test_labels_are_outside_native_source_schema(self):
        s, r = modules()
        from coact_cps.vehicle.policy import SOURCES, strict_loads

        keys = {k: b"k" * 32 for k in SOURCES}
        wires = r.packets(self.frames()[0], 0, keys)
        self.assertNotIn("case", str(wires))
        self.assertNotIn("truth", str(wires))
        self.assertEqual(set(wires), {"identity", "gnss", "imu", "wheel"})
        self.assertEqual(
            set(strict_loads(wires["imu"])["body"]["signals"]),
            {"acceleration_mps2", "interval_start_ms", "interval_end_ms", "maximum_gap_ms"},
        )

    def test_source_fingerprint_detects_mutation(self):
        _, r = modules()
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "a.py").write_text("a\n")
            h = r.fingerprints(p, ["a.py"])
            (p / "a.py").write_text("b\n")
            self.assertNotEqual(h, r.fingerprints(p, ["a.py"]))
            with self.assertRaises(ValueError):
                r.fingerprints(p, ["../outside"])


if __name__ == "__main__":
    unittest.main()
