import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


class IntegrityRunnerTests(unittest.TestCase):
    def test_detached_receipt_rejects_edited_calibration(self):
        import json
        import tempfile

        import run_vehicle_native_stage7 as r

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            p = root / r.CALIBRATION
            p.parent.mkdir(parents=True)
            p.write_text('{"threshold":1}')
            (root / r.RECEIPT).write_text(json.dumps({"calibration_sha256": r.sha(p.read_bytes())}))
            p.write_text('{"threshold":100}')
            with self.assertRaises(ValueError):
                r.check_frozen(root)

    def test_route_partition_rejects_any_old_holdout_route(self):
        import run_vehicle_native_stage7 as r

        m = {
            "prior_stage6_routes": ["old"],
            "trips": [
                dict(id=str(i), route_id=str(i), phase=p)
                for i, p in enumerate(["calibration"] * 4 + ["validation"] * 2 + ["test"] * 5)
            ],
        }
        r.validate_split(m)
        m["trips"][-1]["route_id"] = "old"
        with self.assertRaises(ValueError):
            r.validate_split(m)

    def test_candidate_factor_scales_both_dual_and_horizon_boundaries(self):
        import run_vehicle_native_stage7 as r

        b = dict(chi=1.0, window=2.0, cusum=3.0, dual_chi=4.0, dual_cusum=5.0)
        dual, h = r.candidate_config(b, [0.2, 0.5, 1.0, 2.0], 2.0, 4)
        self.assertEqual(dual["dual_chi"], 16.0)
        self.assertEqual(dual["chi"], 1.0)
        self.assertEqual(h["threshold"], 8.0)

    def test_budget_attack_is_a_shared_stream_and_keeps_evaluation_fields_out(self):
        import vehicle_integrity_scenarios as s
        from vehicle_native_data import NativeFrame

        f = [
            NativeFrame(
                "t",
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
            for i in range(200)
        ]
        h = dict(threshold=2.0, scales=[0.2, 0.5, 1.0, 2.0])
        a = s.perturb(f, "horizon_budget", None, h)
        self.assertEqual(a, s.perturb(f, "horizon_budget", None, h))
        self.assertTrue(any(x.wheel_speed != y.wheel_speed for x, y in zip(a, f, strict=True)))
        self.assertEqual(set(a[0].__dict__), set(f[0].__dict__))


if __name__ == "__main__":
    unittest.main()
