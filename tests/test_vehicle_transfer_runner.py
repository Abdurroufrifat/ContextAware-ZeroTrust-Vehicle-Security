import hashlib
import json
import tempfile
import unittest
from pathlib import Path


class TransferRunnerTests(unittest.TestCase):
    def manifest(self):
        return dict(
            trips=[
                dict(
                    id=f"civic_{i + 1:02d}",
                    route_id=f"99c94dc769b5d96e|2018-01-{i + 1:02d}--00-00-00",
                    day=f"2018-01-{i + 1:02d}",
                    phase="validation" if i < 2 else "test",
                    segments=[5, 6],
                    parts=["a", "b"],
                )
                for i in range(7)
            ]
        )

    def test_seven_distinct_civic_dates_with_two_contiguous_segments(self):
        import run_vehicle_native_stage9 as r

        m = self.manifest()
        r.validate_split(m)
        m["trips"][2]["segments"] = [5, 7]
        with self.assertRaises(ValueError):
            r.validate_split(m)

    def test_other_vehicle_duplicate_route_and_cross_phase_date_rejected(self):
        import run_vehicle_native_stage9 as r

        for field, value in [
            ("route_id", "b0c9d2329ad1606b|2018-01-03--00-00-00"),
            ("route_id", self.manifest()["trips"][0]["route_id"]),
            ("day", "2018-01-01"),
        ]:
            m = self.manifest()
            m["trips"][2][field] = value
            with self.assertRaises(ValueError):
                r.validate_split(m)

    def test_imported_calibration_receipt_rejects_changed_bytes(self):
        import run_vehicle_native_stage9 as r

        raw = b'{"model":{},"thresholds":{}}'
        sha = hashlib.sha256(raw).hexdigest()
        receipt = json.dumps(dict(calibration_sha256=sha)).encode()
        self.assertEqual(r.verify_reference_bytes(raw, receipt, sha), json.loads(raw))
        with self.assertRaises(ValueError):
            r.verify_reference_bytes(raw + b" ", receipt, sha)

    def test_completed_or_partial_phase_cannot_be_silently_recomputed(self):
        import run_vehicle_native_stage9 as r

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            r.ensure_phase_unrun(root, "test")
            (root / "artifacts/vehicle_native_stage9/test").mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "already"):
                r.ensure_phase_unrun(root, "test")

    def test_normal_availability_pools_period_denominators(self):
        import run_vehicle_native_stage9 as r

        metrics = [
            dict(trip="t", case="normal", method="auth", period="before", requests=100, grants=0),
            dict(trip="t", case="normal", method="auth", period="after", requests=10, grants=10),
        ]
        row = r.normal_overall_metrics(metrics)[0]
        self.assertEqual(row["requests"], 110)
        self.assertEqual(row["grants"], 10)
        self.assertAlmostEqual(row["grant_fraction"], 10 / 110)

    def test_matched_normal_denial_is_not_credited_as_attack_detection(self):
        import run_vehicle_native_stage9 as r
        from vehicle_native_data import NativeFrame

        f = NativeFrame("t", 0, 1000, 800, 1000, 995, 998, 10.0, 10.0, 0.0, 10)
        row = dict(method="auth", frame=0, perturbed=True, final_grant=False)
        r.label_comparison(row, f, f, {("auth", 0): False})
        self.assertFalse(row["motion_changed"])
        self.assertFalse(row["attack_only_denial"])
        self.assertTrue(row["also_denied_normal"])

    def test_shared_stream_never_adds_evaluation_fields_to_physical_frame(self):
        import run_vehicle_native_stage9 as r
        from vehicle_native_data import NativeFrame

        from coact_cps.vehicle.native_detectors import ModelConfig

        frames = [
            NativeFrame(
                "t",
                i,
                1000 + 200 * i,
                800 + 200 * i,
                1000 + 200 * i,
                995 + 200 * i,
                998 + 200 * i,
                10.0,
                10.0,
                0.0,
                10,
            )
            for i in range(120)
        ]
        config = dict(
            stage7_horizon=dict(threshold=2.0, scales=[0.2, 0.5, 1.0, 2.0]),
            anchor=dict(scale=0.2, threshold=2.0),
        )
        out = r.perturb(frames, "enrollment_ramp", ModelConfig(), config)
        self.assertEqual(out, r.perturb(frames, "enrollment_ramp", ModelConfig(), config))
        self.assertEqual(set(out[0].__dict__), set(frames[0].__dict__))

    def test_overall_normal_paired_interval_names_candidate(self):
        import run_vehicle_native_stage9 as r

        metrics = [
            dict(
                trip=f"t{i}",
                case="normal",
                method=m,
                period=p,
                requests=n,
                grants=n if m == "horizon_recovery" else 0,
                grant_fraction=1.0 if m == "horizon_recovery" else 0.0,
            )
            for i in range(5)
            for m in r.METHODS
            for p, n in [("before", 100), ("after", 10)]
        ]
        rows = r.paired_intervals(metrics, 9142026)
        overall = [x for x in rows if x["period"] == "overall"]
        self.assertEqual(len(overall), 8)
        self.assertTrue(
            all(
                x["candidate"] == "horizon_recovery"
                and x["route_clusters"] == 5
                and x["candidate_minus_reference"] == 1.0
                for x in overall
            )
        )
