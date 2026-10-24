import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


class SessionRunnerTests(unittest.TestCase):
    def test_multiple_test_episodes_share_route_without_cross_phase_leakage(self):
        import run_vehicle_native_stage8 as r

        entries = [dict(id=f"c{i}", route_id=f"c{i}", phase="calibration") for i in range(4)]
        entries += [dict(id="v", route_id="v", phase="validation")]
        entries += [
            dict(id=f"t{i}", route_id="a" if i < 3 else "b", phase="test") for i in range(6)
        ]
        m = dict(trips=entries, prior_routes=[f"c{i}" for i in range(4)])
        r.validate_split(m)
        entries[-1]["route_id"] = "v"
        with self.assertRaises(ValueError):
            r.validate_split(m)

    def test_pooling_uses_episode_denominators_within_route(self):
        import run_vehicle_native_stage8 as r

        m = dict(trips=[dict(id="e1", route_id="a"), dict(id="e2", route_id="a")])
        rows = [
            dict(trip="e1", case="normal", method="auth", period="before", requests=100, grants=0),
            dict(trip="e2", case="normal", method="auth", period="before", requests=10, grants=10),
        ]
        out = r.pool_route_metrics(rows, m)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["requests"], 110)
        self.assertAlmostEqual(out[0]["grant_fraction"], 10 / 110)
        self.assertEqual(out[0]["episodes"], 2)

    def test_detached_receipt_rejects_edited_calibration(self):
        import run_vehicle_native_stage8 as r

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p = root / r.CALIBRATION
            p.parent.mkdir(parents=True)
            p.write_text('{"edited":true}')
            (root / r.RECEIPT).write_text(json.dumps(dict(calibration_sha256="0" * 64)))
            with self.assertRaisesRegex(ValueError, "calibration changed"):
                r.check_frozen(root)

    def test_enrollment_ramp_is_shared_offline_without_evaluation_fields(self):
        from vehicle_native_data import NativeFrame
        from vehicle_session_scenarios import active, perturb

        from coact_cps.vehicle.native_detectors import ModelConfig
        from coact_cps.vehicle.session_anchor import SessionAnchor

        frames = [
            NativeFrame(
                "trip",
                i,
                1000 + i * 200,
                800 + i * 200,
                1000 + i * 200,
                995 + i * 200,
                1000 + i * 200,
                10.0,
                10.0,
                0.0,
                10,
            )
            for i in range(55)
        ]
        changed = perturb(
            frames,
            "enrollment_ramp",
            ModelConfig(),
            dict(stage7_horizon={}, anchor=dict(scale=0.2, threshold=2.0)),
        )
        monitor = SessionAnchor(0, 0.2, 2)
        for f in changed:
            self.assertTrue(active(f, frames, "enrollment_ramp"))
            self.assertFalse(hasattr(f, "evaluation_only"))
            monitor.ingest(f.imu_accel, f.interval_start_ms, f.time_ms)
            result = monitor.sample(f.gnss_speed, f.wheel_speed, f.time_ms)
        self.assertFalse(result.alarm)
        self.assertAlmostEqual(result.bias, -0.12)

    def test_session_gate_keeps_frozen_matched_dual_thresholds(self):
        import run_vehicle_native_stage8 as r

        from coact_cps.vehicle.native_detectors import ModelConfig

        old = dict(chi=1e6, window=1e7, cusum=1e9, dual_chi=2e6, dual_cusum=2e9)
        gate, _ = r.new_engine(
            ModelConfig(),
            "anchor_recovery",
            dict(baseline=old, candidate=old),
            dict(anchor=dict(scale=0.2, threshold=2), stage7_horizon={}),
        )
        self.assertEqual(dict(gate._thresholds), old)
