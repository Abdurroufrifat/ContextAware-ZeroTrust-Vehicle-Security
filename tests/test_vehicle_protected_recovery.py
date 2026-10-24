import os
import unittest
from dataclasses import replace

from run_vehicle_native_stage6 import packets
from vehicle_recovery_scenarios10b import perturb

from coact_cps.vehicle.native_detectors import ModelConfig
from coact_cps.vehicle.policy import SOURCES, VehicleRequest


class ProtectedRecoveryTests(unittest.TestCase):
    def fixture(self):
        from test_vehicle_window_recovery import WindowRecoveryTests

        return WindowRecoveryTests()

    def engine(self):
        from coact_cps.vehicle.protected_recovery_policy import ProtectedRecoveryPolicyEngine

        c = self.fixture().config()
        keys = {s: os.urandom(32) for s in SOURCES}
        return ProtectedRecoveryPolicyEngine(
            keys,
            os.urandom(32),
            {("native-reader", "native-ego", "vehicle.telemetry.longitudinal", "read")},
            ModelConfig(**c["model"]),
            c["thresholds"]["candidate"],
            c["session_config"]["stage7_horizon"],
            c["session_config"]["anchor"],
        ), keys

    def run_stream(self, frames, revoked=(), missing=(), repeats=False):
        engine, keys = self.engine()
        out = []
        checked = False
        for f in frames:
            q = VehicleRequest(
                f"r{f.frame}",
                f"n{f.frame}",
                "native-reader",
                "native-ego",
                "vehicle.telemetry.longitudinal",
                "read",
            )
            wire = packets(f, 2 * f.frame + 1, keys, revoked=f.frame in revoked)
            if f.frame in missing:
                wire.pop("imu")
            d = engine.evaluate(q, wire, f.time_ms)
            diag = engine.diagnostics("native-ego")
            out.append((d, diag))
            if (
                repeats
                and diag.get("quarantined")
                and diag["confirmation_epochs"] > 0
                and not checked
            ):
                q = replace(q, request_id="same-time", nonce="same-time-nonce")
                engine.evaluate(q, packets(f, 2 * f.frame + 2, keys), f.time_ms + 1)
                after = engine.diagnostics("native-ego")
                self.assertEqual(after["confirmation_epochs"], diag["confirmation_epochs"])
                self.assertEqual(after["protected_age_ms"], diag["protected_age_ms"])
                checked = True
        if repeats:
            self.assertTrue(checked)
        return out

    def test_persistent_step_cannot_fill_its_own_protected_fallback(self):
        frames = [
            replace(f, gnss_speed=f.gnss_speed + 5, wheel_speed=f.wheel_speed + 5)
            if f.frame >= 200
            else f
            for f in self.fixture().frames()
        ]
        rows = self.run_stream(frames)
        self.assertTrue(all(d.outcome != "ALLOW" for d, x in rows[200:]))
        self.assertFalse(any(x.get("protected_window_release") for d, x in rows))
        self.assertTrue(
            any(
                x.get("protected_gnss_error", 0) > 4
                for d, x in rows
                if x.get("protected_gnss_error") is not None
            )
        )

    def test_poisoned_observer_can_recover_when_protected_prediction_agrees(self):
        c = self.fixture().config()
        frames = perturb(
            self.fixture().frames(),
            "observer_aware",
            ModelConfig(**c["model"]),
            c["session_config"],
        )
        rows = self.run_stream(frames)
        releases = [i for i, (d, x) in enumerate(rows) if x.get("protected_window_release")]
        self.assertTrue(releases)
        self.assertEqual(rows[-1][0].outcome, "ALLOW")
        for i in releases:
            self.assertEqual(
                [d.outcome for d, x in rows[i : i + 4]],
                ["CHALLENGE", "CHALLENGE", "CHALLENGE", "ALLOW"],
            )

    def test_live_confirmation_does_not_advance_on_same_time(self):
        c = self.fixture().config()
        frames = perturb(
            self.fixture().frames(),
            "observer_aware",
            ModelConfig(**c["model"]),
            c["session_config"],
        )
        self.run_stream(frames, repeats=True)

    def test_revoked_identity_cannot_release(self):
        c = self.fixture().config()
        frames = perturb(
            self.fixture().frames(),
            "observer_aware",
            ModelConfig(**c["model"]),
            c["session_config"],
        )
        rows = self.run_stream(frames, revoked=range(400, 600))
        self.assertTrue(
            all(
                d.outcome == "DENY" and not x.get("protected_window_release") for d, x in rows[400:]
            )
        )

    def test_missing_imu_cannot_support_protected_confirmation(self):
        c = self.fixture().config()
        frames = perturb(
            self.fixture().frames(),
            "observer_aware",
            ModelConfig(**c["model"]),
            c["session_config"],
        )
        rows = self.run_stream(frames, missing=range(400, 600))
        self.assertTrue(
            all(
                d.outcome == "DENY" and not x.get("protected_window_release") for d, x in rows[400:]
            )
        )

    def test_startup_common_bias_remains_a_counterexample(self):
        frames = [
            replace(f, gnss_speed=f.gnss_speed + 8, wheel_speed=f.wheel_speed + 8)
            for f in self.fixture().frames()
        ]
        self.assertEqual(self.run_stream(frames)[-1][0].outcome, "ALLOW")

    def test_completed_release_does_not_export_a_stale_active_snapshot(self):
        c = self.fixture().config()
        frames = perturb(
            self.fixture().frames(),
            "observer_aware",
            ModelConfig(**c["model"]),
            c["session_config"],
        )
        rows = self.run_stream(frames)
        self.assertEqual(rows[-1][0].outcome, "ALLOW")
        self.assertFalse(rows[-1][1]["protected_present"])
