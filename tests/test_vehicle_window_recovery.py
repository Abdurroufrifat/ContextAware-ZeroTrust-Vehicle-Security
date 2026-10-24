import json
import os
import unittest
from pathlib import Path

from run_vehicle_native_stage6 import packets
from vehicle_native_data import NativeFrame
from vehicle_session_scenarios import perturb

from coact_cps.vehicle.native_detectors import ModelConfig
from coact_cps.vehicle.policy import SOURCES, VehicleRequest


class WindowRecoveryTests(unittest.TestCase):
    def config(self):
        p = Path(__file__).resolve().parents[1]
        return json.loads(
            (
                p / "data/reference/vehicle_native_stage9/stage9_linux_frozen_transfer.json"
            ).read_bytes()
        )

    def engine(self):
        from coact_cps.vehicle.window_recovery_policy import WindowRecoveryPolicyEngine

        f = self.config()
        keys = {s: os.urandom(32) for s in SOURCES}
        engine = WindowRecoveryPolicyEngine(
            keys,
            os.urandom(32),
            {("native-reader", "native-ego", "vehicle.telemetry.longitudinal", "read")},
            ModelConfig(**f["model"]),
            f["thresholds"]["candidate"],
            f["session_config"]["stage7_horizon"],
        )
        return engine, keys

    def frames(self):
        m = ModelConfig(**self.config()["model"])
        return [
            NativeFrame(
                "synthetic",
                i,
                1000 + 200 * i,
                800 + 200 * i,
                1000 + 200 * i,
                1000 + 200 * i,
                1000 + 200 * i,
                10 + m.gnss_offset,
                10,
                m.imu_offset,
                10,
            )
            for i in range(600)
        ]

    def run_frames(self, frames, missing=(), revoked=()):
        engine, keys = self.engine()
        rows = []
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
            rows.append((d, engine.diagnostics("native-ego")))
        return engine, keys, rows

    def test_poisoned_bias_recovers_after_complete_honest_window(self):
        f = self.config()
        frames = self.frames()
        changed = perturb(frames, "observer_aware", ModelConfig(**f["model"]), f["session_config"])
        engine, keys, rows = self.run_frames(changed)
        self.assertEqual(rows[399][0].outcome, "DENY")
        self.assertEqual(rows[-1][0].outcome, "ALLOW")
        releases = [i for i, (d, x) in enumerate(rows) if x["released"]]
        self.assertTrue(
            any(400 < i < 550 and rows[i][1]["recovery_path"] == "fresh_window" for i in releases)
        )
        self.assertTrue(all(rows[i][0].outcome == "CHALLENGE" for i in releases))

    def test_primary_distance_is_not_a_timer_only_release(self):
        from dataclasses import replace

        frames = self.frames()
        changed = [
            replace(
                f,
                gnss_speed=f.gnss_speed + 5 if f.frame >= 100 else f.gnss_speed,
                wheel_speed=f.wheel_speed + 5 if f.frame >= 100 else f.wheel_speed,
            )
            for f in frames
        ]
        _, _, rows = self.run_frames(changed)
        self.assertTrue(all(d.outcome != "ALLOW" for d, x in rows[100:195]))
        releases = [
            i
            for i, (d, x) in enumerate(rows)
            if x["recovery_path"] == "fresh_window" and x["released"]
        ]
        self.assertTrue(releases)
        # Explicit counterexample: once the mutually consistent step fills the window,
        # a consistency-only fallback can grant during the persistent forgery.
        self.assertTrue(any(d.outcome == "ALLOW" for d, x in rows[250:]))

    def test_missing_imu_prevents_fallback_confirmation(self):
        f = self.config()
        frames = self.frames()
        changed = perturb(frames, "observer_aware", ModelConfig(**f["model"]), f["session_config"])
        _, _, rows = self.run_frames(changed, missing=range(400, 600))
        self.assertTrue(all(d.outcome == "DENY" and not x["released"] for d, x in rows[400:]))

    def test_revoked_identity_prevents_release(self):
        f = self.config()
        frames = self.frames()
        changed = perturb(frames, "observer_aware", ModelConfig(**f["model"]), f["session_config"])
        _, _, rows = self.run_frames(changed, revoked=range(400, 600))
        self.assertTrue(all(d.outcome == "DENY" and not x["released"] for d, x in rows[400:]))

    def test_gap_requires_new_full_window_before_fallback(self):
        f = self.config()
        frames = self.frames()
        changed = perturb(frames, "observer_aware", ModelConfig(**f["model"]), f["session_config"])
        _, _, rows = self.run_frames(changed, missing=range(480, 485))
        self.assertFalse(
            any(x["released"] and x["recovery_path"] == "fresh_window" for d, x in rows[485:580])
        )

    def test_repeated_same_time_does_not_add_confirmation(self):
        engine, keys, rows = self.run_frames(self.frames()[:40])
        f = self.frames()[39]
        before = engine.diagnostics("native-ego")["confirmation_epochs"]
        q = VehicleRequest(
            "repeat",
            "different-nonce",
            "native-reader",
            "native-ego",
            "vehicle.telemetry.longitudinal",
            "read",
        )
        d = engine.evaluate(q, packets(f, 2 * f.frame + 2, keys), f.time_ms + 1)
        self.assertEqual(engine.diagnostics("native-ego")["confirmation_epochs"], before)
        self.assertEqual(d.outcome, "ALLOW")

    def test_same_time_in_live_quarantine_and_release_quorum(self):
        config = self.config()
        frames = self.frames()
        changed = perturb(
            frames, "observer_aware", ModelConfig(**config["model"]), config["session_config"]
        )
        engine, keys = self.engine()
        checked = False
        released = None
        outcomes = []
        for f in changed:
            q = VehicleRequest(
                f"r{f.frame}",
                f"n{f.frame}",
                "native-reader",
                "native-ego",
                "vehicle.telemetry.longitudinal",
                "read",
            )
            decision = engine.evaluate(q, packets(f, 2 * f.frame + 1, keys), f.time_ms)
            diagnostic = engine.diagnostics("native-ego")
            outcomes.append(decision.outcome)
            if diagnostic["quarantined"] and diagnostic["confirmation_epochs"] > 0 and not checked:
                before = diagnostic["confirmation_epochs"]
                repeat = VehicleRequest(
                    "repeat-live",
                    "repeat-live-nonce",
                    q.subject_id,
                    q.vehicle_id,
                    q.resource,
                    q.operation,
                )
                again = engine.evaluate(repeat, packets(f, 2 * f.frame + 2, keys), f.time_ms + 1)
                self.assertEqual(engine.diagnostics("native-ego")["confirmation_epochs"], before)
                self.assertEqual(again.outcome, "DENY")
                checked = True
            if diagnostic["released"] and diagnostic["recovery_path"] == "fresh_window":
                released = f.frame
        self.assertTrue(checked, "must exercise a live confirmation state")
        self.assertIsNotNone(released)
        self.assertEqual(
            outcomes[released : released + 4], ["CHALLENGE", "CHALLENGE", "CHALLENGE", "ALLOW"]
        )

    def test_invalid_request_retains_native_typed_rejection(self):
        engine, keys = self.engine()
        with self.assertRaises(ValueError):
            engine.evaluate(None, {}, 1000)
