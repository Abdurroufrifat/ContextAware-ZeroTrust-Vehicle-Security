import unittest

from coact_cps.vehicle.native_detectors import ModelConfig
from coact_cps.vehicle.policy import SOURCES, VehicleRequest, seal_source


class SessionPolicyTests(unittest.TestCase):
    def setup_gate(self, variant="anchor_recovery"):
        from coact_cps.vehicle.session_policy import SessionPolicyEngine

        self.keys = {s: (s.encode() * 40)[:32] for s in SOURCES}
        self.gate = SessionPolicyEngine(
            self.keys,
            b"c" * 32,
            {("tester", "ego", "vehicle.telemetry.longitudinal", "read")},
            ModelConfig(),
            dict(chi=1e6, window=1e7, cusum=1e9, dual_chi=1e6, dual_cusum=1e9),
            variant,
            dict(scale=0.2, threshold=2.0),
        )

    def wires(self, i, speed=10.0, seq=None, revoked=False):
        t = 1000 + i * 200
        values = {
            "identity": dict(
                subject_id="tester", credential_valid=True, revoked=revoked, attestation_ok=True
            ),
            "gnss": dict(speed_mps=speed),
            "wheel": dict(speed_mps=speed),
            "imu": dict(
                acceleration_mps2=0.0,
                interval_start_ms=t - 200,
                interval_end_ms=t,
                maximum_gap_ms=10,
            ),
        }
        return {
            s: seal_source(
                s,
                "ego",
                f"f{i}",
                t - 5 if s == "imu" else t,
                2 * i + 1 if seq is None else seq,
                v,
                self.keys[s],
            )
            for s, v in values.items()
        }

    def request(self, i):
        return VehicleRequest(
            f"r{i}", f"n{i}", "tester", "ego", "vehicle.telemetry.longitudinal", "read"
        )

    def step(self, i, **kwargs):
        return self.gate.evaluate(self.request(i), self.wires(i, **kwargs), 1000 + 200 * i)

    def ready(self):
        for i in range(26):
            d = self.step(i)
        self.assertEqual(d.outcome, "ALLOW")
        self.assertTrue(self.gate.diagnostics("ego")["anchor_ready"])
        return d

    def test_baseline_quorum_available_while_anchor_enrolls(self):
        self.setup_gate()
        self.assertEqual(
            [self.step(i).outcome for i in range(3)], ["CHALLENGE", "CHALLENGE", "ALLOW"]
        )
        self.assertFalse(self.gate.diagnostics("ego")["anchor_ready"])

    def test_anchor_alarm_freezes_primary_and_latches_quarantine(self):
        self.setup_gate()
        self.ready()
        n = self.gate.diagnostics("ego")["measurement_updates"]
        d = self.step(26, speed=18)
        self.assertEqual(d.outcome, "DENY")
        self.assertEqual(self.gate.diagnostics("ego")["measurement_updates"], n)
        self.assertTrue(self.gate.diagnostics("ego")["quarantined"])

    def test_honest_return_releases_without_resetting_anchor(self):
        self.setup_gate()
        self.ready()
        bias = self.gate.diagnostics("ego")["anchor_bias"]
        for i in range(26, 31):
            self.step(i, speed=18)
        release = None
        for i in range(31, 100):
            d = self.step(i)
            if self.gate.diagnostics("ego")["recovery_resets"]:
                release = i
                self.assertEqual(d.outcome, "CHALLENGE")
                break
        self.assertIsNotNone(release)
        self.assertEqual(
            [self.step(i).outcome for i in range(release + 1, release + 4)],
            ["CHALLENGE", "CHALLENGE", "ALLOW"],
        )
        self.assertEqual(self.gate.diagnostics("ego")["anchor_bias"], bias)

    def test_forged_shadow_cannot_release_enrolled_anchor(self):
        self.setup_gate()
        self.ready()
        for i in range(26, 100):
            self.assertEqual(self.step(i, speed=18).outcome, "DENY")
        self.assertEqual(self.gate.diagnostics("ego")["recovery_resets"], 0)

    def test_reference_loss_remains_denied_after_forged_reenrollment(self):
        self.setup_gate()
        self.ready()
        self.step(26, speed=18)
        wires = self.wires(27, speed=18)
        wires.pop("imu")
        self.gate.evaluate(self.request(27), wires, 6400)
        for i in range(28, 70):
            self.assertEqual(self.step(i, speed=18).outcome, "DENY")
        self.assertTrue(self.gate.diagnostics("ego")["anchor_lost"])
        self.assertEqual(self.gate.diagnostics("ego")["recovery_resets"], 0)

    def test_invalid_context_breaks_recovery_confirmation(self):
        self.setup_gate()
        self.ready()
        self.step(26, speed=18)
        state = self.gate._sessions["ego"]
        state.confirm_count = 4
        state.confirm_start = 2000
        wires = self.wires(27)
        wires.pop("gnss")
        self.gate.evaluate(self.request(27), wires, 6400)
        self.assertEqual(state.confirm_count, 0)

    def test_capability_use_copy_does_not_advance_anchor_or_shadow(self):
        self.setup_gate()
        d = self.ready()
        before = self.gate.diagnostics("ego")
        cap = self.gate.issue_capability(d)
        use = self.gate.consume_capability(cap, d.request, self.wires(25, seq=52), 6001)
        self.assertTrue(use.authorized)
        self.assertEqual(use.expires_ms, 6250)
        after = self.gate.diagnostics("ego")
        self.assertEqual(before["shadow_updates"], after["shadow_updates"])
        self.assertEqual(before["anchor_samples"], after["anchor_samples"])
        self.assertFalse(
            self.gate.consume_capability(cap, d.request, self.wires(25, seq=52), 6002).authorized
        )

    def test_use_time_revocation_is_still_denied(self):
        self.setup_gate()
        d = self.ready()
        use = self.gate.consume_capability(
            self.gate.issue_capability(d), d.request, self.wires(25, seq=52, revoked=True), 6001
        )
        self.assertFalse(use.authorized)

    def test_primary_only_denial_reports_alarm_without_false_quarantine(self):
        from coact_cps.vehicle.session_policy import SessionPolicyEngine

        self.setup_gate("anchor_dual")
        self.gate = SessionPolicyEngine(
            self.keys,
            b"c" * 32,
            {("tester", "ego", "vehicle.telemetry.longitudinal", "read")},
            ModelConfig(),
            dict(chi=1e6, window=1e7, cusum=1e9, dual_chi=0.01, dual_cusum=1e9),
            "anchor_dual",
            dict(scale=0.2, threshold=2.0),
        )
        self.ready()
        d = self.step(26, speed=10.1)
        diag = self.gate.diagnostics("ego")
        self.assertEqual(d.outcome, "DENY")
        self.assertFalse(diag["anchor_alarm"])
        self.assertFalse(diag["quarantined"])
        self.assertEqual(d.reasons, ("native_motion_statistic_alarm",))
