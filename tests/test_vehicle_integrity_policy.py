import unittest

from coact_cps.vehicle.native_detectors import ModelConfig
from coact_cps.vehicle.policy import SOURCES, VehicleRequest, seal_source


class IntegrityPolicyTests(unittest.TestCase):
    def engine(self, variant="horizon_recovery"):
        from coact_cps.vehicle.integrity_policy import IntegrityPolicyEngine

        self.keys = {s: (s.encode() * 40)[:32] for s in SOURCES}
        self.gate = IntegrityPolicyEngine(
            self.keys,
            b"c" * 32,
            {("tester", "ego", "vehicle.telemetry.longitudinal", "read")},
            ModelConfig(),
            dict(chi=1e6, window=1e7, cusum=1e9, dual_chi=1e6, dual_cusum=1e9),
            variant,
            dict(horizons=(2000,), scales=(1.0,), threshold=2.0),
        )
        return self.gate

    def packets(self, i, speed=10.0, seq=None, revoked=False):
        t = 1000 + i * 200
        sequence = 2 * i + 1 if seq is None else seq
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
            s: seal_source(s, "ego", f"f{i}", t - 5 if s == "imu" else t, sequence, v, self.keys[s])
            for s, v in values.items()
        }

    def q(self, i):
        return VehicleRequest(
            f"r{i}", f"n{i}", "tester", "ego", "vehicle.telemetry.longitudinal", "read"
        )

    def step(self, i, **kwargs):
        return self.gate.evaluate(self.q(i), self.packets(i, **kwargs), 1000 + i * 200)

    def ready(self):
        for i in range(11):
            d = self.step(i)
        self.assertEqual(d.outcome, "ALLOW")
        return d

    def test_horizon_requires_real_elapsed_warmup(self):
        self.engine()
        outcomes = [self.step(i).outcome for i in range(11)]
        self.assertEqual(outcomes[:-1], ["CHALLENGE"] * 10)
        self.assertEqual(outcomes[-1], "ALLOW")

    def test_horizon_alarm_freezes_primary_and_enters_quarantine(self):
        self.engine()
        self.ready()
        before = self.gate.diagnostics("ego")["measurement_updates"]
        d = self.step(11, speed=18.0)
        self.assertEqual(d.outcome, "DENY")
        self.assertIn("native_integrity_horizon_alarm", d.reasons)
        self.assertEqual(self.gate.diagnostics("ego")["measurement_updates"], before)
        self.assertTrue(self.gate.diagnostics("ego")["quarantined"])

    def test_shadow_recovery_release_challenges_then_needs_three_new_epochs(self):
        self.engine()
        self.ready()
        for i in range(11, 14):
            self.step(i, speed=18.0)
        release = None
        for i in range(14, 55):
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

    def test_virtual_capability_use_does_not_advance_monitor_or_shadow(self):
        self.engine()
        d = self.ready()
        before = self.gate.diagnostics("ego")
        a = self.gate.consume_capability(
            self.gate.issue_capability(d), d.request, self.packets(10, seq=22), 3001
        )
        self.assertTrue(a.authorized)
        after = self.gate.diagnostics("ego")
        self.assertEqual(before["shadow_updates"], after["shadow_updates"])
        self.assertEqual(before["horizon_samples"], after["horizon_samples"])
        self.assertEqual(a.expires_ms, 3250)

    def test_bad_context_interrupts_shadow_confirmation(self):
        self.engine("recovery_dual")
        self.ready()
        state = self.gate._integrity["ego"]
        state.quarantined = True
        state.confirm_count = 4
        state.confirm_start = 1000
        packets = self.packets(11)
        packets.pop("gnss")
        d = self.gate.evaluate(self.q(11), packets, 3200)
        self.assertEqual(d.outcome, "DENY")
        self.assertEqual(state.confirm_count, 0)

    def test_stable_forged_speed_cannot_recover_by_inflating_primary_covariance(self):
        self.engine()
        self.ready()
        for i in range(11, 65):
            d = self.step(i, speed=18.0)
            self.assertEqual(d.outcome, "DENY")
        self.assertEqual(self.gate.diagnostics("ego")["recovery_resets"], 0)

    def test_missing_imu_during_quarantine_can_reenroll_honest_stream(self):
        self.engine()
        self.ready()
        self.step(11, speed=18.0)
        wires = self.packets(12)
        wires.pop("imu")
        self.assertEqual(self.gate.evaluate(self.q(12), wires, 3400).outcome, "DENY")
        outcomes = [self.step(i).outcome for i in range(13, 60)]
        self.assertIn("ALLOW", outcomes)
        self.assertGreater(self.gate.diagnostics("ego")["recovery_resets"], 0)

    def test_resumed_gnss_after_long_gap_can_build_new_window(self):
        self.engine()
        self.ready()
        for i in range(11, 36):
            p = self.packets(i)
            p.pop("gnss")
            self.assertEqual(self.gate.evaluate(self.q(i), p, 1000 + i * 200).outcome, "DENY")
        outcomes = [self.step(i).outcome for i in range(36, 65)]
        self.assertIn("ALLOW", outcomes)

    def test_lost_imu_reference_has_forged_reenrollment_blind_spot(self):
        self.engine()
        self.ready()
        self.step(11, speed=18.0)
        p = self.packets(12, speed=18.0)
        p.pop("imu")
        self.gate.evaluate(self.q(12), p, 3400)
        outcomes = [self.step(i, speed=18.0).outcome for i in range(13, 60)]
        self.assertIn("ALLOW", outcomes)

    def test_revocation_at_use_still_blocks_capability(self):
        self.engine()
        d = self.ready()
        a = self.gate.consume_capability(
            self.gate.issue_capability(d), d.request, self.packets(10, seq=22, revoked=True), 3001
        )
        self.assertFalse(a.authorized)


if __name__ == "__main__":
    unittest.main()
