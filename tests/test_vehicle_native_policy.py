import importlib
import unittest

from coact_cps.vehicle.explanation import decision_card, verify_decision_card
from coact_cps.vehicle.native_detectors import ModelConfig
from coact_cps.vehicle.policy import SOURCES, VehicleRequest, canonical_json, seal_source


def module():
    try:
        return importlib.import_module("coact_cps.vehicle.native_policy")
    except ModuleNotFoundError:
        raise AssertionError("native signed policy not implemented") from None


class NativePolicyTests(unittest.TestCase):
    def test_sensor_same_timestamp_conflict_cannot_hide_behind_new_imu_epoch(self):
        self.setup_engine("auth")
        d = self.ready()
        from coact_cps.vehicle.policy import strict_loads

        wires = self.packets(2, seq=6)
        for source in ("gnss", "wheel"):
            b = strict_loads(wires[source])["body"]
            b["signals"]["speed_mps"] = 18.0
            wires[source] = seal_source(
                source,
                b["vehicle_id"],
                b["frame_id"],
                b["timestamp_ms"],
                b["sequence"],
                b["signals"],
                self.keys[source],
            )
        b = strict_loads(wires["imu"])["body"]
        b["timestamp_ms"] = 1401
        b["signals"].update(interval_start_ms=1400, interval_end_ms=1401, maximum_gap_ms=1)
        wires["imu"] = seal_source("imu", "ego", "f2", 1401, 6, b["signals"], self.keys["imu"])
        self.assertFalse(
            self.engine.consume_capability(
                self.engine.issue_capability(d), d.request, wires, 1401
            ).authorized
        )

    def test_startup_requires_consecutive_complete_contexts(self):
        self.setup_engine()
        outcomes = []
        for i in range(5):
            p = self.packets(i)
            if i % 2:
                p.pop("gnss")
            outcomes.append(self.engine.evaluate(self.request(i), p, 1000 + i * 200).outcome)
        self.assertEqual(outcomes, ["CHALLENGE", "DENY", "CHALLENGE", "DENY", "CHALLENGE"])

    def test_ungated_dual_ablation_retains_updates_on_alarm(self):
        self.setup_engine("dual_ungated")
        self.ready()
        self.assertEqual(self.engine.method, "dual_ungated")

    def setup_engine(self, method="chi"):
        m = module()
        self.keys = {s: (s.encode() * 40)[:32] for s in SOURCES}
        p = {("tester", "ego", "vehicle.telemetry.longitudinal", "read")}
        self.engine = m.NativePolicyEngine(
            self.keys,
            b"c" * 32,
            p,
            ModelConfig(),
            method,
            dict(chi=1e6, window=1e7, cusum=1e9, dual_chi=1e6, dual_cusum=1e9),
        )
        return self.engine

    def request(self, i):
        return VehicleRequest(
            f"r{i}", f"n{i}", "tester", "ego", "vehicle.telemetry.longitudinal", "read"
        )

    def packets(self, i, seq=None, revoked=False, accel=0.0, start=None):
        t = 1000 + i * 200
        seq = 2 * i + 1 if seq is None else seq
        signals = {
            "identity": dict(
                subject_id="tester", credential_valid=True, revoked=revoked, attestation_ok=True
            ),
            "gnss": dict(speed_mps=10.0),
            "wheel": dict(speed_mps=10.0),
            "imu": dict(
                acceleration_mps2=accel,
                interval_start_ms=t - 200 if start is None else start,
                interval_end_ms=t,
                maximum_gap_ms=10,
            ),
        }
        return {
            s: seal_source(s, "ego", f"f{i}", t - 5 if s == "imu" else t, seq, v, self.keys[s])
            for s, v in signals.items()
        }

    def ready(self):
        for i in range(3):
            d = self.engine.evaluate(self.request(i), self.packets(i), 1000 + i * 200)
        self.assertEqual(d.outcome, "ALLOW")
        return d

    def test_startup_quorum_then_bounded_one_use_capability(self):
        self.setup_engine()
        d = self.ready()
        c = self.engine.issue_capability(d)
        a = self.engine.consume_capability(c, d.request, self.packets(2, seq=6), 1401)
        self.assertTrue(a.authorized)
        self.assertEqual(a.expires_ms, 1650)
        self.assertFalse(
            self.engine.consume_capability(c, d.request, self.packets(2, seq=7), 1402).authorized
        )

    def test_recheck_same_physical_sample_does_not_assimilate_twice(self):
        self.setup_engine()
        d = self.ready()
        before = self.engine.diagnostics("ego")["measurement_updates"]
        self.engine.consume_capability(
            self.engine.issue_capability(d), d.request, self.packets(2, seq=6), 1401
        )
        self.assertEqual(self.engine.diagnostics("ego")["measurement_updates"], before)

    def test_changed_same_time_motion_is_denied(self):
        self.setup_engine()
        d = self.ready()
        a = self.engine.consume_capability(
            self.engine.issue_capability(d), d.request, self.packets(2, seq=6, accel=2.0), 1401
        )
        self.assertFalse(a.authorized)

    def test_identity_revocation_at_use_is_denied(self):
        self.setup_engine()
        d = self.ready()
        a = self.engine.consume_capability(
            self.engine.issue_capability(d), d.request, self.packets(2, seq=6, revoked=True), 1401
        )
        self.assertFalse(a.authorized)

    def test_missing_imu_requires_new_recovery_quorum(self):
        self.setup_engine()
        self.ready()
        p = self.packets(3)
        p.pop("imu")
        self.assertEqual(self.engine.evaluate(self.request(3), p, 1600).outcome, "DENY")
        outcomes = [
            self.engine.evaluate(self.request(i), self.packets(i), 1000 + i * 200).outcome
            for i in range(4, 7)
        ]
        self.assertEqual(outcomes, ["CHALLENGE", "CHALLENGE", "ALLOW"])

    def test_independent_imu_ingestion_has_replay_watermark(self):
        self.setup_engine()
        self.ready()
        p = self.packets(3, seq=9)
        p.pop("gnss")
        self.engine.evaluate(self.request(3), p, 1600)
        p = self.packets(4, seq=10)
        p["imu"] = self.packets(3, seq=8)["imu"]
        d = self.engine.evaluate(self.request(4), p, 1800)
        self.assertEqual(d.outcome, "DENY")
        self.assertTrue(any("independent_sequence" in r for r in d.reasons))

    def test_invented_v2x_source_and_bad_mac_cannot_grant(self):
        self.setup_engine()
        self.ready()
        p = self.packets(3)
        p["v2x"] = "{}"
        self.assertEqual(self.engine.evaluate(self.request(3), p, 1600).outcome, "DENY")
        p = self.packets(4)
        p["gnss"] = p["gnss"].replace('"speed_mps":10.0', '"speed_mps":20.0')
        self.assertEqual(self.engine.evaluate(self.request(4), p, 1800).outcome, "DENY")

    def test_source_contract_only_does_not_call_it_physical_truth(self):
        self.setup_engine("auth")
        d = self.ready()
        self.assertIn("authenticated_source_contract_only", d.reasons)

    def test_exact_decision_card_rejects_altered_mode(self):
        self.setup_engine()
        d = self.ready()
        card = decision_card(d)
        self.assertTrue(verify_decision_card(d, canonical_json(card)))
        card["mode"] = "snapshot"
        self.assertFalse(verify_decision_card(d, canonical_json(card)))


if __name__ == "__main__":
    unittest.main()
