import importlib.util
import json
import secrets
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from coact_cps.vehicle.policy import SOURCES, VehicleRequest, seal_source

FILE = Path(__file__).resolve().parents[1] / "src/coact_cps/vehicle/temporal_policy.py"


def module():
    if not FILE.exists():
        raise AssertionError("candidate policy does not exist")
    spec = importlib.util.spec_from_file_location("coact_cps.vehicle.temporal_candidate", FILE)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def signals(speed=10.0, acceleration=0.0):
    return dict(
        identity=dict(
            subject_id="fleet-service", credential_valid=True, revoked=False, attestation_ok=True
        ),
        gnss=dict(speed_mps=speed),
        imu=dict(acceleration_mps2=acceleration, yaw_rate_rps=0.0),
        wheel=dict(speed_mps=speed, acceleration_mps2=acceleration, yaw_rate_rps=0.0),
        can=dict(anomaly_score=0.05),
        v2x=dict(inconsistency_score=0.05, uncertainty=0.05),
    )


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.m = module()
        self.keys = {s: secrets.token_bytes(32) for s in SOURCES}
        self.permissions = {("fleet-service", "ego", "vehicle.telemetry", "read")}
        self.engine = self.m.TemporalVehiclePolicyEngine(
            self.keys, secrets.token_bytes(32), self.permissions
        )

    def request(self, i=1, resource="vehicle.telemetry", operation="read"):
        return VehicleRequest(f"r{i}", f"n{i}", "fleet-service", "ego", resource, operation)

    def packets(self, seq, now=1000, s=None, missing=None):
        s = s or signals()
        return {
            source: seal_source(source, "ego", f"f{seq}", now, seq, s[source], self.keys[source])
            for source in SOURCES
            if source != missing
        }

    def enroll(self):
        d = self.engine.evaluate(self.request(0), self.packets(1), 1000)
        self.assertEqual(d.outcome, "CHALLENGE")

    def test_normal_access_after_enrollment(self):
        self.enroll()
        d = self.engine.evaluate(self.request(), self.packets(2, 1200), 1200)
        self.assertEqual(d.outcome, "ALLOW")

    def test_reference_age_cannot_grow_its_error_budget_forever(self):
        self.engine = self.m.TemporalVehiclePolicyEngine(
            self.keys,
            secrets.token_bytes(32),
            self.permissions,
            temporal_config=self.m.TemporalConfig(maximum_reference_age_ms=600),
        )
        self.enroll()
        for seq, now in [(2, 1200), (3, 1400), (4, 1600)]:
            self.assertEqual(
                self.engine.evaluate(self.request(seq), self.packets(seq, now), now).outcome,
                "ALLOW",
            )
        self.assertEqual(
            self.engine.evaluate(self.request(5), self.packets(5, 1800), 1800).outcome, "DENY"
        )

    def test_coordinated_speed_jump_is_denied_and_not_reanchored(self):
        self.enroll()
        for seq in [2, 3, 4]:
            now = 1000 + (seq - 1) * 200
            d = self.engine.evaluate(self.request(seq), self.packets(seq, now, signals(18)), now)
            self.assertEqual(d.outcome, "DENY")
        d = self.engine.evaluate(self.request(5), self.packets(5, 1800), 1800)
        self.assertEqual(d.outcome, "ALLOW")

    def test_startup_bias_is_a_retained_blind_spot(self):
        self.engine.evaluate(self.request(0), self.packets(1, s=signals(18)), 1000)
        d = self.engine.evaluate(self.request(), self.packets(2, 1200, signals(18)), 1200)
        self.assertEqual(d.outcome, "ALLOW")

    def test_coherent_motion_forgery_is_a_retained_blind_spot(self):
        self.enroll()
        for seq in range(2, 12):
            now = 1000 + (seq - 1) * 200
            d = self.engine.evaluate(
                self.request(seq), self.packets(seq, now, signals(10 + (seq - 1), 5)), now
            )
            self.assertEqual(d.outcome, "ALLOW")

    def test_missing_v2x_grants_snapshot_only(self):
        self.enroll()
        d = self.engine.evaluate(self.request(), self.packets(2, 1200, missing="v2x"), 1200)
        self.assertEqual((d.outcome, d.mode), ("RESTRICT", "snapshot"))
        self.assertIn("v2x_unavailable_snapshot_only", d.reasons)

    def test_present_forged_v2x_cannot_take_degraded_path(self):
        self.enroll()
        p = self.packets(2, 1200)
        packet = json.loads(p["v2x"])
        packet["mac"] = "0" * 64
        p["v2x"] = json.dumps(packet)
        self.assertEqual(self.engine.evaluate(self.request(), p, 1200).outcome, "DENY")

    def test_missing_mandatory_source_denied(self):
        self.enroll()
        for source in SOURCES[:-1]:
            self.assertEqual(
                self.engine.evaluate(
                    self.request(), self.packets(2, 1200, missing=source), 1200
                ).outcome,
                "DENY",
            )

    def test_degradation_ablation_denies_missing_v2x(self):
        e = self.m.TemporalVehiclePolicyEngine(
            self.keys,
            secrets.token_bytes(32),
            self.permissions,
            temporal_config=self.m.TemporalConfig(allow_degraded_v2x=False),
        )
        self.assertEqual(
            e.evaluate(self.request(), self.packets(1, 1000, missing="v2x"), 1000).outcome, "DENY"
        )

    def test_snapshot_capability_cannot_escalate_when_v2x_returns(self):
        self.enroll()
        r = self.request()
        d = self.engine.evaluate(r, self.packets(2, 1200, missing="v2x"), 1200)
        token = self.engine.issue_capability(d)
        a = self.engine.consume_capability(token, r, self.packets(3, 1201), 1201)
        self.assertTrue(a.authorized)
        self.assertEqual(a.mode, "snapshot")
        self.assertEqual(a.expires_ms, d.expires_ms)
        self.assertFalse(
            self.engine.consume_capability(token, r, self.packets(4, 1202), 1202).authorized
        )

    def test_revocation_at_use_denies(self):
        self.enroll()
        r = self.request()
        d = self.engine.evaluate(r, self.packets(2, 1200), 1200)
        s = signals()
        s["identity"]["revoked"] = True
        self.assertFalse(
            self.engine.consume_capability(
                self.engine.issue_capability(d), r, self.packets(3, 1201, s), 1201
            ).authorized
        )

    def test_clock_rollback_denies(self):
        self.enroll()
        self.assertEqual(
            self.engine.evaluate(self.request(), self.packets(2, 999), 999).outcome, "DENY"
        )

    def test_imu_continues_through_other_source_failure(self):
        self.enroll()
        for seq in range(2, 12):
            now = 1000 + (seq - 1) * 200
            self.engine.evaluate(
                self.request(seq),
                self.packets(seq, now, signals(10 + (seq - 1) * 0.4, 2), missing="gnss"),
                now,
            )
        d = self.engine.evaluate(self.request(12), self.packets(12, 3200, signals(14.4, 2)), 3200)
        self.assertEqual(d.outcome, "ALLOW")

    def test_imu_gap_cannot_implicitly_reset_history(self):
        self.enroll()
        for seq, now in [(2, 2200), (3, 2400)]:
            self.assertEqual(
                self.engine.evaluate(self.request(seq), self.packets(seq, now), now).outcome, "DENY"
            )

    def test_scope_denial_does_not_enroll(self):
        self.assertEqual(
            self.engine.evaluate(
                self.request(resource="vehicle.actuator"), self.packets(1), 1000
            ).outcome,
            "DENY",
        )
        self.assertEqual(
            self.engine.evaluate(self.request(), self.packets(2, 1200), 1200).outcome, "CHALLENGE"
        )

    def test_authenticated_source_replay_denied(self):
        self.enroll()
        p = self.packets(2, 1200)
        p["gnss"] = self.packets(1)["gnss"]
        self.assertEqual(self.engine.evaluate(self.request(), p, 1200).outcome, "DENY")

    def test_config_is_immutable_and_bound(self):
        with self.assertRaises(AttributeError):
            self.engine.temporal_config.allow_degraded_v2x = False
        other = self.m.TemporalVehiclePolicyEngine(
            self.keys,
            secrets.token_bytes(32),
            self.permissions,
            temporal_config=self.m.TemporalConfig(enable_temporal=False),
        )
        self.assertNotEqual(self.engine.policy_digest, other.policy_digest)

    def test_independent_imu_watermark_blocks_older_complete_context(self):
        self.enroll()
        self.engine.evaluate(self.request(100), self.packets(1000000, 1200, missing="gnss"), 1200)
        for seq, now, missing in [(2, 1100, None), (3, 1400, None), (4, 1500, "v2x")]:
            self.assertEqual(
                self.engine.evaluate(
                    self.request(seq), self.packets(seq, now, missing=missing), max(1200, now)
                ).outcome,
                "DENY",
            )

    def test_identical_integrated_imu_retry_allowed_but_conflicting_retry_denied(self):
        self.enroll()
        self.engine.evaluate(self.request(100), self.packets(100, 1200, missing="gnss"), 1200)
        self.assertEqual(
            self.engine.evaluate(self.request(101), self.packets(100, 1300), 1300).outcome, "DENY"
        )
        self.assertEqual(
            self.engine.evaluate(self.request(102), self.packets(100, 1200), 1310).outcome, "ALLOW"
        )

    def test_reference_age_uses_evaluation_clock_for_delayed_same_sample(self):
        self.engine = self.m.TemporalVehiclePolicyEngine(
            self.keys,
            secrets.token_bytes(32),
            self.permissions,
            temporal_config=self.m.TemporalConfig(maximum_reference_age_ms=600),
        )
        self.enroll()
        self.engine.evaluate(self.request(100), self.packets(100, 1200, missing="gnss"), 1200)
        self.assertEqual(
            self.engine.evaluate(self.request(101), self.packets(100, 1200), 1700).outcome, "DENY"
        )


if __name__ == "__main__":
    unittest.main()
