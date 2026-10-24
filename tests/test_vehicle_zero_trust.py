from __future__ import annotations

import hashlib
import json
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

from coact_cps.vehicle.policy import VehiclePolicyEngine, VehicleRequest, seal_source

SOURCES = ("identity", "gnss", "imu", "wheel", "can", "v2x")
KEYS = {s: hashlib.sha256(("test-only-" + s).encode()).digest() for s in SOURCES}
SIGNALS = {
    "identity": {
        "subject_id": "fleet-service",
        "credential_valid": True,
        "revoked": False,
        "attestation_ok": True,
    },
    "gnss": {"speed_mps": 12.0},
    "imu": {"acceleration_mps2": 0.1, "yaw_rate_rps": 0.01},
    "wheel": {"speed_mps": 12.1, "acceleration_mps2": 0.1, "yaw_rate_rps": 0.01},
    "can": {"anomaly_score": 0.05},
    "v2x": {"inconsistency_score": 0.05, "uncertainty": 0.05},
}


def make_engine():
    return VehiclePolicyEngine(
        KEYS,
        b"capability-test-key-distinct-32bytes",
        {
            ("fleet-service", "av-1", "vehicle.telemetry", "read"),
            ("fleet-service", "av-1", "vehicle.planner", "update_route"),
        },
    )


def make_request(**changes):
    return replace(
        VehicleRequest("req-1", "nonce-1", "fleet-service", "av-1", "vehicle.telemetry", "read"),
        **changes,
    )


def make_frames(seq=1, time=1000, changes=None, vehicle="av-1", frame="frame-1"):
    signals = {s: dict(v) for s, v in SIGNALS.items()}
    for source, values in (changes or {}).items():
        signals[source].update(values)
    return {s: seal_source(s, vehicle, frame, time, seq, signals[s], KEYS[s]) for s in SOURCES}


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.engine = make_engine()
        self.request = make_request()

    def test_fresh_consistent_request_is_allowed(self):
        d = self.engine.evaluate(self.request, make_frames(), 1000)
        self.assertEqual(d.outcome, "ALLOW")
        self.assertEqual(d.mode, "stream")
        self.assertLess(d.risk, 0.25)
        self.assertEqual(len(d.context_digest), 64)

    def test_unauthorized_resource_is_denied(self):
        d = self.engine.evaluate(make_request(resource="vehicle.secret"), make_frames(), 1000)
        self.assertEqual(d.outcome, "DENY")
        self.assertIn("scope_not_authorized", d.reasons)

    def test_authenticated_other_vehicle_requires_explicit_permission(self):
        d = self.engine.evaluate(make_request(vehicle_id="av-2"), make_frames(vehicle="av-2"), 1000)
        self.assertEqual(d.outcome, "DENY")
        self.assertIn("scope_not_authorized", d.reasons)

    def test_actuator_write_is_denied_even_if_acl_lists_it(self):
        engine = VehiclePolicyEngine(
            KEYS,
            b"capability-test-key-distinct-32bytes",
            {("fleet-service", "av-1", "vehicle.actuator", "write")},
        )
        d = engine.evaluate(
            make_request(resource="vehicle.actuator", operation="write"), make_frames(), 1000
        )
        self.assertEqual(d.outcome, "DENY")

    def test_invalid_revoked_or_unattested_identity_is_denied(self):
        for values in (
            {"credential_valid": False},
            {"revoked": True},
            {"attestation_ok": False},
            {"subject_id": "other"},
        ):
            with self.subTest(values=values):
                d = make_engine().evaluate(
                    self.request, make_frames(changes={"identity": values}), 1000
                )
                self.assertEqual(d.outcome, "DENY")

    def test_signed_gnss_speed_disagreement_is_denied(self):
        d = self.engine.evaluate(
            self.request, make_frames(changes={"gnss": {"speed_mps": 20}}), 1000
        )
        self.assertEqual(d.outcome, "DENY")
        self.assertIn("high_context_risk", d.reasons)

    def test_acceleration_and_yaw_disagreement_are_denied(self):
        for values in ({"acceleration_mps2": 5.0}, {"yaw_rate_rps": 1.0}):
            with self.subTest(values=values):
                d = make_engine().evaluate(self.request, make_frames(changes={"imu": values}), 1000)
                self.assertEqual(d.outcome, "DENY")

    def test_moderate_risk_restricts_reads(self):
        d = self.engine.evaluate(
            self.request, make_frames(changes={"can": {"anomaly_score": 0.3}}), 1000
        )
        self.assertEqual((d.outcome, d.mode), ("RESTRICT", "snapshot"))

    def test_moderate_risk_does_not_grant_route_update(self):
        d = self.engine.evaluate(
            make_request(resource="vehicle.planner", operation="update_route"),
            make_frames(changes={"can": {"anomaly_score": 0.3}}),
            1000,
        )
        self.assertEqual(d.outcome, "CHALLENGE")

    def test_uncertainty_cannot_be_averaged_away(self):
        d = self.engine.evaluate(
            self.request, make_frames(changes={"v2x": {"uncertainty": 0.8}}), 1000
        )
        self.assertEqual(d.outcome, "DENY")

    def test_missing_source_fails_closed(self):
        for source in SOURCES:
            frames = make_frames()
            del frames[source]
            with self.subTest(source=source):
                self.assertEqual(make_engine().evaluate(self.request, frames, 1000).outcome, "DENY")

    def test_stale_and_future_source_fail_closed(self):
        for time in (499, 1001):
            with self.subTest(time=time):
                self.assertEqual(
                    make_engine().evaluate(self.request, make_frames(time=time), 1000).outcome,
                    "DENY",
                )

    def test_unsynchronized_sources_fail_closed(self):
        frames = make_frames()
        frames["gnss"] = seal_source(
            "gnss", "av-1", "frame-1", 900, 1, SIGNALS["gnss"], KEYS["gnss"]
        )
        self.assertEqual(self.engine.evaluate(self.request, frames, 1000).outcome, "DENY")

    def test_wrong_vehicle_or_frame_fails_closed(self):
        for field in ("vehicle_id", "frame_id"):
            frames = make_frames()
            body = json.loads(frames["gnss"])["body"]
            args = dict(
                source="gnss",
                vehicle_id=body["vehicle_id"],
                frame_id=body["frame_id"],
                timestamp_ms=1000,
                sequence=1,
                signals=SIGNALS["gnss"],
                key=KEYS["gnss"],
            )
            args[field] = "other"
            frames["gnss"] = seal_source(**args)
            with self.subTest(field=field):
                self.assertEqual(make_engine().evaluate(self.request, frames, 1000).outcome, "DENY")

    def test_unsigned_source_change_is_denied(self):
        frames = make_frames()
        v = json.loads(frames["gnss"])
        v["body"]["signals"]["speed_mps"] = 0
        frames["gnss"] = json.dumps(v)
        self.assertEqual(self.engine.evaluate(self.request, frames, 1000).outcome, "DENY")

    def test_authenticated_sequence_replay_is_denied(self):
        self.engine.evaluate(self.request, make_frames(), 1000)
        d = self.engine.evaluate(
            make_request(request_id="req-2", nonce="nonce-2"), make_frames(), 1000
        )
        self.assertEqual(d.outcome, "DENY")
        self.assertTrue(any("replay" in r for r in d.reasons))

    def test_malformed_duplicate_key_or_oversize_input_is_denied(self):
        for wire in ("[]", '{"body":1,"body":2}', "NaN", "{", "x" * 17000):
            frames = make_frames()
            frames["gnss"] = wire
            with self.subTest(wire=wire[:30]):
                self.assertEqual(make_engine().evaluate(self.request, frames, 1000).outcome, "DENY")

    def test_nan_boolean_and_out_of_range_signals_fail_closed(self):
        for value in (float("nan"), True, -1, 1000, "12"):
            with self.subTest(value=value):
                try:
                    frames = make_frames(changes={"gnss": {"speed_mps": value}})
                except ValueError:
                    continue
                self.assertEqual(make_engine().evaluate(self.request, frames, 1000).outcome, "DENY")

    def test_bad_source_keys_and_config_are_rejected(self):
        with self.assertRaises(ValueError):
            VehiclePolicyEngine({s: b"short" for s in SOURCES}, b"c" * 32, set())
        with self.assertRaises(ValueError):
            VehiclePolicyEngine({s: b"k" * 32 for s in SOURCES}, b"c" * 32, set())

    def test_invalid_request_fields_are_rejected(self):
        for changes in (
            {"nonce": ""},
            {"nonce": True},
            {"resource": "x\n"},
            {"vehicle_id": "v" * 300},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                make_request(**changes)


class CapabilityTests(unittest.TestCase):
    def setUp(self):
        self.engine = make_engine()
        self.request = make_request()
        self.decision = self.engine.evaluate(self.request, make_frames(), 1000)

    def test_grant_is_revalidated_against_fresh_context(self):
        cap = self.engine.issue_capability(self.decision)
        result = self.engine.consume_capability(
            cap, self.request, make_frames(seq=2, time=1010), 1010
        )
        self.assertTrue(result.authorized)
        self.assertEqual(result.mode, "stream")
        self.assertEqual(result.expires_ms, 1250)

    def test_denied_and_forged_decisions_cannot_be_signed(self):
        denied = self.engine.evaluate(make_request(resource="other"), make_frames(seq=2), 1000)
        for decision in (
            denied,
            replace(self.decision, risk=0.0),
            replace(self.decision, request=make_request(resource="other")),
        ):
            with self.subTest(decision=decision.outcome), self.assertRaises(ValueError):
                self.engine.issue_capability(decision)

    def test_revoked_credential_blocks_previously_issued_grant(self):
        cap = self.engine.issue_capability(self.decision)
        result = self.engine.consume_capability(
            cap,
            self.request,
            make_frames(seq=2, time=1010, changes={"identity": {"revoked": True}}),
            1010,
        )
        self.assertFalse(result.authorized)

    def test_context_risk_increase_blocks_grant(self):
        cap = self.engine.issue_capability(self.decision)
        result = self.engine.consume_capability(
            cap,
            self.request,
            make_frames(seq=2, time=1010, changes={"gnss": {"speed_mps": 25}}),
            1010,
        )
        self.assertFalse(result.authorized)

    def test_snapshot_grant_cannot_escalate_to_stream(self):
        engine = make_engine()
        decision = engine.evaluate(
            self.request, make_frames(changes={"can": {"anomaly_score": 0.3}}), 1000
        )
        cap = engine.issue_capability(decision)
        result = engine.consume_capability(cap, self.request, make_frames(seq=2, time=1010), 1010)
        self.assertTrue(result.authorized)
        self.assertEqual(result.mode, "snapshot")

    def test_risk_increase_can_reduce_stream_to_snapshot(self):
        cap = self.engine.issue_capability(self.decision)
        result = self.engine.consume_capability(
            cap,
            self.request,
            make_frames(seq=2, time=1010, changes={"can": {"anomaly_score": 0.3}}),
            1010,
        )
        self.assertTrue(result.authorized)
        self.assertEqual(result.mode, "snapshot")

    def test_expired_and_future_capability_are_denied(self):
        cap = self.engine.issue_capability(self.decision)
        for now in (999, 1250):
            with self.subTest(now=now):
                result = self.engine.consume_capability(
                    cap, self.request, make_frames(seq=2, time=now), now
                )
                self.assertFalse(result.authorized)

    def test_capability_tamper_and_scope_substitution_are_denied(self):
        cap = self.engine.issue_capability(self.decision)
        wire = json.loads(cap)
        wire["body"]["request"]["resource"] = "vehicle.actuator"
        self.assertFalse(
            self.engine.consume_capability(
                json.dumps(wire), self.request, make_frames(seq=2, time=1010), 1010
            ).authorized
        )
        self.assertFalse(
            self.engine.consume_capability(
                cap, make_request(vehicle_id="av-2"), make_frames(seq=2, time=1010), 1010
            ).authorized
        )

    def test_capability_nonce_is_single_use(self):
        cap = self.engine.issue_capability(self.decision)
        self.assertTrue(
            self.engine.consume_capability(
                cap, self.request, make_frames(seq=2, time=1010), 1010
            ).authorized
        )
        result = self.engine.consume_capability(
            cap, self.request, make_frames(seq=3, time=1020), 1020
        )
        self.assertFalse(result.authorized)
        self.assertIn("capability_replay", result.reasons)

    def test_concurrent_capability_use_has_one_winner(self):
        cap = self.engine.issue_capability(self.decision)

        def use(seq):
            return self.engine.consume_capability(
                cap, self.request, make_frames(seq=seq, time=1010), 1010
            ).authorized

        with ThreadPoolExecutor(max_workers=2) as pool:
            self.assertEqual(sum(pool.map(use, (2, 3))), 1)

    def test_policy_change_invalidates_old_capability(self):
        from coact_cps.vehicle.policy import PolicyConfig

        cap = self.engine.issue_capability(self.decision)
        other = VehiclePolicyEngine(
            KEYS,
            b"capability-test-key-distinct-32bytes",
            {("fleet-service", "av-1", "vehicle.telemetry", "read")},
            PolicyConfig(source_ttl_ms=400),
        )
        self.assertFalse(
            other.consume_capability(
                cap, self.request, make_frames(seq=2, time=1010), 1010
            ).authorized
        )

    def test_malformed_capability_fails_closed(self):
        for cap in ("null", "[]", "{", '{"body":1,"body":2}', "x" * 17000):
            with self.subTest(cap=cap[:30]):
                self.assertFalse(
                    self.engine.consume_capability(
                        cap, self.request, make_frames(seq=2, time=1010), 1010
                    ).authorized
                )

    def test_policy_configuration_is_immutable_after_engine_initialization(self):
        from coact_cps.vehicle.policy import PolicyConfig

        with self.assertRaises(AttributeError):
            self.engine.config = PolicyConfig(
                allow_threshold=0.9, restrict_threshold=0.95, deny_threshold=0.99
            )

    def test_clock_rollback_cannot_resurrect_consumed_capability(self):
        cap_a = self.engine.issue_capability(self.decision)
        self.assertTrue(
            self.engine.consume_capability(
                cap_a, self.request, make_frames(seq=2, time=1010), 1010
            ).authorized
        )
        request_b = make_request(request_id="req-b", nonce="nonce-b")
        d_b = self.engine.evaluate(request_b, make_frames(seq=3, time=1300), 1300)
        cap_b = self.engine.issue_capability(d_b)
        self.assertTrue(
            self.engine.consume_capability(
                cap_b, request_b, make_frames(seq=4, time=1310), 1310
            ).authorized
        )
        result = self.engine.consume_capability(
            cap_a, self.request, make_frames(seq=5, time=1020), 1020
        )
        self.assertFalse(result.authorized)
        self.assertIn("clock_rollback", result.reasons)


if __name__ == "__main__":
    unittest.main()
