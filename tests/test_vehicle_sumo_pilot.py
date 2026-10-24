from __future__ import annotations

import copy
import importlib.util
import json
import math
import sys
import unittest
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts/vehicle_sumo_pilot_core.py"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def core():
    spec = importlib.util.spec_from_file_location("vehicle_pilot_core", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def trace(n=70):
    return [
        {
            "frame": i,
            "time_s": 0.2 * i,
            "x_m": 2 * i,
            "y_m": 0.0,
            "speed_mps": 10.0,
            "acceleration_mps2": 0.0,
            "angle_deg": 90.0,
        }
        for i in range(n)
    ]


class PilotTests(unittest.TestCase):
    def test_observations_are_repeatable_and_contain_no_attack_labels(self):
        m = core()
        one = m.make_observations(trace(), 11)
        self.assertEqual(one, m.make_observations(trace(), 11))
        self.assertNotEqual(one, m.make_observations(trace(), 23))
        self.assertEqual(set(one[0]), {"frame", "timestamp_ms", "signals"})
        self.assertEqual(set(one[0]["signals"]), set(m.SOURCES))
        self.assertNotIn("attack", json.dumps(one))

    def test_stationary_proxy_is_finite_and_uses_correct_units(self):
        m = core()
        t = trace()
        for row in t:
            row.update(x_m=0.0, speed_mps=0.0)
        observations = m.make_observations(t, 11)
        for row in observations:
            for source in ("gnss", "imu", "wheel"):
                for value in row["signals"][source].values():
                    self.assertTrue(math.isfinite(value))
            self.assertGreaterEqual(row["signals"]["gnss"]["speed_mps"], 0)

    def test_angle_wrap_does_not_create_full_rotation_jump(self):
        m = core()
        self.assertAlmostEqual(m.wrapped_angle_delta(359, 1), 2)
        self.assertAlmostEqual(m.wrapped_angle_delta(1, 359), -2)

    def test_injection_does_not_modify_base_data_or_leak_labels(self):
        m = core()
        observed = m.make_observations(trace(), 11)[40]
        before = copy.deepcopy(observed)
        changed, transport = m.inject_scenario(observed, "gnss_spoof", 0.5)
        self.assertEqual(before, observed)
        self.assertGreater(
            changed["signals"]["gnss"]["speed_mps"], observed["signals"]["gnss"]["speed_mps"] + 5
        )
        self.assertNotIn("attack", changed)
        self.assertEqual(transport, {})

    def test_benign_loss_and_stale_context_are_transport_changes(self):
        m = core()
        observed = m.make_observations(trace(), 11)[40]
        _, loss = m.inject_scenario(observed, "benign_v2x_loss", 0.5)
        _, stale = m.inject_scenario(observed, "stale_gnss", 0.5)
        self.assertEqual(loss["missing"], "v2x")
        self.assertGreater(stale["age_ms"], 500)

    def test_coordinated_spoof_preserves_speed_residual(self):
        m = core()
        observed = m.make_observations(trace(), 11)[40]
        before = abs(
            observed["signals"]["gnss"]["speed_mps"] - observed["signals"]["wheel"]["speed_mps"]
        )
        changed, _ = m.inject_scenario(observed, "coordinated_speed_spoof", 0.5)
        after = abs(
            changed["signals"]["gnss"]["speed_mps"] - changed["signals"]["wheel"]["speed_mps"]
        )
        self.assertAlmostEqual(before, after)

    def test_identity_only_and_full_policy_differ_on_signed_gnss_spoof(self):
        m = core()
        observed = m.make_observations(trace(), 11)[40]
        changed, transport = m.inject_scenario(observed, "gnss_spoof", 0.5)
        harness = m.PolicyHarness("ego")
        result = harness.step(changed, transport, active=True, scenario="gnss_spoof")
        self.assertEqual(result["identity_only"]["outcome"], "ALLOW")
        self.assertEqual(result["full"]["outcome"], "DENY")
        self.assertFalse(result["full"]["any_grant"])

    def test_post_issuance_revocation_is_blocked_at_consumption(self):
        m = core()
        observed = m.make_observations(trace(), 11)[40]
        changed, transport = m.inject_scenario(observed, "post_issue_revocation", 0.5)
        result = m.PolicyHarness("ego").step(
            changed, transport, active=True, scenario="post_issue_revocation"
        )
        self.assertTrue(result["full_no_use_recheck"]["any_grant"])
        self.assertFalse(result["full"]["any_grant"])

    def test_display_binding_rejects_misrepresentation_without_changing_enforcement(self):
        m = core()
        observed = m.make_observations(trace(), 11)[40]
        changed, transport = m.inject_scenario(observed, "display_downplay", 0.5)
        result = m.PolicyHarness("ego").step(
            changed, transport, active=True, scenario="display_downplay"
        )
        self.assertFalse(result["full"]["display_accepted"])
        self.assertTrue(result["full_unbound_display"]["display_accepted"])
        self.assertEqual(result["full"]["any_grant"], result["full_unbound_display"]["any_grant"])

    def test_replay_and_source_loss_fail_closed_for_full_policy(self):
        m = core()
        observations = m.make_observations(trace(), 11)
        for scenario in ("source_replay", "benign_v2x_loss"):
            harness = m.PolicyHarness("ego")
            harness.step(observations[39], {}, active=False, scenario=scenario)
            observed, transport = m.inject_scenario(observations[40], scenario, 0.5)
            result = harness.step(observed, transport, active=True, scenario=scenario)
            self.assertFalse(result["full"]["any_grant"])

    def test_exposure_card_rejects_changed_authorization_deadline(self):
        m = core()
        card = {"policy": {"outcome": "ALLOW"}, "service": {"authorized": True, "expires_ms": 1250}}
        altered = copy.deepcopy(card)
        altered["service"]["expires_ms"] = 2000
        self.assertTrue(m.verify_exposure_card(card, json.dumps(card)))
        self.assertFalse(m.verify_exposure_card(card, json.dumps(altered)))

    def test_summary_retains_coordinated_failure_and_benign_denial(self):
        m = core()
        rows = [
            dict(
                policy="full",
                seed=11,
                scenario="coordinated_speed_spoof",
                active=True,
                is_attack=True,
                any_grant=True,
                full_grant=True,
                outcome="ALLOW",
                display_accepted=True,
                display_mutated=False,
                latency_ms=0.5,
            ),
            dict(
                policy="full",
                seed=11,
                scenario="benign_v2x_loss",
                active=True,
                is_attack=False,
                any_grant=False,
                full_grant=False,
                outcome="DENY",
                display_accepted=True,
                display_mutated=False,
                latency_ms=0.6,
            ),
        ]
        summary = m.summarize(rows)
        self.assertEqual(summary["full"]["attack_window_any_grant_fraction"], 1)
        self.assertEqual(summary["full"]["benign_any_grant_fraction"], 0)


if __name__ == "__main__":
    unittest.main()
