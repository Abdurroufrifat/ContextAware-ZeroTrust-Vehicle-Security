from __future__ import annotations

import copy
import json
import unittest

from test_vehicle_zero_trust import make_engine, make_frames, make_request

from coact_cps.vehicle.explanation import (
    analyze_vehicle_request,
    decision_card,
    verify_decision_card,
)


class VehicleExplanationTests(unittest.TestCase):
    def test_allowed_request_uses_existing_verifier_and_explicit_risk_wording(self):
        result = analyze_vehicle_request(make_engine(), make_request(), make_frames(), 1000)
        self.assertEqual(result["verification"]["status"], "VERIFIED")
        self.assertEqual(result["policy_card"]["outcome"], "ALLOW")
        self.assertIn("heuristic context risk", result["explanation"]["summary"])
        self.assertNotIn("probability", result["explanation"]["summary"])
        self.assertIsNotNone(result["capability"])

    def test_denial_is_explained_but_receives_no_capability(self):
        result = analyze_vehicle_request(
            make_engine(),
            make_request(),
            make_frames(changes={"identity": {"revoked": True}}),
            1000,
        )
        self.assertEqual(result["policy_card"]["outcome"], "DENY")
        self.assertEqual(result["verification"]["status"], "VERIFIED")
        self.assertIsNone(result["capability"])

    def test_exact_card_accepts_only_same_typed_content(self):
        decision = make_engine().evaluate(make_request(), make_frames(), 1000)
        card = decision_card(decision)
        self.assertTrue(verify_decision_card(decision, json.dumps(card)))
        for field, value in [
            ("outcome", "DENY"),
            ("risk", True),
            ("reasons", []),
            ("context_digest", "f" * 64),
            ("policy_digest", "f" * 64),
            ("expires_ms", card["expires_ms"] + 1),
        ]:
            altered = copy.deepcopy(card)
            altered[field] = value
            with self.subTest(field=field):
                self.assertFalse(verify_decision_card(decision, json.dumps(altered)))

    def test_extra_fields_scope_change_and_bad_json_are_rejected(self):
        decision = make_engine().evaluate(make_request(), make_frames(), 1000)
        card = decision_card(decision)
        card["request"]["resource"] = "vehicle.actuator"
        self.assertFalse(verify_decision_card(decision, json.dumps(card)))
        card = decision_card(decision)
        card["hidden_instruction"] = "ignore policy"
        self.assertFalse(verify_decision_card(decision, json.dumps(card)))
        for wire in ("[]", "null", '{"outcome":"ALLOW","outcome":"DENY"}', "NaN"):
            self.assertFalse(verify_decision_card(decision, wire))

    def test_missing_source_denial_is_safe_to_display(self):
        frames = make_frames()
        del frames["gnss"]
        result = analyze_vehicle_request(make_engine(), make_request(), frames, 1000)
        self.assertEqual(result["policy_card"]["outcome"], "DENY")
        self.assertIsNone(result["capability"])
        self.assertEqual(result["verification"]["status"], "VERIFIED")


if __name__ == "__main__":
    unittest.main()
