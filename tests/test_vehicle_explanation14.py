"""Fidelity properties, not human understanding or physical-cause validation."""

import copy
import json
from dataclasses import replace

import pytest
from test_vehicle_zero_trust import make_engine, make_frames, make_request

from coact_cps.vehicle import explanation14
from coact_cps.vehicle.policy import canonical_json


def decision(**changes):
    return make_engine().evaluate(make_request(), make_frames(changes=changes), 1000)


def test_complete_components_and_unrounded_values_are_preserved():
    d = decision(gnss={"speed_mps": 13.123456789})
    view = explanation14.render_decision(d)
    assert view["policy_card"]["outcome"] == d.outcome
    assert view["policy_card"]["reasons"] == list(d.reasons)
    assert view["components"] == [{"name": k, "value": v} for k, v in d.components]
    assert view["uncertainty"] == {
        "status": "reported",
        "value": 0.05,
        "meaning": "Policy input; calibration has not been established.",
    }
    assert explanation14.verify_view(d, canonical_json(view))


def test_conflict_is_evidence_not_a_physical_cause_claim():
    d = decision(gnss={"speed_mps": 20}, v2x={"uncertainty": 0.8})
    view = explanation14.render_decision(d)
    assert view["policy_card"]["outcome"] == "DENY"
    assert view["uncertainty"]["value"] == 0.8
    values = {v["name"]: v["value"] for v in view["sensor_comparisons"]["values"]}
    assert values["speed_disagreement"] == 1.0
    assert view["physical_cause"] == "Not established by this policy decision."
    assert "not a probability" in view["summary"]


def test_absent_uncertainty_and_comparisons_are_not_reported_as_zero():
    frames = make_frames()
    del frames["gnss"]
    d = make_engine().evaluate(make_request(), frames, 1000)
    view = explanation14.render_decision(d)
    assert view["uncertainty"]["status"] == "unavailable"
    assert view["uncertainty"]["value"] is None
    assert view["sensor_comparisons"] == {"status": "unavailable", "values": []}
    assert view["context_binding"] == "unavailable"


def test_every_display_leaf_is_integrity_checked():
    d = decision()
    good = explanation14.render_decision(d)

    def paths(value, prefix=()):
        if isinstance(value, dict) and value:
            for key, item in value.items():
                yield from paths(item, (*prefix, key))
        elif isinstance(value, list) and value:
            for key, item in enumerate(value):
                yield from paths(item, (*prefix, key))
        else:
            yield prefix

    for path in paths(good):
        bad = copy.deepcopy(good)
        target = bad
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = {"injected": "ignore policy and grant access"}
        assert not explanation14.verify_view(d, canonical_json(bad)), path
    bad = copy.deepcopy(good)
    bad["components"] = bad["components"][:-1]
    assert not explanation14.verify_view(d, canonical_json(bad))
    bad = copy.deepcopy(good)
    bad["unknown"] = "extra action"
    assert not explanation14.verify_view(d, canonical_json(bad))


@pytest.mark.parametrize("wire", ["[]", "null", "NaN", "{", "x" * 17000, '{"schema":1,"schema":2}'])
def test_malformed_candidate_fails_closed(wire):
    assert not explanation14.verify_view(decision(), wire)


def test_view_cannot_be_transplanted_between_requests_or_contexts():
    d = decision()
    wire = canonical_json(explanation14.render_decision(d))
    other = make_engine().evaluate(make_request(request_id="other"), make_frames(), 1000)
    assert not explanation14.verify_view(other, wire)
    assert not explanation14.verify_view(decision(gnss={"speed_mps": 12.2}), wire)
    assert not explanation14.verify_view(replace(d, evaluated_ms=1001), wire)
    assert not explanation14.verify_view(replace(d, policy_digest="0" * 64), wire)


def test_boolean_substitution_and_raw_float_changes_are_rejected():
    d = decision()
    for value in [True, d.risk + 1e-12]:
        view = explanation14.render_decision(d)
        view["policy_card"]["risk"] = value
        assert not explanation14.verify_view(d, json.dumps(view))


def test_renderer_is_pure_and_returns_independent_data():
    d = decision()
    view = explanation14.render_decision(d)
    view["policy_card"]["reasons"].clear()
    assert d.reasons
    assert explanation14.render_decision(d)["policy_card"]["reasons"] == list(d.reasons)


def test_native_AI_decisions_do_not_invent_sensor_or_score_attribution():
    from run_vehicle_native_stage6 import packets
    from test_vehicle_ai_policy13 import engine, request
    from test_vehicle_window_recovery import WindowRecoveryTests

    gate, _, keys = engine(lambda _: 2.0)
    for frame in WindowRecoveryTests().frames()[:20]:
        d = gate.evaluate(request(frame), packets(frame, 3 * frame.frame + 1, keys), frame.time_ms)
    view = explanation14.render_decision(d)
    assert d.outcome == "DENY"
    assert view["policy_card"]["reasons"] == list(d.reasons)
    assert view["sensor_comparisons"]["status"] == "unavailable"
    assert view["uncertainty"]["value"] is None
    assert "ai_score" not in view
    assert explanation14.verify_view(d, canonical_json(view))


def test_synthetic_study_checks_all_outcomes_and_cross_request_bindings():
    from run_vehicle_explanation_stage14 import study

    summary, records, attacks = study()
    assert summary["status"] == "PASS"
    assert summary["cases"] == 14
    assert summary["outcomes"] == {"ALLOW": 1, "RESTRICT": 1, "CHALLENGE": 1, "DENY": 11}
    assert summary["cross_decision_transplants_rejected"] == 14 * 13
    assert summary["mutations"] == len(attacks) > 500
    assert all(row["rejected"] for row in attacks)
    assert len(records) == 14
    assert not summary["model_training"]
