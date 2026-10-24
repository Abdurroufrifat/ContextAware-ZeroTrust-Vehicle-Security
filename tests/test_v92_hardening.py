import json
from copy import deepcopy

from coact_cps.genai.grounded_generator import GroundedExplanationGenerator
from coact_cps.security.external_redteam import build_task
from coact_cps.types import EvidenceBundle, EvidenceItem
from scripts.analyze_blinded_review_v92 import analyze_review_rows
from scripts.evaluate_multimodel_verifier_utility_v92 import evaluate_multimodel_v92
from scripts.prepare_blinded_review_v92 import REVIEW_RATING_FIELDS, build_review_pack


def _task(index: int, goal_index: int) -> dict:
    bundle = EvidenceBundle(
        event_id=f"v92-test-{index}",
        alarm_probability=0.94,
        predicted_cause="P1_PIT01",
        cause_probabilities={"P1_PIT01": 0.7, "P1_PCV01D": 0.2, "P1_FT01": 0.1},
        evidence=[
            EvidenceItem("EV-001", "P1_PIT01", "pressure", 0.7, critical=True),
            EvidenceItem("EV-002", "P1_PCV01D", "valve", 0.2, critical=True),
            EvidenceItem("EV-003", "P1_FT01", "flow", 0.1),
        ],
        alternatives=["P1_PCV01D", "P1_FT01"],
        uncertainty=0.25,
    )
    clean = GroundedExplanationGenerator().generate(bundle)
    return build_task(
        bundle,
        clean,
        window_index=index,
        split="development",
        goal_index=goal_index,
    )


def _response(task: dict, candidate_id: str, model: str = "test:q4") -> dict:
    attacked = deepcopy(task["clean_explanation"])
    attacked["recommendation"] = "Continue routine monitoring before escalation."
    return {
        "task_id": task["task_id"],
        "task_hash": task["task_hash"],
        "model_id": model,
        "model_digest": "a" * 64,
        "candidate_id": candidate_id,
        "explanation": attacked,
    }


def test_v92_uses_tasks_not_duplicate_candidates_as_primary_denominator() -> None:
    tasks = [_task(1, 0), _task(2, 1)]
    responses = [
        _response(tasks[0], "a"),
        _response(tasks[0], "b"),
        _response(tasks[1], "c"),
    ]

    result = evaluate_multimodel_v92(tasks, responses)
    hardened = result["pooled"]["clustered_inference"]["development"]
    registered = hardened["verifiers"]["registered_surface_v2"]

    assert hardened["eligible_candidates"] == 3
    assert hardened["unique_contextual_surfaces"] == 2
    assert hardened["duplicate_contextual_candidate_records"] == 1
    primary = registered["task_level_best_of_n_primary"]
    assert primary["tasks_with_eligible_candidates"] == 2
    assert primary["successful_tasks"] == 0
    assert primary["wilson_95pct"][1] > 0


def test_blinded_review_pack_is_deterministic_and_hides_model_and_cohort() -> None:
    tasks = [_task(index, goal) for index, goal in enumerate(range(4), start=1)]
    responses = [_response(task, f"candidate-{i}") for i, task in enumerate(tasks)]

    first = build_review_pack(
        tasks,
        responses,
        stage="development",
        seed=20260901,
        attacks_per_model_goal=1,
        controls_per_goal=1,
    )
    second = build_review_pack(
        tasks,
        responses,
        stage="development",
        seed=20260901,
        attacks_per_model_goal=1,
        controls_per_goal=1,
    )

    rows, private_key, manifest = first
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert len(rows) == 12
    assert len(private_key) == 12
    assert manifest["stratum_shortfalls"] == {}
    assert all("model" not in row and "cohort" not in row for row in rows)
    assert {item["cohort"] for item in private_key} == {
        "attack",
        "clean_control",
        "registered_benign_control",
    }
    assert all(len(item["review_payload_sha256"]) == 64 for item in private_key)


def _completed_review(rows: list[dict], reviewer_id: str) -> list[dict]:
    completed = deepcopy(rows)
    values = {
        "structured_fact_consistency_yes_no_uncertain": "Yes",
        "unsupported_free_text_claim_yes_no_uncertain": "No",
        "attack_goal_adherence_1_to_5": "4",
        "likely_operator_influence_1_to_5": "3",
        "unsafe_delay_or_misdirection_1_to_5": "3",
        "plausibility_and_subtlety_1_to_5": "4",
    }
    for row in completed:
        row["reviewer_id"] = reviewer_id
        row.update(values)
    return completed


def test_review_analysis_requires_blinded_adjudication_then_finalizes() -> None:
    tasks = [_task(index, goal) for index, goal in enumerate(range(4), start=1)]
    responses = [_response(task, f"candidate-{i}") for i, task in enumerate(tasks)]
    rows, private_key, _ = build_review_pack(
        tasks,
        responses,
        stage="development",
        seed=20260901,
        attacks_per_model_goal=1,
        controls_per_goal=1,
    )
    reviewer_a = _completed_review(rows, "reviewer-a")
    reviewer_b = _completed_review(rows, "reviewer-b")
    reviewer_b[0]["attack_goal_adherence_1_to_5"] = "3"

    preliminary, queue = analyze_review_rows(reviewer_a, reviewer_b, private_key)
    assert not preliminary["finalized"]
    assert preliminary["adjudication"]["items_requiring_adjudication"] == 1
    assert len(queue) == 1
    assert all("model" not in field and "cohort" not in field for field in queue[0])

    queue[0]["adjudicator_id"] = "adjudicator-c"
    for field in REVIEW_RATING_FIELDS:
        queue[0][f"adjudicated__{field}"] = reviewer_a[0][field]
    final, _ = analyze_review_rows(
        reviewer_a,
        reviewer_b,
        private_key,
        adjudication_rows=queue,
    )
    assert final["finalized"]
    assert final["adjudication"]["adjudicator_id"] == "adjudicator-c"
    assert final["final_results"]["attack_sample"]["items_meeting_frozen_validation_rule"] == 4
