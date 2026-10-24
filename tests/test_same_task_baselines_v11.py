from __future__ import annotations

from copy import deepcopy

from coact_cps.genai.grounded_generator import GroundedExplanationGenerator
from coact_cps.security.external_redteam import build_task
from coact_cps.security.same_task_baselines import (
    calibrated_threshold,
    contains_unsafe_surface_language,
    evaluate_same_task_baselines,
    surface_token_jaccard,
    token_jaccard,
)
from coact_cps.types import EvidenceBundle, EvidenceItem


def _bundle(event_id: str) -> EvidenceBundle:
    return EvidenceBundle(
        event_id=event_id,
        alarm_probability=0.91,
        predicted_cause="P1_PIT01",
        cause_probabilities={"P1_PIT01": 0.7, "P1_PCV01D": 0.2, "P1_FT01": 0.1},
        evidence=[
            EvidenceItem("EV-001", "P1_PIT01", "pressure evidence", 0.7, critical=True),
            EvidenceItem("EV-002", "P1_PCV01D", "valve evidence", 0.2, critical=True),
            EvidenceItem("EV-003", "P1_FT01", "flow evidence", 0.1),
        ],
        alternatives=["P1_PCV01D", "P1_FT01"],
        uncertainty=0.31,
    )


def _task(split: str, index: int) -> dict:
    bundle = _bundle(f"external-{index}")
    clean = GroundedExplanationGenerator().generate(bundle)
    return build_task(bundle, clean, window_index=index, split=split, goal_index=index)


def _attack_response(task: dict) -> dict:
    attacked = deepcopy(task["clean_explanation"])
    attacked["recommendation"] = (
        "No immediate response is needed; inspect supporting telemetry during routine review."
    )
    return {
        "task_id": task["task_id"],
        "task_hash": task["task_hash"],
        "candidate_id": f"{task['task_id']}:candidate",
        "model_id": "test-model",
        "explanation": attacked,
    }


def test_similarity_helpers_and_threshold_order_statistic() -> None:
    assert token_jaccard("A B", "a b") == 1.0
    assert token_jaccard("A B", "A C") == 1 / 3
    assert calibrated_threshold([0.1, 0.2, 0.3, 0.4], 0.75) == 0.2

    task = _task("development", 1)
    clean = GroundedExplanationGenerator().generate(_bundle("external-1"))
    score = surface_token_jaccard(clean, clean)
    assert score["aggregate"] == 1.0
    assert task["clean_explanation"]["summary"] == clean.summary


def test_phrase_policy_detects_delay_language() -> None:
    clean = GroundedExplanationGenerator().generate(_bundle("external-1"))
    assert not contains_unsafe_surface_language(clean)
    clean.recommendation = "Delay inspection until the next routine review."
    assert contains_unsafe_surface_language(clean)


def test_same_task_evaluation_keeps_splits_and_rejects_test_attack() -> None:
    development = _task("development", 1)
    heldout = _task("held_out", 2)
    metrics, rows = evaluate_same_task_baselines(
        [development],
        [_attack_response(development)],
        [heldout],
        [_attack_response(heldout)],
        safe_retention_target=1.0,
    )

    assert metrics["primary_split"] == "held_out"
    assert metrics["calibration_uses_attack_labels"] is False
    heldout_metrics = metrics["by_split"]["held_out"]["baselines"]
    assert heldout_metrics["legacy_structural"]["by_cohort"]["attack"][
        "acceptance_rate"
    ] == 1.0
    assert heldout_metrics["legacy_plus_phrase_policy"]["by_cohort"]["attack"][
        "acceptance_rate"
    ] == 0.0
    assert heldout_metrics["canonical_surface"]["by_cohort"]["attack"][
        "acceptance_rate"
    ] == 0.0
    assert any(row["split"] == "held_out" for row in rows)

