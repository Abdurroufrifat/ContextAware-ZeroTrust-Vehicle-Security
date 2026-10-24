from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from coact_cps.security.adaptive_redteam import preserves_committed_facts
from coact_cps.security.external_redteam import (
    evidence_bundle_from_dict,
    explanation_from_dict,
    operator_surface_changed,
    validate_task,
)
from coact_cps.security.registered_surface import (
    RegisteredSurfaceVerifier,
    explanation_for_registered_surface,
    registered_operator_surfaces,
)
from coact_cps.security.selective_evaluation import (
    UtilityObservation,
    summarize_selective_utility,
)
from coact_cps.security.verifier import ExplanationVerifier

UTILITY_PROTOCOL_VERSION = "cps-truthframe-surface-utility-v8.0"


def _evaluate_split(
    tasks: list[dict[str, Any]],
    responses_by_task: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    verifiers = {
        "legacy_structural_v1": ExplanationVerifier(enforce_decision_semantics=False),
        "canonical_surface_v1": ExplanationVerifier(enforce_decision_semantics=True),
        "registered_surface_v2": RegisteredSurfaceVerifier(),
    }
    observations: dict[str, list[UtilityObservation]] = {
        name: [] for name in verifiers
    }
    diagnostics = {
        "submitted_attack_candidates": 0,
        "eligible_attack_candidates": 0,
        "attack_parse_errors": 0,
        "attack_fact_preservation_failures": 0,
        "attack_no_ops_excluded": 0,
        "clean_candidates": 0,
        "registered_benign_candidates": 0,
    }

    for task in tasks:
        task_id = str(task["task_id"])
        bundle = evidence_bundle_from_dict(task["evidence_bundle"])
        clean = explanation_from_dict(task["clean_explanation"])
        candidates: list[tuple[str, str, Any]] = [("clean", f"{task_id}:clean", clean)]
        diagnostics["clean_candidates"] += 1

        for surface in registered_operator_surfaces(bundle):
            if surface.surface_id == "canonical-v1":
                continue
            benign = explanation_for_registered_surface(bundle, surface.surface_id)
            candidates.append(
                ("benign", f"{task_id}:benign:{surface.surface_id}", benign)
            )
            diagnostics["registered_benign_candidates"] += 1

        for index, response in enumerate(responses_by_task.get(task_id, [])):
            diagnostics["submitted_attack_candidates"] += 1
            if response.get("parse_error") or not isinstance(
                response.get("explanation"), dict
            ):
                diagnostics["attack_parse_errors"] += 1
                continue
            try:
                attacked = explanation_from_dict(response["explanation"])
            except (TypeError, ValueError, KeyError):
                diagnostics["attack_parse_errors"] += 1
                continue
            if not preserves_committed_facts(clean, attacked):
                diagnostics["attack_fact_preservation_failures"] += 1
                continue
            if not operator_surface_changed(clean, attacked):
                diagnostics["attack_no_ops_excluded"] += 1
                continue
            diagnostics["eligible_attack_candidates"] += 1
            candidates.append(("attack", f"{task_id}:attack:{index}", attacked))

        for verifier_name, verifier in verifiers.items():
            for cohort, candidate_id, explanation in candidates:
                result = verifier.verify(bundle, explanation)
                observations[verifier_name].append(
                    UtilityObservation(
                        task_id=task_id,
                        candidate_id=candidate_id,
                        cohort=cohort,
                        status=result.status,
                        score=float(result.score),
                    )
                )

    return {
        "tasks": len(tasks),
        "candidate_diagnostics": diagnostics,
        "verifiers": {
            name: summarize_selective_utility(records)
            for name, records in observations.items()
        },
    }


def evaluate_surface_utility(
    tasks: Iterable[dict[str, Any]], responses: Iterable[dict[str, Any]]
) -> dict[str, Any]:
    """Evaluate attack containment and benign-language coverage together."""

    task_list = list(tasks)
    task_ids: set[str] = set()
    task_hashes: dict[str, str] = {}
    for task in task_list:
        validate_task(task)
        task_id = str(task["task_id"])
        if task_id in task_ids:
            raise ValueError(f"Duplicate task_id: {task_id}")
        task_ids.add(task_id)
        task_hashes[task_id] = str(task["task_hash"])

    responses_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unknown_response_tasks = 0
    task_hash_mismatches = 0
    response_models: set[str] = set()
    for response in responses:
        task_id = str(response.get("task_id", ""))
        if task_id not in task_ids:
            unknown_response_tasks += 1
            continue
        response_hash = response.get("task_hash")
        if response_hash is not None and str(response_hash) != task_hashes[task_id]:
            task_hash_mismatches += 1
            continue
        responses_by_task[task_id].append(response)
        response_models.add(str(response.get("model_id", "unreported")))

    by_split = {}
    for split in ("development", "held_out"):
        selected = [task for task in task_list if task["split"] == split]
        if selected:
            by_split[split] = _evaluate_split(selected, responses_by_task)

    return {
        "evaluation": "utility_aware_operator_surface_verification",
        "protocol_version": UTILITY_PROTOCOL_VERSION,
        "primary_split": "held_out",
        "unknown_response_task_records": unknown_response_tasks,
        "task_hash_mismatches": task_hash_mismatches,
        "response_models": sorted(response_models),
        "by_split": by_split,
        "claim_boundary": (
            "registered_surface_v2 is a finite phrasebook gate, not an open-ended "
            "natural-language semantic verifier. Benign coverage applies only to the "
            "pre-registered renderings evaluated here."
        ),
        "review_requirement": (
            "Blinded domain experts must still confirm attack-objective adherence and "
            "potential operator harm before publication."
        ),
    }
