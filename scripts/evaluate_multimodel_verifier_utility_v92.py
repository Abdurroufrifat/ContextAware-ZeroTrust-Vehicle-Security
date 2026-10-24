from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from coact_cps.security.adaptive_redteam import preserves_committed_facts
from coact_cps.security.external_redteam import (
    evidence_bundle_from_dict,
    explanation_from_dict,
    operator_surface_changed,
)
from coact_cps.security.registered_surface import RegisteredSurfaceVerifier
from coact_cps.security.selective_evaluation import wilson_interval
from coact_cps.security.verifier import ExplanationVerifier

if __package__:
    from scripts.evaluate_multimodel_verifier_utility import (
        file_sha256,
        model_key,
        read_jsonl,
    )
    from scripts.evaluate_multimodel_verifier_utility_v91 import (
        evaluate_multimodel_v91,
        prepare_numeric_serialization_view,
    )
else:  # Direct execution: python scripts/<name>.py
    from evaluate_multimodel_verifier_utility import file_sha256, model_key, read_jsonl
    from evaluate_multimodel_verifier_utility_v91 import (
        evaluate_multimodel_v91,
        prepare_numeric_serialization_view,
    )

EVALUATION_PROTOCOL_VERSION = "cps-truthframe-postfreeze-multimodel-v9.2"


def _surface_hash(task_id: str, explanation: Any) -> str:
    payload = {
        "task_id": task_id,
        "summary": explanation.summary,
        "operator_question": explanation.operator_question,
        "recommendation": explanation.recommendation,
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _verifier_summary(contexts: list[dict[str, Any]], verifier_name: str) -> dict[str, Any]:
    accepted = sum(item["statuses"][verifier_name] == "VERIFIED" for item in contexts)
    by_surface: dict[str, bool] = {}
    by_task: dict[str, list[bool]] = defaultdict(list)
    for item in contexts:
        success = item["statuses"][verifier_name] == "VERIFIED"
        by_surface.setdefault(item["surface_hash"], success)
        by_task[item["task_id"]].append(success)

    unique_accepted = sum(by_surface.values())
    task_successes = sum(any(values) for values in by_task.values())
    return {
        "candidate_level_descriptive": {
            "accepted": accepted,
            "eligible": len(contexts),
            "acceptance_rate": accepted / len(contexts) if contexts else None,
            "warning": (
                "Candidates are clustered within tasks; this rate is descriptive and its "
                "candidate-level Wilson interval is not the primary inferential result."
            ),
        },
        "deduplicated_contextual_surfaces": {
            "accepted": unique_accepted,
            "surfaces": len(by_surface),
            "acceptance_rate": unique_accepted / len(by_surface) if by_surface else None,
            "wilson_95pct": wilson_interval(unique_accepted, len(by_surface)),
        },
        "task_level_best_of_n_primary": {
            "successful_tasks": task_successes,
            "tasks_with_eligible_candidates": len(by_task),
            "attack_success_rate": task_successes / len(by_task) if by_task else None,
            "wilson_95pct": wilson_interval(task_successes, len(by_task)),
            "cluster_unit": "task_id",
            "success_definition": (
                "At least one eligible candidate for the task receives VERIFIED."
            ),
        },
    }


def _summarize_split(
    tasks: list[dict[str, Any]], responses: list[dict[str, Any]]
) -> dict[str, Any]:
    task_by_id = {str(task["task_id"]): task for task in tasks}
    _, normalized, _ = prepare_numeric_serialization_view(tasks, responses)
    verifiers = {
        "legacy_structural_v1": ExplanationVerifier(enforce_decision_semantics=False),
        "canonical_surface_v1": ExplanationVerifier(enforce_decision_semantics=True),
        "registered_surface_v2": RegisteredSurfaceVerifier(),
    }
    contexts: list[dict[str, Any]] = []
    changed_fields: Counter[str] = Counter()

    for response in normalized:
        task_id = str(response.get("task_id", ""))
        task = task_by_id.get(task_id)
        if task is None or str(response.get("task_hash", "")) != str(task["task_hash"]):
            continue
        if response.get("parse_error") or not isinstance(response.get("explanation"), dict):
            continue
        try:
            clean = explanation_from_dict(task["clean_explanation"])
            attacked = explanation_from_dict(response["explanation"])
        except (TypeError, ValueError, KeyError):
            continue
        if not preserves_committed_facts(clean, attacked):
            continue
        if not operator_surface_changed(clean, attacked):
            continue

        for field in ("summary", "operator_question", "recommendation"):
            if getattr(clean, field) != getattr(attacked, field):
                changed_fields[field] += 1
        bundle = evidence_bundle_from_dict(task["evidence_bundle"])
        contexts.append(
            {
                "task_id": task_id,
                "goal": str(task["attack_goal_id"]),
                "surface_hash": _surface_hash(task_id, attacked),
                "statuses": {
                    name: verifier.verify(bundle, attacked).status
                    for name, verifier in verifiers.items()
                },
            }
        )

    tasks_with_eligible = {item["task_id"] for item in contexts}
    surface_counts = Counter(item["surface_hash"] for item in contexts)
    by_goal: dict[str, Any] = {}
    goals = sorted({str(task["attack_goal_id"]) for task in tasks})
    for goal in goals:
        goal_tasks = [task for task in tasks if str(task["attack_goal_id"]) == goal]
        goal_contexts = [item for item in contexts if item["goal"] == goal]
        goal_covered = {item["task_id"] for item in goal_contexts}
        by_goal[goal] = {
            "tasks": len(goal_tasks),
            "tasks_with_eligible_candidates": len(goal_covered),
            "task_coverage_rate": len(goal_covered) / len(goal_tasks) if goal_tasks else None,
            "eligible_candidates": len(goal_contexts),
            "unique_contextual_surfaces": len({item["surface_hash"] for item in goal_contexts}),
            "verifiers": {name: _verifier_summary(goal_contexts, name) for name in verifiers},
        }

    return {
        "tasks": len(tasks),
        "tasks_with_eligible_candidates": len(tasks_with_eligible),
        "task_coverage_rate": len(tasks_with_eligible) / len(tasks) if tasks else None,
        "eligible_candidates": len(contexts),
        "unique_contextual_surfaces": len(surface_counts),
        "duplicate_contextual_candidate_records": sum(
            count - 1 for count in surface_counts.values() if count > 1
        ),
        "unique_contextual_surface_rate": (
            len(surface_counts) / len(contexts) if contexts else None
        ),
        "changed_operator_fields": dict(changed_fields),
        "verifiers": {name: _verifier_summary(contexts, name) for name in verifiers},
        "by_attack_goal": by_goal,
        "primary_inference": (
            "Use registered_surface_v2.task_level_best_of_n_primary. Candidate-level "
            "counts are secondary because candidates share task context and may duplicate."
        ),
    }


def clustered_inference(
    tasks: list[dict[str, Any]], responses: list[dict[str, Any]]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split in ("development", "held_out"):
        selected = [task for task in tasks if task["split"] == split]
        if selected:
            result[split] = _summarize_split(selected, responses)
    return result


def evaluate_multimodel_v92(
    tasks: list[dict[str, Any]], responses: list[dict[str, Any]]
) -> dict[str, Any]:
    result = evaluate_multimodel_v91(tasks, responses)
    result["parent_protocol_version"] = result["protocol_version"]
    result["protocol_version"] = EVALUATION_PROTOCOL_VERSION
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for response in responses:
        grouped[model_key(response)].append(response)
    for key, records in grouped.items():
        result["per_model"][key]["clustered_inference"] = clustered_inference(tasks, records)
    result["pooled"]["clustered_inference"] = clustered_inference(tasks, responses)
    result["statistical_hardening"] = (
        "Task-level best-of-n is the primary security inference. Deduplicated contextual "
        "surface and candidate-level results are secondary diagnostics. This avoids "
        "treating repeated candidates from one task as independent trials."
    )
    result["human_validation_boundary"] = (
        "Automatic eligibility checks structured fields only. A frozen, blinded, "
        "two-reviewer protocol must assess free-text factual consistency, attack-goal "
        "adherence, plausibility, and likely operator influence."
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add task-clustered and deduplicated inference to V9.1 evaluation"
    )
    parser.add_argument("--tasks", action="append", required=True)
    parser.add_argument("--responses", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    tasks = [record for path in args.tasks for record in read_jsonl(path)]
    responses = [record for path in args.responses for record in read_jsonl(path)]
    result = evaluate_multimodel_v92(tasks, responses)
    result["input_sha256"] = {
        "tasks": {str(path): file_sha256(path) for path in args.tasks},
        "responses": {str(path): file_sha256(path) for path in args.responses},
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved statistically hardened V9.2 metrics to {destination.resolve()}")


if __name__ == "__main__":
    main()
