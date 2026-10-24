from __future__ import annotations

import argparse
import json
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

from coact_cps.security.surface_utility import evaluate_surface_utility

if __package__:
    from scripts.evaluate_multimodel_verifier_utility import (
        file_sha256,
        model_key,
        provenance_summary,
        read_jsonl,
    )
    from scripts.v91_numeric_protocol import (
        NUMERIC_SERIALIZATION_PROTOCOL,
        UNCERTAINTY_ABSOLUTE_TOLERANCE,
        canonicalize_uncertainty_serialization,
        exact_committed_facts_match,
        material_committed_facts_match,
    )
else:  # Direct execution: python scripts/<name>.py
    from evaluate_multimodel_verifier_utility import (
        file_sha256,
        model_key,
        provenance_summary,
        read_jsonl,
    )
    from v91_numeric_protocol import (
        NUMERIC_SERIALIZATION_PROTOCOL,
        UNCERTAINTY_ABSOLUTE_TOLERANCE,
        canonicalize_uncertainty_serialization,
        exact_committed_facts_match,
        material_committed_facts_match,
    )

EVALUATION_PROTOCOL_VERSION = "cps-truthframe-postfreeze-multimodel-v9.1"


def _raw_explanation(record: dict[str, Any]) -> Any:
    value = record.get("raw_explanation")
    return value if isinstance(value, dict) else record.get("explanation")


def prepare_numeric_serialization_view(
    tasks: list[dict[str, Any]], responses: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    tasks_by_id = {str(task["task_id"]): task for task in tasks}
    raw_view: list[dict[str, Any]] = []
    normalized_view: list[dict[str, Any]] = []
    counters = {
        "response_records": len(responses),
        "parsed_explanations": 0,
        "raw_exact_fact_preserving": 0,
        "raw_numeric_only_failures": 0,
        "normalization_applied": 0,
        "post_normalization_fact_preserving": 0,
        "material_fact_failures": 0,
        "outside_tolerance": 0,
        "unknown_task_records": 0,
    }
    applied_deltas: list[float] = []

    for response in responses:
        raw_record = deepcopy(response)
        normalized_record = deepcopy(response)
        task = tasks_by_id.get(str(response.get("task_id", "")))
        explanation = _raw_explanation(response)
        if task is None:
            counters["unknown_task_records"] += 1
            raw_view.append(raw_record)
            normalized_view.append(normalized_record)
            continue
        if not isinstance(explanation, dict):
            raw_view.append(raw_record)
            normalized_view.append(normalized_record)
            continue

        counters["parsed_explanations"] += 1
        clean = task["clean_explanation"]
        raw_record["explanation"] = deepcopy(explanation)
        normalized, audit = canonicalize_uncertainty_serialization(explanation, clean)
        normalized_record["explanation"] = normalized
        normalized_record["numeric_normalization"] = audit

        material_match = material_committed_facts_match(clean, explanation)
        exact_match = exact_committed_facts_match(clean, explanation)
        if exact_match:
            counters["raw_exact_fact_preserving"] += 1
        if not material_match:
            counters["material_fact_failures"] += 1
        if material_match and not exact_match and audit["applied"]:
            counters["raw_numeric_only_failures"] += 1
        if audit["applied"]:
            counters["normalization_applied"] += 1
            applied_deltas.append(float(audit["absolute_delta"]))
        if audit["reason"] == "outside_tolerance":
            counters["outside_tolerance"] += 1
        if exact_committed_facts_match(clean, normalized):
            counters["post_normalization_fact_preserving"] += 1

        raw_view.append(raw_record)
        normalized_view.append(normalized_record)

    diagnostics: dict[str, Any] = {
        "protocol_version": NUMERIC_SERIALIZATION_PROTOCOL,
        "field": "uncertainty",
        "absolute_tolerance": UNCERTAINTY_ABSOLUTE_TOLERANCE,
        "semantic_repairs_permitted": False,
        **counters,
        "maximum_applied_absolute_delta": max(applied_deltas, default=0.0),
    }
    return raw_view, normalized_view, diagnostics


def _diagnostics_by_split(evaluation: dict[str, Any]) -> dict[str, Any]:
    return {
        split: value["candidate_diagnostics"]
        for split, value in evaluation.get("by_split", {}).items()
    }


def evaluation_bundle(
    tasks: list[dict[str, Any]], responses: list[dict[str, Any]]
) -> dict[str, Any]:
    raw_view, normalized_view, diagnostics = prepare_numeric_serialization_view(
        tasks, responses
    )
    raw_evaluation = evaluate_surface_utility(tasks, raw_view)
    normalized_evaluation = evaluate_surface_utility(tasks, normalized_view)
    return {
        "provenance": provenance_summary(responses),
        "numeric_serialization": diagnostics,
        "pre_normalization_candidate_diagnostics": _diagnostics_by_split(raw_evaluation),
        "evaluation": normalized_evaluation,
    }


def evaluate_multimodel_v91(
    tasks: list[dict[str, Any]], responses: list[dict[str, Any]]
) -> dict[str, Any]:
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for response in responses:
        by_model[model_key(response)].append(response)

    return {
        "evaluation": "postfreeze_multimodel_operator_surface_validation",
        "protocol_version": EVALUATION_PROTOCOL_VERSION,
        "frozen_v8_modified": False,
        "model_count": len(by_model),
        "per_model": {
            key: evaluation_bundle(tasks, by_model[key]) for key in sorted(by_model)
        },
        "pooled": evaluation_bundle(tasks, responses),
        "numeric_serialization_policy": (
            "Only uncertainty values within absolute distance 1e-12 of the task's "
            "committed value are canonicalized. Raw model text is retained. No text, "
            "cause label, evidence ID, or alternative is repaired."
        ),
        "claim_boundary": (
            "These are new attacker outputs evaluated against an unchanged frozen V8 "
            "defense on the existing frozen task set. They test attacker-family "
            "generalization but are not a newly sampled untouched task split."
        ),
        "review_requirement": (
            "Report eligible, malformed, materially fact-changing, numeric-only, and "
            "no-op candidates separately. Two blinded domain reviewers must confirm "
            "objective adherence before candidates are described as harmful manipulations."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate exact attacker models with audited V9.1 numeric normalization"
    )
    parser.add_argument("--tasks", action="append", required=True)
    parser.add_argument("--responses", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    tasks = [record for path in args.tasks for record in read_jsonl(path)]
    responses = [record for path in args.responses for record in read_jsonl(path)]
    result = evaluate_multimodel_v91(tasks, responses)
    result["input_sha256"] = {
        "tasks": {str(path): file_sha256(path) for path in args.tasks},
        "responses": {str(path): file_sha256(path) for path in args.responses},
    }

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved audited V9.1 metrics to {destination.resolve()}")


if __name__ == "__main__":
    main()
