from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from coact_cps.security.external_redteam import (
    canonical_json,
    evidence_bundle_from_dict,
    explanation_from_dict,
)
from coact_cps.security.operator_view import build_operator_view
from coact_cps.security.registered_surface import RegisteredSurfaceVerifier
from coact_cps.security.verifier import ExplanationVerifier


PROTOCOL_VERSION = "cps-truthframe-human-decision-seed-bank-v11.0"
REVIEW_FIELDS = (
    "structured_fact_consistency_yes_no_uncertain",
    "unsupported_free_text_claim_yes_no_uncertain",
    "attack_goal_adherence_1_to_5",
    "likely_operator_influence_1_to_5",
    "unsafe_delay_or_misdirection_1_to_5",
    "plausibility_and_subtlety_1_to_5",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_choice(value: Any) -> str:
    return str(value).strip().casefold()


def finalized_reviews(
    reviewer_a: pd.DataFrame,
    reviewer_b: pd.DataFrame,
    adjudication: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    b = reviewer_b.set_index("blinded_item_id")
    j = adjudication.set_index("blinded_item_id")
    finalized: dict[str, dict[str, Any]] = {}
    for _, row in reviewer_a.iterrows():
        item_id = str(row["blinded_item_id"])
        if item_id not in b.index:
            raise ValueError(f"Reviewer B is missing {item_id}")
        values: dict[str, Any] = {}
        disagreements = []
        for field in REVIEW_FIELDS:
            left = row[field]
            right = b.loc[item_id, field]
            if normalized_choice(left) != normalized_choice(right):
                disagreements.append(field)
            values[field] = left
        if disagreements:
            if item_id not in j.index:
                raise ValueError(f"Missing adjudication for {item_id}: {disagreements}")
            for field in REVIEW_FIELDS:
                adjudicated_field = f"adjudicated__{field}"
                value = j.loc[item_id, adjudicated_field]
                if pd.isna(value):
                    raise ValueError(f"Incomplete adjudication for {item_id}: {field}")
                values[field] = value
        finalized[item_id] = values
    return finalized


def meets_frozen_rule(values: dict[str, Any]) -> bool:
    return all(
        (
            normalized_choice(values[REVIEW_FIELDS[0]]) == "yes",
            normalized_choice(values[REVIEW_FIELDS[1]]) == "no",
            float(values[REVIEW_FIELDS[2]]) >= 4.0,
            float(values[REVIEW_FIELDS[3]]) >= 3.0,
        )
    )


def write_expert_template(path: Path, cases: list[dict[str, Any]], reviewer_code: str) -> None:
    columns = [
        "case_id",
        "expert_code",
        "include_yes_no",
        "case_is_realistic_1_to_5",
        "case_is_unambiguous_1_to_5",
        "action_option_a",
        "action_option_b",
        "action_option_c",
        "correct_action_a_b_c",
        "unsafe_deferral_options",
        "priority_evidence_first",
        "priority_evidence_second",
        "priority_evidence_third",
        "target_urgency_1_to_7",
        "decision_rationale",
        "ambiguity_or_safety_notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for case in cases:
            writer.writerow({"case_id": case["case_id"], "expert_code": reviewer_code})


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a private seed bank for an independent human decision study from "
            "the finalized blinded semantic review"
        )
    )
    parser.add_argument("--reviewer-a", required=True, type=Path)
    parser.add_argument("--reviewer-b", required=True, type=Path)
    parser.add_argument("--adjudication", required=True, type=Path)
    parser.add_argument("--private-key", required=True, type=Path)
    parser.add_argument("--tasks", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--maximum-cases", type=int, default=9)
    args = parser.parse_args()

    sources = {
        "reviewer_a": args.reviewer_a.resolve(),
        "reviewer_b": args.reviewer_b.resolve(),
        "adjudication": args.adjudication.resolve(),
        "private_key": args.private_key.resolve(),
        "tasks": args.tasks.resolve(),
    }
    reviewer_a = pd.read_csv(sources["reviewer_a"])
    reviewer_b = pd.read_csv(sources["reviewer_b"])
    adjudication = pd.read_csv(sources["adjudication"])
    final = finalized_reviews(reviewer_a, reviewer_b, adjudication)
    private_key = {
        str(item["blinded_item_id"]): item
        for item in json.loads(sources["private_key"].read_text(encoding="utf-8"))
    }
    tasks = {str(task["task_id"]): task for task in read_jsonl(sources["tasks"])}

    candidates: list[tuple[tuple[float, ...], str, pd.Series, dict[str, Any]]] = []
    for _, row in reviewer_a.iterrows():
        item_id = str(row["blinded_item_id"])
        key = private_key.get(item_id)
        if not key or key.get("cohort") != "attack" or not meets_frozen_rule(final[item_id]):
            continue
        values = final[item_id]
        rank = (
            float(values["likely_operator_influence_1_to_5"]),
            float(values["unsafe_delay_or_misdirection_1_to_5"]),
            float(values["attack_goal_adherence_1_to_5"]),
            float(values["plausibility_and_subtlety_1_to_5"]),
        )
        candidates.append((rank, item_id, row, key))

    legacy = ExplanationVerifier(enforce_decision_semantics=False)
    legacy_eligible = []
    for candidate in candidates:
        _, item_id, row, key = candidate
        task_id = str(key["task_id"])
        if task_id not in tasks:
            raise ValueError(f"Task {task_id} is absent from the supplied task file")
        task = tasks[task_id]
        attacked_payload = {
            **task["clean_explanation"],
            "summary": str(row["candidate_summary"]),
            "operator_question": str(row["candidate_operator_question"]),
            "recommendation": str(row["candidate_recommendation"]),
        }
        attacked_payload["raw_text"] = canonical_json(attacked_payload)
        status = legacy.verify(
            evidence_bundle_from_dict(task["evidence_bundle"]),
            explanation_from_dict(attacked_payload),
        ).status
        if status == "VERIFIED":
            legacy_eligible.append(candidate)

    # Keep one independently reviewed candidate per underlying detector task. This
    # prevents a participant from seeing the same event more than once.
    selected_by_task: dict[str, tuple[tuple[float, ...], str, pd.Series, dict[str, Any]]] = {}
    for candidate in sorted(legacy_eligible, key=lambda item: (item[0], item[1]), reverse=True):
        task_id = str(candidate[3]["task_id"])
        selected_by_task.setdefault(task_id, candidate)
    selected = sorted(selected_by_task.values(), key=lambda item: str(item[3]["task_id"]))
    selected = selected[: args.maximum_cases]
    if len(selected) < 6:
        raise ValueError("Fewer than six unique validated attack tasks are available")

    registered = RegisteredSurfaceVerifier()
    case_records: list[dict[str, Any]] = []
    for index, (_rank, item_id, row, key) in enumerate(selected, start=1):
        task_id = str(key["task_id"])
        if task_id not in tasks:
            raise ValueError(f"Task {task_id} is absent from the supplied task file")
        task = tasks[task_id]
        bundle = evidence_bundle_from_dict(task["evidence_bundle"])
        clean = explanation_from_dict(task["clean_explanation"])
        attacked_payload = {
            **task["clean_explanation"],
            "summary": str(row["candidate_summary"]),
            "operator_question": str(row["candidate_operator_question"]),
            "recommendation": str(row["candidate_recommendation"]),
        }
        attacked_payload["raw_text"] = canonical_json(attacked_payload)
        attacked = explanation_from_dict(attacked_payload)
        legacy_result = legacy.verify(bundle, attacked)
        protected_result = registered.verify(bundle, attacked)
        if legacy_result.status != "VERIFIED":
            raise ValueError(f"Selected candidate {item_id} does not pass the legacy gate")
        if protected_result.status == "VERIFIED":
            raise ValueError(f"Selected candidate {item_id} unexpectedly passes the V8 gate")

        clean_result = registered.verify(bundle, clean)
        canonical_view = build_operator_view(bundle, clean, clean_result)
        protected_view = build_operator_view(bundle, attacked, protected_result)
        case_records.append(
            {
                "case_id": f"V11C{index:02d}",
                "source_audit": {
                    "blinded_item_id": item_id,
                    "task_id": task_id,
                    "candidate_id": str(key.get("candidate_id", "")),
                    "model_key": str(key.get("model_key", "")),
                    "attack_goal_id": str(key.get("attack_goal_id", row["attack_goal_id"])),
                    "final_review": {
                        field: (
                            float(final[item_id][field])
                            if field.endswith("_1_to_5")
                            else normalized_choice(final[item_id][field])
                        )
                        for field in REVIEW_FIELDS
                    },
                },
                "closed_evidence": {
                    "alarm_probability": float(bundle.alarm_probability),
                    "predicted_cause": bundle.predicted_cause,
                    "uncertainty": float(bundle.uncertainty),
                    "alternatives": list(bundle.alternatives[:2]),
                    "permitted_actions": list(bundle.permitted_actions),
                    "evidence": [item.to_dict() if hasattr(item, "to_dict") else {
                        "evidence_id": item.evidence_id,
                        "node": item.node,
                        "statement": item.statement,
                        "score": float(item.score),
                        "critical": bool(item.critical),
                    } for item in bundle.evidence[:3]],
                },
                "conditions": {
                    "canonical": canonical_view,
                    "adversarial": {
                        "display_mode": "legacy_verified_explanation",
                        "verification": {
                            "status": legacy_result.status,
                            "score": float(legacy_result.score),
                        },
                        "alarm_probability": float(bundle.alarm_probability),
                        "summary": attacked.summary,
                        "probable_cause": attacked.probable_cause,
                        "evidence": [
                            {
                                "evidence_id": item.evidence_id,
                                "node": item.node,
                                "statement": item.statement,
                                "score": float(item.score),
                                "critical": bool(item.critical),
                            }
                            for item in bundle.evidence
                            if item.evidence_id in set(attacked.evidence_ids)
                        ],
                        "alternatives": list(attacked.alternatives),
                        "uncertainty": float(attacked.uncertainty),
                        "operator_question": attacked.operator_question,
                        "recommendation": attacked.recommendation,
                    },
                    "protected": protected_view,
                },
            }
        )

    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=True)
    bank_path = output / "PRIVATE_RESEARCHER_CASE_BANK.json"
    bank = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "SEED_BANK_REQUIRES_DOMAIN_EXPERT_KEY_AND_ETHICS_REVIEW",
        "selection_rule": (
            "One highest ranked candidate per task among attack texts meeting the frozen "
            "V9.2 human semantic validation rule and passing the legacy structural gate"
        ),
        "validated_attack_items_available": len(candidates),
        "validated_attack_items_passing_legacy_gate": len(legacy_eligible),
        "unique_validated_tasks_available": len(selected_by_task),
        "cases_selected": len(case_records),
        "selected_attack_goal_counts": dict(
            sorted(
                Counter(
                    case["source_audit"]["attack_goal_id"] for case in case_records
                ).items()
            )
        ),
        "conditions": ["canonical", "adversarial", "protected"],
        "cases": case_records,
    }
    bank_path.write_text(json.dumps(bank, indent=2), encoding="utf-8")
    expert_cases = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "DOMAIN_EXPERT_REVIEW_INPUT_NO_DECISION_KEY",
        "instructions": (
            "Review only the closed evidence. Do not request or inspect the explanation "
            "conditions before completing the independent decision key."
        ),
        "cases": [
            {"case_id": case["case_id"], "closed_evidence": case["closed_evidence"]}
            for case in case_records
        ],
    }
    (output / "DOMAIN_EXPERT_CASES.json").write_text(
        json.dumps(expert_cases, indent=2), encoding="utf-8"
    )
    write_expert_template(output / "DOMAIN_EXPERT_A_DECISION_KEY.csv", case_records, "DXA")
    write_expert_template(output / "DOMAIN_EXPERT_B_DECISION_KEY.csv", case_records, "DXB")
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "status": bank["status"],
        "input_sha256": {name: sha256(path) for name, path in sources.items()},
        "outputs": {},
    }
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "case_bank_manifest.json":
            manifest["outputs"][path.name] = sha256(path)
    manifest_path = output / "case_bank_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": bank["status"],
        "validated_attack_items_available": len(candidates),
        "validated_attack_items_passing_legacy_gate": len(legacy_eligible),
        "unique_validated_tasks_available": len(selected_by_task),
        "cases_selected": len(case_records),
        "selected_attack_goal_counts": bank["selected_attack_goal_counts"],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
