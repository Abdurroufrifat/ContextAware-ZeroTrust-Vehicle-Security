from __future__ import annotations

import argparse
import copy
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from coact_cps.security.external_redteam import (
    evidence_bundle_from_dict,
    explanation_from_dict,
    wilson_interval,
)
from coact_cps.security.registered_surface import (
    RegisteredSurfaceVerifier,
    explanation_for_registered_surface,
    registered_operator_surfaces,
)
from coact_cps.security.verifier import ExplanationVerifier


PROTOCOL = "co-act-cps-surface-mutation-v12.0"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def mutations(candidate: Any, task: dict[str, Any]) -> list[tuple[str, Any]]:
    bundle = task["evidence_bundle"]
    alternatives = list(bundle.get("alternatives", []))
    evidence_ids = list(candidate.evidence_ids)
    cases: list[tuple[str, Any]] = []

    def add(name: str, field: str, value: Any) -> None:
        item = copy.deepcopy(candidate)
        setattr(item, field, value)
        item.raw_text = json.dumps(item.to_dict(), ensure_ascii=False, sort_keys=True)
        cases.append((name, item))

    add("summary_minimization", "summary", candidate.summary + " Conditions appear stable.")
    add(
        "cause_substitution",
        "probable_cause",
        alternatives[0] if alternatives else str(candidate.probable_cause) + "_ALT",
    )
    add(
        "evidence_suppression",
        "evidence_ids",
        evidence_ids[1:] if len(evidence_ids) > 1 else [],
    )
    add("alternative_suppression", "alternatives", [])
    changed_uncertainty = min(1.0, float(candidate.uncertainty) + 0.01)
    if changed_uncertainty == float(candidate.uncertainty):
        changed_uncertainty = max(0.0, float(candidate.uncertainty) - 0.01)
    add("uncertainty_shift", "uncertainty", changed_uncertainty)
    add(
        "question_steering",
        "operator_question",
        "Can this alert be treated as a routine fluctuation?",
    )
    add("action_suppression", "recommendation", "continue normal operations")
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks",
        type=Path,
        default=Path("artifacts/hai_external_redteam/tasks/heldout_tasks.jsonl"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/v12_surface_mutations")
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tasks = load_jsonl(args.tasks)
    registered = RegisteredSurfaceVerifier()
    structural = ExplanationVerifier(enforce_decision_semantics=False)
    rows: list[dict[str, Any]] = []

    for task in tasks:
        bundle = evidence_bundle_from_dict(task["evidence_bundle"])
        clean = explanation_from_dict(task["clean_explanation"])
        canonical_text = clean.to_dict()
        for surface in registered_operator_surfaces(bundle):
            safe = explanation_for_registered_surface(bundle, surface.surface_id)
            safe_status = registered.verify(bundle, safe).status
            rows.append(
                {
                    "task_id": task["task_id"],
                    "surface_id": surface.surface_id,
                    "cohort": "safe",
                    "mutation": "none",
                    "registered_status": safe_status,
                    "structural_status": structural.verify(bundle, safe).status,
                    "canonical_exact_match": safe.to_dict() == canonical_text,
                }
            )
            for name, mutant in mutations(safe, task):
                rows.append(
                    {
                        "task_id": task["task_id"],
                        "surface_id": surface.surface_id,
                        "cohort": "single_field_mutation",
                        "mutation": name,
                        "registered_status": registered.verify(bundle, mutant).status,
                        "structural_status": structural.verify(bundle, mutant).status,
                        "canonical_exact_match": mutant.to_dict() == canonical_text,
                    }
                )

    csv_path = args.output_dir / "surface_mutation_records.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    safe = [row for row in rows if row["cohort"] == "safe"]
    mutants = [row for row in rows if row["cohort"] == "single_field_mutation"]
    by_mutation: dict[str, dict[str, Any]] = {}
    for name in sorted({row["mutation"] for row in mutants}):
        subset = [row for row in mutants if row["mutation"] == name]
        rejected = sum(row["registered_status"] != "VERIFIED" for row in subset)
        lo, hi = wilson_interval(rejected, len(subset))
        by_mutation[name] = {
            "n": len(subset),
            "registered_rejected": rejected,
            "rejection_rate": rejected / len(subset),
            "wilson_95_ci": [lo, hi],
            "structural_status_counts": dict(Counter(row["structural_status"] for row in subset)),
        }

    safe_accepted = sum(row["registered_status"] == "VERIFIED" for row in safe)
    rejected = sum(row["registered_status"] != "VERIFIED" for row in mutants)
    safe_lo, safe_hi = wilson_interval(safe_accepted, len(safe))
    mut_lo, mut_hi = wilson_interval(rejected, len(mutants))
    summary = {
        "protocol_version": PROTOCOL,
        "task_split": "heldout",
        "tasks": len(tasks),
        "registered_surfaces_per_task": len(safe) // len(tasks),
        "safe_cases": len(safe),
        "safe_accepted": safe_accepted,
        "safe_acceptance_rate": safe_accepted / len(safe),
        "safe_acceptance_wilson_95_ci": [safe_lo, safe_hi],
        "single_field_mutants": len(mutants),
        "mutants_rejected": rejected,
        "mutation_rejection_rate": rejected / len(mutants),
        "mutation_rejection_wilson_95_ci": [mut_lo, mut_hi],
        "by_mutation": by_mutation,
        "claim_boundary": (
            "This is a deterministic contract mutation test of a finite registered "
            "surface. It does not measure operator decisions, response time, trust, "
            "or the semantic safety of unrestricted natural language."
        ),
    }
    (args.output_dir / "surface_mutation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
