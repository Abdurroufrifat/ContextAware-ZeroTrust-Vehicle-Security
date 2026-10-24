from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from coact_cps.security.surface_utility import evaluate_surface_utility

EVALUATION_PROTOCOL_VERSION = "cps-truthframe-postfreeze-multimodel-v9.0"


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"Record at {path}:{line_number} must be an object")
        records.append(record)
    return records


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_key(response: dict[str, Any]) -> str:
    model_id = str(response.get("model_id", "unreported"))
    digest = str(response.get("model_digest", ""))
    return f"{model_id}@sha256:{digest}" if digest else model_id


def provenance_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    candidate_ids = [str(record.get("candidate_id", "")) for record in records]
    nonempty_ids = [value for value in candidate_ids if value]
    return {
        "response_records": len(records),
        "unique_candidate_ids": len(set(nonempty_ids)),
        "duplicate_candidate_ids": len(nonempty_ids) - len(set(nonempty_ids)),
        "model_ids": sorted({str(record.get("model_id", "unreported")) for record in records}),
        "model_digests": sorted(
            {str(record["model_digest"]) for record in records if record.get("model_digest")}
        ),
        "run_fingerprints": sorted(
            {
                str(record["run_fingerprint"])
                for record in records
                if record.get("run_fingerprint")
            }
        ),
        "runtime_versions": sorted(
            {
                str(record.get("runtime", {}).get("version"))
                for record in records
                if isinstance(record.get("runtime"), dict)
                and record.get("runtime", {}).get("version")
            }
        ),
    }


def evaluate_multimodel(
    tasks: list[dict[str, Any]], responses: list[dict[str, Any]]
) -> dict[str, Any]:
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for response in responses:
        by_model[model_key(response)].append(response)

    per_model = {}
    for key in sorted(by_model):
        records = by_model[key]
        per_model[key] = {
            "provenance": provenance_summary(records),
            "evaluation": evaluate_surface_utility(tasks, records),
        }

    return {
        "evaluation": "postfreeze_multimodel_operator_surface_validation",
        "protocol_version": EVALUATION_PROTOCOL_VERSION,
        "frozen_v8_modified": False,
        "model_count": len(by_model),
        "per_model": per_model,
        "pooled": {
            "provenance": provenance_summary(responses),
            "evaluation": evaluate_surface_utility(tasks, responses),
        },
        "claim_boundary": (
            "These are new attacker outputs evaluated against an unchanged frozen V8 "
            "defense on the existing frozen task set. They test attacker-family "
            "generalization but are not a newly sampled untouched task split."
        ),
        "review_requirement": (
            "Report eligible, malformed, fact-changing, and no-op candidates separately. "
            "Two blinded domain reviewers must confirm objective adherence before the final "
            "paper describes candidates as harmful manipulations."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report V8 utility and security per exact attacker model and pooled"
    )
    parser.add_argument("--tasks", action="append", required=True)
    parser.add_argument("--responses", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    tasks = [record for path in args.tasks for record in read_jsonl(path)]
    responses = [record for path in args.responses for record in read_jsonl(path)]
    result = evaluate_multimodel(tasks, responses)
    result["input_sha256"] = {
        "tasks": {str(path): file_sha256(path) for path in args.tasks},
        "responses": {str(path): file_sha256(path) for path in args.responses},
    }

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved per-model and pooled metrics to {destination.resolve()}")


if __name__ == "__main__":
    main()
