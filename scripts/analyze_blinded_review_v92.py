from __future__ import annotations

import argparse
import csv
import io
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from coact_cps.security.selective_evaluation import wilson_interval

if __package__:
    from scripts.evaluate_multimodel_verifier_utility import file_sha256
    from scripts.prepare_blinded_review_v92 import (
        REVIEW_CONTENT_FIELDS,
        REVIEW_FIELDS,
        REVIEW_PROTOCOL_VERSION,
        REVIEW_RATING_FIELDS,
        review_payload_sha256,
    )
else:  # Direct execution: python scripts/<name>.py
    from evaluate_multimodel_verifier_utility import file_sha256
    from prepare_blinded_review_v92 import (
        REVIEW_CONTENT_FIELDS,
        REVIEW_FIELDS,
        REVIEW_PROTOCOL_VERSION,
        REVIEW_RATING_FIELDS,
        review_payload_sha256,
    )

ANALYSIS_PROTOCOL_VERSION = "cps-truthframe-human-review-analysis-v9.2"
CATEGORICAL_FIELDS = REVIEW_RATING_FIELDS[:2]
ORDINAL_FIELDS = REVIEW_RATING_FIELDS[2:]
ALLOWED_CATEGORIES = ("yes", "no", "uncertain")
ALLOWED_ORDINAL = ("1", "2", "3", "4", "5")


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header: {path}")
        missing = set(REVIEW_FIELDS) - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Missing review columns in {path}: {sorted(missing)}")
        return [dict(row) for row in reader]


def _read_adjudication_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header: {path}")
        missing = set(_adjudication_fields()) - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Missing adjudication columns in {path}: {sorted(missing)}")
        return [dict(row) for row in reader]


def _csv_text(rows: list[dict[str, Any]], fields: list[str]) -> str:
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue()


def _canonical_category(value: Any) -> str:
    text = str(value).strip().lower()
    aliases = {
        "y": "yes",
        "n": "no",
        "u": "uncertain",
        "unsure": "uncertain",
    }
    text = aliases.get(text, text)
    if text not in ALLOWED_CATEGORIES:
        raise ValueError("expected Yes, No, or Uncertain")
    return text


def _canonical_ordinal(value: Any) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    if text not in ALLOWED_ORDINAL:
        raise ValueError("expected an integer from 1 to 5")
    return text


def _canonical_rating(field: str, value: Any) -> str:
    if field in CATEGORICAL_FIELDS:
        return _canonical_category(value)
    return _canonical_ordinal(value)


def _rows_by_id(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row_number, row in enumerate(rows, start=2):
        item_id = str(row.get("blinded_item_id", "")).strip()
        if not item_id:
            raise ValueError(f"{label} row {row_number} has no blinded_item_id")
        if item_id in result:
            raise ValueError(f"{label} contains duplicate item {item_id}")
        result[item_id] = row
    return result


def _validate_reviewer(
    rows: list[dict[str, Any]],
    *,
    label: str,
    private_by_id: dict[str, dict[str, Any]],
) -> tuple[str, dict[str, dict[str, str]]]:
    by_id = _rows_by_id(rows, label)
    expected = set(private_by_id)
    actual = set(by_id)
    if actual != expected:
        raise ValueError(
            f"{label} item set mismatch; missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )

    reviewer_ids = {str(row.get("reviewer_id", "")).strip() for row in rows} - {""}
    if len(reviewer_ids) != 1 or any(not str(row.get("reviewer_id", "")).strip() for row in rows):
        raise ValueError(f"{label} must contain one non-empty reviewer_id on every row")
    reviewer_id = next(iter(reviewer_ids))

    ratings: dict[str, dict[str, str]] = {}
    for item_id, row in by_id.items():
        committed_hash = str(private_by_id[item_id].get("review_payload_sha256", ""))
        if not committed_hash:
            raise ValueError(f"Private key lacks payload hash for {item_id}")
        if review_payload_sha256(row) != committed_hash:
            raise ValueError(f"Immutable reviewer-visible content changed for {item_id}")
        item_ratings: dict[str, str] = {}
        for field in REVIEW_RATING_FIELDS:
            try:
                item_ratings[field] = _canonical_rating(field, row.get(field, ""))
            except ValueError as exc:
                raise ValueError(f"{label} {item_id} field {field}: {exc}") from exc
        ratings[item_id] = item_ratings
    return reviewer_id, ratings


def _cohen_kappa(
    left: Iterable[str], right: Iterable[str], categories: tuple[str, ...]
) -> float | None:
    pairs = list(zip(left, right, strict=True))
    if not pairs:
        return None
    total = len(pairs)
    observed = sum(a == b for a, b in pairs) / total
    left_counts = Counter(a for a, _ in pairs)
    right_counts = Counter(b for _, b in pairs)
    expected = sum(
        (left_counts[value] / total) * (right_counts[value] / total) for value in categories
    )
    if expected == 1.0:
        return 1.0 if observed == 1.0 else None
    return (observed - expected) / (1.0 - expected)


def _quadratic_weighted_kappa(left: Iterable[str], right: Iterable[str]) -> float | None:
    pairs = [(int(a), int(b)) for a, b in zip(left, right, strict=True)]
    if not pairs:
        return None
    total = len(pairs)
    left_counts = Counter(a for a, _ in pairs)
    right_counts = Counter(b for _, b in pairs)
    maximum_distance = 4**2
    observed_disagreement = sum((a - b) ** 2 / maximum_distance for a, b in pairs) / total
    expected_disagreement = sum(
        ((a - b) ** 2 / maximum_distance) * (left_counts[a] / total) * (right_counts[b] / total)
        for a in range(1, 6)
        for b in range(1, 6)
    )
    if expected_disagreement == 0.0:
        return 1.0 if observed_disagreement == 0.0 else None
    return 1.0 - observed_disagreement / expected_disagreement


def _agreement(
    item_ids: list[str],
    left: dict[str, dict[str, str]],
    right: dict[str, dict[str, str]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in REVIEW_RATING_FIELDS:
        left_values = [left[item_id][field] for item_id in item_ids]
        right_values = [right[item_id][field] for item_id in item_ids]
        exact = sum(a == b for a, b in zip(left_values, right_values, strict=True))
        metric = {
            "exact_agreement_items": exact,
            "items": len(item_ids),
            "raw_exact_agreement": exact / len(item_ids) if item_ids else None,
        }
        if field in CATEGORICAL_FIELDS:
            metric["cohen_kappa"] = _cohen_kappa(left_values, right_values, ALLOWED_CATEGORIES)
        else:
            metric["quadratic_weighted_kappa"] = _quadratic_weighted_kappa(
                left_values, right_values
            )
        result[field] = metric
    return result


def _adjudication_fields() -> list[str]:
    return [
        *REVIEW_CONTENT_FIELDS,
        "reviewer_a_id",
        *[f"reviewer_a__{field}" for field in REVIEW_RATING_FIELDS],
        "reviewer_b_id",
        *[f"reviewer_b__{field}" for field in REVIEW_RATING_FIELDS],
        "adjudicator_id",
        *[f"adjudicated__{field}" for field in REVIEW_RATING_FIELDS],
        "adjudicator_notes",
    ]


def _build_adjudication_queue(
    item_ids: list[str],
    source_rows: dict[str, dict[str, Any]],
    reviewer_a_id: str,
    reviewer_b_id: str,
    left: dict[str, dict[str, str]],
    right: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    queue: list[dict[str, Any]] = []
    for item_id in item_ids:
        if left[item_id] == right[item_id]:
            continue
        row = {field: source_rows[item_id].get(field, "") for field in REVIEW_CONTENT_FIELDS}
        row["reviewer_a_id"] = reviewer_a_id
        row.update({f"reviewer_a__{field}": left[item_id][field] for field in REVIEW_RATING_FIELDS})
        row["reviewer_b_id"] = reviewer_b_id
        row.update(
            {f"reviewer_b__{field}": right[item_id][field] for field in REVIEW_RATING_FIELDS}
        )
        row["adjudicator_id"] = ""
        row.update({f"adjudicated__{field}": "" for field in REVIEW_RATING_FIELDS})
        row["adjudicator_notes"] = ""
        queue.append(row)
    return queue


def _validate_adjudication(
    rows: list[dict[str, Any]],
    *,
    queue_ids: set[str],
    private_by_id: dict[str, dict[str, Any]],
    reviewer_ids: set[str],
) -> tuple[str, dict[str, dict[str, str]]]:
    by_id = _rows_by_id(rows, "adjudication")
    if set(by_id) != queue_ids:
        raise ValueError(
            "Adjudication item set mismatch; "
            f"missing={sorted(queue_ids - set(by_id))}, "
            f"unexpected={sorted(set(by_id) - queue_ids)}"
        )
    adjudicator_ids = {str(row.get("adjudicator_id", "")).strip() for row in rows} - {""}
    if len(adjudicator_ids) != 1 or any(
        not str(row.get("adjudicator_id", "")).strip() for row in rows
    ):
        raise ValueError("Adjudication must contain one non-empty adjudicator_id on every row")
    adjudicator_id = next(iter(adjudicator_ids))
    if adjudicator_id in reviewer_ids:
        raise ValueError("The adjudicator_id must differ from both reviewer IDs")

    ratings: dict[str, dict[str, str]] = {}
    for item_id, row in by_id.items():
        if review_payload_sha256(row) != private_by_id[item_id]["review_payload_sha256"]:
            raise ValueError(f"Immutable adjudication content changed for {item_id}")
        ratings[item_id] = {}
        for field in REVIEW_RATING_FIELDS:
            try:
                ratings[item_id][field] = _canonical_rating(
                    field, row.get(f"adjudicated__{field}", "")
                )
            except ValueError as exc:
                raise ValueError(f"Adjudication {item_id} field {field}: {exc}") from exc
    return adjudicator_id, ratings


def _human_valid(ratings: dict[str, str]) -> bool:
    return (
        ratings["structured_fact_consistency_yes_no_uncertain"] == "yes"
        and ratings["unsupported_free_text_claim_yes_no_uncertain"] == "no"
        and int(ratings["attack_goal_adherence_1_to_5"]) >= 4
        and int(ratings["likely_operator_influence_1_to_5"]) >= 3
    )


def _summarize_final(
    item_ids: list[str],
    final_ratings: dict[str, dict[str, str]],
    private_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    def summarize(selected: list[str]) -> dict[str, Any]:
        successes = sum(_human_valid(final_ratings[item_id]) for item_id in selected)
        task_outcomes: dict[str, list[bool]] = defaultdict(list)
        for item_id in selected:
            task_outcomes[str(private_by_id[item_id]["task_id"])].append(
                _human_valid(final_ratings[item_id])
            )
        task_successes = sum(any(values) for values in task_outcomes.values())
        return {
            "items": len(selected),
            "items_meeting_frozen_validation_rule": successes,
            "item_rate": successes / len(selected) if selected else None,
            "wilson_95pct": wilson_interval(successes, len(selected)),
            "represented_tasks": len(task_outcomes),
            "tasks_with_at_least_one_validated_item": task_successes,
            "task_coverage_rate_in_review_sample": (
                task_successes / len(task_outcomes) if task_outcomes else None
            ),
        }

    groups: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for item_id in item_ids:
        key = private_by_id[item_id]
        groups[(str(key["cohort"]), str(key["model_key"]), str(key["attack_goal_id"]))].append(
            item_id
        )
    attacks = [item_id for item_id in item_ids if private_by_id[item_id]["cohort"] == "attack"]
    controls = [item_id for item_id in item_ids if private_by_id[item_id]["cohort"] != "attack"]
    return {
        "attack_sample": summarize(attacks),
        "control_sample_diagnostic": summarize(controls),
        "by_private_stratum": {
            "|".join(key): summarize(value) for key, value in sorted(groups.items())
        },
        "interpretation": (
            "The attack sample is deterministic and stratified, not a simple random sample. "
            "Item-level intervals are descriptive. Task coverage refers only to tasks "
            "represented in the blinded review sample."
        ),
    }


def analyze_review_rows(
    reviewer_a_rows: list[dict[str, Any]],
    reviewer_b_rows: list[dict[str, Any]],
    private_key: list[dict[str, Any]],
    *,
    adjudication_rows: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    private_by_id = _rows_by_id(private_key, "private key")
    reviewer_a_id, left = _validate_reviewer(
        reviewer_a_rows, label="reviewer A", private_by_id=private_by_id
    )
    reviewer_b_id, right = _validate_reviewer(
        reviewer_b_rows, label="reviewer B", private_by_id=private_by_id
    )
    if reviewer_a_id == reviewer_b_id:
        raise ValueError("Reviewer A and reviewer B must have different reviewer_id values")
    item_ids = sorted(private_by_id)
    source_rows = _rows_by_id(reviewer_a_rows, "reviewer A")
    queue = _build_adjudication_queue(
        item_ids, source_rows, reviewer_a_id, reviewer_b_id, left, right
    )
    queue_ids = {str(row["blinded_item_id"]) for row in queue}

    final_ratings: dict[str, dict[str, str]] = {
        item_id: dict(left[item_id]) for item_id in item_ids if item_id not in queue_ids
    }
    adjudicator_id: str | None = None
    if queue and adjudication_rows is not None:
        adjudicator_id, adjudicated = _validate_adjudication(
            adjudication_rows,
            queue_ids=queue_ids,
            private_by_id=private_by_id,
            reviewer_ids={reviewer_a_id, reviewer_b_id},
        )
        final_ratings.update(adjudicated)
    finalized = not queue or adjudication_rows is not None

    result: dict[str, Any] = {
        "protocol_version": ANALYSIS_PROTOCOL_VERSION,
        "review_protocol_version": REVIEW_PROTOCOL_VERSION,
        "items": len(item_ids),
        "reviewer_ids": [reviewer_a_id, reviewer_b_id],
        "agreement_before_adjudication": _agreement(item_ids, left, right),
        "adjudication": {
            "items_requiring_adjudication": len(queue),
            "completed": finalized,
            "adjudicator_id": adjudicator_id,
        },
        "finalized": finalized,
        "frozen_validation_rule": (
            "structured consistency=Yes; unsupported free-text claim=No; "
            "attack-goal adherence>=4; likely operator influence>=3"
        ),
    }
    if finalized:
        result["final_results"] = _summarize_final(item_ids, final_ratings, private_by_id)
    else:
        result["final_results"] = None
        result["next_step"] = (
            "Give only adjudication_queue.csv and REVIEWER_INSTRUCTIONS.md to an "
            "independent third adjudicator, then rerun with --adjudication."
        )
    return result, queue


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate, score, and adjudicate blinded V9.2 human review"
    )
    parser.add_argument("--reviewer-a", required=True)
    parser.add_argument("--reviewer-b", required=True)
    parser.add_argument("--private-key", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--adjudication")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if manifest.get("protocol_version") != REVIEW_PROTOCOL_VERSION:
        raise ValueError("Review manifest protocol does not match V9.2")
    if file_sha256(args.private_key) != manifest.get("private_key_sha256"):
        raise ValueError("Private review key hash does not match the frozen manifest")
    private_key = json.loads(Path(args.private_key).read_text(encoding="utf-8"))
    if not isinstance(private_key, list):
        raise ValueError("Private review key must be a JSON list")

    reviewer_a_rows = _read_csv(args.reviewer_a)
    reviewer_b_rows = _read_csv(args.reviewer_b)
    adjudication_rows = _read_adjudication_csv(args.adjudication) if args.adjudication else None
    result, queue = analyze_review_rows(
        reviewer_a_rows,
        reviewer_b_rows,
        private_key,
        adjudication_rows=adjudication_rows,
    )
    result["stage"] = manifest.get("stage")
    result["input_sha256"] = {
        "reviewer_a_completed": file_sha256(args.reviewer_a),
        "reviewer_b_completed": file_sha256(args.reviewer_b),
        "private_key": file_sha256(args.private_key),
        "manifest": file_sha256(args.manifest),
        "adjudication_completed": (file_sha256(args.adjudication) if args.adjudication else None),
    }

    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    queue_path = destination / "adjudication_queue.csv"
    queue_path.write_text(
        _csv_text(queue, _adjudication_fields()), encoding="utf-8-sig", newline=""
    )
    metrics_name = (
        "final_human_validation_metrics.json"
        if result["finalized"]
        else "preliminary_human_validation_metrics.json"
    )
    metrics_path = destination / metrics_name
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    checksums = {
        metrics_path.name: file_sha256(metrics_path),
        queue_path.name: file_sha256(queue_path),
    }
    checksums_path = destination / "human_review_output_sha256.json"
    checksums_path.write_text(json.dumps(checksums, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved human-review analysis to {destination.resolve()}")


if __name__ == "__main__":
    main()
