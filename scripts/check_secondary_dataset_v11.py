from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from coact_cps.config import load_config


PROTOCOL_VERSION = "cps-truthframe-secondary-dataset-check-v11.0"


def resolve_paths(values: list[str], base: Path) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = base / candidate
        if any(character in str(candidate) for character in "*?[]"):
            paths.extend(sorted(candidate.parent.glob(candidate.name)))
        else:
            paths.append(candidate)
    return [path.resolve() for path in paths]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_profile(path: Path, label_column: str | None) -> dict[str, Any]:
    header = pd.read_csv(path, nrows=0)
    original_columns = [str(column) for column in header.columns]
    columns = [column.strip() for column in original_columns]
    rows = 0
    labels: set[str] = set()
    actual_label = next(
        (column for column in original_columns if column.strip() == label_column), None
    )
    usecols = [actual_label] if actual_label else None
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=100_000, low_memory=False):
        rows += len(chunk)
        if usecols:
            labels.update(
                str(value).strip() for value in chunk[actual_label].dropna().unique()
            )
    return {
        "file": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
        "rows": rows,
        "columns": columns,
        "label_values": sorted(labels),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check a second CPS dataset before any training or test evaluation"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_config(config_path)
    root = config_path.parent.parent
    dataset = config["dataset"]
    normal_paths = resolve_paths(list(dataset["train_csvs"]), root)
    attack_paths = resolve_paths(list(dataset["test_csvs"]), root)
    label_column = dataset.get("label_column")
    normal_values = {str(value).strip().casefold() for value in dataset.get("normal_values", [])}
    issues: list[str] = []
    warnings: list[str] = []

    normal_paths = [path for path in normal_paths if path.exists() and path.is_file()]
    attack_paths = [path for path in attack_paths if path.exists() and path.is_file()]
    if not normal_paths:
        issues.append("No normal training CSV matched dataset.train_csvs")
    if not attack_paths:
        issues.append("No attack test CSV matched dataset.test_csvs")
    overlap = sorted(set(normal_paths) & set(attack_paths))
    if overlap:
        issues.append("The same path appears in both training and test inputs")

    normal_profiles = [csv_profile(path, label_column) for path in normal_paths]
    attack_profiles = [csv_profile(path, label_column) for path in attack_paths]
    all_profiles = normal_profiles + attack_profiles
    if all_profiles:
        reference_columns = all_profiles[0]["columns"]
        for profile in all_profiles[1:]:
            if profile["columns"] != reference_columns:
                issues.append(f"Column order differs in {profile['file']}")
        if label_column and label_column not in reference_columns:
            issues.append(f"Configured label column is absent: {label_column}")
        configured_features = dataset.get("features")
        if configured_features is None:
            sample = (
                pd.read_csv(normal_paths[0], nrows=5000, low_memory=False)
                if normal_paths
                else None
            )
            numeric = [] if sample is None else [
                str(column).strip()
                for column in sample.columns
                if str(column).strip() != label_column
                and pd.to_numeric(sample[column], errors="coerce").notna().mean() > 0.95
            ]
            warnings.append(
                "Feature selection is not frozen. The first normal file sample has "
                f"{len(numeric)} numeric candidates."
            )
        else:
            missing = [
                feature for feature in configured_features if feature not in reference_columns
            ]
            if missing:
                issues.append(f"Configured features are absent: {missing[:20]}")

    for profile in normal_profiles:
        unexpected = [
            value for value in profile["label_values"] if value.casefold() not in normal_values
        ]
        if unexpected:
            issues.append(
                f"Normal training file {profile['file']} contains nonnormal labels: {unexpected}"
            )
    observed_test_labels = {
        value.casefold() for profile in attack_profiles for value in profile["label_values"]
    }
    if label_column and attack_profiles:
        if not (observed_test_labels & normal_values):
            issues.append(
                "The test files contain no configured normal label; binary metrics "
                "would be invalid"
            )
        if not (observed_test_labels - normal_values):
            issues.append("The test files contain no attack label")

    hashes: dict[str, str] = {}
    for role, profiles in (("normal", normal_profiles), ("attack", attack_profiles)):
        for profile in profiles:
            digest = str(profile["sha256"])
            if digest in hashes and hashes[digest] != role:
                issues.append("Identical file content appears in both training and test inputs")
            hashes[digest] = role

    result = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "PASS" if not issues else "FAIL",
        "config": config_path.name,
        "dataset_name": dataset.get("name"),
        "dataset_version": dataset.get("version"),
        "source_url": dataset.get("source_url"),
        "split_rule": (
            "Normal source files are used for training and chronological validation. "
            "Attack source files remain untouched until final evaluation."
        ),
        "normal_training_files": normal_profiles,
        "attack_test_files": attack_profiles,
        "issues": issues,
        "warnings": warnings,
        "claim_boundary": (
            "A passing readiness check does not produce detector results. It verifies file "
            "separation, schema, labels, and hashes before the evaluation run."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "normal_files": len(normal_profiles),
        "attack_files": len(attack_profiles),
        "issues": issues,
        "warnings": warnings,
        "output": str(args.output.resolve()),
    }, indent=2))
    raise SystemExit(0 if not issues else 2)


if __name__ == "__main__":
    main()
