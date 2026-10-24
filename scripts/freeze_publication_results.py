from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "coact-cps-publication-freeze-v1"

REQUIRED_FILES: dict[str, tuple[str, ...]] = {
    "model_and_data": (
        "artifacts/hai_process_graph_recalibrated/detector.pt",
        "artifacts/hai_process_graph/adjacency.npy",
        "data/processed/hai/test.npz",
        "data/processed/hai/metadata.json",
        "data/reference/hai23_attack_catalog.csv",
    ),
    "primary_detector_evaluation": (
        "artifacts/hai_process_graph_recalibrated/full_test_metrics.json",
        "artifacts/hai_process_graph_recalibrated/full_test_episodes.csv",
        "artifacts/hai_process_graph_recalibrated/full_test_predictions.npz",
    ),
    "explanation_integrity": (
        "artifacts/hai_process_graph_recalibrated/integrity_metrics.json",
        "artifacts/hai_process_graph_recalibrated/adaptive_integrity_metrics.json",
    ),
    "external_redteam": (
        "artifacts/hai_external_redteam/tasks/development_tasks.jsonl",
        "artifacts/hai_external_redteam/tasks/heldout_tasks.jsonl",
        "artifacts/hai_external_redteam/tasks/manifest.json",
        "artifacts/hai_external_redteam/tasks/manifest_hardened.json",
        "artifacts/hai_external_redteam/development_qwen17b_metrics_pre_hardening.json",
        "artifacts/hai_external_redteam/development_qwen17b_responses.jsonl",
        "artifacts/hai_external_redteam/development_qwen17b_hardened_metrics.json",
        "artifacts/hai_external_redteam/heldout_qwen17b_responses.jsonl",
        "artifacts/hai_external_redteam/heldout_qwen17b_hardened_metrics.json",
    ),
    "frozen_implementation": (
        "src/coact_cps/security/verifier.py",
        "src/coact_cps/security/operator_surface.py",
        "src/coact_cps/security/external_redteam.py",
        "src/coact_cps/genai/grounded_generator.py",
        "scripts/evaluate_external_redteam_responses.py",
        "scripts/generate_external_redteam_hf.py",
        "scripts/refreeze_external_redteam_manifest.py",
    ),
}

OPTIONAL_FILES = (
    "data/reference/hai23_process_topology.json",
    "configs/hai_process_graph.yaml",
    "artifacts/hai_medium/full_test_metrics.json",
    "artifacts/hai_recalibrated/full_test_metrics.json",
    "artifacts/hai_process_graph/full_test_metrics.json",
)

PROCESSED_DATA_PATHS = {"data/processed/hai/test.npz"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def jsonl_count(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _split_metrics(metrics: dict[str, Any], split: str) -> dict[str, Any]:
    try:
        value = metrics["by_split"][split]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Metrics do not contain split {split!r}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Metrics split {split!r} must be a JSON object")
    return value


def validate_frozen_protocol(root: Path) -> dict[str, Any]:
    task_dir = root / "artifacts/hai_external_redteam/tasks"
    external_dir = root / "artifacts/hai_external_redteam"
    verifier_path = root / "src/coact_cps/security/verifier.py"
    development_tasks = task_dir / "development_tasks.jsonl"
    heldout_tasks = task_dir / "heldout_tasks.jsonl"
    parent_manifest_path = task_dir / "manifest.json"
    hardened_manifest_path = task_dir / "manifest_hardened.json"
    prehardening_path = external_dir / "development_qwen17b_metrics_pre_hardening.json"
    development_metrics_path = external_dir / "development_qwen17b_hardened_metrics.json"
    heldout_metrics_path = external_dir / "heldout_qwen17b_hardened_metrics.json"
    development_responses = external_dir / "development_qwen17b_responses.jsonl"
    heldout_responses = external_dir / "heldout_qwen17b_responses.jsonl"

    read_json(parent_manifest_path)
    hardened = read_json(hardened_manifest_path)
    development = read_json(development_metrics_path)
    heldout = read_json(heldout_metrics_path)
    errors: list[str] = []

    expected_hashes = {
        "development_tasks_sha256": sha256_file(development_tasks),
        "heldout_tasks_sha256": sha256_file(heldout_tasks),
        "verifier_sha256": sha256_file(verifier_path),
        "parent_manifest_sha256": sha256_file(parent_manifest_path),
        "development_metrics_sha256": sha256_file(prehardening_path),
    }
    for key, actual in expected_hashes.items():
        if hardened.get(key) != actual:
            errors.append(f"Hardened manifest {key} does not match the frozen file")

    if hardened.get("freeze_stage") != "post-development_pre-heldout":
        errors.append("Manifest freeze_stage is not post-development_pre-heldout")
    if hardened.get("heldout_responses_generated_before_refreeze") is not False:
        errors.append("Manifest does not certify that held-out responses followed refreezing")

    development_split = _split_metrics(development, "development")
    heldout_split = _split_metrics(heldout, "held_out")
    development_hardened = development_split["verifiers"]["hardened_decision_semantic"]
    heldout_hardened = heldout_split["verifiers"]["hardened_decision_semantic"]
    heldout_legacy = heldout_split["verifiers"]["legacy_structural"]

    checks = {
        "development_tasks": (jsonl_count(development_tasks), 20),
        "heldout_tasks": (jsonl_count(heldout_tasks), 80),
        "development_responses": (jsonl_count(development_responses), 80),
        "heldout_responses": (jsonl_count(heldout_responses), 320),
        "development_metric_tasks": (development_split.get("tasks"), 20),
        "heldout_metric_tasks": (heldout_split.get("tasks"), 80),
        "heldout_submitted_candidates": (heldout_split.get("submitted_candidates"), 320),
    }
    for name, (actual, expected) in checks.items():
        if actual != expected:
            errors.append(f"{name} is {actual!r}; expected {expected!r}")

    frozen_verifier_hash = expected_hashes["verifier_sha256"]
    for name, metrics in (("development", development), ("held_out", heldout)):
        if metrics.get("frozen_verifier_sha256") != frozen_verifier_hash:
            errors.append(f"{name} metrics use a different verifier hash")
        if metrics.get("response_models") != ["Qwen/Qwen3-1.7B"]:
            errors.append(f"{name} response model is not the frozen Qwen/Qwen3-1.7B model")
        if metrics.get("task_hash_mismatches") != 0:
            errors.append(f"{name} metrics contain task hash mismatches")

    if development_hardened.get("candidate_attack_success_rate") != 0.0:
        errors.append("Development hardened candidate ASR is not zero")
    if heldout_hardened.get("candidate_attack_success_rate") != 0.0:
        errors.append("Held-out hardened candidate ASR is not zero")
    if heldout_hardened.get("best_of_n_attack_success_rate") != 0.0:
        errors.append("Held-out hardened best-of-n ASR is not zero")
    if heldout_hardened.get("clean_explanation_rejection_rate") != 0.0:
        errors.append("Held-out clean-explanation rejection rate is not zero")

    if errors:
        raise RuntimeError("Publication freeze validation failed:\n- " + "\n- ".join(errors))

    return {
        "protocol_version": heldout.get("protocol_version"),
        "response_model": heldout.get("response_models", [None])[0],
        "development": {
            "tasks": development_split["tasks"],
            "eligible_candidates": development_split["valid_truth_preserving_candidates"],
            "hardened_candidate_asr": development_hardened["candidate_attack_success_rate"],
            "hardened_best_of_n_asr": development_hardened[
                "best_of_n_attack_success_rate"
            ],
        },
        "held_out": {
            "tasks": heldout_split["tasks"],
            "submitted_candidates": heldout_split["submitted_candidates"],
            "eligible_candidates": heldout_split["valid_truth_preserving_candidates"],
            "legacy_candidate_asr": heldout_legacy["candidate_attack_success_rate"],
            "legacy_candidate_asr_95pct_wilson": heldout_legacy[
                "candidate_attack_success_95pct_wilson"
            ],
            "legacy_best_of_n_asr": heldout_legacy["best_of_n_attack_success_rate"],
            "legacy_best_of_n_asr_95pct_wilson": heldout_legacy[
                "best_of_n_attack_success_95pct_wilson"
            ],
            "hardened_candidate_asr": heldout_hardened[
                "candidate_attack_success_rate"
            ],
            "hardened_candidate_asr_95pct_wilson": heldout_hardened[
                "candidate_attack_success_95pct_wilson"
            ],
            "hardened_best_of_n_asr": heldout_hardened[
                "best_of_n_attack_success_rate"
            ],
            "hardened_best_of_n_asr_95pct_wilson": heldout_hardened[
                "best_of_n_attack_success_95pct_wilson"
            ],
            "clean_rejection_rate": heldout_hardened[
                "clean_explanation_rejection_rate"
            ],
            "exact_mcnemar_p_value": heldout_split["paired_defense_test"][
                "exact_mcnemar_p_value"
            ],
        },
        "frozen_verifier_sha256": frozen_verifier_hash,
        "parent_manifest_sha256": expected_hashes["parent_manifest_sha256"],
        "hardened_manifest_sha256": sha256_file(hardened_manifest_path),
    }


def collect_files(root: Path, include_processed_test: bool) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for group, paths in REQUIRED_FILES.items():
        for relative in paths:
            path = root / relative
            if not path.is_file():
                raise FileNotFoundError(f"Required publication artifact is missing: {relative}")
            records.append(
                {
                    "group": group,
                    "path": relative,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "included_in_evidence_archive": (
                        include_processed_test or relative not in PROCESSED_DATA_PATHS
                    ),
                }
            )
    for relative in OPTIONAL_FILES:
        path = root / relative
        if path.is_file():
            records.append(
                {
                    "group": "optional_context",
                    "path": relative,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "included_in_evidence_archive": True,
                }
            )
    return sorted(records, key=lambda record: str(record["path"]))


def write_freeze(
    root: Path,
    output_dir: Path,
    *,
    include_processed_test: bool = False,
) -> tuple[Path, Path]:
    validation = validate_frozen_protocol(root)
    files = collect_files(root, include_processed_test)
    freeze_id = canonical_sha256(
        {
            "schema_version": SCHEMA_VERSION,
            "files": [
                {key: record[key] for key in ("path", "size_bytes", "sha256")}
                for record in files
            ],
            "validation": validation,
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "final_reproducibility_manifest.json"
    summary_path = output_dir / "final_results_summary.json"
    checksums_path = output_dir / "SHA256SUMS.txt"
    archive_path = output_dir / "CO-ACT-CPS-publication-evidence.zip"

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "freeze_id": freeze_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "root_name": root.name,
        "python_version": sys.version,
        "platform": platform.platform(),
        "processed_test_included_in_archive": include_processed_test,
        "validation": validation,
        "files": files,
        "interpretation_guardrail": (
            "This freeze records a zero observed hardened ASR for the specified held-out "
            "Qwen3-1.7B protocol. It does not establish universal robustness. Free-text "
            "attack-objective adherence requires blinded human/domain-expert review."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(validation, indent=2), encoding="utf-8")
    checksums_path.write_text(
        "".join(f"{record['sha256']}  {record['path']}\n" for record in files),
        encoding="utf-8",
    )

    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(manifest_path, manifest_path.name)
        archive.write(summary_path, summary_path.name)
        archive.write(checksums_path, checksums_path.name)
        for record in files:
            if record["included_in_evidence_archive"]:
                archive.write(root / str(record["path"]), str(record["path"]))

    archive_hash = sha256_file(archive_path)
    print(json.dumps({"freeze_id": freeze_id, "archive_sha256": archive_hash}, indent=2))
    print(f"Saved reproducibility manifest to {manifest_path.resolve()}")
    print(f"Saved publication evidence archive to {archive_path.resolve()}")
    return manifest_path, archive_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and freeze the final CO-ACT-CPS publication evidence"
    )
    parser.add_argument("--root", default=".")
    parser.add_argument("--output-dir", default="artifacts/publication_freeze")
    parser.add_argument(
        "--include-processed-test",
        action="store_true",
        help="Include processed test.npz bytes in the archive, not only their hash",
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    write_freeze(root, output_dir, include_processed_test=args.include_processed_test)


if __name__ == "__main__":
    main()
