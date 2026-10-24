from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

import sklearn

from coact_cps.security.same_task_baselines import evaluate_same_task_baselines


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("No candidate rows were produced")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(path: Path, metrics: dict) -> None:
    rows = []
    for split, split_result in metrics["by_split"].items():
        for baseline, result in split_result["baselines"].items():
            rows.append(
                {
                    "split": split,
                    "baseline": baseline,
                    "clean_coverage": result["by_cohort"]["clean"]["acceptance_rate"],
                    "benign_coverage": result["by_cohort"]["benign"]["acceptance_rate"],
                    "safe_coverage": result["safe_coverage"]["rate"],
                    "candidate_attack_success_rate": result["by_cohort"]["attack"][
                        "acceptance_rate"
                    ],
                    "candidate_attack_containment_rate": result["attack_containment"][
                        "rate"
                    ],
                    "task_best_of_n_attack_success_rate": result[
                        "task_best_of_n_attack_success"
                    ]["acceptance_rate"],
                    "balanced_security_utility": result["balanced_security_utility"],
                }
            )
    write_csv(path, rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate same task operator surface baselines on development safe surfaces "
            "and evaluate the untouched heldout red team records"
        )
    )
    parser.add_argument("--development-tasks", required=True, type=Path)
    parser.add_argument("--development-responses", required=True, type=Path)
    parser.add_argument("--heldout-tasks", required=True, type=Path)
    parser.add_argument("--heldout-responses", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--safe-retention-target", type=float, default=0.95)
    args = parser.parse_args()

    inputs = {
        "development_tasks": args.development_tasks.resolve(),
        "development_responses": args.development_responses.resolve(),
        "heldout_tasks": args.heldout_tasks.resolve(),
        "heldout_responses": args.heldout_responses.resolve(),
    }
    metrics, rows = evaluate_same_task_baselines(
        read_jsonl(inputs["development_tasks"]),
        read_jsonl(inputs["development_responses"]),
        read_jsonl(inputs["heldout_tasks"]),
        read_jsonl(inputs["heldout_responses"]),
        safe_retention_target=args.safe_retention_target,
    )
    metrics["input_sha256"] = {
        name: {"file": path.name, "sha256": file_sha256(path)}
        for name, path in inputs.items()
    }
    metrics["software"] = {
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
    }

    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "same_task_baselines_v11_metrics.json"
    scores_path = output / "same_task_baselines_v11_candidate_scores.csv"
    summary_path = output / "same_task_baselines_v11_summary.csv"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_csv(scores_path, rows)
    write_summary_csv(summary_path, metrics)

    heldout = metrics["by_split"]["held_out"]["baselines"]
    concise = {
        name: {
            "safe_coverage": result["safe_coverage"]["rate"],
            "candidate_attack_success_rate": result["by_cohort"]["attack"][
                "acceptance_rate"
            ],
            "task_best_of_n_attack_success_rate": result[
                "task_best_of_n_attack_success"
            ]["acceptance_rate"],
        }
        for name, result in heldout.items()
    }
    print(json.dumps(concise, indent=2))
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved candidate scores to {scores_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()

