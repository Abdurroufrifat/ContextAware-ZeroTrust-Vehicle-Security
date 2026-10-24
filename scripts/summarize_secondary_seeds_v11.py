from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

from scipy.stats import t


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(values: list[float]) -> dict[str, Any]:
    count = len(values)
    mean = statistics.fmean(values)
    standard_deviation = statistics.stdev(values) if count > 1 else 0.0
    critical = float(t.ppf(0.975, count - 1)) if count > 1 else 0.0
    margin = critical * standard_deviation / math.sqrt(count) if count > 1 else 0.0
    return {
        "runs": count,
        "mean": mean,
        "standard_deviation": standard_deviation,
        "minimum": min(values),
        "maximum": max(values),
        "mean_95pct_t_interval": [mean - margin, mean + margin],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize repeated secondary dataset runs")
    parser.add_argument("--metrics-root", required=True, type=Path)
    parser.add_argument("--expected-runs", type=int, default=5)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    paths = sorted(args.metrics_root.glob("seed_*/test_metrics.json"))
    if len(paths) != args.expected_runs:
        raise ValueError(f"Expected {args.expected_runs} run files, found {len(paths)}")
    records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    numeric_keys = sorted(
        key
        for key in set.intersection(
            *(set(record) for record in records)
        )
        if all(
            isinstance(record[key], (int, float)) and math.isfinite(float(record[key]))
            for record in records
        )
    )
    result = {
        "protocol_version": "cps-truthframe-secondary-dataset-summary-v11.0",
        "runs": len(paths),
        "metrics": {
            key: summarize([float(record[key]) for record in records]) for key in numeric_keys
        },
        "run_files": [
            {"path": str(path), "sha256": sha256(path)} for path in paths
        ],
        "interpretation": (
            "Intervals describe variability across configured random seeds. They do not "
            "replace uncertainty over attack episodes or prove transfer to other plants."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "runs": len(paths),
                "metrics": numeric_keys,
                "output": str(args.output.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
