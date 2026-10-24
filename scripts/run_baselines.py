from __future__ import annotations

import argparse
import json
from pathlib import Path

from coact_cps.baselines import evaluate_isolation_forest, evaluate_persistence
from coact_cps.data.windowing import WindowArrays


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run calibrated persistence and Isolation Forest baselines"
    )
    parser.add_argument("--data-directory", required=True)
    parser.add_argument("--output", default="artifacts/baselines.json")
    parser.add_argument("--calibration-quantile", type=float, default=0.995)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    source = Path(args.data_directory)
    train = WindowArrays.load(str(source / "train.npz"))
    validation = WindowArrays.load(str(source / "val.npz"))
    test = WindowArrays.load(str(source / "test.npz"))
    results = {
        "persistence": evaluate_persistence(validation, test, quantile=args.calibration_quantile),
        "isolation_forest": evaluate_isolation_forest(
            train,
            validation,
            test,
            quantile=args.calibration_quantile,
            seed=args.seed,
        ),
        "warning": "Add deep temporal and published dataset-specific baselines for the paper.",
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(results, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps(results, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
