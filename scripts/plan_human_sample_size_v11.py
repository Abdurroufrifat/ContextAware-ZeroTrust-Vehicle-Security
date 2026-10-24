from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from scipy.stats import norm


def paired_normal_approximation(effect_size: float, alpha: float, power: float) -> int:
    if effect_size <= 0:
        raise ValueError("effect_size must be positive")
    if not 0 < alpha < 1 or not 0 < power < 1:
        raise ValueError("alpha and power must lie between zero and one")
    value = (norm.ppf(1 - alpha / 2) + norm.ppf(power)) ** 2 / effect_size**2
    return int(math.ceil(value))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a participant level sample size sensitivity table"
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--power", type=float, default=0.90)
    parser.add_argument("--attrition", type=float, default=0.10)
    parser.add_argument("--output-directory", required=True, type=Path)
    args = parser.parse_args()
    if not 0 <= args.attrition < 1:
        raise ValueError("attrition must lie in [0, 1)")

    rows = []
    for effect_size in (0.30, 0.35, 0.40, 0.45, 0.50, 0.60):
        analyzable = paired_normal_approximation(effect_size, args.alpha, args.power)
        recruit = int(math.ceil(analyzable / (1.0 - args.attrition)))
        rows.append(
            {
                "standardized_participant_level_effect": effect_size,
                "two_sided_alpha": args.alpha,
                "target_power": args.power,
                "analyzable_participants": analyzable,
                "recruitment_target_with_attrition": recruit,
            }
        )

    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "sample_size_sensitivity.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "method": "paired participant level normal approximation",
        "purpose": "planning sensitivity only",
        "primary_design_target": {
            "standardized_effect": 0.40,
            "power": args.power,
            "alpha": args.alpha,
            "analyzable_participants": next(
                row["analyzable_participants"]
                for row in rows
                if row["standardized_participant_level_effect"] == 0.40
            ),
            "recruitment_target_with_attrition": next(
                row["recruitment_target_with_attrition"]
                for row in rows
                if row["standardized_participant_level_effect"] == 0.40
            ),
        },
        "limitation": (
            "Freeze the final sample size with a methods adviser before recruitment. "
            "The confirmatory generalized mixed model and the expected practitioner "
            "effect may require a simulation based adjustment."
        ),
        "sensitivity_rows": rows,
    }
    json_path = output / "sample_size_plan.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["primary_design_target"], indent=2))


if __name__ == "__main__":
    main()

