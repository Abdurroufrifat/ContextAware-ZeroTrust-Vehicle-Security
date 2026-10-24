from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd

from coact_cps.data.hai23_catalog import HAI23_ATTACKS_BY_LABEL_FILE, observable_targets


def positive_runs(frame: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp, int]]:
    labels = pd.to_numeric(frame["label"], errors="raise").astype(bool)
    starts = frame.loc[labels & ~labels.shift(fill_value=False), "timestamp"].reset_index(drop=True)
    ends = frame.loc[labels & ~labels.shift(-1, fill_value=False), "timestamp"].reset_index(
        drop=True
    )
    counts = []
    active = False
    current = 0
    for value in labels:
        if value:
            current += 1
            active = True
        elif active:
            counts.append(current)
            current = 0
            active = False
    if active:
        counts.append(current)
    return list(zip(starts, ends, counts, strict=True))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate HAI 23.05 labels and build the official attack-target catalog"
    )
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    feature_names = set(
        json.loads(args.metadata.read_text(encoding="utf-8"))["feature_names"]
    )
    rows = []
    for filename, attacks in HAI23_ATTACKS_BY_LABEL_FILE.items():
        path = args.labels / filename
        frame = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False)
        required = {"timestamp", "label"}
        missing = required - set(frame.columns)
        if missing:
            raise KeyError(f"{path} is missing columns: {sorted(missing)}")
        runs = positive_runs(frame)
        if len(runs) != len(attacks):
            raise ValueError(
                f"{filename}: finalized labels contain {len(runs)} episodes, "
                f"but the official catalog contains {len(attacks)} scenarios"
            )
        for episode, (attack, (start, end, positive_rows)) in enumerate(
            zip(attacks, runs, strict=True), start=1
        ):
            observed = observable_targets(attack, feature_names)
            rows.append(
                {
                    "label_file": filename,
                    "episode": episode,
                    "attack_id": attack.attack_id,
                    "primitives": ";".join(attack.primitives),
                    "target_controllers": ";".join(attack.target_controllers),
                    "target_points": ";".join(attack.target_points),
                    "observable_target_points": ";".join(observed),
                    "label_start": start.isoformat(sep=" "),
                    "label_end": end.isoformat(sep=" "),
                    "positive_rows": positive_rows,
                    "exact_target_evaluable": bool(observed),
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    evaluable = sum(row["exact_target_evaluable"] for row in rows)
    print(f"Validated {len(rows)} finalized HAI 23.05 attack episodes")
    print("  label-test1.csv: 14 episodes")
    print("  label-test2.csv: 38 episodes")
    print(f"  exact target evaluable from the 86 telemetry features: {evaluable}/{len(rows)}")
    print(f"Saved official attack catalog: {args.output.resolve()}")


if __name__ == "__main__":
    main()
