from __future__ import annotations

import argparse
from pathlib import Path

from coact_cps.config import load_config
from coact_cps.data.csv_adapter import prepare_from_csv, save_prepared_dataset


def _resolve_paths(values: list[str], base: Path) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = base / candidate
        if any(character in str(candidate) for character in "*?[]"):
            paths.extend(sorted(candidate.parent.glob(candidate.name)))
        else:
            paths.append(candidate)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare HAI or SWaT CSV files")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    base = config_path.parent.parent
    dataset = config["dataset"]
    prepared = prepare_from_csv(
        train_paths=_resolve_paths(dataset["train_csvs"], base),
        test_paths=_resolve_paths(dataset["test_csvs"], base),
        feature_names=dataset.get("features"),
        label_column=dataset.get("label_column"),
        normal_values=dataset.get("normal_values", ["0", "normal"]),
        window_size=int(dataset.get("window_size", 64)),
        stride=int(dataset.get("stride", 4)),
        label_rule=dataset.get("label_rule", "last"),
        root_column=dataset.get("root_column"),
        validation_fraction=float(dataset.get("validation_fraction", 0.20)),
    )
    output = Path(dataset["output_directory"])
    if not output.is_absolute():
        output = base / output
    save_prepared_dataset(prepared, output)
    print(f"Prepared {len(prepared.feature_names)} features in {output}")


if __name__ == "__main__":
    main()
