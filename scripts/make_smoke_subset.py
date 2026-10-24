from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def select_indices(labels: np.ndarray, limit: int, stratified: bool) -> np.ndarray:
    if len(labels) <= limit:
        return np.arange(len(labels), dtype=np.int64)

    if not stratified or len(np.unique(labels)) < 2:
        return np.linspace(0, len(labels) - 1, limit, dtype=np.int64)

    classes = np.unique(labels)
    selected: list[np.ndarray] = []
    per_class = max(1, limit // len(classes))
    for value in classes:
        candidates = np.flatnonzero(labels == value)
        count = min(per_class, len(candidates))
        positions = np.linspace(0, len(candidates) - 1, count, dtype=np.int64)
        selected.append(candidates[positions])

    indices = np.unique(np.concatenate(selected))
    if len(indices) < limit:
        remaining = np.setdiff1d(np.arange(len(labels)), indices, assume_unique=True)
        count = min(limit - len(indices), len(remaining))
        positions = np.linspace(0, len(remaining) - 1, count, dtype=np.int64)
        indices = np.concatenate((indices, remaining[positions]))
    return np.sort(indices[:limit])


def subset_file(source: Path, destination: Path, limit: int, stratified: bool) -> None:
    with np.load(source) as data:
        indices = select_indices(data["y"], limit, stratified)
        payload = {name: data[name][indices] for name in data.files}
    np.savez_compressed(destination, **payload)
    positives = int(payload["y"].sum())
    print(
        f"Created {destination.name}: {len(indices):,} windows, "
        f"{positives:,} attack windows"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a small CPU smoke subset")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--train", type=int, default=1024)
    parser.add_argument("--validation", type=int, default=512)
    parser.add_argument("--test", type=int, default=512)
    args = parser.parse_args()

    source = args.input.resolve()
    destination = args.output.resolve()
    if source == destination:
        raise ValueError("Input and output directories must be different")
    destination.mkdir(parents=True, exist_ok=True)

    for name in ("metadata.json", "train.npz", "val.npz", "test.npz"):
        if not (source / name).exists():
            raise FileNotFoundError(source / name)

    subset_file(source / "train.npz", destination / "train.npz", args.train, False)
    subset_file(source / "val.npz", destination / "val.npz", args.validation, False)
    subset_file(source / "test.npz", destination / "test.npz", args.test, True)

    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    metadata["development_subset"] = True
    metadata["warning"] = (
        "CPU smoke subset for pipeline validation only; never report as publication evidence."
    )
    (destination / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"CPU smoke subset ready: {destination}")


if __name__ == "__main__":
    main()
