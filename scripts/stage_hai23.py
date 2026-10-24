from __future__ import annotations

import argparse
from itertools import zip_longest
from pathlib import Path

import pandas as pd

TRAIN_FILES = tuple(f"hai-train{index}.csv" for index in range(1, 5))
TEST_PAIRS = tuple(
    (f"hai-test{index}.csv", f"label-test{index}.csv") for index in range(1, 3)
)


def _write_chunks(chunks, destination: Path) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    first = True
    for chunk in chunks:
        chunk.to_csv(
            destination,
            index=False,
            mode="w" if first else "a",
            header=first,
        )
        rows += len(chunk)
        first = False
    return rows


def stage_training(source: Path, destination: Path, chunk_size: int) -> int:
    def chunks():
        for chunk in pd.read_csv(source, chunksize=chunk_size, low_memory=False):
            chunk["attack"] = 0
            yield chunk

    return _write_chunks(chunks(), destination)


def stage_test(
    source: Path,
    label_source: Path,
    destination: Path,
    chunk_size: int,
) -> tuple[int, int]:
    attack_rows = 0
    label_rows = 0

    def chunks():
        nonlocal attack_rows, label_rows
        telemetry_reader = pd.read_csv(source, chunksize=chunk_size, low_memory=False)
        label_reader = pd.read_csv(label_source, chunksize=chunk_size, low_memory=False)
        for chunk_number, pair in enumerate(
            zip_longest(telemetry_reader, label_reader), start=1
        ):
            chunk, labels = pair
            if chunk is None or labels is None:
                raise ValueError(
                    f"Telemetry/label chunk mismatch for {source.name}"
                )
            if "timestamp" not in chunk.columns:
                raise KeyError(f"{source} is missing the timestamp column")
            required = {"timestamp", "label"}
            missing = required - set(labels.columns)
            if missing:
                raise KeyError(f"{label_source} is missing columns: {sorted(missing)}")
            if len(chunk) != len(labels):
                raise ValueError(
                    f"Row mismatch in chunk {chunk_number} for {source.name}: "
                    f"telemetry={len(chunk)}, labels={len(labels)}"
                )
            telemetry_time = pd.to_datetime(
                chunk["timestamp"], errors="raise"
            ).reset_index(drop=True)
            label_time = pd.to_datetime(
                labels["timestamp"], errors="raise"
            ).reset_index(drop=True)
            exact_match = telemetry_time.eq(label_time)
            same_minute = telemetry_time.dt.floor("min").eq(label_time.dt.floor("min"))
            mismatched = ~(exact_match | same_minute)
            if mismatched.any():
                offset = int(mismatched.to_numpy().argmax())
                raise ValueError(
                    f"Timestamp mismatch in chunk {chunk_number}, row {offset}, "
                    f"for {source.name}"
                )
            chunk["attack"] = pd.to_numeric(labels["label"], errors="raise").to_numpy()
            chunk["attack"] = chunk["attack"].astype("int8")
            attack_rows += int(chunk["attack"].sum())
            label_rows += len(labels)
            yield chunk

    rows = _write_chunks(chunks(), destination)
    if rows != label_rows:
        raise ValueError(
            f"Row mismatch for {source.name}: telemetry={rows}, labels={label_rows}"
        )
    return rows, attack_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create HAI 23.05 CSV files with an integrated attack column"
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    args = parser.parse_args()

    source = args.input.resolve()
    destination = args.output.resolve()
    if source == destination:
        raise ValueError("Input and output directories must be different")
    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be positive")

    expected = [*TRAIN_FILES]
    for telemetry_name, label_name in TEST_PAIRS:
        expected.extend((telemetry_name, label_name))
    missing = [name for name in expected if not (source / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing HAI files: {missing}")

    for name in TRAIN_FILES:
        rows = stage_training(source / name, destination / name, args.chunk_size)
        print(f"Staged {name}: {rows:,} normal rows")

    for telemetry_name, label_name in TEST_PAIRS:
        rows, attacks = stage_test(
            source / telemetry_name,
            source / label_name,
            destination / telemetry_name,
            args.chunk_size,
        )
        print(f"Staged {telemetry_name}: {rows:,} rows, {attacks:,} attack rows")

    print(f"HAI 23.05 staging complete: {destination}")


if __name__ == "__main__":
    main()
