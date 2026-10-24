from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .windowing import WindowArrays, make_windows


@dataclass(slots=True)
class PreparedDataset:
    train: WindowArrays
    val: WindowArrays
    test: WindowArrays
    feature_names: list[str]
    scaler_mean: list[float]
    scaler_scale: list[float]


def inspect_csv(path: str | Path, rows: int = 5) -> dict:
    csv_path = Path(path)
    frame = pd.read_csv(csv_path, nrows=max(rows, 1), low_memory=False)
    return {
        "path": str(csv_path.resolve()),
        "columns": list(frame.columns),
        "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "sample": frame.head(rows).astype(str).to_dict(orient="records"),
    }


def _read_frames(paths: Sequence[str | Path]) -> list[pd.DataFrame]:
    if not paths:
        raise ValueError("At least one CSV path is required")
    frames = []
    for path in paths:
        csv_path = Path(path)
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        frame = pd.read_csv(csv_path, low_memory=False)
        frame.columns = [str(column).strip() for column in frame.columns]
        frames.append(frame)
    return frames


def _combine_windows(items: Sequence[WindowArrays]) -> WindowArrays:
    if not items:
        raise ValueError("No source file was long enough to form a window")
    return WindowArrays(
        x=np.concatenate([item.x for item in items]),
        y=np.concatenate([item.y for item in items]),
        root=np.concatenate([item.root for item in items]),
        end_indices=np.concatenate([item.end_indices for item in items]),
        groups=np.concatenate(
            [
                item.groups if item.groups is not None else np.zeros(len(item.y), dtype=np.int64)
                for item in items
            ]
        ),
    )


def _training_validation_frames(
    frames: Sequence[pd.DataFrame], validation_fraction: float, window_size: int
) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    if len(frames) == 1:
        frame = frames[0]
        split_row = int(len(frame) * (1.0 - validation_fraction))
        if split_row <= window_size or len(frame) - split_row <= window_size:
            raise ValueError("Training data is too small for the requested validation split")
        return (
            [frame.iloc[:split_row].reset_index(drop=True)],
            [frame.iloc[split_row:].reset_index(drop=True)],
        )
    target_rows = max(1, int(sum(len(frame) for frame in frames) * validation_fraction))
    validation: list[pd.DataFrame] = []
    accumulated = 0
    split_index = len(frames)
    for index in range(len(frames) - 1, 0, -1):
        validation.insert(0, frames[index])
        accumulated += len(frames[index])
        split_index = index
        if accumulated >= target_rows:
            break
    training = list(frames[:split_index])
    if not training or not validation:
        raise ValueError("Could not create a whole-file validation split")
    return training, validation


def _binary_labels(
    frame: pd.DataFrame,
    label_column: str | None,
    normal_values: Iterable[str | int | float],
) -> np.ndarray:
    if not label_column:
        return np.zeros(len(frame), dtype=np.int64)
    if label_column not in frame.columns:
        raise KeyError(f"Label column '{label_column}' was not found")
    normal = {str(value).strip().lower() for value in normal_values}
    normalized = frame[label_column].astype(str).str.strip().str.lower()
    return (~normalized.isin(normal)).astype(np.int64).to_numpy()


def _numeric_matrix(frame: pd.DataFrame, features: Sequence[str]) -> np.ndarray:
    missing = [feature for feature in features if feature not in frame.columns]
    if missing:
        raise KeyError(f"Missing configured feature columns: {missing[:20]}")
    numeric = frame.loc[:, features].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    numeric = numeric.ffill().bfill().fillna(0.0)
    return numeric.to_numpy(dtype=np.float32)


def _root_indices(
    frame: pd.DataFrame, root_column: str | None, feature_names: Sequence[str]
) -> np.ndarray:
    roots = np.full(len(frame), -1, dtype=np.int64)
    if not root_column or root_column not in frame.columns:
        return roots
    index = {name: idx for idx, name in enumerate(feature_names)}
    for row_idx, value in enumerate(frame[root_column].astype(str)):
        roots[row_idx] = index.get(value.strip(), -1)
    return roots


def prepare_from_csv(
    train_paths: Sequence[str | Path],
    test_paths: Sequence[str | Path],
    feature_names: Sequence[str] | None,
    label_column: str | None,
    normal_values: Sequence[str | int | float],
    window_size: int,
    stride: int,
    label_rule: str = "last",
    root_column: str | None = None,
    validation_fraction: float = 0.20,
) -> PreparedDataset:
    """Prepare chronological windows without fitting preprocessing on test data."""

    train_frames = _read_frames(train_paths)
    test_frames = _read_frames(test_paths)
    all_training = pd.concat(train_frames, axis=0, ignore_index=True)
    excluded = {column for column in (label_column, root_column) if column}
    if feature_names is None:
        candidates = [column for column in all_training.columns if column not in excluded]
        feature_names = [
            column
            for column in candidates
            if pd.to_numeric(all_training[column], errors="coerce").notna().mean() > 0.95
        ]
    feature_names = list(feature_names)
    if not feature_names:
        raise ValueError("No usable numeric features were selected")

    fit_frames, val_frames = _training_validation_frames(
        train_frames, validation_fraction, window_size
    )

    scaler = StandardScaler()
    fit_matrix = np.concatenate(
        [_numeric_matrix(frame, feature_names) for frame in fit_frames], axis=0
    )
    scaler.fit(fit_matrix)

    def convert(frames: Sequence[pd.DataFrame], group_offset: int) -> WindowArrays:
        windows = []
        row_offset = 0
        for index, frame in enumerate(frames):
            if len(frame) < window_size:
                row_offset += len(frame)
                continue
            matrix = scaler.transform(_numeric_matrix(frame, feature_names)).astype(np.float32)
            labels = _binary_labels(frame, label_column, normal_values)
            roots = _root_indices(frame, root_column, feature_names)
            item = make_windows(
                matrix,
                labels,
                roots,
                window_size=window_size,
                stride=stride,
                label_rule=label_rule,
                group_id=group_offset + index,
            )
            item.end_indices = item.end_indices + row_offset
            windows.append(item)
            row_offset += len(frame)
        return _combine_windows(windows)

    return PreparedDataset(
        train=convert(fit_frames, 0),
        val=convert(val_frames, len(fit_frames)),
        test=convert(test_frames, len(fit_frames) + len(val_frames)),
        feature_names=feature_names,
        scaler_mean=scaler.mean_.astype(float).tolist(),
        scaler_scale=scaler.scale_.astype(float).tolist(),
    )


def save_prepared_dataset(dataset: PreparedDataset, output_directory: str | Path) -> None:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    dataset.train.save(str(output / "train.npz"))
    dataset.val.save(str(output / "val.npz"))
    dataset.test.save(str(output / "test.npz"))
    metadata = {
        "feature_names": dataset.feature_names,
        "scaler_mean": dataset.scaler_mean,
        "scaler_scale": dataset.scaler_scale,
        "source_group_counts": {
            "train": int(len(np.unique(dataset.train.groups))),
            "validation": int(len(np.unique(dataset.val.groups))),
            "test": int(len(np.unique(dataset.test.groups))),
        },
        "warning": "Dataset windows contain no publication results until models are evaluated.",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
