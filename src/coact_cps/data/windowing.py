from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(slots=True)
class WindowArrays:
    x: np.ndarray
    y: np.ndarray
    root: np.ndarray
    end_indices: np.ndarray
    groups: np.ndarray | None = None

    def save(self, path: str) -> None:
        np.savez_compressed(
            path,
            x=self.x,
            y=self.y,
            root=self.root,
            end_indices=self.end_indices,
            groups=(
                self.groups if self.groups is not None else np.zeros(len(self.y), dtype=np.int64)
            ),
        )

    @classmethod
    def load(cls, path: str) -> WindowArrays:
        data = np.load(path)
        return cls(
            x=data["x"],
            y=data["y"],
            root=data["root"],
            end_indices=data["end_indices"],
            groups=(
                data["groups"]
                if "groups" in data.files
                else np.zeros(len(data["y"]), dtype=np.int64)
            ),
        )


def make_windows(
    values: np.ndarray,
    labels: np.ndarray,
    root_indices: np.ndarray | None,
    window_size: int,
    stride: int = 1,
    label_rule: str = "last",
    group_id: int = 0,
) -> WindowArrays:
    values = np.asarray(values, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    if values.ndim == 2:
        values = values[..., None]
    if values.ndim != 3:
        raise ValueError("values must have shape [time, nodes] or [time, nodes, features]")
    if len(values) != len(labels):
        raise ValueError("values and labels must have equal time length")
    if window_size < 2 or stride < 1:
        raise ValueError("window_size must be >=2 and stride must be >=1")
    if len(values) < window_size:
        raise ValueError("Not enough rows to form one window")
    roots = (
        np.full(len(labels), -1, dtype=np.int64)
        if root_indices is None
        else np.asarray(root_indices, dtype=np.int64)
    )

    x_windows: list[np.ndarray] = []
    y_windows: list[int] = []
    root_windows: list[int] = []
    ends: list[int] = []
    for start in range(0, len(values) - window_size + 1, stride):
        end = start + window_size
        current_labels = labels[start:end]
        if label_rule == "last":
            window_label = int(current_labels[-1])
        elif label_rule == "any":
            window_label = int(current_labels.max())
        else:
            raise ValueError("label_rule must be 'last' or 'any'")
        root_window = roots[start:end]
        attacked_roots = root_window[root_window >= 0]
        root = -1
        if window_label and len(attacked_roots):
            counts = np.bincount(attacked_roots)
            root = int(counts.argmax())
        x_windows.append(values[start:end])
        y_windows.append(window_label)
        root_windows.append(root)
        ends.append(end - 1)

    return WindowArrays(
        x=np.stack(x_windows).astype(np.float32),
        y=np.asarray(y_windows, dtype=np.int64),
        root=np.asarray(root_windows, dtype=np.int64),
        end_indices=np.asarray(ends, dtype=np.int64),
        groups=np.full(len(y_windows), group_id, dtype=np.int64),
    )


class CPSWindowDataset(Dataset):
    def __init__(self, arrays: WindowArrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return len(self.arrays.y)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "x": torch.from_numpy(self.arrays.x[index]),
            "y": torch.tensor(self.arrays.y[index], dtype=torch.float32),
            "root": torch.tensor(self.arrays.root[index], dtype=torch.long),
            "end_index": torch.tensor(self.arrays.end_indices[index], dtype=torch.long),
        }


def chronological_window_split(
    arrays: WindowArrays, train_fraction: float = 0.60, val_fraction: float = 0.20
) -> tuple[WindowArrays, WindowArrays, WindowArrays]:
    if not 0.0 < train_fraction < 1.0 or not 0.0 <= val_fraction < 1.0:
        raise ValueError("Invalid split fractions")
    n = len(arrays.y)
    train_end = int(n * train_fraction)
    val_end = int(n * (train_fraction + val_fraction))

    def subset(start: int, end: int) -> WindowArrays:
        return WindowArrays(
            x=arrays.x[start:end],
            y=arrays.y[start:end],
            root=arrays.root[start:end],
            end_indices=arrays.end_indices[start:end],
            groups=(None if arrays.groups is None else arrays.groups[start:end]),
        )

    return subset(0, train_end), subset(train_end, val_end), subset(val_end, n)
