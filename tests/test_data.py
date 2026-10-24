from __future__ import annotations

import numpy as np
import pandas as pd

from coact_cps.data.csv_adapter import prepare_from_csv
from coact_cps.data.synthetic import SyntheticCPSConfig, generate_synthetic_cps
from coact_cps.data.windowing import chronological_window_split, make_windows


def test_synthetic_data_and_topology_shapes() -> None:
    frame, topology = generate_synthetic_cps(SyntheticCPSConfig(n_steps=400, seed=3))
    assert len(frame) == 400
    assert frame["attack"].sum() > 0
    adjacency = topology.relation_adjacency(bidirectional=True)
    assert adjacency.shape == (2, 6, 6)
    assert adjacency.sum() > 0


def test_windowing_and_chronological_split() -> None:
    values = np.arange(120, dtype=np.float32).reshape(40, 3)
    labels = np.zeros(40, dtype=np.int64)
    labels[25:30] = 1
    roots = np.full(40, -1, dtype=np.int64)
    roots[25:30] = 2
    arrays = make_windows(values, labels, roots, window_size=8, stride=2, label_rule="any")
    assert arrays.x.shape[1:] == (8, 3, 1)
    assert arrays.y.sum() > 0
    assert 2 in arrays.root
    train, val, test = chronological_window_split(arrays)
    assert len(train.y) + len(val.y) + len(test.y) == len(arrays.y)
    assert train.end_indices.max() < test.end_indices.max()


def test_csv_adapter_never_windows_across_files(tmp_path) -> None:
    train_paths = []
    test_paths = []
    for split, paths, labels in (
        ("train", train_paths, [0, 0]),
        ("test", test_paths, [0, 1]),
    ):
        for index, label in enumerate(labels):
            destination = tmp_path / f"{split}-{index}.csv"
            pd.DataFrame(
                {
                    "S1": np.linspace(index, index + 1, 60),
                    "A1": np.tile([0, 1], 30),
                    "attack": np.full(60, label),
                }
            ).to_csv(destination, index=False)
            paths.append(destination)
    prepared = prepare_from_csv(
        train_paths=train_paths,
        test_paths=test_paths,
        feature_names=["S1", "A1"],
        label_column="attack",
        normal_values=[0],
        window_size=12,
        stride=3,
    )
    assert len(np.unique(prepared.train.groups)) == 1
    assert len(np.unique(prepared.val.groups)) == 1
    assert len(np.unique(prepared.test.groups)) == 2
    assert len(prepared.test.y) == 34
