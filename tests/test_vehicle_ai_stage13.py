from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "method",
    [
        "coact_physics",
        "coact_no_graph",
        "coact_no_physics",
        "gdn_adapted",
        "catch_adapted",
        "isolation_forest",
        "lof",
        "persistence",
    ],
)
def test_adapters_train_and_roundtrip(method, tmp_path):
    from run_vehicle_ai_stage13 import checkpoint, load_adapter
    from vehicle_ai_data13 import Scaling

    from coact_cps.vehicle.ai_models13 import fit_adapter, score_nodes

    cfg = dict(window=32, epochs=1, batch_size=8)
    x = np.random.default_rng(3).normal(size=(16, 32, 3)).astype(np.float32)
    a, h = fit_adapter(
        method, x, np.full(16, 0.2), Scaling((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), cfg, 7, ROOT
    )
    p, s = checkpoint(tmp_path, tmp_path, a, cfg)
    spec = dict(
        method=method,
        seed=7,
        checkpoint=None if p is None else str(tmp_path / p),
        checkpoint_sha256=s,
    )
    b = load_adapter(ROOT, spec, cfg)
    first = score_nodes(a, x, cfg, 101)
    assert np.isfinite(first).all() and np.array_equal(first, score_nodes(b, x, cfg, 101))
    assert np.array_equal(first, score_nodes(a, x, cfg, 101))


def test_physics_and_graph_ablation():
    from vehicle_ai_data13 import Scaling

    from coact_cps.vehicle.ai_models13 import adjacency, physics_loss

    x = torch.zeros(2, 32, 3)
    pred = torch.tensor([[0.1, 0.1, 1.0], [0.1, 0.1, 1.0]])
    scaling = Scaling((10.0, 10.0, 0.0), (2.0, 2.0, 1.0))
    assert physics_loss(pred, x, torch.tensor([0.2, 0.2]), scaling) < 1e-10
    assert (
        physics_loss(pred + torch.tensor([1.0, 0.0, 0.0]), x, torch.tensor([0.2, 0.2]), scaling) > 1
    )
    assert torch.count_nonzero(adjacency(True)[:, ~torch.eye(3, dtype=torch.bool)]) == 0
    assert torch.count_nonzero(adjacency(False)[:, ~torch.eye(3, dtype=torch.bool)]) > 0


def test_windows_restart_and_never_read_future():
    from test_vehicle_window_recovery import WindowRecoveryTests
    from vehicle_ai_data13 import windows

    fs = WindowRecoveryTests().frames()[:70]
    a = windows(fs, 8)
    fs2 = list(fs)
    fs2[30] = replace(fs2[30], interval_start_ms=fs2[30].interval_start_ms + 1)
    b = windows(fs2, 8)
    assert not any(30 <= i <= 36 for i in b.positions)
    fs3 = [replace(f, wheel_speed=999.0) if f.frame == 50 else f for f in fs]
    c = windows(fs3, 8)
    assert np.array_equal(a.x[:43], c.x[:43])
