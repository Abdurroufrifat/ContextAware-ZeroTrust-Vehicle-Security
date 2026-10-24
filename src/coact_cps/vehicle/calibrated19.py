"""Experimental fine-tuning helpers. Existing detector implementation is unchanged."""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from coact_cps.vehicle.ai_models13 import adjacency

EPOCHS = 5
LEARNING_RATE = 0.0001
BATCH_SIZE = 64


def require_clean(context):
    labels = context["labels"]
    if labels.dtype != bool or labels.ndim != 1 or labels.any():
        raise ValueError("only normal contexts may enter fine-tuning or calibration")


def make_windows(context, scaler):
    f, t, s, p = (np.asarray(context[k]) for k in ("features", "time_ms", "start_ms", "positions"))
    mean, scale = np.asarray(scaler["mean"]), np.asarray(scaler["scale"])
    if (
        f.ndim != 2
        or f.shape[1] != 3
        or not np.isfinite(f).all()
        or t.shape != (len(f),)
        or s.shape != t.shape
        or t.dtype.kind not in "iu"
        or s.dtype.kind not in "iu"
        or p.ndim != 1
        or not len(p)
        or p.dtype.kind not in "iu"
        or np.any(p < 31)
        or np.any(p >= len(f))
        or np.any(np.diff(p) <= 0)
        or mean.shape != (3,)
        or scale.shape != (3,)
        or np.any(scale <= 0)
        or not np.isfinite(mean).all()
        or not np.isfinite(scale).all()
    ):
        raise ValueError("invalid context or original feature scaling")
    t, s = t.astype(np.int64), s.astype(np.int64)
    dt = (t - s) / 1000
    if np.any(dt <= 0) or np.any(dt > 2) or np.any(np.diff(t) <= 0):
        raise ValueError("invalid intervals")
    x, durations = [], []
    for i in p:
        if not np.array_equal(s[i - 30 : i + 1], t[i - 31 : i]):
            raise ValueError("noncontiguous scoring window")
        x.append((f[i - 31 : i + 1] - mean) / scale)
        durations.append(dt[i - 31 : i + 1])
    return np.asarray(x, dtype=np.float32), np.asarray(durations, dtype=np.float32)


def calibration_loss(pred, x, dt, scaler, slope, offset):
    """Five-interval speed consistency with one-epoch-old IMU inputs.

    Current target values are excluded from the physical integral. All three
    current targets remain in the separate reconstruction MSE during training.
    """
    mean = torch.as_tensor(scaler["mean"], dtype=x.dtype, device=x.device)
    scale = torch.as_tensor(scaler["scale"], dtype=x.dtype, device=x.device)
    physical = x * scale + mean
    predicted = pred * scale + mean
    acceleration = slope * physical[:, -6:-1, 2] + offset
    integral = (acceleration * dt[:, -5:]).sum(dim=1)
    speed_start = physical[:, -6, :2].mean(dim=1)
    consistency = predicted[:, :2].mean(dim=1) - speed_start - integral
    return ((predicted[:, 0] - predicted[:, 1]).square() + consistency.square()).mean()


def fine_tune(adapter, x, dt, scaler, seed, slope, offset):
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(dt)),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    optimizer = torch.optim.Adam(adapter.model.parameters(), lr=LEARNING_RATE)
    history = []
    adapter.model.train()
    for _ in range(EPOCHS):
        total = 0.0
        for xb, durations in loader:
            optimizer.zero_grad()
            masked = xb.clone()
            masked[:, -1] = 0
            pred = adapter.model(masked[:, :, :, None], adjacency()).reconstructed[:, -1, :, 0]
            reconstruction = (pred - xb[:, -1]).square().mean()
            loss = reconstruction + 0.05 * calibration_loss(
                pred, xb, durations, scaler, slope, offset
            )
            if not torch.isfinite(loss):
                raise ValueError("nonfinite fine-tuning loss")
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(xb)
        history.append(total / len(x))
    adapter.model.eval()
    return history
