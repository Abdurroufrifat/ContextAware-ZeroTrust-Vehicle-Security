"""Inactive development guard preserving the original GNSS/wheel decisions."""

import numpy as np

FIT_ROUTES = {"civic13_07", "civic13_08"}
MIN_BIN = 30
SCALE_FLOOR = 0.1
PRIMARY_CAP = 1.0


def causal_context(features, time_ms, start_ms):
    f, t, s = np.asarray(features), np.asarray(time_ms), np.asarray(start_ms)
    if (
        f.ndim != 2
        or f.shape[1] != 3
        or not np.isfinite(f).all()
        or t.shape != (len(f),)
        or s.shape != t.shape
        or t.dtype.kind not in "iu"
        or s.dtype.kind not in "iu"
    ):
        raise ValueError("finite features and integer timestamp vectors required")
    t, s = t.astype(np.int64), s.astype(np.int64)
    if np.any(t <= s) or np.any(t - s > 2000) or np.any(np.diff(t) <= 0):
        raise ValueError("invalid time intervals")
    result = np.full(len(f), np.nan)
    speed = f[:, :2].mean(axis=1)
    for i in range(6, len(f)):
        if np.array_equal(s[i - 5 : i], t[i - 6 : i - 1]):
            duration = (int(t[i - 1]) - int(t[i - 6])) / 1000
            result[i] = abs(speed[i - 1] - speed[i - 6]) / duration
    return result


def fit_partition(context_routes):
    if set(context_routes) != FIT_ROUTES:
        raise ValueError("exact two clean development fit routes required")
    pool = []
    for x in context_routes.values():
        x = np.asarray(x, dtype=float)
        if x.ndim != 1 or np.any(x < 0):
            raise ValueError("nonnegative context vectors required")
        finite = x[np.isfinite(x)]
        if len(finite) < 2 * MIN_BIN:
            raise ValueError("insufficient finite fit context")
        pool.append(finite)
    values = np.concatenate(pool)
    return {
        "edges": np.unique(np.quantile(values, [0.2, 0.4, 0.6, 0.8])).tolist(),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


def _locate(context, partition):
    x = np.asarray(context, dtype=float)
    if x.ndim != 1:
        raise ValueError("one-dimensional context required")
    valid = np.isfinite(x) & (x >= partition["minimum"]) & (x <= partition["maximum"])
    return np.searchsorted(partition["edges"], x, side="right"), valid


def fit_profile(context_routes, imu_routes, partition):
    if set(context_routes) != FIT_ROUTES or set(imu_routes) != FIT_ROUTES:
        raise ValueError("exact normal fit routes required")
    xs, ys = [], []
    for route in sorted(FIT_ROUTES):
        x, y = np.asarray(context_routes[route]), np.asarray(imu_routes[route])
        if x.ndim != 1 or x.shape != y.shape or not np.isfinite(y).all():
            raise ValueError("finite aligned IMU residual vectors required")
        xs.append(x)
        ys.append(y)
    x, y = np.concatenate(xs), np.concatenate(ys)
    bins, valid = _locate(x, partition)
    median, scale, counts, supported = [], [], [], []
    for j in range(len(partition["edges"]) + 1):
        values = y[valid & (bins == j)]
        enough = len(values) >= MIN_BIN
        median.append(float(np.median(values)) if enough else 0.0)
        scale.append(
            float(max(np.quantile(values, 0.75) - np.quantile(values, 0.25), SCALE_FLOOR))
            if enough
            else 1.0
        )
        counts.append(len(values))
        supported.append(enough)
    return {**partition, "median": median, "scale": scale, "counts": counts, "supported": supported}


def fit_quantile(context, imu, profile):
    x, y = np.asarray(context), np.asarray(imu)
    if x.shape != y.shape or not np.isfinite(y).all():
        raise ValueError("aligned finite normal calibration residuals required")
    bins, valid = _locate(x, profile)
    valid &= np.asarray(profile["supported"])[bins]
    if valid.sum() < MIN_BIN:
        raise ValueError("insufficient supported normal calibration endpoints")
    z = (y[valid] - np.asarray(profile["median"])[bins[valid]]) / np.asarray(profile["scale"])[
        bins[valid]
    ]
    return float(np.quantile(z, 0.95))


def boundaries(context, profile, quantile, original_threshold, cap=None):
    if not np.isfinite([quantile, original_threshold]).all() or (
        cap is not None and (not np.isfinite(cap) or cap <= 0)
    ):
        raise ValueError("finite thresholds and positive adjustment cap required")
    bins, valid = _locate(context, profile)
    valid &= np.asarray(profile["supported"])[bins]
    threshold = np.full(len(bins), float(original_threshold))
    learned = (
        np.asarray(profile["median"])[bins[valid]]
        + quantile * np.asarray(profile["scale"])[bins[valid]]
    )
    threshold[valid] = np.maximum(original_threshold, learned)
    if cap is not None:
        threshold = np.minimum(threshold, original_threshold + cap)
    return threshold


def decide(normalized_nodes, original_threshold, imu_boundaries):
    z, cut = np.asarray(normalized_nodes), np.asarray(imu_boundaries)
    if (
        z.ndim != 2
        or z.shape[1] != 3
        or cut.shape != (len(z),)
        or not np.isfinite(original_threshold)
        or not np.isfinite(z).all()
        or not np.isfinite(cut).all()
        or np.any(cut < original_threshold)
    ):
        raise ValueError("finite aligned residuals and non-weakened threshold floor required")
    return (z[:, :2] > original_threshold).any(axis=1) | (z[:, 2] > cut)
