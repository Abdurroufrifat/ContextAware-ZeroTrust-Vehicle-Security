"""Exploratory causal consistency residual; no deployment or test eligibility."""

import numpy as np

STATUS = "DEVELOPMENT_ONLY_NOT_PROMOTED"
HISTORY = 5
MAX_INTERVAL_MS = 500


def physical_residual(features, time_ms, start_ms):
    """Absolute speed increment minus right-end acceleration integral / duration.

    Features are GNSS speed, wheel speed (m/s), acceleration (m/s²).
    Insufficient or noncontiguous history returns infinity, never a clean score.
    Constant speed offsets and physically consistent spoofing can be invisible.
    """
    x = np.asarray(features, dtype=float)
    t, s = np.asarray(time_ms), np.asarray(start_ms)
    if (
        x.ndim != 2
        or x.shape[1] != 3
        or t.shape != (len(x),)
        or s.shape != t.shape
        or t.dtype.kind not in "iu"
        or s.dtype.kind not in "iu"
        or not np.isfinite(x).all()
    ):
        raise ValueError("finite N by 3 features and integer timestamp vectors required")
    # Cast avoids unsigned underflow before checking differences.
    t, s = t.astype(np.int64), s.astype(np.int64)
    dt = t - s
    if np.any(dt <= 0) or np.any(dt > MAX_INTERVAL_MS) or np.any(np.diff(t) <= 0):
        raise ValueError("invalid time interval")
    result = np.full(len(x), np.inf)
    speed = x[:, :2].mean(axis=1)
    for i in range(HISTORY, len(x)):
        first = i - HISTORY
        if not np.array_equal(s[first + 1 : i + 1], t[first:i]):
            continue
        duration = (int(t[i]) - int(t[first])) / 1000
        integral = np.sum(x[first + 1 : i + 1, 2] * dt[first + 1 : i + 1] / 1000)
        result[i] = abs(speed[i] - speed[first] - integral) / duration
    return result


def fit_physical_scale(clean_routes):
    if set(clean_routes) != {"civic13_07", "civic13_08"}:
        raise ValueError("exact two development scale routes required")
    values = []
    for residual in clean_routes.values():
        a = np.asarray(residual, dtype=float)
        if a.ndim != 1 or np.isnan(a).any() or np.any(a < 0):
            raise ValueError("invalid residual")
        finite = a[np.isfinite(a)]
        if len(finite) < 3:
            raise ValueError("too little finite history")
        values.append(finite)
    pooled = np.concatenate(values)
    q25, med, q75 = np.quantile(pooled, [0.25, 0.5, 0.75])
    return float(med), float(max(q75 - q25, 0.01))


def hybrid_score(nodes, median, scale, physical, physical_median, physical_scale):
    nodes, median, scale, physical = map(np.asarray, (nodes, median, scale, physical))
    if (
        nodes.ndim != 2
        or nodes.shape[1] != 3
        or median.shape != (3,)
        or scale.shape != (3,)
        or physical.shape != (len(nodes),)
        or not all(np.isfinite(a).all() for a in (nodes, median, scale))
        or np.any(scale <= 0)
        or np.isnan(physical).any()
        or np.any(physical < 0)
        or not np.isfinite([physical_median, physical_scale]).all()
        or physical_scale <= 0
        or physical_median < 0
    ):
        raise ValueError("invalid hybrid inputs")
    model = ((nodes[:, :2] - median[:2]) / scale[:2]).max(axis=1)
    return np.maximum(model, (physical - physical_median) / physical_scale)


def threshold_from_clean(scores, original):
    scores = np.asarray(scores)
    if (
        scores.ndim != 1
        or not len(scores)
        or not np.isfinite(scores).all()
        or not np.isfinite(original)
    ):
        raise ValueError("finite clean scores required")
    return float(max(original, np.quantile(scores, 0.95)))
