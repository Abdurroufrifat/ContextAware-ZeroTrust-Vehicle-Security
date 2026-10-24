"""Three-component saved-output diagnostic; never a deployable detector."""

import math

import numpy as np

FACTORS = ("residuals", "normalization", "threshold")


def normalized(nodes, calibration):
    nodes = np.asarray(nodes, dtype=float)
    median = np.asarray(calibration["median"], dtype=float)
    scale = np.asarray(calibration["scale"], dtype=float)
    if (
        nodes.ndim != 2
        or nodes.shape[1] != 3
        or not len(nodes)
        or median.shape != (3,)
        or scale.shape != (3,)
        or np.any(scale <= 0)
        or not all(np.isfinite(v).all() for v in (nodes, median, scale))
    ):
        raise ValueError("finite three-channel residuals and positive scales required")
    return (nodes - median) / scale


def factorial_alarms(
    old_nodes, new_nodes, old_calibration, new_calibration, old_threshold, new_threshold
):
    if (
        np.shape(old_nodes) != np.shape(new_nodes)
        or not np.isfinite([old_threshold, new_threshold]).all()
    ):
        raise ValueError("aligned endpoints and finite thresholds required")
    result = {}
    for mask in range(8):
        nodes = new_nodes if mask & 1 else old_nodes
        calibration = new_calibration if mask & 2 else old_calibration
        threshold = new_threshold if mask & 4 else old_threshold
        result[mask] = normalized(nodes, calibration).max(axis=1) > threshold
    return result


def shapley(values):
    if set(values) != set(range(8)) or not np.isfinite(list(values.values())).all():
        raise ValueError("complete finite eight-state cube required")
    contributions = {}
    for i, name in enumerate(FACTORS):
        total = 0.0
        for subset in range(8):
            if subset & (1 << i):
                continue
            size = subset.bit_count()
            weight = math.factorial(size) * math.factorial(2 - size) / 6
            total += weight * (values[subset | (1 << i)] - values[subset])
        contributions[name] = total
    if not np.isclose(sum(contributions.values()), values[7] - values[0], rtol=0, atol=1e-12):
        raise ValueError("attribution does not reconstruct the metric change")
    return contributions
