"""Development diagnostics only: causal IMU/speed agreement, never policy output."""

import numpy as np

from coact_cps.vehicle.physics17 import physical_residual

FIT_ROUTES = ("civic13_07", "civic13_08")
LAGS = tuple(range(6))
HISTORY = 5


def causal_pairs(features, time_ms, start_ms):
    # Reuse the previously tested finite-value and bounded interval contract.
    physical_residual(features, time_ms, start_ms)
    x = np.asarray(features, dtype=float)
    t, s = np.asarray(time_ms), np.asarray(start_ms)
    speed = x[:, :2].mean(axis=1)
    positions, exposures, target = [], [], []
    for i in range(HISTORY + max(LAGS), len(x)):
        first = i - HISTORY - max(LAGS)
        if not np.array_equal(s[first + 1 : i + 1], t[first:i]):
            continue
        duration = (int(t[i]) - int(t[i - HISTORY])) / 1000
        target.append((speed[i] - speed[i - HISTORY]) / duration)
        row = []
        for lag in LAGS:
            lo, hi = i - HISTORY + 1 - lag, i + 1 - lag
            dt = (t[lo:hi] - s[lo:hi]).astype(float)
            row.append(float(np.sum(x[lo:hi, 2] * dt) / dt.sum()))
        positions.append(i)
        exposures.append(row)
    if len(positions) < 3:
        raise ValueError("insufficient contiguous diagnostic history")
    return {"positions": np.array(positions), "x": np.array(exposures), "y": np.array(target)}


def _validate_routes(routes, lag):
    if set(routes) != set(FIT_ROUTES) or type(lag) is not int or lag not in LAGS:
        raise ValueError("exact development fit routes and integer lag 0..5 required")
    for r in routes.values():
        x, y = np.asarray(r["x"]), np.asarray(r["y"])
        if (
            y.ndim != 1
            or len(y) < 3
            or x.shape != (len(y), len(LAGS))
            or not np.isfinite(x).all()
            or not np.isfinite(y).all()
        ):
            raise ValueError("finite paired diagnostic rows required")


def _fit(groups):
    design, target = [], []
    for x, y in groups:
        # Every route contributes total weight one, regardless of epoch count.
        weight = 1 / np.sqrt(len(y))
        design.append(np.column_stack((x, np.ones(len(y)))) * weight)
        target.append(y * weight)
    coef, _, rank, _ = np.linalg.lstsq(np.concatenate(design), np.concatenate(target), rcond=None)
    if rank != 2:
        raise ValueError("degenerate acceleration exposure")
    return float(coef[0]), float(coef[1])


def fit_routes(routes, lag):
    _validate_routes(routes, lag)
    return _fit([(routes[r]["x"][:, lag], routes[r]["y"]) for r in FIT_ROUTES])


def select_lag(routes):
    _validate_routes(routes, 0)
    table = []
    for lag in LAGS:
        errors = []
        for train, check in (FIT_ROUTES, FIT_ROUTES[::-1]):
            beta, offset = _fit([(routes[train]["x"][:, lag], routes[train]["y"])])
            error = metrics(routes[check]["y"], beta * routes[check]["x"][:, lag] + offset)
            errors.append({"fit": train, "check": check, "mae": error["mae"]})
        table.append(
            {
                "lag_epochs": lag,
                "mean_mae": float(np.mean([v["mae"] for v in errors])),
                "directional_mae": errors,
            }
        )
    # Stable first-minimum tie break favors lower lag.
    return min(table, key=lambda v: v["mean_mae"])["lag_epochs"], table


def metrics(target, prediction):
    y, p = np.asarray(target), np.asarray(prediction)
    if (
        y.ndim != 1
        or not len(y)
        or p.shape != y.shape
        or not np.isfinite(y).all()
        or not np.isfinite(p).all()
    ):
        raise ValueError("finite equal-length nonempty vectors required")
    error = p - y
    return {
        "count": len(y),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "absolute_error_p90": float(np.quantile(abs(error), 0.9)),
        "signed_error_mean": float(error.mean()),
    }
