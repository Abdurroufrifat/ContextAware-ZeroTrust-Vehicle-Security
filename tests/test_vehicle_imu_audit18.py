import numpy as np
import pytest

from coact_cps.vehicle.imu_audit18 import causal_pairs, fit_routes, metrics, select_lag


def trace(dt=None):
    dt = np.full(40, 200) if dt is None else np.asarray(dt)
    t = np.cumsum(dt).astype(np.int64)
    s = t - dt
    a = np.sin(np.arange(len(t)) / 4)
    v = np.cumsum(a * dt / 1000)
    return np.column_stack((v, v, a)), t, s


def routes():
    rng = np.random.default_rng(1)
    result = {}
    for r in ("civic13_07", "civic13_08"):
        x = rng.normal(size=(60, 6))
        result[r] = {"x": x, "y": 2 * x[:, 0] - 0.5}
    return result


def test_interval_weighted_identity_kinematics():
    x, t, s = trace(np.tile([100, 200, 300, 190, 210], 8))
    pairs = causal_pairs(x, t, s)
    np.testing.assert_allclose(pairs["y"], pairs["x"][:, 0], atol=1e-14)
    assert pairs["positions"][0] == 10


def test_no_future_samples_used():
    x, t, s = trace()
    before = causal_pairs(x, t, s)
    x[30:] += 10
    after = causal_pairs(x, t, s)
    mask = before["positions"] < 30
    np.testing.assert_array_equal(before["x"][mask], after["x"][mask])
    np.testing.assert_array_equal(before["y"][mask], after["y"][mask])


def test_gap_excludes_entire_comparison_history():
    x, t, s = trace()
    t[20:] += 20
    s[20:] += 20
    pairs = causal_pairs(x, t, s)
    assert not np.isin(pairs["positions"], np.arange(20, 30)).any()
    assert 30 in pairs["positions"]


def test_known_affine_relation():
    beta, offset = fit_routes(routes(), 0)
    np.testing.assert_allclose([beta, offset], [2, -0.5], atol=1e-14)


def test_equal_route_weights_despite_unequal_lengths():
    r = {}
    for key, repeats, bias in [("civic13_07", 2, 0), ("civic13_08", 100, 2)]:
        x = np.tile([-1.0, 1.0], repeats)
        r[key] = {"x": np.tile(x[:, None], (1, 6)), "y": x + bias}
    np.testing.assert_allclose(fit_routes(r, 0), [1, 1], atol=1e-14)


def test_cross_route_lag_selection_uses_both_directions():
    lag, table = select_lag(routes())
    assert lag == 0
    assert len(table) == 6
    assert len(table[0]["directional_mae"]) == 2
    assert table[0]["mean_mae"] < 1e-12


def test_invalid_fit_route_rejected():
    r = routes()
    r["civic13_09"] = r.pop("civic13_08")
    with pytest.raises(ValueError):
        fit_routes(r, 0)


def test_degenerate_imu_rejected():
    r = routes()
    for value in r.values():
        value["x"][:] = 1
    with pytest.raises(ValueError, match="degenerate"):
        fit_routes(r, 0)


def test_nonfinite_fit_rejected():
    r = routes()
    r["civic13_07"]["x"][0, 0] = np.nan
    with pytest.raises(ValueError):
        fit_routes(r, 0)


def test_invalid_measurements_rejected():
    x, t, s = trace()
    with pytest.raises(ValueError):
        causal_pairs(x[:-1], t, s)


def test_metrics_compare_against_zero_baseline():
    result = metrics(np.array([0.0, 2.0]), np.array([1.0, 1.0]))
    assert result["mae"] == 1
    assert result["rmse"] == 1
    assert result["count"] == 2
    with pytest.raises(ValueError):
        metrics(np.array([np.nan]), np.array([0.0]))


def test_invalid_lag_rejected():
    for lag in [True, -1, 6]:
        with pytest.raises(ValueError):
            fit_routes(routes(), lag)
