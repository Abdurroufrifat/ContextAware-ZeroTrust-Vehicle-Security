import itertools

import numpy as np
import pytest

from coact_cps.vehicle.loss_attribution20 import FACTORS, factorial_alarms, shapley

CAL = {"median": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0]}


def test_all_eight_combinations_and_strict_threshold():
    old = np.array([[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    new = np.zeros((2, 3))
    alarms = factorial_alarms(old, new, CAL, CAL, 2.0, 4.0)
    assert set(alarms) == set(range(8))
    np.testing.assert_array_equal(alarms[0], [False, True])
    assert not alarms[1].any()
    assert not alarms[4].any()


def test_normalizer_can_change_alarm_without_prediction_change():
    nodes = np.array([[3.0, 0.0, 0.0]])
    cal = {"median": [0.0] * 3, "scale": [2.0] * 3}
    alarms = factorial_alarms(nodes, nodes, CAL, cal, 2.0, 2.0)
    assert alarms[0][0]
    assert not alarms[2][0]


def test_shapley_recovers_additive_component_effects():
    values = {
        i: 0.2 + sum([0.1, -0.15, 0.03][j] for j in range(3) if i & (1 << j)) for i in range(8)
    }
    np.testing.assert_allclose(list(shapley(values).values()), [0.1, -0.15, 0.03], atol=1e-15)


def test_interaction_is_shared_and_sum_matches_total():
    result = shapley({i: float(i == 7) for i in range(8)})
    np.testing.assert_allclose(list(result.values()), [1 / 3] * 3)
    assert sum(result.values()) == 1


def test_matches_independent_permutation_definition():
    values = dict(enumerate([0.2, 0.4, 0.1, 0.7, 0.3, 0.1, 0.6, 0.5]))
    expected = np.zeros(3)
    for order in itertools.permutations(range(3)):
        mask = 0
        for factor in order:
            after = mask | (1 << factor)
            expected[factor] += (values[after] - values[mask]) / 6
            mask = after
    np.testing.assert_allclose(list(shapley(values).values()), expected, atol=1e-15)


def test_incomplete_or_nonfinite_cube_rejected():
    for values in [{0: 1}, {i: np.nan for i in range(8)}]:
        with pytest.raises(ValueError):
            shapley(values)


def test_invalid_score_inputs_rejected():
    nodes = np.ones((3, 3))
    for a, b, cal, th in [
        (nodes, nodes[:2], CAL, 1.0),
        (nodes * np.nan, nodes, CAL, 1.0),
        (nodes, nodes, {"median": [0.0] * 3, "scale": [0.0] * 3}, 1.0),
        (nodes, nodes, CAL, np.inf),
    ]:
        with pytest.raises(ValueError):
            factorial_alarms(a, b, cal, CAL, th, 1.0)


def test_factor_names_describe_saved_output_components():
    assert FACTORS == ("residuals", "normalization", "threshold")
