import copy

import numpy as np
import pytest
import run_vehicle_false_alarm_stage15 as audit


def example():
    context = {
        "base_grant": np.array([True, True, True, False, True]),
        "labels": np.zeros(5, dtype=bool),
        "positions": np.array([2, 3, 4]),
    }
    block = {
        "positions": np.array([2, 3, 4]),
        "nodes": np.array([[2.0, 0, 0], [0, 3.0, 0], [1.0, 0, 0]]),
        "score": np.array([2.0, 3.0, 1.0]),
        "alarm": np.array([True, True, False]),
        "grant": np.array([False, False, False, False, True]),
    }
    spec = {
        "method": "coact_physics",
        "threshold": 1.0,
        "node_calibration": {"median": [0, 0, 0], "scale": [1, 1, 1]},
    }
    return context, block, spec


def test_accounting_separates_preexisting_denial_warmup_and_added_AI_denial():
    row = audit.decompose(*example())
    assert row["base_denied_epochs"] == 1
    assert row["unscored_base_granted_epochs"] == 2
    assert row["AI_added_denied_epochs"] == 1
    assert row["AI_alarm_on_base_denied_epochs"] == 1
    assert row["composed_granted_epochs"] == 1
    assert row["base_granted_epochs"] - row["composed_granted_epochs"] == 3
    assert row["alarm_rate_scored"] == 2 / 3
    assert row["exceedance_patterns"] == {"gnss": 1, "wheel": 1}


def test_threshold_equality_is_not_an_alarm():
    row = audit.decompose(*example())
    assert row["at_threshold_windows"] == 1
    assert row["alarm_windows"] == 2


@pytest.mark.parametrize("change", ["alarm", "score", "grant", "positions", "labels", "nonfinite"])
def test_inconsistent_saved_arrays_fail_closed(change):
    context, block, spec = copy.deepcopy(example())
    if change == "labels":
        context["labels"][2] = True
    elif change == "nonfinite":
        block["nodes"][0, 0] = np.nan
    elif change == "positions":
        block["positions"][1] = 2
    elif change == "score":
        block["score"][0] += 0.001
    else:
        block[change][0 if change == "alarm" else 4] ^= True
    with pytest.raises(ValueError):
        audit.decompose(context, block, spec)


def test_macro_average_counts_routes_equally_and_seeds_within_route():
    rows = [
        dict(
            phase="test",
            method="a",
            trip=trip,
            alarm_rate_scored=value,
            grant_rate_all_epochs=value,
            base_grant_rate_all_epochs=1.0,
            AI_added_denied_rate_all_epochs=0.0,
            unscored_base_granted_rate_all_epochs=0.0,
        )
        for trip, value in [("one", 0.0), ("one", 1.0), ("two", 0.0)]
    ]
    assert audit.summarize(rows)["a"]["alarm_rate_scored"] == 0.25


def test_catch_mean_and_gdn_causal_smoothing_are_reconstructed():
    nodes = np.array([[4.0, 0, 0], [0, 4.0, 0]])
    calibration = {"median": [0, 0, 0], "scale": [1, 1, 1]}
    assert np.array_equal(
        audit.score_channels(nodes, calibration, "catch_adapted")[1], np.array([4 / 3, 4 / 3])
    )
    assert np.array_equal(
        audit.score_channels(nodes, calibration, "gdn_adapted")[1], np.array([4.0, 2.0])
    )


def test_scalar_baselines_have_one_score_channel():
    calibration = {"median": [2.0], "scale": [0.5]}
    _, scores = audit.score_channels(np.array([[2.0], [3.0]]), calibration, "lof")
    assert np.array_equal(scores, np.array([0.0, 2.0]))
