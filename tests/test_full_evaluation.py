import numpy as np
import pytest

from coact_cps.full_evaluation import (
    grouped_episode_metrics,
    grouped_positive_runs,
    target_hit_at_k,
    validate_regular_windows,
)


def test_grouped_runs_do_not_join_files() -> None:
    values = np.asarray([0, 1, 1, 0, 1, 1])
    groups = np.asarray([4, 4, 4, 5, 5, 5])
    episodes = grouped_positive_runs(values, groups)
    assert [(item.group, item.start, item.end) for item in episodes] == [
        (4, 1, 3),
        (5, 4, 6),
    ]


def test_regular_window_validation_rejects_subsampled_data() -> None:
    assert validate_regular_windows(np.asarray([4, 4, 5, 5]), np.asarray([63, 127, 63, 127])) == 64
    with pytest.raises(ValueError, match="subsampled or irregular"):
        validate_regular_windows(np.asarray([4, 4, 4]), np.asarray([63, 127, 255]))


def test_grouped_episode_metrics_count_detection_and_false_alarm_runs() -> None:
    truth = np.asarray([0, 1, 1, 0, 0, 1, 1, 0])
    probability = np.asarray([0.8, 0.2, 0.7, 0.0, 0.9, 0.1, 0.8, 0.0])
    groups = np.asarray([4, 4, 4, 4, 5, 5, 5, 5])
    metrics, _ = grouped_episode_metrics(truth, probability, groups)
    assert metrics["attack_episode_count"] == 2
    assert metrics["attack_episode_recall"] == 1
    assert metrics["false_alarm_episode_count"] == 2


def test_target_hit_reports_all_and_detected_denominators() -> None:
    scores = np.asarray([[0.9, 0.1, 0.0], [0.1, 0.8, 0.2], [0.1, 0.2, 0.7]])
    targets = [(0,), (1,), ()]
    metrics = target_hit_at_k(scores, targets, np.asarray([True, False, True]), 1)
    assert metrics["target_hit_at_1_all_evaluable"] == 1
    assert metrics["target_hit_at_1_detected_evaluable"] == 1
    assert metrics["evaluable_target_episodes"] == 2
    assert metrics["detected_evaluable_target_episodes"] == 1
