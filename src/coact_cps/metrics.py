from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_detection_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities).astype(float)
    predictions = (probabilities >= 0.5).astype(int)
    results = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "brier": float(brier_score_loss(y_true, probabilities)),
        "ece": expected_calibration_error(y_true, probabilities),
    }
    if len(np.unique(y_true)) > 1:
        results["roc_auc"] = float(roc_auc_score(y_true, probabilities))
        results["average_precision"] = float(average_precision_score(y_true, probabilities))
    else:
        results["roc_auc"] = float("nan")
        results["average_precision"] = float("nan")
    return results


def expected_calibration_error(
    y_true: np.ndarray, probabilities: np.ndarray, bins: int = 10
) -> float:
    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities).astype(float)
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for index in range(bins):
        if index == bins - 1:
            selected = (probabilities >= boundaries[index]) & (
                probabilities <= boundaries[index + 1]
            )
        else:
            selected = (probabilities >= boundaries[index]) & (
                probabilities < boundaries[index + 1]
            )
        if selected.any():
            error += float(selected.mean()) * abs(
                float(y_true[selected].mean()) - float(probabilities[selected].mean())
            )
    return float(error)


def _positive_runs(values: np.ndarray) -> list[tuple[int, int]]:
    binary = np.asarray(values).astype(bool)
    changes = np.diff(np.pad(binary.astype(int), (1, 1)))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def attack_episode_metrics(
    y_true: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Episode recall and detection delay; add official eTaPR for HAI reporting."""

    y_true = np.asarray(y_true).astype(int)
    predictions = np.asarray(probabilities) >= threshold
    episodes = _positive_runs(y_true)
    detected = 0
    delays: list[int] = []
    for start, end in episodes:
        hits = np.where(predictions[start:end])[0]
        if len(hits):
            detected += 1
            delays.append(int(hits[0]))
    normal = y_true == 0
    false_alarm_runs = _positive_runs(predictions & normal)
    return {
        "attack_episode_recall": float(detected / len(episodes)) if episodes else float("nan"),
        "mean_detection_delay_windows": float(np.mean(delays)) if delays else float("nan"),
        "false_alarm_window_rate": float(predictions[normal].mean())
        if normal.any()
        else float("nan"),
        "false_alarm_episode_count": float(len(false_alarm_runs)),
    }


def root_cause_hit_at_k(
    true_roots: np.ndarray, root_probabilities: np.ndarray, k: int = 1
) -> float:
    true_roots = np.asarray(true_roots).astype(int)
    root_probabilities = np.asarray(root_probabilities)
    valid = true_roots >= 0
    if not valid.any():
        return float("nan")
    top = np.argsort(root_probabilities[valid], axis=1)[:, -k:]
    return float(
        np.mean(
            [root in candidates for root, candidates in zip(true_roots[valid], top, strict=True)]
        )
    )
