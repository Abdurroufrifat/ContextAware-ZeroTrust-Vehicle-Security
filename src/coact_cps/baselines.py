from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import IsolationForest

from .data.windowing import WindowArrays
from .metrics import attack_episode_metrics, binary_detection_metrics, root_cause_hit_at_k


@dataclass(slots=True)
class CalibratedScores:
    probabilities: np.ndarray
    threshold: float
    temperature: float


def calibrate_scores(
    validation_scores: np.ndarray,
    test_scores: np.ndarray,
    validation_labels: np.ndarray,
    quantile: float = 0.995,
) -> CalibratedScores:
    normal = np.asarray(validation_labels) == 0
    if not normal.any():
        raise ValueError("Calibration requires validation-normal samples")
    reference = np.asarray(validation_scores, dtype=float)[normal]
    threshold = float(np.quantile(reference, quantile))
    lower = float(np.quantile(reference, 0.95))
    temperature = max(threshold - lower, float(np.std(reference)) * 0.25, 1e-8)
    logits = np.clip((np.asarray(test_scores) - threshold) / temperature, -30.0, 30.0)
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    return CalibratedScores(probabilities, threshold, temperature)


def persistence_scores(arrays: WindowArrays) -> tuple[np.ndarray, np.ndarray]:
    residual = np.mean(np.abs(arrays.x[:, -1] - arrays.x[:, -2]), axis=-1)
    top_count = max(1, int(round(residual.shape[1] * 0.10)))
    graph = np.sort(residual, axis=1)[:, -top_count:].mean(axis=1)
    scale = np.maximum(np.quantile(residual, 0.75), 1e-6)
    root_probabilities = np.exp(np.clip(residual / scale, -30.0, 30.0))
    root_probabilities /= root_probabilities.sum(axis=1, keepdims=True)
    return graph, root_probabilities


def summary_features(arrays: WindowArrays) -> np.ndarray:
    values = arrays.x[..., 0]
    components = (
        values[:, -1],
        values.mean(axis=1),
        values.std(axis=1),
        values[:, -1] - values[:, 0],
    )
    return np.concatenate(components, axis=1).astype(np.float32)


def evaluate_persistence(
    validation: WindowArrays, test: WindowArrays, quantile: float = 0.995
) -> dict[str, float]:
    validation_scores, _ = persistence_scores(validation)
    test_scores, root_probabilities = persistence_scores(test)
    calibrated = calibrate_scores(validation_scores, test_scores, validation.y, quantile=quantile)
    metrics = binary_detection_metrics(test.y, calibrated.probabilities)
    metrics.update(attack_episode_metrics(test.y, calibrated.probabilities))
    metrics["root_hit_at_1"] = root_cause_hit_at_k(test.root, root_probabilities, 1)
    metrics["root_hit_at_3"] = root_cause_hit_at_k(
        test.root, root_probabilities, min(3, root_probabilities.shape[1])
    )
    metrics["calibration_threshold"] = calibrated.threshold
    return metrics


def evaluate_isolation_forest(
    train: WindowArrays,
    validation: WindowArrays,
    test: WindowArrays,
    quantile: float = 0.995,
    seed: int = 7,
) -> dict[str, float]:
    normal_train = train.y == 0
    if not normal_train.any():
        raise ValueError("Isolation Forest requires normal training windows")
    model = IsolationForest(
        n_estimators=200,
        max_samples="auto",
        contamination="auto",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(summary_features(train)[normal_train])
    validation_scores = -model.decision_function(summary_features(validation))
    test_scores = -model.decision_function(summary_features(test))
    calibrated = calibrate_scores(validation_scores, test_scores, validation.y, quantile=quantile)
    metrics = binary_detection_metrics(test.y, calibrated.probabilities)
    metrics.update(attack_episode_metrics(test.y, calibrated.probabilities))
    metrics["calibration_threshold"] = calibrated.threshold
    return metrics
