from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Episode:
    group: int
    start: int
    end: int


def grouped_positive_runs(values: np.ndarray, groups: np.ndarray) -> list[Episode]:
    values = np.asarray(values).astype(bool)
    groups = np.asarray(groups)
    if values.ndim != 1 or groups.ndim != 1 or len(values) != len(groups):
        raise ValueError("values and groups must be equal-length one-dimensional arrays")

    episodes: list[Episode] = []
    seen: set[int] = set()
    for raw_group in groups:
        group = int(raw_group)
        if group in seen:
            continue
        seen.add(group)
        indices = np.flatnonzero(groups == raw_group)
        if len(indices) and not np.array_equal(indices, np.arange(indices[0], indices[-1] + 1)):
            raise ValueError(f"Group {group} is not stored contiguously")
        local = values[indices]
        changes = np.diff(np.pad(local.astype(int), (1, 1)))
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]
        episodes.extend(
            Episode(group=group, start=int(indices[0] + start), end=int(indices[0] + end))
            for start, end in zip(starts, ends, strict=True)
        )
    return episodes


def validate_regular_windows(groups: np.ndarray, end_indices: np.ndarray) -> int:
    groups = np.asarray(groups)
    end_indices = np.asarray(end_indices)
    if groups.ndim != 1 or end_indices.ndim != 1 or len(groups) != len(end_indices):
        raise ValueError("groups and end_indices must be equal-length one-dimensional arrays")

    steps = []
    for group in dict.fromkeys(int(value) for value in groups):
        indices = np.flatnonzero(groups == group)
        differences = np.diff(end_indices[indices])
        if len(differences):
            if (differences <= 0).any():
                raise ValueError(f"Group {group} has non-increasing window end indices")
            steps.extend(int(value) for value in differences)
    if not steps:
        raise ValueError("At least two windows are required")
    stride = min(steps)
    if any(step != stride for step in steps):
        raise ValueError(
            "Windows are subsampled or irregular; episode metrics would be invalid. "
            "Use the complete chronological test.npz file."
        )
    return stride


def grouped_episode_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    groups: np.ndarray,
    threshold: float = 0.5,
) -> tuple[dict[str, float], list[Episode]]:
    y_true = np.asarray(y_true).astype(bool)
    predictions = np.asarray(probabilities) >= threshold
    groups = np.asarray(groups)
    episodes = grouped_positive_runs(y_true, groups)

    detected = 0
    delays = []
    for episode in episodes:
        hits = np.flatnonzero(predictions[episode.start : episode.end])
        if len(hits):
            detected += 1
            delays.append(int(hits[0]))

    false_alarm_episodes = 0
    normal = ~y_true
    for group in dict.fromkeys(int(value) for value in groups):
        indices = np.flatnonzero(groups == group)
        false_alarm_episodes += len(
            grouped_positive_runs(
                predictions[indices] & normal[indices], np.zeros(len(indices))
            )
        )

    return (
        {
            "attack_episode_count": float(len(episodes)),
            "attack_episode_recall": float(detected / len(episodes)) if episodes else float("nan"),
            "detected_attack_episodes": float(detected),
            "mean_detection_delay_windows": float(np.mean(delays)) if delays else float("nan"),
            "median_detection_delay_windows": float(np.median(delays))
            if delays
            else float("nan"),
            "false_alarm_window_rate": float(predictions[normal].mean())
            if normal.any()
            else float("nan"),
            "false_alarm_episode_count": float(false_alarm_episodes),
        },
        episodes,
    )


def target_hit_at_k(
    episode_scores: np.ndarray,
    target_indices: list[tuple[int, ...]],
    detected: np.ndarray,
    k: int,
) -> dict[str, float]:
    episode_scores = np.asarray(episode_scores)
    detected = np.asarray(detected).astype(bool)
    if episode_scores.ndim != 2 or len(episode_scores) != len(target_indices):
        raise ValueError("Episode scores and target lists do not align")
    if len(detected) != len(target_indices):
        raise ValueError("Detection mask and target lists do not align")

    evaluable = np.asarray([bool(targets) for targets in target_indices])
    top = np.argsort(episode_scores, axis=1)[:, -min(k, episode_scores.shape[1]) :]
    hits = np.asarray(
        [
            bool(set(targets).intersection(candidates))
            for targets, candidates in zip(target_indices, top, strict=True)
        ]
    )
    detected_evaluable = evaluable & detected
    return {
        f"target_hit_at_{k}_all_evaluable": float(hits[evaluable].mean())
        if evaluable.any()
        else float("nan"),
        f"target_hit_at_{k}_detected_evaluable": float(hits[detected_evaluable].mean())
        if detected_evaluable.any()
        else float("nan"),
        "evaluable_target_episodes": float(evaluable.sum()),
        "detected_evaluable_target_episodes": float(detected_evaluable.sum()),
    }
