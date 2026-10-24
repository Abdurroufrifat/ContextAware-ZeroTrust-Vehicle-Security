from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .records import CONDITIONS

BASE_FEATURES = (
    "alarm_probability",
    "explanation_uncertainty",
    "verification_score",
    "response_time_s",
    "diagnosis_correct",
    "unsafe_decision",
    "self_confidence",
    "reliance",
    "workload",
)


@dataclass(slots=True)
class OperatorSequenceArrays:
    interactions: np.ndarray
    padding_mask: np.ndarray
    diagnosis: np.ndarray
    states: np.ndarray
    participants: np.ndarray
    diagnosis_labels: list[str]


def _row_features(row: pd.Series) -> np.ndarray:
    numeric = np.asarray(
        [
            float(row["alarm_probability"]),
            float(row["explanation_uncertainty"]),
            float(row["verification_score"]),
            min(np.log1p(max(float(row["response_time_s"]), 0.0)) / np.log(121.0), 1.0),
            float(row["diagnosis_correct"]),
            float(row["unsafe_decision"]),
            float(row["self_confidence"]),
            float(row["reliance"]),
            float(row["workload"]),
        ],
        dtype=np.float32,
    )
    condition = np.zeros(len(CONDITIONS), dtype=np.float32)
    condition[CONDITIONS.index(str(row["condition"]))] = 1.0
    return np.concatenate([numeric, condition])


def build_operator_sequences(
    frame: pd.DataFrame, max_sequence_length: int = 16
) -> OperatorSequenceArrays:
    """Use prior trials to predict the operator state on the next trial."""

    required = {
        "participant_id",
        "trial_index",
        "condition",
        "diagnosis_label",
        *BASE_FEATURES,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"Missing human-study columns: {missing}")
    invalid_conditions = sorted(set(frame["condition"].astype(str)) - set(CONDITIONS))
    if invalid_conditions:
        raise ValueError(f"Unknown study conditions: {invalid_conditions}")
    labels = sorted(frame["diagnosis_label"].astype(str).unique().tolist())
    label_index = {label: index for index, label in enumerate(labels)}
    feature_size = len(BASE_FEATURES) + len(CONDITIONS)
    sequences: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    diagnoses: list[int] = []
    states: list[np.ndarray] = []
    participants: list[str] = []

    for participant, group in frame.groupby("participant_id", sort=False):
        ordered = group.sort_values("trial_index").reset_index(drop=True)
        for target_index in range(1, len(ordered)):
            history = ordered.iloc[max(0, target_index - max_sequence_length) : target_index]
            values = np.stack([_row_features(row) for _, row in history.iterrows()])
            padded = np.zeros((max_sequence_length, feature_size), dtype=np.float32)
            mask = np.ones(max_sequence_length, dtype=bool)
            padded[: len(values)] = values
            mask[: len(values)] = False
            target = ordered.iloc[target_index]
            sequences.append(padded)
            masks.append(mask)
            diagnoses.append(label_index[str(target["diagnosis_label"])])
            states.append(
                np.asarray(
                    [target["self_confidence"], target["reliance"], target["workload"]],
                    dtype=np.float32,
                )
            )
            participants.append(str(participant))
    if not sequences:
        raise ValueError("At least one participant with two completed trials is required")
    return OperatorSequenceArrays(
        interactions=np.stack(sequences),
        padding_mask=np.stack(masks),
        diagnosis=np.asarray(diagnoses, dtype=np.int64),
        states=np.stack(states),
        participants=np.asarray(participants),
        diagnosis_labels=labels,
    )
