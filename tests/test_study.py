from __future__ import annotations

import pandas as pd

from coact_cps.study.records import StudyTrial, append_trial, new_participant_code
from coact_cps.study.sequence_dataset import build_operator_sequences


def _trial(participant: str, trial_index: int) -> StudyTrial:
    return StudyTrial(
        participant_id=participant,
        trial_index=trial_index,
        scenario_id=f"S-{trial_index}",
        condition="coadaptive_verified",
        alarm_probability=0.8,
        explanation_uncertainty=0.3,
        verification_score=0.9,
        verification_status="VERIFIED",
        evidence_count=3,
        alternative_count=2,
        collaboration_action="show_alternatives",
        diagnosis_label="sensor_bias",
        diagnosis_correct=1,
        chosen_response="Inspect telemetry",
        unsafe_decision=0,
        response_time_s=12.0,
        self_confidence=0.7,
        reliance=0.6,
        workload=0.4,
    )


def test_study_log_and_sequence_builder(tmp_path) -> None:
    participant = new_participant_code()
    destination = tmp_path / "trials.csv"
    append_trial(destination, _trial(participant, 1))
    append_trial(destination, _trial(participant, 2))
    frame = pd.read_csv(destination)
    arrays = build_operator_sequences(frame, max_sequence_length=4)
    assert arrays.interactions.shape == (1, 4, 14)
    assert arrays.padding_mask.shape == (1, 4)
    assert arrays.participants[0] == participant
