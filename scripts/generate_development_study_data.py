from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from coact_cps.study.records import CONDITIONS, StudyTrial, append_trial, new_participant_code

DIAGNOSES = ("sensor_bias", "actuator_forcing", "replay", "normal_operation")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate simulated interaction logs for software testing only"
    )
    parser.add_argument("--output", default="data/study/development_simulated.csv")
    parser.add_argument("--participants", type=int, default=12)
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    destination = Path(args.output)
    if destination.exists() and not args.overwrite:
        raise SystemExit(f"{destination} exists; use --overwrite to replace development data")
    if destination.exists():
        destination.unlink()
    rng = np.random.default_rng(args.seed)

    for _ in range(args.participants):
        participant = new_participant_code()
        reliance_state = float(rng.uniform(0.35, 0.65))
        for trial_index in range(1, args.trials + 1):
            condition = str(rng.choice(CONDITIONS))
            uncertainty = float(rng.uniform(0.05, 0.55))
            status = str(rng.choice(("VERIFIED", "WARNING", "ABSTAIN"), p=(0.7, 0.2, 0.1)))
            if status == "ABSTAIN":
                action = "abstain"
            elif status == "WARNING":
                action = "show_evidence"
            elif uncertainty >= 0.25:
                action = "show_alternatives"
            else:
                action = "explain_summary"
            true_diagnosis = str(rng.choice(DIAGNOSES))
            support_gain = {
                "no_ai": 0.0,
                "alarm_only": 0.03,
                "generic_xai": 0.07,
                "verified_static": 0.12,
                "coadaptive_verified": 0.16,
            }[condition]
            correct = int(rng.random() < min(0.90, 0.52 + support_gain - 0.15 * uncertainty))
            if correct:
                diagnosis = true_diagnosis
            else:
                diagnosis = str(rng.choice([item for item in DIAGNOSES if item != true_diagnosis]))
            unsafe_probability = max(0.01, 0.13 - support_gain + 0.10 * uncertainty)
            unsafe = int(rng.random() < unsafe_probability)
            response_time = float(rng.lognormal(mean=2.7 + 0.4 * uncertainty, sigma=0.25))
            confidence = float(
                np.clip(0.45 + 0.35 * correct - 0.25 * uncertainty + rng.normal(0, 0.08), 0, 1)
            )
            reliance_state = float(
                np.clip(
                    0.75 * reliance_state + 0.25 * (0.45 + support_gain) + rng.normal(0, 0.04), 0, 1
                )
            )
            workload = float(np.clip(0.45 + 0.35 * uncertainty + rng.normal(0, 0.08), 0, 1))
            reward = float(correct - 1.5 * unsafe - 0.002 * min(response_time, 300.0))
            append_trial(
                destination,
                StudyTrial(
                    participant_id=participant,
                    trial_index=trial_index,
                    scenario_id=f"SIM-{trial_index:03d}",
                    condition=condition,
                    alarm_probability=float(rng.uniform(0.55, 0.98)),
                    explanation_uncertainty=uncertainty,
                    verification_score={"VERIFIED": 0.92, "WARNING": 0.68, "ABSTAIN": 0.35}[status],
                    verification_status=status,
                    evidence_count=int(rng.integers(2, 6)),
                    alternative_count=int(uncertainty >= 0.25) * 2,
                    collaboration_action=action,
                    diagnosis_label=diagnosis,
                    diagnosis_correct=correct,
                    chosen_response=(
                        "Immediately change actuator state" if unsafe else "Inspect telemetry"
                    ),
                    unsafe_decision=unsafe,
                    response_time_s=response_time,
                    self_confidence=confidence,
                    reliance=reliance_state,
                    workload=workload,
                    behavior_propensity=1.0 / 6.0,
                    outcome_reward=reward,
                    approval_reference="DEVELOPMENT_ONLY",
                ),
            )
    print(f"Wrote simulated development records to {destination}")
    print("Never combine these rows with human-study data or use them for publication claims.")


if __name__ == "__main__":
    main()
