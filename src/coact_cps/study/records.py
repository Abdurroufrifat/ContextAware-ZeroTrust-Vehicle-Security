from __future__ import annotations

import csv
import re
import secrets
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

CONDITIONS = (
    "no_ai",
    "alarm_only",
    "generic_xai",
    "verified_static",
    "coadaptive_verified",
)

COLLABORATION_ACTIONS = (
    "explain_summary",
    "show_evidence",
    "show_alternatives",
    "ask_context",
    "warn_uncertainty",
    "abstain",
)


def new_participant_code() -> str:
    """Return a random pseudonymous code; it is not derived from personal data."""

    return f"P-{secrets.token_hex(4).upper()}"


def _safe_text(value: str, maximum: int = 160) -> str:
    cleaned = re.sub(r"[\r\n\t]+", " ", str(value)).strip()[:maximum]
    if cleaned.startswith(("=", "+", "-", "@")):
        cleaned = "'" + cleaned
    return cleaned


@dataclass(slots=True)
class StudyTrial:
    participant_id: str
    trial_index: int
    scenario_id: str
    condition: str
    alarm_probability: float
    explanation_uncertainty: float
    verification_score: float
    verification_status: str
    evidence_count: int
    alternative_count: int
    collaboration_action: str
    diagnosis_label: str
    diagnosis_correct: int
    chosen_response: str
    unsafe_decision: int
    response_time_s: float
    self_confidence: float
    reliance: float
    workload: float
    behavior_propensity: float = 1.0
    outcome_reward: float = 0.0
    approval_reference: str = "DEVELOPMENT_ONLY"
    recorded_at_utc: str = ""

    def validate(self) -> None:
        if self.condition not in CONDITIONS:
            raise ValueError(f"Unknown study condition: {self.condition}")
        if self.collaboration_action not in COLLABORATION_ACTIONS:
            raise ValueError(f"Unknown collaboration action: {self.collaboration_action}")
        for name in (
            "alarm_probability",
            "explanation_uncertainty",
            "verification_score",
            "self_confidence",
            "reliance",
            "workload",
            "behavior_propensity",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.response_time_s < 0:
            raise ValueError("response_time_s must be non-negative")
        if self.diagnosis_correct not in (0, 1) or self.unsafe_decision not in (0, 1):
            raise ValueError("Binary study outcomes must be 0 or 1")
        if not self.participant_id.startswith("P-"):
            raise ValueError("Use a pseudonymous P- participant code; never store names or email")

    def to_row(self) -> dict[str, object]:
        self.validate()
        row = asdict(self)
        row["participant_id"] = _safe_text(self.participant_id, 32)
        row["scenario_id"] = _safe_text(self.scenario_id, 64)
        row["diagnosis_label"] = _safe_text(self.diagnosis_label)
        row["chosen_response"] = _safe_text(self.chosen_response)
        row["approval_reference"] = _safe_text(self.approval_reference, 80)
        row["recorded_at_utc"] = self.recorded_at_utc or datetime.now(timezone.utc).isoformat()
        return row


def append_trial(path: str | Path, trial: StudyTrial) -> None:
    """Append one de-identified trial to a local CSV with a stable schema."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    row = trial.to_row()
    exists = destination.exists() and destination.stat().st_size > 0
    with destination.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
