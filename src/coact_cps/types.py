from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class CollaborationAction(str, Enum):
    EXPLAIN_SUMMARY = "explain_summary"
    SHOW_EVIDENCE = "show_evidence"
    SHOW_ALTERNATIVES = "show_alternatives"
    ASK_CONTEXT = "ask_context"
    WARN_UNCERTAINTY = "warn_uncertainty"
    ABSTAIN = "abstain"


@dataclass(slots=True)
class EvidenceItem:
    evidence_id: str
    node: str
    statement: str
    score: float
    value: float | None = None
    timestamp: str | None = None
    reliability: float = 1.0
    critical: bool = False


@dataclass(slots=True)
class EvidenceBundle:
    event_id: str
    alarm_probability: float
    predicted_cause: str
    cause_probabilities: dict[str, float]
    evidence: list[EvidenceItem]
    alternatives: list[str] = field(default_factory=list)
    uncertainty: float = 0.0
    process_context: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    process_grounding: dict[str, Any] = field(default_factory=dict)
    counterfactual: dict[str, Any] = field(default_factory=dict)
    permitted_actions: list[str] = field(
        default_factory=lambda: [
            "inspect supporting telemetry",
            "verify actuator command history",
            "escalate to the responsible operator",
        ]
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class GeneratedExplanation:
    summary: str
    probable_cause: str
    evidence_ids: list[str]
    alternatives: list[str]
    uncertainty: float
    operator_question: str
    recommendation: str
    raw_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class VerificationResult:
    status: str
    score: float
    reasons: list[str]
    missing_critical_evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CollaborationDecision:
    action: CollaborationAction
    reason: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["action"] = self.action.value
        return result
