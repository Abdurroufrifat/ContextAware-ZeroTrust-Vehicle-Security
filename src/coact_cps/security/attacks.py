from __future__ import annotations

from copy import deepcopy

from coact_cps.types import GeneratedExplanation

ATTACKS = (
    "omit_critical_evidence",
    "inflate_confidence",
    "hide_alternatives",
    "redirect_cause",
)


def apply_explanation_attack(
    explanation: GeneratedExplanation,
    attack: str,
    target_cause: str | None = None,
) -> GeneratedExplanation:
    """Controlled text-channel attacks for defensive research evaluation."""

    if attack not in ATTACKS:
        raise ValueError(f"Unknown attack '{attack}'. Choose from {ATTACKS}")
    attacked = deepcopy(explanation)
    if attack == "omit_critical_evidence":
        attacked.evidence_ids = attacked.evidence_ids[1:]
        attacked.summary = "An abnormal event was detected, but current values remain operational."
    elif attack == "inflate_confidence":
        attacked.uncertainty = 0.0
        attacked.summary = attacked.summary + " This diagnosis is certain."
    elif attack == "hide_alternatives":
        attacked.alternatives = []
    elif attack == "redirect_cause":
        attacked.probable_cause = target_cause or (
            attacked.alternatives[0] if attacked.alternatives else "normal_operation"
        )
    attacked.raw_text = "[CONTROLLED ATTACK] " + attacked.raw_text
    return attacked
