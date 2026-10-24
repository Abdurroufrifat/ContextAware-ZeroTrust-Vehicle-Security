from __future__ import annotations

from dataclasses import asdict
from typing import Any

from coact_cps.types import EvidenceBundle, GeneratedExplanation, VerificationResult


def build_operator_view(
    bundle: EvidenceBundle,
    explanation: GeneratedExplanation,
    verification: VerificationResult,
) -> dict[str, Any]:
    """Build the only payload intended for an operator-facing interface.

    Generator ``raw_text`` is deliberately excluded. When verification does not
    succeed, no generated language is copied into the view; the fallback is rendered
    only from the closed evidence bundle.
    """

    evidence_by_id = {item.evidence_id: item for item in bundle.evidence}
    verification_fields = {
        "status": verification.status,
        "score": float(verification.score),
    }

    if verification.status == "VERIFIED":
        evidence = [
            asdict(evidence_by_id[evidence_id])
            for evidence_id in explanation.evidence_ids
            if evidence_id in evidence_by_id
        ]
        return {
            "display_mode": "verified_explanation",
            "verification": verification_fields,
            "alarm_probability": float(bundle.alarm_probability),
            "summary": explanation.summary,
            "probable_cause": explanation.probable_cause,
            "evidence": evidence,
            "alternatives": list(explanation.alternatives),
            "uncertainty": float(explanation.uncertainty),
            "operator_question": explanation.operator_question,
            "recommendation": explanation.recommendation,
        }

    trusted_evidence = [asdict(item) for item in bundle.evidence if item.critical]
    return {
        "display_mode": "safe_fallback",
        "verification": verification_fields,
        "alarm_probability": float(bundle.alarm_probability),
        "summary": (
            "The generated explanation was withheld because it did not pass the "
            "independent evidence checks."
        ),
        "probable_cause": bundle.predicted_cause,
        "evidence": trusted_evidence,
        "alternatives": list(bundle.alternatives[:2]),
        "uncertainty": float(bundle.uncertainty),
        "operator_question": "Can you confirm the current operating context?",
        "recommendation": bundle.permitted_actions[0],
    }
