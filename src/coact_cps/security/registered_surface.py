from __future__ import annotations

import json
import math
from dataclasses import dataclass

from coact_cps.security.operator_surface import canonical_operator_surface
from coact_cps.security.verifier import ExplanationVerifier
from coact_cps.types import EvidenceBundle, GeneratedExplanation, VerificationResult


@dataclass(frozen=True)
class RegisteredOperatorSurface:
    """A pre-reviewed rendering of the integrity-critical operator fields."""

    surface_id: str
    summary: str
    evidence_ids: tuple[str, ...]
    alternatives: tuple[str, ...]
    operator_question: str
    recommendation: str


def registered_operator_surfaces(
    bundle: EvidenceBundle,
) -> tuple[RegisteredOperatorSurface, ...]:
    """Return the finite, reviewable language surface allowed by the V8 gate.

    This is deliberately a registered phrasebook, not an open-ended semantic judge.
    Every permitted rendering is constructed from the same closed evidence bundle.
    """

    canonical = canonical_operator_surface(bundle)
    evidence_by_id = {item.evidence_id: item for item in bundle.evidence}
    evidence_text = "; ".join(
        evidence_by_id[evidence_id].statement
        for evidence_id in canonical.evidence_ids
        if evidence_id in evidence_by_id
    )

    if bundle.alarm_probability >= 0.5:
        concise_summary = (
            f"Abnormal-event probability: {bundle.alarm_probability:.1%}. "
            f"Supporting evidence: {evidence_text}."
        )
        evidence_first_summary = (
            f"Supporting evidence: {evidence_text}. The detector estimates an abnormal "
            f"CPS event with probability {bundle.alarm_probability:.1%}."
        )
    else:
        concise_summary = (
            f"Current abnormal-event probability: {bundle.alarm_probability:.1%}; the "
            f"window is not classified as abnormal. Evidence reviewed: {evidence_text}."
        )
        evidence_first_summary = (
            f"Evidence reviewed: {evidence_text}. The detector does not currently classify "
            f"this window as abnormal ({bundle.alarm_probability:.1%})."
        )

    if bundle.uncertainty >= 0.25 and canonical.alternatives:
        concise_question = (
            f"Does the operating context support {bundle.predicted_cause} rather than "
            f"{canonical.alternatives[0]}?"
        )
        evidence_first_question = (
            f"Which observed operating condition best distinguishes {bundle.predicted_cause} "
            f"from {canonical.alternatives[0]}?"
        )
    else:
        concise_question = "Is there authorized maintenance that could explain this event?"
        evidence_first_question = (
            "Can you verify whether authorized maintenance is active for this process?"
        )

    shared = {
        "evidence_ids": canonical.evidence_ids,
        "alternatives": canonical.alternatives,
        "recommendation": canonical.recommendation,
    }
    return (
        RegisteredOperatorSurface(
            surface_id="canonical-v1",
            summary=canonical.summary,
            operator_question=canonical.operator_question,
            **shared,
        ),
        RegisteredOperatorSurface(
            surface_id="concise-v1",
            summary=concise_summary,
            operator_question=concise_question,
            **shared,
        ),
        RegisteredOperatorSurface(
            surface_id="evidence-first-v1",
            summary=evidence_first_summary,
            operator_question=evidence_first_question,
            **shared,
        ),
    )


def explanation_for_registered_surface(
    bundle: EvidenceBundle, surface_id: str
) -> GeneratedExplanation:
    """Materialize one registered surface as a complete explanation record."""

    surfaces = {
        surface.surface_id: surface for surface in registered_operator_surfaces(bundle)
    }
    if surface_id not in surfaces:
        raise KeyError(f"Unknown registered surface: {surface_id}")
    surface = surfaces[surface_id]
    result = GeneratedExplanation(
        summary=surface.summary,
        probable_cause=bundle.predicted_cause,
        evidence_ids=list(surface.evidence_ids),
        alternatives=list(surface.alternatives),
        uncertainty=float(bundle.uncertainty),
        operator_question=surface.operator_question,
        recommendation=surface.recommendation,
    )
    result.raw_text = json.dumps(result.to_dict(), ensure_ascii=False, sort_keys=True)
    return result


def match_registered_surface(
    bundle: EvidenceBundle, explanation: GeneratedExplanation
) -> str | None:
    """Return the matching surface identifier, or ``None`` for free-form text."""

    for surface in registered_operator_surfaces(bundle):
        if all(
            (
                explanation.summary == surface.summary,
                tuple(explanation.evidence_ids) == surface.evidence_ids,
                tuple(explanation.alternatives) == surface.alternatives,
                explanation.operator_question == surface.operator_question,
                explanation.recommendation == surface.recommendation,
            )
        ):
            return surface.surface_id
    return None


class RegisteredSurfaceVerifier:
    """V8 verifier for a finite, independently reviewable operator surface.

    The V7 verifier remains untouched for reproducibility. This extension permits a
    small set of pre-reviewed benign renderings while keeping cause, evidence,
    alternatives, uncertainty, and recommendation locked to the evidence bundle.
    ``raw_text`` is audit-only and must never be displayed by the operator interface.
    """

    def __init__(self, allowed_surface_ids: tuple[str, ...] | None = None):
        self.allowed_surface_ids = set(allowed_surface_ids or ()) or None
        self._structural = ExplanationVerifier(enforce_decision_semantics=False)

    def verify(
        self, bundle: EvidenceBundle, explanation: GeneratedExplanation
    ) -> VerificationResult:
        structural = self._structural.verify(bundle, explanation)
        surface_id = match_registered_surface(bundle, explanation)
        surface_allowed = surface_id is not None and (
            self.allowed_surface_ids is None or surface_id in self.allowed_surface_ids
        )
        commitments_match = all(
            (
                explanation.probable_cause == bundle.predicted_cause,
                math.isclose(
                    float(explanation.uncertainty),
                    float(bundle.uncertainty),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ),
            )
        )

        reasons = list(structural.reasons)
        if not surface_allowed:
            reasons.append(
                "The operator-facing text is not a registered evidence-derived surface."
            )
        if not commitments_match:
            reasons.append(
                "An integrity-critical cause or uncertainty commitment changed."
            )

        if not surface_allowed or not commitments_match:
            return VerificationResult(
                status="ABSTAIN",
                score=0.0,
                reasons=reasons,
                missing_critical_evidence=structural.missing_critical_evidence,
            )

        if structural.status != "VERIFIED":
            return VerificationResult(
                status=structural.status,
                score=structural.score,
                reasons=reasons,
                missing_critical_evidence=structural.missing_critical_evidence,
            )

        return VerificationResult(
            status="VERIFIED",
            score=1.0,
            reasons=[f"Registered operator surface verified: {surface_id}."],
            missing_critical_evidence=[],
        )
