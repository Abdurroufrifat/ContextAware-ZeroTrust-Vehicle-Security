from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from coact_cps.genai.grounded_generator import GroundedExplanationGenerator
from coact_cps.security.attacks import ATTACKS, apply_explanation_attack
from coact_cps.security.provenance import TelemetryAuthenticator, window_digest
from coact_cps.security.verifier import ExplanationVerifier
from coact_cps.types import EvidenceBundle, EvidenceItem


@pytest.fixture
def evidence_bundle() -> EvidenceBundle:
    return EvidenceBundle(
        event_id="test-event",
        alarm_probability=0.88,
        predicted_cause="LIT101",
        cause_probabilities={"LIT101": 0.65, "MV101": 0.25, "P101": 0.10},
        evidence=[
            EvidenceItem("EV-001", "LIT101", "Level residual is high", 0.65, critical=True),
            EvidenceItem("EV-002", "MV101", "Valve relation is inconsistent", 0.25, critical=True),
            EvidenceItem("EV-003", "P101", "Pump is a weaker alternative", 0.10),
        ],
        alternatives=["MV101", "P101"],
        uncertainty=0.40,
    )


def test_clean_explanation_is_verified(evidence_bundle: EvidenceBundle) -> None:
    explanation = GroundedExplanationGenerator().generate(evidence_bundle)
    result = ExplanationVerifier().verify(evidence_bundle, explanation)
    assert result.status == "VERIFIED"
    assert result.score >= 0.8


@pytest.mark.parametrize("attack", ATTACKS)
def test_controlled_manipulation_is_not_verified(
    evidence_bundle: EvidenceBundle, attack: str
) -> None:
    clean = GroundedExplanationGenerator().generate(evidence_bundle)
    manipulated = apply_explanation_attack(clean, attack)
    result = ExplanationVerifier().verify(evidence_bundle, manipulated)
    assert result.status in {"WARNING", "ABSTAIN"}


def test_acquisition_boundary_signature_detects_change() -> None:
    window = np.arange(24, dtype=np.float32).reshape(4, 6, 1)
    authenticator = TelemetryAuthenticator(b"unit-test-key-with-enough-entropy")
    signature = authenticator.sign(window, "event-1")
    assert authenticator.verify(window, "event-1", signature)
    changed = window.copy()
    changed[0, 0, 0] += 1.0
    assert not authenticator.verify(changed, "event-1", signature)
    assert window_digest(window, "event-1") != window_digest(changed, "event-1")


def test_required_invalid_provenance_forces_abstention(
    evidence_bundle: EvidenceBundle,
) -> None:
    bundle = deepcopy(evidence_bundle)
    bundle.provenance = {"required": True, "status": "INVALID"}
    explanation = GroundedExplanationGenerator().generate(bundle)
    result = ExplanationVerifier().verify(bundle, explanation)
    assert result.status == "ABSTAIN"
    assert any("provenance" in reason for reason in result.reasons)


def test_low_uncertainty_confidence_inflation_is_flagged(
    evidence_bundle: EvidenceBundle,
) -> None:
    bundle = deepcopy(evidence_bundle)
    bundle.uncertainty = 0.01
    clean = GroundedExplanationGenerator().generate(bundle)
    manipulated = apply_explanation_attack(clean, "inflate_confidence")
    result = ExplanationVerifier().verify(bundle, manipulated)
    assert result.status in {"WARNING", "ABSTAIN"}
    assert any("certainty" in reason.lower() for reason in result.reasons)


def test_alternatives_are_committed_even_when_uncertainty_is_low(
    evidence_bundle: EvidenceBundle,
) -> None:
    bundle = deepcopy(evidence_bundle)
    bundle.uncertainty = 0.01
    clean = GroundedExplanationGenerator().generate(bundle)
    manipulated = apply_explanation_attack(clean, "hide_alternatives")
    result = ExplanationVerifier().verify(bundle, manipulated)
    assert result.status in {"WARNING", "ABSTAIN"}
    assert any("hypotheses were hidden" in reason for reason in result.reasons)


def test_verifier_rejects_subtle_operator_surface_rewrite(
    evidence_bundle: EvidenceBundle,
) -> None:
    explanation = GroundedExplanationGenerator().generate(evidence_bundle)
    explanation.summary = explanation.summary.replace(
        "an abnormal CPS event", "a potential normal operation"
    )
    explanation.raw_text = explanation.summary

    result = ExplanationVerifier().verify(evidence_bundle, explanation)

    assert result.status == "ABSTAIN"
    assert any("canonical evidence-derived rendering" in reason for reason in result.reasons)


def test_verifier_rejects_neutral_question_paraphrase(
    evidence_bundle: EvidenceBundle,
) -> None:
    explanation = GroundedExplanationGenerator().generate(evidence_bundle)
    explanation.operator_question = "Could you confirm the current operating context?"
    explanation.raw_text = explanation.operator_question

    result = ExplanationVerifier().verify(evidence_bundle, explanation)

    assert result.status == "ABSTAIN"
