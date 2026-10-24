from coact_cps.genai.grounded_generator import GroundedExplanationGenerator
from coact_cps.security.adaptive_redteam import (
    generate_adaptive_candidates,
    preserves_committed_facts,
)
from coact_cps.security.verifier import ExplanationVerifier
from coact_cps.types import EvidenceBundle, EvidenceItem


def _bundle() -> EvidenceBundle:
    return EvidenceBundle(
        event_id="adaptive-test",
        alarm_probability=0.91,
        predicted_cause="P1_PIT01",
        cause_probabilities={"P1_PIT01": 0.7, "P1_PCV01D": 0.2, "P1_FT01": 0.1},
        evidence=[
            EvidenceItem("EV-001", "P1_PIT01", "pressure evidence", 0.7, critical=True),
            EvidenceItem("EV-002", "P1_PCV01D", "valve evidence", 0.2, critical=True),
            EvidenceItem("EV-003", "P1_FT01", "flow evidence", 0.1),
        ],
        alternatives=["P1_PCV01D", "P1_FT01"],
        uncertainty=0.31,
    )


def test_adaptive_candidates_preserve_committed_facts() -> None:
    bundle = _bundle()
    clean = GroundedExplanationGenerator().generate(bundle)
    candidates = generate_adaptive_candidates(bundle, clean)
    assert len(candidates) >= 5
    assert all(preserves_committed_facts(clean, item.explanation) for item in candidates)


def test_legacy_structural_verifier_exposes_adaptive_gap() -> None:
    bundle = _bundle()
    clean = GroundedExplanationGenerator().generate(bundle)
    candidate = next(
        item for item in generate_adaptive_candidates(bundle, clean)
        if item.strategy == "action_smuggling"
    )
    result = ExplanationVerifier(enforce_decision_semantics=False).verify(
        bundle, candidate.explanation
    )
    assert result.status == "VERIFIED"


def test_hardened_verifier_blocks_adaptive_candidates() -> None:
    bundle = _bundle()
    clean = GroundedExplanationGenerator().generate(bundle)
    verifier = ExplanationVerifier(enforce_decision_semantics=True)
    assert verifier.verify(bundle, clean).status == "VERIFIED"
    for candidate in generate_adaptive_candidates(bundle, clean):
        assert verifier.verify(bundle, candidate.explanation).status != "VERIFIED"
