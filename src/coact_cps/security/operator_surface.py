from __future__ import annotations

from dataclasses import dataclass

from coact_cps.types import EvidenceBundle


@dataclass(frozen=True)
class CanonicalOperatorSurface:
    """Deterministic, evidence-derived content shown on the operator surface."""

    summary: str
    evidence_ids: tuple[str, ...]
    alternatives: tuple[str, ...]
    operator_question: str
    recommendation: str


def canonical_operator_surface(bundle: EvidenceBundle) -> CanonicalOperatorSurface:
    """Render integrity-critical explanation fields without free-form generation."""

    sorted_evidence = sorted(bundle.evidence, key=lambda item: item.score, reverse=True)
    critical = [item for item in sorted_evidence if item.critical]
    selected = list(critical)
    for item in sorted_evidence:
        if item not in selected and len(selected) < max(3, len(critical)):
            selected.append(item)

    evidence_text = "; ".join(item.statement for item in selected)
    alternatives = tuple(bundle.alternatives[:2])
    if bundle.uncertainty >= 0.25 and alternatives:
        question = (
            f"Can you confirm whether operational context supports {bundle.predicted_cause} "
            f"rather than {alternatives[0]}?"
        )
    else:
        question = "Can you confirm whether any authorized maintenance explains this event?"

    if bundle.process_context.get("decision_kind") == "vehicle_access_policy":
        outcome = bundle.process_context["outcome"]
        risk = bundle.process_context["context_risk"]
        assessment = (
            f"Vehicle access policy returned {outcome}; heuristic context risk is {risk:.3f}."
        )
    elif bundle.alarm_probability >= 0.5:
        assessment = (
            "The detector estimates an abnormal CPS event with probability "
            f"{bundle.alarm_probability:.1%}."
        )
    else:
        assessment = (
            "The detector does not currently classify this window as abnormal; "
            f"estimated alarm probability is {bundle.alarm_probability:.1%}."
        )

    return CanonicalOperatorSurface(
        summary=f"{assessment} Evidence: {evidence_text}.",
        evidence_ids=tuple(item.evidence_id for item in selected),
        alternatives=alternatives,
        operator_question=question,
        recommendation=bundle.permitted_actions[0],
    )
