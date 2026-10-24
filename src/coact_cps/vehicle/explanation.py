"""Bind a vehicle policy decision to the existing CO-ACT-CPS evidence contract."""

from __future__ import annotations

from dataclasses import asdict

from coact_cps.genai.grounded_generator import GroundedExplanationGenerator
from coact_cps.security.verifier import ExplanationVerifier
from coact_cps.types import EvidenceBundle, EvidenceItem
from coact_cps.vehicle.policy import (
    PolicyDecision,
    VehiclePolicyEngine,
    VehicleRequest,
    canonical_json,
    strict_loads,
)


def decision_card(decision: PolicyDecision) -> dict:
    """The trusted in-process policy card; independent of generator output."""
    result = asdict(decision)
    result["reasons"] = list(decision.reasons)
    result["components"] = dict(decision.components)
    return result


def verify_decision_card(decision: PolicyDecision, candidate_json: str) -> bool:
    """Exact typed membership, not a general natural-language safety classifier."""
    try:
        candidate = strict_loads(candidate_json)
        return canonical_json(candidate) == canonical_json(decision_card(decision))
    except (ValueError, TypeError, OverflowError, RecursionError):
        return False


def build_vehicle_bundle(decision: PolicyDecision) -> EvidenceBundle:
    evidence = [
        EvidenceItem(f"ZT-R-{i:03d}", "access_policy", f"Policy rule: {reason}", 1.0, critical=True)
        for i, reason in enumerate(decision.reasons, start=1)
    ]
    evidence.extend(
        EvidenceItem(
            f"ZT-C-{i:03d}",
            name,
            f"{name} risk component is {value:.4f}",
            value,
            value=value,
            critical=value >= 0.25,
        )
        for i, (name, value) in enumerate(decision.components, start=1)
    )
    # This field is a binary policy alert flag for compatibility. The vehicle
    # renderer reads context_risk and explicitly does not call it a probability.
    policy_alert = float(decision.outcome != "ALLOW")
    return EvidenceBundle(
        event_id=decision.request.request_id,
        alarm_probability=policy_alert,
        predicted_cause=decision.reasons[0],
        # The existing verifier uses this map for candidate membership. The unit
        # weight selects a policy rule; it is not a physical-cause probability.
        cause_probabilities={decision.reasons[0]: 1.0},
        evidence=evidence,
        alternatives=[],
        uncertainty=dict(decision.components).get("uncertainty", 1.0),
        process_context={
            "decision_kind": "vehicle_access_policy",
            "outcome": decision.outcome,
            "context_risk": decision.risk,
            "mode": decision.mode,
            "policy_digest": decision.policy_digest,
            "context_digest": decision.context_digest,
            "request": asdict(decision.request),
        },
        provenance={
            "required": False,
            "status": "VERIFIED" if decision.context_digest else "INVALID",
            "scope": "post-ingestion collector authentication; no physical truth claim",
        },
        permitted_actions=[
            f"Inspect the {decision.outcome} access decision and refresh "
            "authenticated context before a new request"
        ],
    )


def analyze_vehicle_request(
    engine: VehiclePolicyEngine, request: VehicleRequest, frames: dict[str, str], now_ms: int
) -> dict:
    decision = engine.evaluate(request, frames, now_ms)
    bundle = build_vehicle_bundle(decision)
    explanation = GroundedExplanationGenerator().generate(bundle)
    verification = ExplanationVerifier().verify(bundle, explanation)
    card = decision_card(decision)
    card_verified = verify_decision_card(decision, canonical_json(card))
    capability = None
    if (
        decision.outcome in {"ALLOW", "RESTRICT"}
        and verification.status == "VERIFIED"
        and card_verified
    ):
        capability = engine.issue_capability(decision)
    return {
        "policy_card": card,
        "policy_card_verified": card_verified,
        "evidence_bundle": bundle.to_dict(),
        "explanation": explanation.to_dict(),
        "verification": verification.to_dict(),
        "capability": capability,
    }
