"""Strict display contract for trusted in-process vehicle policy decisions."""

from __future__ import annotations

from coact_cps.vehicle.explanation import decision_card
from coact_cps.vehicle.policy import PolicyDecision, canonical_json, digest, strict_loads

COMPARISONS = frozenset(
    {
        "speed_disagreement",
        "acceleration_disagreement",
        "yaw_disagreement",
    }
)


def render_decision(decision: PolicyDecision) -> dict:
    """Expose recorded evidence only; this function cannot issue a capability.

    Caller must supply the trusted decision, not a card received from a client.
    A digest is a reference binding, not a signature or proof of sensor truth.
    No diagnostic snapshot is attached: it is not bound in PolicyDecision.
    """
    card = decision_card(decision)
    components = [{"name": name, "value": value} for name, value in decision.components]
    values = dict(decision.components)
    comparisons = [item.copy() for item in components if item["name"] in COMPARISONS]
    return {
        "schema": "coact-vehicle-explanation-v14",
        "decision_sha256": digest(card),
        "policy_card": card,
        "summary": (
            f"Access decision: {decision.outcome}; mode: {decision.mode}. "
            f"Heuristic policy risk: {decision.risk:.6g}; not a probability."
        ),
        "components": components,
        "uncertainty": {
            "status": "reported" if "uncertainty" in values else "unavailable",
            "value": values.get("uncertainty"),
            "meaning": "Policy input; calibration has not been established.",
        },
        "sensor_comparisons": {
            "status": "reported" if comparisons else "unavailable",
            "values": comparisons,
        },
        "context_binding": "available" if decision.context_digest else "unavailable",
        "physical_cause": "Not established by this policy decision.",
        "recommended_action": (
            "Inspect the recorded reasons and refresh authenticated context for a new request. "
            "This display does not authorize access or override the policy."
        ),
        "limits": [
            "Collector authentication does not prove physical sensor truth.",
            "Unavailable evidence is not evidence of safe operation.",
            "Human understanding, trust and usability have not been evaluated.",
            "This record describes a past evaluation; use-time authorization is separate.",
        ],
    }


def verify_view(decision: PolicyDecision, candidate_json: str) -> bool:
    """Accept only the exact typed rendering relative to a trusted decision.

    Object key order and JSON whitespace may vary. Every value, array order and
    field set must match. This is a closed rendering contract, not a general
    verifier of natural language. No clock or authorization claim is made.
    """
    try:
        candidate = strict_loads(candidate_json)
        return canonical_json(candidate) == canonical_json(render_decision(decision))
    except (ValueError, TypeError, KeyError, OverflowError, RecursionError):
        return False
