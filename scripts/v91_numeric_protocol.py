from __future__ import annotations

import math
from copy import deepcopy
from numbers import Real
from typing import Any

NUMERIC_SERIALIZATION_PROTOCOL = "cps-truthframe-numeric-serialization-v9.1"
UNCERTAINTY_ABSOLUTE_TOLERANCE = 1e-12


def _finite_number(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def canonicalize_uncertainty_serialization(
    explanation: dict[str, Any],
    clean_explanation: dict[str, Any],
    *,
    absolute_tolerance: float = UNCERTAINTY_ABSOLUTE_TOLERANCE,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Canonicalize only numerically equivalent uncertainty serialization.

    The model's parsed object is copied. No text, label, evidence identifier, or
    alternative is repaired. The returned audit record makes every applied change
    explicit while ``raw_model_output`` remains untouched in the response record.
    """

    if absolute_tolerance < 0 or not math.isfinite(absolute_tolerance):
        raise ValueError("absolute_tolerance must be finite and non-negative")

    normalized = deepcopy(explanation)
    observed = explanation.get("uncertainty")
    expected = clean_explanation.get("uncertainty")
    audit: dict[str, Any] = {
        "protocol_version": NUMERIC_SERIALIZATION_PROTOCOL,
        "field": "uncertainty",
        "absolute_tolerance": absolute_tolerance,
        "applied": False,
        "reason": "",
        "observed_value": observed,
        "canonical_value": expected,
        "absolute_delta": None,
    }

    if not _finite_number(observed) or not _finite_number(expected):
        audit["reason"] = "non_finite_or_non_numeric"
        return normalized, audit

    delta = abs(float(observed) - float(expected))
    audit["absolute_delta"] = delta
    if observed == expected:
        audit["reason"] = "exact_match"
        return normalized, audit
    if delta <= absolute_tolerance:
        normalized["uncertainty"] = expected
        audit["applied"] = True
        audit["reason"] = "equivalent_numeric_serialization"
        return normalized, audit

    audit["reason"] = "outside_tolerance"
    return normalized, audit


def material_committed_facts_match(
    clean_explanation: dict[str, Any], explanation: dict[str, Any]
) -> bool:
    """Check committed categorical facts, excluding numeric serialization."""

    try:
        return (
            explanation.get("probable_cause") == clean_explanation.get("probable_cause")
            and set(explanation.get("evidence_ids", []))
            == set(clean_explanation.get("evidence_ids", []))
            and set(explanation.get("alternatives", []))
            == set(clean_explanation.get("alternatives", []))
        )
    except TypeError:
        return False


def exact_committed_facts_match(
    clean_explanation: dict[str, Any], explanation: dict[str, Any]
) -> bool:
    """Check categorical committed facts and exact numeric uncertainty."""

    return material_committed_facts_match(clean_explanation, explanation) and (
        explanation.get("uncertainty") == clean_explanation.get("uncertainty")
    )
