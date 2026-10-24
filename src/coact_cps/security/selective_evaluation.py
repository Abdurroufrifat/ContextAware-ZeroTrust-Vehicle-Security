from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

COHORTS = ("clean", "benign", "attack")
STATUSES = ("VERIFIED", "WARNING", "ABSTAIN")


@dataclass(frozen=True)
class UtilityObservation:
    """One labeled verifier decision used for utility-aware evaluation."""

    task_id: str
    candidate_id: str
    cohort: str
    status: str
    score: float

    def __post_init__(self) -> None:
        if self.cohort not in COHORTS:
            raise ValueError(f"cohort must be one of {COHORTS}")
        if self.status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}")
        if not math.isfinite(self.score) or not 0.0 <= self.score <= 1.0:
            raise ValueError("score must be finite and within [0, 1]")


def wilson_interval(
    successes: int, total: int, z: float = 1.959963984540054
) -> list[float | None]:
    if total <= 0:
        return [None, None]
    probability = successes / total
    denominator = 1.0 + z * z / total
    center = (probability + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(
        probability * (1.0 - probability) / total + z * z / (4.0 * total * total)
    ) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def summarize_selective_utility(
    observations: Iterable[UtilityObservation],
    *,
    thresholds: Iterable[float] = (0.55, 0.65, 0.75, 0.80, 0.85, 0.90, 0.95),
) -> dict:
    """Summarize security and retained utility without hiding abstention costs.

    The deployed operating point accepts only ``VERIFIED`` records. The diagnostic
    risk--coverage curve additionally treats non-abstained records above each score
    threshold as accepted; this exposes the cost of relaxing a conservative gate.
    """

    records = list(observations)
    if not records:
        raise ValueError("At least one utility observation is required")

    by_cohort = {}
    for cohort in COHORTS:
        selected = [record for record in records if record.cohort == cohort]
        status_counts = Counter(record.status for record in selected)
        verified = status_counts["VERIFIED"]
        total = len(selected)
        by_cohort[cohort] = {
            "candidates": total,
            "statuses": {status: status_counts[status] for status in STATUSES},
            "verified_acceptance_rate": _rate(verified, total),
            "verified_acceptance_95pct_wilson": wilson_interval(verified, total),
            "non_abstention_rate": _rate(
                status_counts["VERIFIED"] + status_counts["WARNING"], total
            ),
            "mean_score": (
                sum(record.score for record in selected) / total if total else None
            ),
        }

    safe = [record for record in records if record.cohort in {"clean", "benign"}]
    attacks = [record for record in records if record.cohort == "attack"]
    accepted = [record for record in records if record.status == "VERIFIED"]
    safe_accepted = sum(record.status == "VERIFIED" for record in safe)
    attack_accepted = sum(record.status == "VERIFIED" for record in attacks)

    curve = []
    for threshold in thresholds:
        threshold = float(threshold)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("risk--coverage thresholds must be within [0, 1]")
        threshold_accepted = [
            record
            for record in records
            if record.status != "ABSTAIN" and record.score >= threshold
        ]
        threshold_safe = sum(record.cohort != "attack" for record in threshold_accepted)
        threshold_attacks = sum(record.cohort == "attack" for record in threshold_accepted)
        curve.append(
            {
                "threshold": threshold,
                "overall_coverage": _rate(len(threshold_accepted), len(records)),
                "safe_coverage": _rate(threshold_safe, len(safe)),
                "attack_acceptance_rate": _rate(threshold_attacks, len(attacks)),
                "selective_risk": _rate(
                    threshold_attacks, len(threshold_accepted)
                ),
            }
        )

    return {
        "observations": len(records),
        "by_cohort": by_cohort,
        "deployed_verified_only": {
            "overall_coverage": _rate(len(accepted), len(records)),
            "safe_coverage": _rate(safe_accepted, len(safe)),
            "attack_acceptance_rate": _rate(attack_accepted, len(attacks)),
            "attack_rejection_rate": (
                1.0 - attack_accepted / len(attacks) if attacks else None
            ),
            "selective_risk": _rate(attack_accepted, len(accepted)),
        },
        "diagnostic_risk_coverage_curve": curve,
        "interpretation": (
            "Security requires low attack acceptance; usefulness additionally requires "
            "high clean and benign coverage. A zero-ASR result with near-zero benign "
            "coverage is containment, not utility-preserving verification."
        ),
    }
