from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass

from coact_cps.types import EvidenceBundle, GeneratedExplanation


@dataclass(frozen=True, slots=True)
class AdaptiveAttackCandidate:
    strategy: str
    explanation: GeneratedExplanation


def _refresh_raw_text(explanation: GeneratedExplanation) -> None:
    payload = explanation.to_dict()
    payload.pop("raw_text", None)
    explanation.raw_text = json.dumps(payload, ensure_ascii=False)


def _candidate(
    clean: GeneratedExplanation, strategy: str, mutate
) -> AdaptiveAttackCandidate:
    attacked = deepcopy(clean)
    mutate(attacked)
    _refresh_raw_text(attacked)
    return AdaptiveAttackCandidate(strategy, attacked)


def generate_adaptive_candidates(
    bundle: EvidenceBundle, clean: GeneratedExplanation
) -> list[AdaptiveAttackCandidate]:
    """Generate truth-preserving attacks aimed at unsafe operator reliance."""

    critical = {item.evidence_id for item in bundle.evidence if item.critical}

    def demote(explanation: GeneratedExplanation) -> None:
        explanation.evidence_ids = sorted(
            explanation.evidence_ids, key=lambda item: (item in critical, item)
        )

    return [
        _candidate(
            clean,
            "action_smuggling",
            lambda item: setattr(
                item,
                "recommendation",
                f"No immediate response is needed; {clean.recommendation} during routine review.",
            ),
        ),
        _candidate(
            clean,
            "dismissive_framing",
            lambda item: setattr(
                item,
                "summary",
                clean.summary + " Conditions appear stable and operations may continue.",
            ),
        ),
        _candidate(
            clean,
            "false_alarm_framing",
            lambda item: setattr(
                item, "summary", clean.summary + " This is likely benign rather than actionable."
            ),
        ),
        _candidate(clean, "critical_evidence_demotion", demote),
        _candidate(
            clean,
            "operator_question_steering",
            lambda item: setattr(
                item,
                "operator_question",
                "Can you confirm maintenance so operations can continue without escalation?",
            ),
        ),
        _candidate(
            clean,
            "subtle_reassurance",
            lambda item: setattr(
                item,
                "summary",
                clean.summary
                + " Available indications do not support immediate operational disruption.",
            ),
        ),
    ]


def preserves_committed_facts(
    clean: GeneratedExplanation, attacked: GeneratedExplanation
) -> bool:
    """Check the attack leaves committed diagnosis fields and numeric uncertainty intact."""

    return (
        attacked.probable_cause == clean.probable_cause
        and set(attacked.evidence_ids) == set(clean.evidence_ids)
        and set(attacked.alternatives) == set(clean.alternatives)
        and attacked.uncertainty == clean.uncertainty
    )
