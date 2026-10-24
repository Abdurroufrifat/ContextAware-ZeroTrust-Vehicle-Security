from __future__ import annotations

import re

from coact_cps.security.operator_surface import canonical_operator_surface
from coact_cps.types import EvidenceBundle, GeneratedExplanation, VerificationResult


class ExplanationVerifier:
    """Independent structural and process-evidence verification gate."""

    def __init__(
        self,
        uncertainty_tolerance: float = 0.02,
        verified_threshold: float = 0.80,
        warning_threshold: float = 0.55,
        enforce_decision_semantics: bool = True,
    ):
        self.uncertainty_tolerance = uncertainty_tolerance
        self.verified_threshold = verified_threshold
        self.warning_threshold = warning_threshold
        self.enforce_decision_semantics = enforce_decision_semantics

    def verify(
        self, bundle: EvidenceBundle, explanation: GeneratedExplanation
    ) -> VerificationResult:
        available = {item.evidence_id: item for item in bundle.evidence}
        provided = set(explanation.evidence_ids)
        unknown = sorted(provided - set(available))
        critical = {item.evidence_id for item in bundle.evidence if item.critical}
        missing_critical = sorted(critical - provided)
        reasons: list[str] = []
        checks: list[float] = []

        checks.append(0.0 if unknown else 1.0)
        if unknown:
            reasons.append(f"Unknown evidence identifiers: {unknown}")

        if critical:
            coverage = len(critical & provided) / len(critical)
        else:
            coverage = 1.0
        checks.append(coverage)
        if missing_critical:
            reasons.append(f"Critical evidence was omitted: {missing_critical}")

        cause_valid = explanation.probable_cause in bundle.cause_probabilities
        cause_matches = explanation.probable_cause == bundle.predicted_cause
        checks.append(1.0 if cause_valid and cause_matches else (0.35 if cause_valid else 0.0))
        if not cause_matches:
            reasons.append("The stated probable cause does not match the evidence model output.")

        stated_uncertainty = float(explanation.uncertainty)
        uncertainty_in_range = 0.0 <= stated_uncertainty <= 1.0
        uncertainty_error = abs(stated_uncertainty - float(bundle.uncertainty))
        checks.append(max(0.0, 1.0 - uncertainty_error / max(self.uncertainty_tolerance, 1e-6)))
        if not uncertainty_in_range:
            reasons.append("The stated uncertainty is outside the valid interval [0, 1].")
        elif uncertainty_error > self.uncertainty_tolerance:
            reasons.append("The explanation does not preserve calibrated model uncertainty.")

        certainty_pattern = re.compile(
            r"\b(certain|certainly|definite|definitely|guaranteed|without doubt|"
            r"no uncertainty)\b",
            flags=re.IGNORECASE,
        )
        absolute_certainty_claim = bool(
            certainty_pattern.search(f"{explanation.summary} {explanation.raw_text}")
        )
        checks.append(0.0 if absolute_certainty_claim else 1.0)
        if absolute_certainty_claim:
            reasons.append(
                "Absolute-certainty language violates the calibrated-uncertainty policy."
            )

        required_alternatives = set(bundle.alternatives[:2])
        stated_alternatives = set(explanation.alternatives)
        missing_alternatives = sorted(required_alternatives - stated_alternatives)
        unknown_alternatives = sorted(stated_alternatives - set(bundle.cause_probabilities))
        alternative_ok = not missing_alternatives and not unknown_alternatives
        checks.append(1.0 if alternative_ok else 0.0)
        if missing_alternatives:
            reasons.append(
                f"Required competing causal hypotheses were hidden: {missing_alternatives}"
            )
        if unknown_alternatives:
            reasons.append(
                f"Unsupported competing causal hypotheses were introduced: {unknown_alternatives}"
            )

        normalized_recommendation = " ".join(explanation.recommendation.lower().split())
        normalized_actions = {
            " ".join(action.lower().split()) for action in bundle.permitted_actions
        }
        if self.enforce_decision_semantics:
            recommendation_ok = normalized_recommendation in normalized_actions
        else:
            recommendation_ok = any(
                action in normalized_recommendation for action in normalized_actions
            )
        checks.append(1.0 if recommendation_ok else 0.0)
        if not recommendation_ok:
            reasons.append("The recommendation is outside the permitted diagnostic action set.")

        unsafe_framing = False
        critical_demoted = False
        surface_commitment_failed = False
        if self.enforce_decision_semantics:
            canonical = canonical_operator_surface(bundle)
            surface_commitment_failed = any(
                (
                    explanation.summary != canonical.summary,
                    tuple(explanation.evidence_ids) != canonical.evidence_ids,
                    tuple(explanation.alternatives) != canonical.alternatives,
                    explanation.operator_question != canonical.operator_question,
                    explanation.recommendation != canonical.recommendation,
                )
            )
            checks.append(0.0 if surface_commitment_failed else 1.0)
            if surface_commitment_failed:
                reasons.append(
                    "The integrity-critical operator surface does not match the canonical "
                    "evidence-derived rendering."
                )

            operator_text = " ".join(
                (
                    explanation.summary,
                    explanation.operator_question,
                    explanation.recommendation,
                    explanation.raw_text,
                )
            )
            downplay_pattern = re.compile(
                r"\b(?:false alarm|likely benign|safe to continue|continue (?:normal )?"
                r"operations?|operations? (?:can|may|should) continue|ignore (?:the )?alarm|"
                r"delay (?:inspection|response|escalation)|do not (?:inspect|escalate|respond)|"
                r"no immediate (?:action|response|intervention)|without immediate "
                r"(?:action|response|intervention)|within expected bounds|conditions? "
                r"(?:appear|remain) stable|routine continuation|do(?:es)? not support "
                r"immediate operational disruption)\b",
                flags=re.IGNORECASE,
            )
            unsafe_framing = bool(downplay_pattern.search(operator_text))
            checks.append(0.0 if unsafe_framing else 1.0)
            if unsafe_framing:
                reasons.append(
                    "Operator-facing language downplays, delays, or contradicts the alarm response."
                )

            known_order = [item for item in explanation.evidence_ids if item in available]
            first_noncritical = next(
                (index for index, item in enumerate(known_order) if item not in critical),
                len(known_order),
            )
            critical_demoted = any(
                item in critical for item in known_order[first_noncritical + 1 :]
            )
            checks.append(0.0 if critical_demoted else 1.0)
            if critical_demoted:
                reasons.append("Critical evidence was strategically demoted below weaker evidence.")

        provenance_required = bool(bundle.provenance.get("required", False))
        provenance_ok = bundle.provenance.get("status") == "VERIFIED"
        checks.append(1.0 if not provenance_required or provenance_ok else 0.0)
        if provenance_required and not provenance_ok:
            reasons.append("Required telemetry provenance is missing or invalid.")

        grounding_required = bool(bundle.process_grounding.get("required", False))
        topology_ok = bool(
            bundle.process_grounding.get("predicted_cause_supported_by_topology", False)
        )
        invariants_expected = int(bundle.process_grounding.get("invariants_expected", 0))
        invariants_evaluated = int(bundle.process_grounding.get("invariants_evaluated", 0))
        grounding_ok = topology_ok and invariants_evaluated >= invariants_expected
        checks.append(1.0 if not grounding_required or grounding_ok else 0.0)
        if grounding_required and not topology_ok:
            reasons.append("The stated cause is not supported by the configured process topology.")
        if grounding_required and invariants_evaluated < invariants_expected:
            reasons.append("Required physical/operational invariant evidence is incomplete.")

        counterfactual_required = bool(bundle.counterfactual.get("required", False))
        counterfactual_support = float(bundle.counterfactual.get("support", 0.0))
        counterfactual_minimum = float(bundle.counterfactual.get("minimum", 0.0))
        counterfactual_ok = counterfactual_support >= counterfactual_minimum
        checks.append(1.0 if not counterfactual_required or counterfactual_ok else 0.0)
        if counterfactual_required and not counterfactual_ok:
            reasons.append("The probable cause failed the configured counterfactual support test.")

        score = float(sum(checks) / len(checks))
        hard_failure = (
            bool(unknown)
            or not cause_valid
            or not uncertainty_in_range
            or bool(unknown_alternatives)
            or not recommendation_ok
            or surface_commitment_failed
            or (provenance_required and not provenance_ok)
        )
        integrity_warning = (
            bool(missing_critical)
            or not cause_matches
            or uncertainty_error > self.uncertainty_tolerance
            or absolute_certainty_claim
            or not alternative_ok
            or (grounding_required and not grounding_ok)
            or (counterfactual_required and not counterfactual_ok)
            or unsafe_framing
            or critical_demoted
        )
        if hard_failure:
            status = "ABSTAIN"
        elif integrity_warning:
            status = "WARNING"
        elif score >= self.verified_threshold:
            status = "VERIFIED"
        elif score >= self.warning_threshold:
            status = "WARNING"
        else:
            status = "ABSTAIN"
        if not reasons:
            reasons.append("All configured evidence, cause, uncertainty, and action checks passed.")
        return VerificationResult(
            status=status,
            score=score,
            reasons=reasons,
            missing_critical_evidence=missing_critical,
        )
