from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from .genai.grounded_generator import GroundedExplanationGenerator
from .models.collaboration_policy import choose_safe_action
from .models.temporal_graph import PhysicsInvariant, TemporalHeterogeneousGraphTransformer
from .security.operator_view import build_operator_view
from .security.provenance import TelemetryAuthenticator, window_digest
from .security.registered_surface import RegisteredSurfaceVerifier
from .security.verifier import ExplanationVerifier
from .types import EvidenceBundle, EvidenceItem


class CoActPipeline:
    def __init__(
        self,
        model: TemporalHeterogeneousGraphTransformer,
        adjacency: torch.Tensor,
        node_names: Sequence[str],
        generator: GroundedExplanationGenerator | None = None,
        verifier: ExplanationVerifier | RegisteredSurfaceVerifier | None = None,
        device: str | None = None,
        physics_invariants: Sequence[PhysicsInvariant] | None = None,
        invariant_threshold: float = 0.05,
        authenticator: TelemetryAuthenticator | None = None,
        require_authentication: bool = False,
        require_process_grounding: bool = False,
        counterfactual_minimum: float = 0.0,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.adjacency = adjacency.to(self.device)
        self.node_names = list(node_names)
        self.generator = generator or GroundedExplanationGenerator()
        self.verifier = verifier or ExplanationVerifier()
        self.physics_invariants = list(physics_invariants or [])
        self.invariant_threshold = float(invariant_threshold)
        self.authenticator = authenticator
        self.require_authentication = require_authentication
        self.require_process_grounding = require_process_grounding
        self.counterfactual_minimum = float(counterfactual_minimum)

    @torch.no_grad()
    def _counterfactual_support(self, x: torch.Tensor, root_index: int) -> tuple[float, float]:
        self.model.eval()
        base_probability, _ = self.model.predict_point(x, self.adjacency)
        counterfactual = x.clone()
        if self.model.detection_mode == "normal_only":
            masked = x.clone()
            masked[:, -1] = 0.0
            predicted = self.model(masked, self.adjacency).reconstructed[:, -1, root_index]
            counterfactual[:, -1, root_index] = predicted
        else:
            replacement = torch.median(x[:, :-1, root_index], dim=1).values
            counterfactual[:, -1, root_index] = replacement
        changed_probability, _ = self.model.predict_point(counterfactual, self.adjacency)
        base = float(base_probability[0].cpu())
        changed = float(changed_probability[0].cpu())
        support = max(0.0, min(1.0, (base - changed) / max(base, 1e-6)))
        return support, changed

    def _invariant_items(self, x: torch.Tensor) -> tuple[list[EvidenceItem], list[dict]]:
        items: list[EvidenceItem] = []
        details: list[dict] = []
        for index, invariant in enumerate(self.physics_invariants, start=1):
            if invariant.kind != "mass_balance":
                continue
            state = x[0, :, invariant.state_node, 0]
            inflow = x[0, 1:, invariant.inflow_node, 0]
            outflow = x[0, 1:, invariant.outflow_node, 0]
            observed = state[1:] - state[:-1]
            expected = invariant.coefficient * (inflow - outflow)
            residual = float(torch.mean(torch.abs(observed - expected)).cpu())
            violated = residual > self.invariant_threshold
            evidence_id = f"PHY-{index:03d}"
            statement = (
                f"Configured mass-balance residual is {residual:.4f} "
                f"(threshold {self.invariant_threshold:.4f})"
            )
            items.append(
                EvidenceItem(
                    evidence_id=evidence_id,
                    node=self.node_names[invariant.state_node],
                    statement=statement,
                    score=min(1.0, residual / max(self.invariant_threshold, 1e-6)),
                    value=residual,
                    reliability=1.0,
                    critical=violated,
                )
            )
            details.append(
                {
                    "evidence_id": evidence_id,
                    "kind": invariant.kind,
                    "residual": residual,
                    "threshold": self.invariant_threshold,
                    "status": "VIOLATED" if violated else "CONSISTENT",
                }
            )
        return items, details

    def build_evidence_bundle(
        self,
        window: np.ndarray,
        event_id: str = "event-001",
        mc_samples: int = 10,
        telemetry_signature: str | None = None,
    ) -> EvidenceBundle:
        x = torch.as_tensor(window, dtype=torch.float32, device=self.device)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        attack_mean, attack_std, root_probability = self.model.predict_with_uncertainty(
            x, self.adjacency, samples=mc_samples
        )
        root_values = root_probability[0].detach().cpu().numpy()
        ranked = np.argsort(root_values)[::-1]
        cause_probabilities = {
            self.node_names[index]: float(root_values[index]) for index in ranked
        }
        predicted = self.node_names[int(ranked[0])]
        alternatives = [self.node_names[int(index)] for index in ranked[1:3]]
        last_values = x[0, -1, :, 0].detach().cpu().numpy()
        evidence = []
        for rank, index in enumerate(ranked[:4]):
            node = self.node_names[int(index)]
            evidence.append(
                EvidenceItem(
                    evidence_id=f"EV-{rank + 1:03d}",
                    node=node,
                    statement=(
                        f"{node} has value {last_values[index]:.4f} and root-cause "
                        f"support {root_values[index]:.1%} in the current window"
                    ),
                    score=float(root_values[index]),
                    value=float(last_values[index]),
                    reliability=1.0,
                    critical=rank < 2,
                )
            )
        invariant_items, invariant_details = self._invariant_items(x)
        evidence.extend(invariant_items)
        root_index = int(ranked[0])
        relation_degree = float(
            self.adjacency[:, root_index, :].sum().cpu()
            + self.adjacency[:, :, root_index].sum().cpu()
        )
        topology_supported = relation_degree > 0
        counterfactual_support, counterfactual_probability = self._counterfactual_support(
            x, root_index
        )
        digest = window_digest(window, event_id)
        if self.authenticator is None:
            provenance_status = "UNAVAILABLE"
        elif telemetry_signature is None:
            provenance_status = "MISSING"
        elif self.authenticator.verify(window, event_id, telemetry_signature):
            provenance_status = "VERIFIED"
        else:
            provenance_status = "INVALID"
        uncertainty = min(
            1.0,
            0.5 * float(attack_std[0].cpu()) + 0.5 * (1.0 - float(root_values[ranked[0]])),
        )
        return EvidenceBundle(
            event_id=event_id,
            alarm_probability=float(attack_mean[0].cpu()),
            predicted_cause=predicted,
            cause_probabilities=cause_probabilities,
            evidence=evidence,
            alternatives=alternatives,
            uncertainty=uncertainty,
            process_context={"window_length": int(x.shape[1]), "node_count": len(self.node_names)},
            provenance={
                "required": self.require_authentication,
                "status": provenance_status,
                "algorithm": "HMAC-SHA256"
                if self.authenticator is not None
                else "SHA-256 digest only",
                "window_digest": digest,
                "scope": "post-ingestion/acquisition-boundary integrity",
            },
            process_grounding={
                "required": self.require_process_grounding,
                "predicted_cause_supported_by_topology": topology_supported,
                "predicted_cause_relation_degree": relation_degree,
                "invariants_expected": len(self.physics_invariants),
                "invariants_evaluated": len(invariant_details),
                "invariants": invariant_details,
            },
            counterfactual={
                "required": self.require_process_grounding,
                "support": counterfactual_support,
                "minimum": self.counterfactual_minimum,
                "alarm_probability_after_replacement": counterfactual_probability,
                "intervention": f"replace final {predicted} value with model-consistent value",
            },
        )

    def analyze(
        self,
        window: np.ndarray,
        event_id: str = "event-001",
        operator_confidence: float = 0.5,
        operator_reliance: float = 0.5,
        operator_workload: float = 0.5,
        telemetry_signature: str | None = None,
    ) -> dict:
        bundle = self.build_evidence_bundle(
            window, event_id, telemetry_signature=telemetry_signature
        )
        explanation = self.generator.generate(bundle)
        verification = self.verifier.verify(bundle, explanation)
        decision = choose_safe_action(
            verification,
            alarm_probability=bundle.alarm_probability,
            uncertainty=bundle.uncertainty,
            operator_confidence=operator_confidence,
            operator_reliance=operator_reliance,
            operator_workload=operator_workload,
        )
        return {
            "evidence_bundle": bundle.to_dict(),
            "operator_view": build_operator_view(bundle, explanation, verification),
            "audit_explanation": explanation.to_dict(),
            "verification": verification.to_dict(),
            "collaboration_decision": decision.to_dict(),
        }
