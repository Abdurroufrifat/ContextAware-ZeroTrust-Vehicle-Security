from __future__ import annotations

import torch
from torch import nn

from coact_cps.types import CollaborationAction, CollaborationDecision, VerificationResult

ACTIONS = tuple(CollaborationAction)


class CollaborationPolicyNet(nn.Module):
    """Trainable policy over human-AI collaboration actions."""

    def __init__(self, ai_state_size: int, operator_state_size: int, hidden_size: int = 96):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(ai_state_size + operator_state_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, len(ACTIONS)),
        )

    def forward(self, ai_state: torch.Tensor, operator_state: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat([ai_state, operator_state], dim=-1))


def choose_safe_action(
    verification: VerificationResult,
    alarm_probability: float,
    uncertainty: float,
    operator_confidence: float = 0.5,
    operator_reliance: float = 0.5,
    operator_workload: float = 0.5,
) -> CollaborationDecision:
    """Auditable fallback policy used until a policy is trained on human data."""

    if verification.status == "ABSTAIN":
        return CollaborationDecision(
            CollaborationAction.ABSTAIN,
            "The explanation did not pass independent evidence checks.",
            1.0,
        )
    if verification.status == "WARNING":
        return CollaborationDecision(
            CollaborationAction.SHOW_EVIDENCE,
            "The operator should inspect the missing or inconsistent evidence.",
            0.9,
        )
    if uncertainty >= 0.25:
        return CollaborationDecision(
            CollaborationAction.SHOW_ALTERNATIVES,
            "Model uncertainty is high enough that competing causes should be visible.",
            min(1.0, 0.6 + uncertainty),
        )
    if operator_reliance >= 0.80 and operator_confidence < 0.55:
        return CollaborationDecision(
            CollaborationAction.ASK_CONTEXT,
            "High reliance with low confidence indicates a need for active operator input.",
            0.85,
        )
    if operator_workload >= 0.80:
        return CollaborationDecision(
            CollaborationAction.EXPLAIN_SUMMARY,
            "A concise, evidence-linked summary reduces unnecessary information load.",
            0.8,
        )
    if alarm_probability >= 0.75:
        return CollaborationDecision(
            CollaborationAction.ASK_CONTEXT,
            "A strong alarm should be combined with operational context before escalation.",
            alarm_probability,
        )
    return CollaborationDecision(
        CollaborationAction.EXPLAIN_SUMMARY,
        "Evidence is verified and current uncertainty is limited.",
        verification.score,
    )
