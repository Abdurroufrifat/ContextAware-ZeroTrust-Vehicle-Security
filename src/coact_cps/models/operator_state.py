from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(slots=True)
class OperatorStateOutput:
    diagnosis_logits: torch.Tensor
    confidence: torch.Tensor
    reliance: torch.Tensor
    workload: torch.Tensor
    embedding: torch.Tensor


class OperatorStateTransformer(nn.Module):
    """Estimates operator state from a sequence of observable interactions.

    It must be trained on consented human-study data for publication. Simulated
    logs are supported only for software development.
    """

    def __init__(
        self,
        input_size: int,
        n_diagnoses: int,
        max_sequence_length: int = 16,
        d_model: int = 48,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.input_projection = nn.Linear(input_size, d_model)
        self.position = nn.Parameter(torch.randn(max_sequence_length, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=3 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.diagnosis_head = nn.Linear(d_model, n_diagnoses)
        self.state_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 3))

    def forward(
        self, interactions: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> OperatorStateOutput:
        if interactions.ndim != 3:
            raise ValueError("interactions must have shape [batch, sequence, features]")
        sequence_length = interactions.shape[1]
        if sequence_length > self.max_sequence_length:
            raise ValueError("Interaction sequence is longer than configured maximum")
        hidden = self.input_projection(interactions)
        hidden = hidden + self.position[None, :sequence_length]
        hidden = self.encoder(hidden, src_key_padding_mask=padding_mask)
        if padding_mask is None:
            pooled = hidden[:, -1]
        else:
            lengths = (~padding_mask).sum(dim=1).clamp_min(1) - 1
            pooled = hidden[torch.arange(len(hidden), device=hidden.device), lengths]
        state = torch.sigmoid(self.state_head(pooled))
        return OperatorStateOutput(
            diagnosis_logits=self.diagnosis_head(pooled),
            confidence=state[:, 0],
            reliance=state[:, 1],
            workload=state[:, 2],
            embedding=pooled,
        )
