from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(slots=True)
class TemporalGraphOutput:
    attack_logit: torch.Tensor
    root_logits: torch.Tensor
    node_embeddings: torch.Tensor
    graph_embedding: torch.Tensor
    reconstructed: torch.Tensor


@dataclass(frozen=True)
class PhysicsInvariant:
    kind: str
    state_node: int
    inflow_node: int
    outflow_node: int
    coefficient: float


class RelationalGraphMixer(nn.Module):
    def __init__(self, d_model: int, n_relations: int, dropout: float):
        super().__init__()
        self.relation_linears = nn.ModuleList(
            nn.Linear(d_model, d_model, bias=False) for _ in range(n_relations)
        )
        self.gate = nn.Linear(2 * d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        if adjacency.ndim != 3:
            raise ValueError("adjacency must have shape [relations, nodes, nodes]")
        if adjacency.shape[0] != len(self.relation_linears):
            raise ValueError("Relation count does not match the model")
        messages = torch.zeros_like(hidden)
        for relation_idx, projection in enumerate(self.relation_linears):
            matrix = adjacency[relation_idx]
            degree = matrix.sum(dim=-1, keepdim=True).clamp_min(1.0)
            normalized = matrix / degree
            aggregated = torch.einsum("ij,btjd->btid", normalized, hidden)
            messages = messages + projection(aggregated)
        gate = torch.sigmoid(self.gate(torch.cat([hidden, messages], dim=-1)))
        mixed = gate * messages + (1.0 - gate) * hidden
        return self.norm(hidden + self.dropout(mixed))


class TemporalHeterogeneousGraphTransformer(nn.Module):
    """Typed graph mixing plus a temporal Transformer for CPS telemetry windows."""

    def __init__(
        self,
        n_nodes: int,
        in_features: int,
        n_relations: int,
        max_window_size: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.15,
        detection_mode: str = "supervised",
        score_top_k: int | None = None,
    ):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if detection_mode not in {"supervised", "normal_only"}:
            raise ValueError("detection_mode must be 'supervised' or 'normal_only'")
        self.n_nodes = n_nodes
        self.in_features = in_features
        self.max_window_size = max_window_size
        self.detection_mode = detection_mode
        self.score_top_k = (
            max(1, int(round(n_nodes * 0.10))) if score_top_k is None else int(score_top_k)
        )
        if not 1 <= self.score_top_k <= n_nodes:
            raise ValueError("score_top_k must be between one and n_nodes")
        self.register_buffer("anomaly_threshold", torch.tensor(0.0))
        self.register_buffer("anomaly_temperature", torch.tensor(1.0))
        self.register_buffer("root_temperature", torch.tensor(1.0))
        self.register_buffer("node_score_scale", torch.ones(n_nodes))
        self.input_projection = nn.Linear(in_features, d_model)
        self.node_embedding = nn.Parameter(torch.randn(n_nodes, d_model) * 0.02)
        self.time_embedding = nn.Parameter(torch.randn(max_window_size, d_model) * 0.02)
        self.graph_before = RelationalGraphMixer(d_model, n_relations, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.graph_after = RelationalGraphMixer(d_model, n_relations, dropout)
        self.attention_pool = nn.Linear(d_model, 1)
        self.attack_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, 1)
        )
        self.root_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, 1)
        )
        self.state_decoder = nn.Linear(d_model, in_features)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> TemporalGraphOutput:
        if x.ndim != 4:
            raise ValueError("x must have shape [batch, time, nodes, features]")
        batch, time, nodes, features = x.shape
        if nodes != self.n_nodes or features != self.in_features:
            raise ValueError(
                f"Expected {self.n_nodes} nodes and {self.in_features} features; "
                f"received {nodes} and {features}"
            )
        if time > self.max_window_size:
            raise ValueError("Input window exceeds max_window_size")

        hidden = self.input_projection(x)
        hidden = hidden + self.node_embedding[None, None, :, :]
        hidden = hidden + self.time_embedding[None, :time, None, :]
        hidden = self.graph_before(hidden, adjacency)

        temporal = hidden.permute(0, 2, 1, 3).reshape(batch * nodes, time, -1)
        temporal = self.temporal_encoder(temporal)
        hidden = temporal.reshape(batch, nodes, time, -1).permute(0, 2, 1, 3)
        hidden = self.graph_after(hidden, adjacency)

        final_nodes = hidden[:, -1]
        weights = torch.softmax(self.attention_pool(final_nodes).squeeze(-1), dim=-1)
        graph_embedding = torch.sum(final_nodes * weights.unsqueeze(-1), dim=1)
        return TemporalGraphOutput(
            attack_logit=self.attack_head(graph_embedding).squeeze(-1),
            root_logits=self.root_head(final_nodes).squeeze(-1),
            node_embeddings=final_nodes,
            graph_embedding=graph_embedding,
            reconstructed=self.state_decoder(hidden),
        )

    def raw_normal_only_scores(
        self, x: torch.Tensor, adjacency: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict the held-out final state and return uncalibrated node residuals."""

        masked = x.clone()
        masked[:, -1] = 0.0
        output = self(masked, adjacency)
        node_scores = torch.mean(torch.abs(output.reconstructed[:, -1] - x[:, -1]), dim=-1)
        graph_scores = torch.topk(
            node_scores, k=min(self.score_top_k, self.n_nodes), dim=-1
        ).values.mean(dim=-1)
        return graph_scores, node_scores

    def normal_only_scores(
        self, x: torch.Tensor, adjacency: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return node-normalized residual scores and their graph aggregation."""

        _, raw_node_scores = self.raw_normal_only_scores(x, adjacency)
        node_scores = raw_node_scores / self.node_score_scale.clamp_min(1e-6)
        graph_scores = torch.topk(
            node_scores, k=min(self.score_top_k, self.n_nodes), dim=-1
        ).values.mean(dim=-1)
        return graph_scores, node_scores

    def predict_point(
        self, x: torch.Tensor, adjacency: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.detection_mode == "normal_only":
            graph_scores, node_scores = self.normal_only_scores(x, adjacency)
            scale = self.anomaly_temperature.clamp_min(1e-6)
            attack_probability = torch.sigmoid((graph_scores - self.anomaly_threshold) / scale)
            root_probability = torch.softmax(
                node_scores / self.root_temperature.clamp_min(1e-6), dim=-1
            )
            return attack_probability, root_probability
        output = self(x, adjacency)
        return torch.sigmoid(output.attack_logit), torch.softmax(output.root_logits, dim=-1)

    @torch.no_grad()
    def predict_with_uncertainty(
        self, x: torch.Tensor, adjacency: torch.Tensor, samples: int = 12
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Monte-Carlo dropout estimate for attack probability and root probability."""

        was_training = self.training
        self.train()
        attack_samples = []
        root_samples = []
        for _ in range(max(samples, 2)):
            attack_probability, root_probability = self.predict_point(x, adjacency)
            attack_samples.append(attack_probability)
            root_samples.append(root_probability)
        if not was_training:
            self.eval()
        attack_stack = torch.stack(attack_samples)
        root_stack = torch.stack(root_samples)
        return (
            attack_stack.mean(dim=0),
            attack_stack.std(dim=0),
            root_stack.mean(dim=0),
        )


def physics_invariant_loss(
    reconstructed: torch.Tensor, invariants: list[PhysicsInvariant]
) -> torch.Tensor:
    if not invariants:
        return reconstructed.new_tensor(0.0)
    losses = []
    for invariant in invariants:
        if invariant.kind != "mass_balance":
            raise ValueError(f"Unsupported invariant: {invariant.kind}")
        state = reconstructed[:, :, invariant.state_node, 0]
        inflow = reconstructed[:, 1:, invariant.inflow_node, 0]
        outflow = reconstructed[:, 1:, invariant.outflow_node, 0]
        observed_delta = state[:, 1:] - state[:, :-1]
        expected_delta = invariant.coefficient * (inflow - outflow)
        losses.append(torch.mean((observed_delta - expected_delta) ** 2))
    return torch.stack(losses).mean()
