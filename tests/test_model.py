from __future__ import annotations

import torch

from coact_cps.models.temporal_graph import (
    PhysicsInvariant,
    TemporalHeterogeneousGraphTransformer,
    physics_invariant_loss,
)


def test_temporal_graph_forward_in_both_modes() -> None:
    x = torch.randn(4, 12, 5, 1)
    adjacency = torch.ones(2, 5, 5) - torch.eye(5).unsqueeze(0)
    for mode in ("supervised", "normal_only"):
        model = TemporalHeterogeneousGraphTransformer(
            n_nodes=5,
            in_features=1,
            n_relations=2,
            max_window_size=12,
            d_model=16,
            n_heads=4,
            n_layers=1,
            detection_mode=mode,
        )
        output = model(x, adjacency)
        assert output.attack_logit.shape == (4,)
        assert output.root_logits.shape == (4, 5)
        attack, roots = model.predict_point(x, adjacency)
        assert attack.shape == (4,)
        assert roots.shape == (4, 5)
        assert torch.all((attack >= 0) & (attack <= 1))


def test_physics_loss_is_finite() -> None:
    reconstructed = torch.randn(3, 10, 4, 1)
    invariant = PhysicsInvariant("mass_balance", 0, 1, 2, 0.1)
    loss = physics_invariant_loss(reconstructed, [invariant])
    assert loss.ndim == 0
    assert torch.isfinite(loss)
