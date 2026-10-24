import numpy as np
import torch

from coact_cps.data.windowing import WindowArrays
from coact_cps.models.loading import load_detector_state
from coact_cps.models.temporal_graph import TemporalHeterogeneousGraphTransformer
from coact_cps.training import calibrate_normal_only


def _model(score_top_k: int = 2) -> TemporalHeterogeneousGraphTransformer:
    return TemporalHeterogeneousGraphTransformer(
        n_nodes=4,
        in_features=1,
        n_relations=1,
        max_window_size=3,
        d_model=8,
        n_heads=2,
        n_layers=1,
        dropout=0.0,
        detection_mode="normal_only",
        score_top_k=score_top_k,
    )


def test_node_score_scale_changes_residual_ranking() -> None:
    model = _model()
    model.node_score_scale.copy_(torch.tensor([10.0, 1.0, 1.0, 1.0]))
    raw = torch.tensor([[5.0, 2.0, 1.0, 0.0]])
    normalized = raw / model.node_score_scale
    assert int(torch.argmax(raw, dim=1)) == 0
    assert int(torch.argmax(normalized, dim=1)) == 1
    assert np.isclose(float(torch.topk(normalized, 2).values.mean()), 1.5)


def test_legacy_checkpoint_load_allows_missing_node_scale() -> None:
    model = _model()
    legacy = {key: value for key, value in model.state_dict().items() if key != "node_score_scale"}
    restored = _model()
    load_detector_state(restored, legacy)
    assert torch.equal(restored.node_score_scale, torch.ones(4))


def test_normal_calibration_learns_positive_per_node_scales() -> None:
    rng = np.random.default_rng(9)
    arrays = WindowArrays(
        x=rng.normal(size=(12, 3, 4, 1)).astype(np.float32),
        y=np.zeros(12, dtype=np.int64),
        root=np.full(12, -1, dtype=np.int64),
        end_indices=np.arange(12),
        groups=np.zeros(12, dtype=np.int64),
    )
    model = _model()
    adjacency = torch.ones(1, 4, 4) - torch.eye(4).unsqueeze(0)
    calibrate_normal_only(
        model,
        arrays,
        adjacency,
        torch.device("cpu"),
        quantile=0.90,
        node_scale_quantile=0.90,
        batch_size=4,
    )
    assert torch.all(model.node_score_scale > 0)
    assert float(model.anomaly_threshold) > 0
