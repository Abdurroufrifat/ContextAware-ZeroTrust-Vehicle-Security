from __future__ import annotations

import warnings

from .temporal_graph import TemporalHeterogeneousGraphTransformer


def load_detector_state(
    model: TemporalHeterogeneousGraphTransformer, state_dict: dict
) -> None:
    """Load current or pre-node-calibration checkpoints safely."""

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    allowed_missing = {"node_score_scale"}
    invalid_missing = set(missing) - allowed_missing
    if invalid_missing or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch; missing={sorted(invalid_missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    if "node_score_scale" in missing:
        warnings.warn(
            "Legacy checkpoint has no per-node residual calibration; using unit scales.",
            stacklevel=2,
        )
