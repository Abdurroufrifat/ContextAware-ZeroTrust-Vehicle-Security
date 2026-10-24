from __future__ import annotations

from coact_cps.baselines import evaluate_isolation_forest, evaluate_persistence
from coact_cps.data.synthetic import SyntheticCPSConfig, generate_synthetic_cps
from coact_cps.data.windowing import make_windows
from coact_cps.training import development_stratified_split


def test_calibrated_baselines_return_detection_metrics() -> None:
    frame, topology = generate_synthetic_cps(SyntheticCPSConfig(n_steps=400, seed=9))
    arrays = make_windows(
        frame.loc[:, topology.nodes].to_numpy(),
        frame["attack"].to_numpy(),
        None,
        window_size=24,
        stride=4,
        label_rule="any",
    )
    train, validation, test = development_stratified_split(arrays, seed=9)
    persistence = evaluate_persistence(validation, test, quantile=0.95)
    isolation = evaluate_isolation_forest(train, validation, test, quantile=0.95, seed=9)
    for metrics in (persistence, isolation):
        assert 0.0 <= metrics["f1"] <= 1.0
        assert "attack_episode_recall" in metrics
