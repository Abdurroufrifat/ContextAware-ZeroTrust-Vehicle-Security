from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from coact_cps.config import load_config
from coact_cps.data.topology import Topology
from coact_cps.data.windowing import WindowArrays
from coact_cps.models.temporal_graph import PhysicsInvariant, TemporalHeterogeneousGraphTransformer
from coact_cps.training import (
    evaluate_model,
    save_detector_checkpoint,
    save_metrics,
    seed_everything,
    train_detector,
)


def _fully_connected_topology(features: list[str]) -> Topology:
    edges = tuple(
        (source, target) for source in features for target in features if source != target
    )
    return Topology(nodes=tuple(features), relations={"statistical": edges})


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the CO-ACT-CPS graph-temporal detector")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    root = config_path.parent.parent
    data_directory = Path(config["dataset"]["output_directory"])
    if not data_directory.is_absolute():
        data_directory = root / data_directory
    metadata = json.loads((data_directory / "metadata.json").read_text(encoding="utf-8"))
    features = list(metadata["feature_names"])
    train = WindowArrays.load(str(data_directory / "train.npz"))
    val = WindowArrays.load(str(data_directory / "val.npz"))
    test = WindowArrays.load(str(data_directory / "test.npz"))

    topology_config = config.get("topology", {})
    topology_file = topology_config.get("file")
    has_explicit_topology = bool(topology_file or topology_config.get("nodes"))
    if topology_file:
        topology_path = Path(topology_file)
        if not topology_path.is_absolute():
            topology_path = root / topology_path
        topology = Topology.from_json(topology_path)
        topology.validate_feature_order(features)
        print(f"Using process topology: {topology_path}")
        print(
            "Topology relations: "
            + ", ".join(
                f"{name}={len(edges)}" for name, edges in topology.relations.items()
            )
        )
    elif has_explicit_topology:
        topology = Topology.from_dict(topology_config)
        topology.validate_feature_order(features)
    else:
        topology = _fully_connected_topology(features)
        print(
            "WARNING: using a fully connected statistical graph. Add physical/control "
            "edges to the YAML before publication experiments."
        )
    adjacency = torch.from_numpy(topology.relation_adjacency(bidirectional=True))
    model_config = dict(config.get("model", {}))
    parameters = {
        "n_nodes": len(features),
        "in_features": int(train.x.shape[-1]),
        "n_relations": len(topology.relations),
        "max_window_size": int(train.x.shape[1]),
        "d_model": int(model_config.get("d_model", 64)),
        "n_heads": int(model_config.get("n_heads", 4)),
        "n_layers": int(model_config.get("n_layers", 2)),
        "dropout": float(model_config.get("dropout", 0.15)),
        "detection_mode": str(model_config.get("detection_mode", "normal_only")),
        "score_top_k": int(model_config.get("score_top_k", max(1, round(len(features) * 0.10)))),
    }
    model = TemporalHeterogeneousGraphTransformer(**parameters)
    node_index = {name: idx for idx, name in enumerate(features)}
    invariants = []
    for item in config.get("physics_invariants", []):
        invariants.append(
            PhysicsInvariant(
                kind=item["kind"],
                state_node=node_index[item["state_node"]],
                inflow_node=node_index[item["inflow_node"]],
                outflow_node=node_index[item["outflow_node"]],
                coefficient=float(item["coefficient"]),
            )
        )
    training = config.get("training", {})
    training_batch_size = int(training.get("batch_size", 64))
    seed_everything(int(training.get("seed", 7)))
    model, history = train_detector(
        model,
        train,
        val,
        adjacency,
        epochs=int(training.get("epochs", 20)),
        batch_size=training_batch_size,
        learning_rate=float(training.get("learning_rate", 2e-4)),
        weight_decay=float(training.get("weight_decay", 1e-4)),
        physics_invariants=invariants,
        physics_weight=float(training.get("physics_weight", 0.05)),
        reconstruction_weight=float(training.get("reconstruction_weight", 0.05)),
        root_weight=float(training.get("root_weight", 0.25)),
        mask_probability=float(training.get("mask_probability", 0.15)),
        calibration_quantile=float(training.get("calibration_quantile", 0.995)),
        node_scale_quantile=(
            None
            if training.get("node_scale_quantile") is None
            else float(training["node_scale_quantile"])
        ),
        lower_reference_probability=(
            None
            if training.get("lower_reference_probability") is None
            else float(training["lower_reference_probability"])
        ),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics = evaluate_model(
        model,
        test,
        adjacency.to(device),
        device,
        batch_size=training_batch_size,
    )
    artifact_directory = Path(config.get("artifacts", {}).get("directory", "artifacts/real"))
    if not artifact_directory.is_absolute():
        artifact_directory = root / artifact_directory
    artifact_directory.mkdir(parents=True, exist_ok=True)
    save_detector_checkpoint(
        model,
        artifact_directory / "detector.pt",
        parameters,
        features,
        list(topology.relation_names),
        history,
        physics_invariants=invariants,
        process_settings={
            "invariant_threshold": float(
                config.get("process_grounding", {}).get("invariant_threshold", 0.05)
            ),
            "require_process_grounding": bool(
                config.get("process_grounding", {}).get("require", has_explicit_topology)
            ),
            "counterfactual_minimum": float(
                config.get("process_grounding", {}).get("counterfactual_minimum", 0.0)
            ),
        },
    )
    save_metrics(metrics, artifact_directory / "test_metrics.json")
    np.save(artifact_directory / "adjacency.npy", adjacency.numpy())
    print(json.dumps(metrics, indent=2, allow_nan=True))
    print(f"Saved trained artifacts in {artifact_directory}")


if __name__ == "__main__":
    main()
