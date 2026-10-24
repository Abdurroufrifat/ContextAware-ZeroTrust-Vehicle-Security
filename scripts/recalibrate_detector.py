from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from coact_cps.data.windowing import WindowArrays
from coact_cps.models.loading import load_detector_state
from coact_cps.models.temporal_graph import TemporalHeterogeneousGraphTransformer
from coact_cps.training import calibrate_normal_only


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recalibrate a normal-only detector on complete normal validation windows"
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--adjacency", required=True, type=Path)
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--score-top-k", type=int, default=5)
    parser.add_argument("--calibration-quantile", type=float, default=0.99)
    parser.add_argument("--node-scale-quantile", type=float, default=0.99)
    parser.add_argument("--lower-reference-probability", type=float, default=0.05)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    parameters = dict(checkpoint["model_parameters"])
    if parameters.get("detection_mode") != "normal_only":
        raise ValueError("This recalibration command requires a normal-only detector")
    parameters["score_top_k"] = args.score_top_k
    model = TemporalHeterogeneousGraphTransformer(**parameters)
    load_detector_state(model, checkpoint["state_dict"])

    validation = WindowArrays.load(str(args.validation))
    if np.asarray(validation.y).any():
        raise ValueError("Recalibration validation data must contain only normal windows")
    if validation.x.shape[2] != parameters["n_nodes"]:
        raise ValueError("Validation node count does not match the detector")
    adjacency = torch.from_numpy(np.load(args.adjacency))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    calibrate_normal_only(
        model,
        validation,
        adjacency.to(device),
        device,
        quantile=args.calibration_quantile,
        node_scale_quantile=args.node_scale_quantile,
        lower_reference_probability=args.lower_reference_probability,
        batch_size=args.batch_size,
    )

    checkpoint["state_dict"] = model.cpu().state_dict()
    checkpoint["model_parameters"] = parameters
    checkpoint["recalibration"] = {
        "source_checkpoint": str(args.checkpoint),
        "validation_windows": int(len(validation.y)),
        "score_top_k": args.score_top_k,
        "calibration_quantile": args.calibration_quantile,
        "node_scale_quantile": args.node_scale_quantile,
        "lower_reference_probability": args.lower_reference_probability,
        "uses_test_labels": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    result = {
        **checkpoint["recalibration"],
        "anomaly_threshold": float(model.anomaly_threshold),
        "anomaly_temperature": float(model.anomaly_temperature),
        "root_temperature": float(model.root_temperature),
        "minimum_node_scale": float(model.node_score_scale.min()),
        "maximum_node_scale": float(model.node_score_scale.max()),
        "output": str(args.output.resolve()),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
