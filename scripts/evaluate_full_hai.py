from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from coact_cps.data.windowing import CPSWindowDataset, WindowArrays
from coact_cps.full_evaluation import (
    grouped_episode_metrics,
    target_hit_at_k,
    validate_regular_windows,
)
from coact_cps.metrics import binary_detection_metrics
from coact_cps.models.loading import load_detector_state
from coact_cps.models.temporal_graph import TemporalHeterogeneousGraphTransformer


def predict(
    checkpoint_path: Path,
    adjacency_path: Path,
    arrays: WindowArrays,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = TemporalHeterogeneousGraphTransformer(**checkpoint["model_parameters"])
    load_detector_state(model, checkpoint["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    adjacency = torch.from_numpy(np.load(adjacency_path)).to(device)

    attack_probabilities = []
    root_probabilities = []
    loader = DataLoader(CPSWindowDataset(arrays), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for number, batch in enumerate(loader, start=1):
            attack, roots = model.predict_point(batch["x"].to(device), adjacency)
            attack_probabilities.append(attack.cpu().numpy())
            root_probabilities.append(roots.cpu().numpy())
            if number == 1 or number % 100 == 0 or number == len(loader):
                print(f"Inference batch {number}/{len(loader)}", flush=True)
    return np.concatenate(attack_probabilities), np.concatenate(root_probabilities)


def official_etapr(y_true: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    if importlib.util.find_spec("eTaPR_pkg") is None:
        raise ModuleNotFoundError(
            "Install the bundled official metric first: "
            "python -m pip install --no-deps vendor/eTaPR-22.6.1-py3-none-any.whl"
        )
    if "cv2" not in sys.modules and importlib.util.find_spec("cv2") is None:
        sys.modules["cv2"] = types.ModuleType("cv2")
    from eTaPR_pkg import etapr

    result = etapr.evaluate_w_streams(
        y_true.astype(int).tolist(),
        predictions.astype(int).tolist(),
        theta_p=0.5,
        theta_r=0.1,
        delta=0.0,
    )
    keys = ("eTaP", "eTaR", "eTaPd", "eTaRd", "eTaPp", "eTaRp", "f1")
    return {f"etapr_{key.lower()}": float(result[key]) for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a detector on complete chronological HAI 23.05 test windows"
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--adjacency", required=True, type=Path)
    parser.add_argument("--windows", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--episodes-output", required=True, type=Path)
    parser.add_argument("--predictions-output", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("threshold must be between zero and one")

    arrays = WindowArrays.load(str(args.windows))
    if arrays.groups is None:
        raise ValueError("Full HAI evaluation requires source groups")
    stride = validate_regular_windows(arrays.groups, arrays.end_indices)
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    feature_names = list(metadata["feature_names"])
    feature_index = {name: index for index, name in enumerate(feature_names)}
    if arrays.x.shape[2] != len(feature_names):
        raise ValueError("Window node count does not match metadata feature count")

    probabilities, root_probabilities = predict(
        args.checkpoint, args.adjacency, arrays, args.batch_size
    )
    predictions = probabilities >= args.threshold
    episode_metrics, episodes = grouped_episode_metrics(
        arrays.y, probabilities, arrays.groups, args.threshold
    )

    catalog = pd.read_csv(args.catalog, keep_default_na=False)
    expected = [14, 38]
    group_order = list(dict.fromkeys(int(value) for value in arrays.groups))
    if len(group_order) != 2:
        raise ValueError(f"Expected two HAI test groups, found {len(group_order)}")
    episodes_by_group = [sum(item.group == group for item in episodes) for group in group_order]
    if episodes_by_group != expected or len(catalog) != 52:
        raise ValueError(
            f"Expected HAI episode structure {expected} and 52 catalog rows; "
            f"received {episodes_by_group} and {len(catalog)}"
        )

    episode_scores = []
    target_indices = []
    detected = []
    episode_rows = []
    for episode, (_, truth) in zip(episodes, catalog.iterrows(), strict=True):
        scores = root_probabilities[episode.start : episode.end].max(axis=0)
        observed_targets = tuple(
            value for value in truth["observable_target_points"].split(";") if value
        )
        indices = tuple(feature_index[value] for value in observed_targets)
        hit_positions = np.flatnonzero(predictions[episode.start : episode.end])
        was_detected = bool(len(hit_positions))
        ranking = np.argsort(scores)[::-1]
        best_rank = (
            min(int(np.where(ranking == index)[0][0]) + 1 for index in indices)
            if indices
            else ""
        )
        episode_scores.append(scores)
        target_indices.append(indices)
        detected.append(was_detected)
        episode_rows.append(
            {
                "attack_id": truth["attack_id"],
                "label_file": truth["label_file"],
                "target_controllers": truth["target_controllers"],
                "target_points": truth["target_points"],
                "observable_target_points": truth["observable_target_points"],
                "detected": was_detected,
                "detection_delay_windows": int(hit_positions[0]) if was_detected else "",
                "best_true_target_rank": best_rank,
                "top_1_predicted_node": feature_names[int(ranking[0])],
                "top_3_predicted_nodes": ";".join(
                    feature_names[int(index)] for index in ranking[:3]
                ),
                "hit_at_1": bool(indices and set(indices).intersection(ranking[:1])),
                "hit_at_3": bool(indices and set(indices).intersection(ranking[:3])),
            }
        )

    episode_scores_array = np.stack(episode_scores)
    detected_array = np.asarray(detected)
    metrics = binary_detection_metrics(arrays.y, probabilities)
    metrics.update(episode_metrics)
    metrics.update(target_hit_at_k(episode_scores_array, target_indices, detected_array, 1))
    metrics.update(target_hit_at_k(episode_scores_array, target_indices, detected_array, 3))
    metrics.update(official_etapr(arrays.y, predictions))
    metrics.update(
        {
            "decision_threshold": args.threshold,
            "inferred_stride_rows": stride,
            "chronological_windows": int(len(arrays.y)),
            "hidden_target_episodes": int(sum(not targets for targets in target_indices)),
            "evaluation_scope": "full_chronological_test_at_window_resolution",
            "publication_warning": (
                "eTaPR is computed at the prepared window resolution. For comparison with "
                "raw-second HAI studies, prepare and evaluate the publication stride separately."
            ),
        }
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2, allow_nan=True), encoding="utf-8")
    args.episodes_output.parent.mkdir(parents=True, exist_ok=True)
    with args.episodes_output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(episode_rows[0]))
        writer.writeheader()
        writer.writerows(episode_rows)
    if args.predictions_output:
        args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.predictions_output,
            y=arrays.y,
            probabilities=probabilities,
            predictions=predictions,
            root_probabilities=root_probabilities,
            groups=arrays.groups,
            end_indices=arrays.end_indices,
        )

    print(json.dumps(metrics, indent=2, allow_nan=True))
    print(f"Saved metrics: {args.output.resolve()}")
    print(f"Saved episode analysis: {args.episodes_output.resolve()}")


if __name__ == "__main__":
    main()
