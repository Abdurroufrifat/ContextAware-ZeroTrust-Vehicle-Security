from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data.windowing import CPSWindowDataset, WindowArrays
from .metrics import attack_episode_metrics, binary_detection_metrics, root_cause_hit_at_k
from .models.temporal_graph import (
    PhysicsInvariant,
    TemporalHeterogeneousGraphTransformer,
    physics_invariant_loss,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def development_stratified_split(
    arrays: WindowArrays, seed: int = 7
) -> tuple[WindowArrays, WindowArrays, WindowArrays]:
    """Development-only split used by the synthetic smoke test.

    Publication experiments must split by complete attack episode and time.
    """

    rng = np.random.default_rng(seed)
    partitions: list[list[int]] = [[], [], []]
    for label in np.unique(arrays.y):
        indices = np.where(arrays.y == label)[0]
        rng.shuffle(indices)
        first = int(len(indices) * 0.60)
        second = int(len(indices) * 0.80)
        for part, values in enumerate((indices[:first], indices[first:second], indices[second:])):
            partitions[part].extend(values.tolist())

    def subset(indices: list[int]) -> WindowArrays:
        selected = np.asarray(sorted(indices), dtype=int)
        return WindowArrays(
            x=arrays.x[selected],
            y=arrays.y[selected],
            root=arrays.root[selected],
            end_indices=arrays.end_indices[selected],
            groups=(None if arrays.groups is None else arrays.groups[selected]),
        )

    return tuple(subset(indices) for indices in partitions)  # type: ignore[return-value]


def _loader(arrays: WindowArrays, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        CPSWindowDataset(arrays),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )


def evaluate_model(
    model: TemporalHeterogeneousGraphTransformer,
    arrays: WindowArrays,
    adjacency: torch.Tensor,
    device: torch.device,
    batch_size: int = 128,
) -> dict[str, float]:
    model.eval()
    probabilities: list[np.ndarray] = []
    roots: list[np.ndarray] = []
    true_y: list[np.ndarray] = []
    true_roots: list[np.ndarray] = []
    with torch.no_grad():
        for batch in _loader(arrays, batch_size, False):
            attack_probability, root_probability = model.predict_point(
                batch["x"].to(device), adjacency
            )
            probabilities.append(attack_probability.cpu().numpy())
            roots.append(root_probability.cpu().numpy())
            true_y.append(batch["y"].numpy())
            true_roots.append(batch["root"].numpy())
    y = np.concatenate(true_y)
    p = np.concatenate(probabilities)
    root_y = np.concatenate(true_roots)
    root_p = np.concatenate(roots)
    metrics = binary_detection_metrics(y, p)
    metrics.update(attack_episode_metrics(y, p))
    metrics["root_hit_at_1"] = root_cause_hit_at_k(root_y, root_p, 1)
    metrics["root_hit_at_3"] = root_cause_hit_at_k(root_y, root_p, min(3, root_p.shape[1]))
    return metrics


@torch.no_grad()
def calibrate_normal_only(
    model: TemporalHeterogeneousGraphTransformer,
    validation: WindowArrays,
    adjacency: torch.Tensor,
    device: torch.device,
    quantile: float = 0.995,
    node_scale_quantile: float | None = None,
    lower_reference_probability: float | None = None,
    batch_size: int = 128,
) -> None:
    """Fit anomaly-score scaling on validation-normal windows only."""

    if model.detection_mode != "normal_only":
        return
    if not 0.90 <= quantile < 1.0:
        raise ValueError("normal-only calibration quantile must be in [0.90, 1.0)")
    if node_scale_quantile is not None and not 0.90 <= node_scale_quantile < 1.0:
        raise ValueError("node scale quantile must be in [0.90, 1.0)")
    if lower_reference_probability is not None and not 0.0 < lower_reference_probability < 0.5:
        raise ValueError("lower reference probability must be between zero and 0.5")
    model.eval()
    raw_node_values: list[np.ndarray] = []
    for batch in _loader(validation, batch_size, False):
        normal = batch["y"] < 0.5
        if not normal.any():
            continue
        _, node_score = model.raw_normal_only_scores(
            batch["x"][normal].to(device), adjacency
        )
        raw_node_values.append(node_score.cpu().numpy())
    if not raw_node_values:
        raise ValueError("Normal-only calibration requires validation-normal windows")
    raw_nodes = np.concatenate(raw_node_values)
    node_scale = (
        np.ones(raw_nodes.shape[1], dtype=float)
        if node_scale_quantile is None
        else np.maximum(np.quantile(raw_nodes, node_scale_quantile, axis=0), 1e-6)
    )
    normalized_nodes = raw_nodes / node_scale[None, :]
    top_count = min(model.score_top_k, model.n_nodes)
    graph = np.sort(normalized_nodes, axis=1)[:, -top_count:].mean(axis=1)
    threshold = float(np.quantile(graph, quantile))
    lower = float(np.quantile(graph, 0.95))
    spread = max(threshold - lower, float(np.std(graph)) * 0.25, 1e-5)
    temperature = (
        spread
        if lower_reference_probability is None
        else spread
        / abs(float(np.log(lower_reference_probability / (1.0 - lower_reference_probability))))
    )
    root_temperature = max(float(np.quantile(normalized_nodes, 0.75)), 1e-5)
    model.node_score_scale.copy_(
        torch.as_tensor(node_scale, dtype=model.node_score_scale.dtype, device=device)
    )
    model.anomaly_threshold.copy_(torch.tensor(threshold, device=device))
    model.anomaly_temperature.copy_(torch.tensor(temperature, device=device))
    model.root_temperature.copy_(torch.tensor(root_temperature, device=device))


def train_detector(
    model: TemporalHeterogeneousGraphTransformer,
    train: WindowArrays,
    val: WindowArrays,
    adjacency: torch.Tensor,
    epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 2e-4,
    weight_decay: float = 1e-4,
    physics_invariants: list[PhysicsInvariant] | None = None,
    physics_weight: float = 0.05,
    reconstruction_weight: float = 0.05,
    root_weight: float = 0.25,
    mask_probability: float = 0.15,
    calibration_quantile: float = 0.995,
    node_scale_quantile: float | None = None,
    lower_reference_probability: float | None = None,
    device: str | None = None,
) -> tuple[TemporalHeterogeneousGraphTransformer, list[dict[str, float]]]:
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(selected_device)
    adjacency = adjacency.to(selected_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    positives = max(float(train.y.sum()), 1.0)
    negatives = max(float(len(train.y) - train.y.sum()), 1.0)
    attack_loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(negatives / positives, device=selected_device)
    )
    root_loss_fn = nn.CrossEntropyLoss()
    reconstruction_loss_fn = nn.SmoothL1Loss()
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in _loader(train, batch_size, True):
            x = batch["x"].to(selected_device)
            y = batch["y"].to(selected_device)
            root = batch["root"].to(selected_device)
            if model.detection_mode == "normal_only":
                normal = y < 0.5
                if not normal.any():
                    continue
                x = x[normal]
                corrupted = x.clone()
                history_mask = torch.rand_like(corrupted) < mask_probability
                history_mask[:, -1] = True
                corrupted[history_mask] = 0.0
                output = model(corrupted, adjacency)
                attack_loss = output.attack_logit.new_tensor(0.0)
                root_loss = output.attack_logit.new_tensor(0.0)
                reconstruction_loss = reconstruction_loss_fn(output.reconstructed[:, -1], x[:, -1])
            else:
                output = model(x, adjacency)
                attack_loss = attack_loss_fn(output.attack_logit, y)
                valid_root = root >= 0
                if valid_root.any():
                    root_loss = root_loss_fn(output.root_logits[valid_root], root[valid_root])
                else:
                    root_loss = output.attack_logit.new_tensor(0.0)
                reconstruction_loss = reconstruction_loss_fn(output.reconstructed, x)
            invariant_loss = physics_invariant_loss(output.reconstructed, physics_invariants or [])
            if model.detection_mode == "normal_only":
                loss = reconstruction_loss + physics_weight * invariant_loss
            else:
                loss = (
                    attack_loss
                    + root_weight * root_loss
                    + reconstruction_weight * reconstruction_loss
                    + physics_weight * invariant_loss
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        calibrate_normal_only(
            model,
            val,
            adjacency,
            selected_device,
            quantile=calibration_quantile,
            node_scale_quantile=node_scale_quantile,
            lower_reference_probability=lower_reference_probability,
            batch_size=batch_size,
        )
        validation = evaluate_model(model, val, adjacency, selected_device, batch_size)
        epoch_result = {
            "epoch": float(epoch),
            "train_loss": float(np.mean(losses)),
            "val_f1": validation["f1"],
            "val_brier": validation["brier"],
        }
        history.append(epoch_result)
        print(
            f"Epoch {epoch}/{epochs} | "
            f"loss={epoch_result['train_loss']:.6f} | "
            f"val_f1={epoch_result['val_f1']:.4f} | "
            f"val_brier={epoch_result['val_brier']:.4f}",
            flush=True,
        )
    return model, history


def save_detector_checkpoint(
    model: TemporalHeterogeneousGraphTransformer,
    path: str | Path,
    model_parameters: dict,
    feature_names: list[str],
    relation_names: list[str],
    history: list[dict[str, float]],
    physics_invariants: list[PhysicsInvariant] | None = None,
    process_settings: dict | None = None,
) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_parameters": model_parameters,
            "feature_names": feature_names,
            "relation_names": relation_names,
            "history": history,
            "physics_invariants": [
                {
                    "kind": item.kind,
                    "state_node": item.state_node,
                    "inflow_node": item.inflow_node,
                    "outflow_node": item.outflow_node,
                    "coefficient": item.coefficient,
                }
                for item in (physics_invariants or [])
            ],
            "process_settings": process_settings or {},
        },
        checkpoint_path,
    )


def physics_invariants_from_checkpoint(checkpoint: dict) -> list[PhysicsInvariant]:
    return [PhysicsInvariant(**item) for item in checkpoint.get("physics_invariants", [])]


def save_metrics(metrics: dict, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(metrics, indent=2, allow_nan=True), encoding="utf-8")
