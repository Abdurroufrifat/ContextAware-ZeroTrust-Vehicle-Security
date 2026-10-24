from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from coact_cps.data.windowing import WindowArrays
from coact_cps.genai.grounded_generator import GroundedExplanationGenerator
from coact_cps.models.loading import load_detector_state
from coact_cps.models.temporal_graph import TemporalHeterogeneousGraphTransformer
from coact_cps.pipeline import CoActPipeline
from coact_cps.security.external_redteam import ATTACK_GOALS, PROTOCOL_VERSION, build_task
from coact_cps.security.verifier import ExplanationVerifier
from coact_cps.training import physics_invariants_from_checkpoint


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def alarmed_indices(
    model: TemporalHeterogeneousGraphTransformer,
    adjacency: torch.Tensor,
    arrays: WindowArrays,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    probabilities = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(arrays.y), batch_size):
            x = torch.as_tensor(
                arrays.x[start : start + batch_size], dtype=torch.float32, device=device
            )
            probability, _ = model.predict_point(x, adjacency)
            probabilities.append(probability.cpu().numpy())
    return np.flatnonzero(np.concatenate(probabilities) >= 0.5)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export label-free frozen tasks for an external LLM explanation red team"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--adjacency", required=True)
    parser.add_argument("--windows", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-windows", type=int, default=100)
    parser.add_argument("--development-fraction", type=float, default=0.20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--mc-samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260829)
    args = parser.parse_args()

    if not 0.0 < args.development_fraction < 0.5:
        raise ValueError("--development-fraction must be greater than 0 and less than 0.5")
    if args.max_windows < 2:
        raise ValueError("--max-windows must be at least 2")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = TemporalHeterogeneousGraphTransformer(**checkpoint["model_parameters"])
    load_detector_state(model, checkpoint["state_dict"])
    arrays = WindowArrays.load(args.windows)
    adjacency = torch.from_numpy(np.load(args.adjacency).astype(np.float32))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    adjacency = adjacency.to(device)
    process = checkpoint.get("process_settings", {})
    template = GroundedExplanationGenerator(mode="template")
    pipeline = CoActPipeline(
        model,
        adjacency,
        checkpoint["feature_names"],
        generator=template,
        physics_invariants=physics_invariants_from_checkpoint(checkpoint),
        invariant_threshold=float(process.get("invariant_threshold", 0.05)),
        require_process_grounding=bool(process.get("require_process_grounding", False)),
        counterfactual_minimum=float(process.get("counterfactual_minimum", 0.0)),
    )

    all_alarmed = alarmed_indices(model, adjacency, arrays, device, args.batch_size)
    selected = all_alarmed.copy()
    rng = np.random.default_rng(args.seed)
    rng.shuffle(selected)
    selected = selected[: min(args.max_windows, len(selected))]
    if len(selected) < 2:
        raise ValueError(f"Only {len(selected)} model-alarmed windows are available")
    development_count = max(1, int(round(len(selected) * args.development_fraction)))

    tasks = []
    for position, index in enumerate(selected):
        split = "development" if position < development_count else "held_out"
        bundle = pipeline.build_evidence_bundle(
            arrays.x[index], event_id=f"external-{int(index)}", mc_samples=args.mc_samples
        )
        clean = template.generate(bundle)
        tasks.append(
            build_task(
                bundle,
                clean,
                window_index=int(index),
                split=split,
                goal_index=position,
            )
        )

    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    development = [task for task in tasks if task["split"] == "development"]
    held_out = [task for task in tasks if task["split"] == "held_out"]
    write_jsonl(destination / "development_tasks.jsonl", development)
    write_jsonl(destination / "heldout_tasks.jsonl", held_out)
    development_path = destination / "development_tasks.jsonl"
    heldout_path = destination / "heldout_tasks.jsonl"

    verifier_path = Path(inspect.getsourcefile(ExplanationVerifier) or "")
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection_rule": "model_alarm_probability_greater_than_or_equal_to_0.5",
        "uses_ground_truth_labels_for_selection": False,
        "ground_truth_labels_exported": False,
        "available_model_alarmed_windows": int(len(all_alarmed)),
        "selected_windows": int(len(tasks)),
        "development_tasks": len(development),
        "heldout_tasks": len(held_out),
        "primary_split": "held_out",
        "attack_goals": [goal_id for goal_id, _ in ATTACK_GOALS],
        "seed": args.seed,
        "mc_samples": args.mc_samples,
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "adjacency_sha256": file_sha256(args.adjacency),
        "windows_sha256": file_sha256(args.windows),
        "verifier_sha256": file_sha256(verifier_path),
        "development_tasks_sha256": file_sha256(development_path),
        "heldout_tasks_sha256": file_sha256(heldout_path),
        "freeze_rule": (
            "Do not change the verifier, task prompts, model, or decoding settings after "
            "inspecting held-out outcomes."
        ),
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    print(f"Saved frozen external red-team tasks in {destination.resolve()}")


if __name__ == "__main__":
    main()
