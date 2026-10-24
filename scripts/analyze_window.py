from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from coact_cps.data.windowing import WindowArrays
from coact_cps.genai.grounded_generator import GroundedExplanationGenerator
from coact_cps.models.loading import load_detector_state
from coact_cps.models.temporal_graph import TemporalHeterogeneousGraphTransformer
from coact_cps.pipeline import CoActPipeline
from coact_cps.training import physics_invariants_from_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze one prepared CPS window")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--adjacency", required=True)
    parser.add_argument("--windows", required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--generator", choices=("template", "hf"), default="template")
    parser.add_argument("--model-id", default="Qwen/Qwen3-4B-Instruct-2507")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = TemporalHeterogeneousGraphTransformer(**checkpoint["model_parameters"])
    load_detector_state(model, checkpoint["state_dict"])
    adjacency = torch.from_numpy(np.load(args.adjacency).astype(np.float32))
    arrays = WindowArrays.load(args.windows)
    if not 0 <= args.index < len(arrays.y):
        raise IndexError(f"Window index must be from 0 to {len(arrays.y) - 1}")
    process = checkpoint.get("process_settings", {})
    pipeline = CoActPipeline(
        model,
        adjacency,
        checkpoint["feature_names"],
        generator=GroundedExplanationGenerator(args.generator, args.model_id),
        physics_invariants=physics_invariants_from_checkpoint(checkpoint),
        invariant_threshold=float(process.get("invariant_threshold", 0.05)),
        require_process_grounding=bool(process.get("require_process_grounding", False)),
        counterfactual_minimum=float(process.get("counterfactual_minimum", 0.0)),
    )
    result = pipeline.analyze(arrays.x[args.index], event_id=f"window-{args.index}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
