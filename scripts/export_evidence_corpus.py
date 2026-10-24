from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from coact_cps.data.windowing import WindowArrays
from coact_cps.genai.grounded_generator import GroundedExplanationGenerator
from coact_cps.models.loading import load_detector_state
from coact_cps.models.temporal_graph import TemporalHeterogeneousGraphTransformer
from coact_cps.pipeline import CoActPipeline
from coact_cps.training import physics_invariants_from_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export closed-evidence prompt/response records for grounded LLM tuning"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--adjacency", required=True)
    parser.add_argument("--windows", required=True)
    parser.add_argument("--output", default="data/processed/explanations.jsonl")
    parser.add_argument("--max-records", type=int, default=2000)
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = TemporalHeterogeneousGraphTransformer(**checkpoint["model_parameters"])
    load_detector_state(model, checkpoint["state_dict"])
    adjacency = torch.from_numpy(np.load(args.adjacency).astype(np.float32))
    arrays = WindowArrays.load(args.windows)
    generator = GroundedExplanationGenerator(mode="template")
    process = checkpoint.get("process_settings", {})
    pipeline = CoActPipeline(
        model,
        adjacency,
        checkpoint["feature_names"],
        generator=generator,
        physics_invariants=physics_invariants_from_checkpoint(checkpoint),
        invariant_threshold=float(process.get("invariant_threshold", 0.05)),
        require_process_grounding=bool(process.get("require_process_grounding", False)),
        counterfactual_minimum=float(process.get("counterfactual_minimum", 0.0)),
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with destination.open("w", encoding="utf-8") as handle:
        for index in range(0, len(arrays.y), max(args.stride, 1)):
            bundle = pipeline.build_evidence_bundle(
                arrays.x[index], event_id=f"corpus-{index}", mc_samples=8
            )
            ideal = generator.generate(bundle)
            record = {
                "event_id": bundle.event_id,
                "prompt": generator.build_prompt(bundle),
                "response": json.dumps(ideal.to_dict(), ensure_ascii=False),
                "source_label": int(arrays.y[index]),
                "warning": "Template target; review and annotate before scientific use.",
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
            if count >= args.max_records:
                break
    print(f"Exported {count} records to {destination}")
    print("Human/domain-expert review is required before treating this as a research corpus.")


if __name__ == "__main__":
    main()
