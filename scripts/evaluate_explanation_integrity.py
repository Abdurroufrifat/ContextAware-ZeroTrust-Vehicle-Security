from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from coact_cps.data.windowing import WindowArrays
from coact_cps.genai.grounded_generator import GroundedExplanationGenerator
from coact_cps.models.loading import load_detector_state
from coact_cps.models.temporal_graph import TemporalHeterogeneousGraphTransformer
from coact_cps.pipeline import CoActPipeline
from coact_cps.security.attacks import ATTACKS, apply_explanation_attack
from coact_cps.training import physics_invariants_from_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the independent explanation-integrity gate"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--adjacency", required=True)
    parser.add_argument("--windows", required=True)
    parser.add_argument("--output", default="artifacts/integrity_metrics.json")
    parser.add_argument("--max-windows", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = TemporalHeterogeneousGraphTransformer(**checkpoint["model_parameters"])
    load_detector_state(model, checkpoint["state_dict"])
    adjacency = torch.from_numpy(np.load(args.adjacency).astype(np.float32))
    arrays = WindowArrays.load(args.windows)
    process = checkpoint.get("process_settings", {})
    pipeline = CoActPipeline(
        model,
        adjacency,
        checkpoint["feature_names"],
        generator=GroundedExplanationGenerator(mode="template"),
        physics_invariants=physics_invariants_from_checkpoint(checkpoint),
        invariant_threshold=float(process.get("invariant_threshold", 0.05)),
        require_process_grounding=bool(process.get("require_process_grounding", False)),
        counterfactual_minimum=float(process.get("counterfactual_minimum", 0.0)),
    )
    candidates = np.arange(len(arrays.y))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(candidates)
    candidates = candidates[: min(args.max_windows, len(candidates))]
    clean_flagged = 0
    attack_flagged = {attack: 0 for attack in ATTACKS}
    attack_statuses = {attack: {"VERIFIED": 0, "WARNING": 0, "ABSTAIN": 0} for attack in ATTACKS}
    verification_times = []

    for index in candidates:
        bundle = pipeline.build_evidence_bundle(
            arrays.x[index], event_id=f"integrity-{index}", mc_samples=8
        )
        clean = pipeline.generator.generate(bundle)
        started = time.perf_counter()
        clean_result = pipeline.verifier.verify(bundle, clean)
        verification_times.append(time.perf_counter() - started)
        clean_flagged += int(clean_result.status != "VERIFIED")
        for attack in ATTACKS:
            manipulated = apply_explanation_attack(clean, attack)
            started = time.perf_counter()
            result = pipeline.verifier.verify(bundle, manipulated)
            verification_times.append(time.perf_counter() - started)
            attack_statuses[attack][result.status] += 1
            attack_flagged[attack] += int(result.status != "VERIFIED")

    total = max(len(candidates), 1)
    by_attack = {}
    for attack in ATTACKS:
        detection_rate = attack_flagged[attack] / total
        by_attack[attack] = {
            "integrity_detection_rate": detection_rate,
            "manipulation_attack_success_rate": 1.0 - detection_rate,
            "statuses": attack_statuses[attack],
        }
    metrics = {
        "windows": len(candidates),
        "clean_explanation_false_alarm_rate": clean_flagged / total,
        "by_attack": by_attack,
        "mean_verification_overhead_ms": float(np.mean(verification_times) * 1000.0),
        "p95_verification_overhead_ms": float(np.quantile(verification_times, 0.95) * 1000.0),
        "warning": (
            "These are controlled channel transformations, not an exhaustive adaptive "
            "red-team evaluation."
        ),
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
