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
from coact_cps.security.adaptive_redteam import (
    generate_adaptive_candidates,
    preserves_committed_facts,
)
from coact_cps.security.verifier import ExplanationVerifier
from coact_cps.training import physics_invariants_from_checkpoint


def _alarmed_indices(
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
        description="Adaptive truth-preserving explanation-channel red-team evaluation"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--adjacency", required=True)
    parser.add_argument("--windows", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-windows", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--mc-samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = TemporalHeterogeneousGraphTransformer(**checkpoint["model_parameters"])
    load_detector_state(model, checkpoint["state_dict"])
    arrays = WindowArrays.load(args.windows)
    adjacency = torch.from_numpy(np.load(args.adjacency).astype(np.float32))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    adjacency = adjacency.to(device)
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

    alarmed = _alarmed_indices(model, adjacency, arrays, device, args.batch_size)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(alarmed)
    selected = alarmed[: min(args.max_windows, len(alarmed))]
    verifiers = {
        "legacy_structural": ExplanationVerifier(enforce_decision_semantics=False),
        "hardened_decision_semantic": ExplanationVerifier(enforce_decision_semantics=True),
    }
    results = {
        name: {
            "clean_rejections": 0,
            "best_of_n_successes": 0,
            "by_strategy": {},
            "verification_times": [],
        }
        for name in verifiers
    }
    preservation_checks = 0
    preservation_passes = 0
    strategy_names: list[str] = []

    for index in selected:
        bundle = pipeline.build_evidence_bundle(
            arrays.x[index], event_id=f"adaptive-{index}", mc_samples=args.mc_samples
        )
        clean = pipeline.generator.generate(bundle)
        candidates = generate_adaptive_candidates(bundle, clean)
        strategy_names = [candidate.strategy for candidate in candidates]
        for candidate in candidates:
            preservation_checks += 1
            preservation_passes += int(
                preserves_committed_facts(clean, candidate.explanation)
            )

        for verifier_name, verifier in verifiers.items():
            started = time.perf_counter()
            clean_result = verifier.verify(bundle, clean)
            results[verifier_name]["verification_times"].append(
                time.perf_counter() - started
            )
            results[verifier_name]["clean_rejections"] += int(
                clean_result.status != "VERIFIED"
            )
            candidate_results = []
            for candidate in candidates:
                started = time.perf_counter()
                verification = verifier.verify(bundle, candidate.explanation)
                results[verifier_name]["verification_times"].append(
                    time.perf_counter() - started
                )
                candidate_results.append((candidate, verification))
                strategy = results[verifier_name]["by_strategy"].setdefault(
                    candidate.strategy,
                    {"VERIFIED": 0, "WARNING": 0, "ABSTAIN": 0},
                )
                strategy[verification.status] += 1
            status_rank = {"ABSTAIN": 0, "WARNING": 1, "VERIFIED": 2}
            _, best = max(
                candidate_results,
                key=lambda item: (status_rank[item[1].status], item[1].score),
            )
            results[verifier_name]["best_of_n_successes"] += int(
                best.status == "VERIFIED"
            )

    total = max(len(selected), 1)
    output = {
        "evaluation": "adaptive_truth_preserving_explanation_red_team",
        "available_alarmed_windows": int(len(alarmed)),
        "evaluated_alarmed_windows": int(len(selected)),
        "candidates_per_window": len(strategy_names),
        "strategies": strategy_names,
        "committed_fact_preservation_rate": preservation_passes
        / max(preservation_checks, 1),
        "detector_alarm_and_prediction_preserved_by_design": True,
        "uses_ground_truth_test_labels": False,
        "verifiers": {},
        "limitations": (
            "Template search is adaptive to verifier outcomes but is not an external LLM "
            "red team or a human-subject decision study."
        ),
    }
    for verifier_name, raw in results.items():
        times = np.asarray(raw.pop("verification_times"), dtype=np.float64)
        by_strategy = {
            strategy: {
                "attack_success_rate": statuses["VERIFIED"] / total,
                "statuses": statuses,
            }
            for strategy, statuses in raw["by_strategy"].items()
        }
        output["verifiers"][verifier_name] = {
            "clean_explanation_rejection_rate": raw["clean_rejections"] / total,
            "adaptive_best_of_n_attack_success_rate": raw["best_of_n_successes"] / total,
            "by_strategy": by_strategy,
            "mean_verification_overhead_ms": float(times.mean() * 1000.0),
            "p95_verification_overhead_ms": float(np.quantile(times, 0.95) * 1000.0),
        }

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
