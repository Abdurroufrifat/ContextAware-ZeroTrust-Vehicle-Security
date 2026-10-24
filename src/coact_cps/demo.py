from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .data.synthetic import SyntheticCPSConfig, generate_synthetic_cps
from .data.windowing import make_windows
from .genai.grounded_generator import GroundedExplanationGenerator
from .models.temporal_graph import PhysicsInvariant, TemporalHeterogeneousGraphTransformer
from .pipeline import CoActPipeline
from .security.attacks import apply_explanation_attack
from .security.provenance import TelemetryAuthenticator
from .security.verifier import ExplanationVerifier
from .training import (
    development_stratified_split,
    evaluate_model,
    save_detector_checkpoint,
    seed_everything,
    train_detector,
)


def run_demo(
    output_directory: str | Path,
    epochs: int = 5,
    seed: int = 7,
    generator_mode: str = "template",
    model_id: str = "Qwen/Qwen3-4B-Instruct-2507",
    n_steps: int = 1800,
    detection_mode: str = "normal_only",
) -> dict:
    seed_everything(seed)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    frame, topology = generate_synthetic_cps(SyntheticCPSConfig(n_steps=n_steps, seed=seed))
    node_index = {node: idx for idx, node in enumerate(topology.nodes)}
    roots = frame["root_cause"].map(node_index).fillna(-1).to_numpy(dtype=np.int64)
    arrays = make_windows(
        frame.loc[:, topology.nodes].to_numpy(dtype=np.float32),
        frame["attack"].to_numpy(dtype=np.int64),
        roots,
        window_size=32,
        stride=3,
        label_rule="last",
    )
    train, val, test = development_stratified_split(arrays, seed=seed)
    adjacency = torch.from_numpy(topology.relation_adjacency(bidirectional=True))
    parameters = {
        "n_nodes": len(topology.nodes),
        "in_features": 1,
        "n_relations": len(topology.relations),
        "max_window_size": 32,
        "d_model": 32,
        "n_heads": 4,
        "n_layers": 1,
        "dropout": 0.15,
        "detection_mode": detection_mode,
    }
    model = TemporalHeterogeneousGraphTransformer(**parameters)
    invariant = PhysicsInvariant(
        kind="mass_balance",
        state_node=node_index["LIT101"],
        inflow_node=node_index["FIT101"],
        outflow_node=node_index["FIT201"],
        coefficient=0.0018,
    )
    model, history = train_detector(
        model,
        train,
        val,
        adjacency,
        epochs=epochs,
        batch_size=64,
        learning_rate=5e-4,
        physics_invariants=[invariant],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics = evaluate_model(model, test, adjacency.to(device), device)

    attacked_candidates = np.where(test.y == 1)[0]
    selected_index = int(attacked_candidates[0] if len(attacked_candidates) else 0)
    authenticator = TelemetryAuthenticator(b"development-only-demo-key-32bytes")
    pipeline = CoActPipeline(
        model,
        adjacency,
        topology.nodes,
        generator=GroundedExplanationGenerator(mode=generator_mode, model_id=model_id),
        physics_invariants=[invariant],
        invariant_threshold=0.03,
        authenticator=authenticator,
        require_authentication=True,
        require_process_grounding=True,
        counterfactual_minimum=0.05,
    )
    analysis_signature = authenticator.sign(test.x[selected_index], "synthetic-demo-event")
    analysis = pipeline.analyze(
        test.x[selected_index],
        event_id="synthetic-demo-event",
        telemetry_signature=analysis_signature,
    )

    attack_signature = authenticator.sign(test.x[selected_index], "attack-test-event")
    bundle = pipeline.build_evidence_bundle(
        test.x[selected_index],
        event_id="attack-test-event",
        telemetry_signature=attack_signature,
    )
    clean_explanation = pipeline.generator.generate(bundle)
    attacked_explanation = apply_explanation_attack(clean_explanation, "omit_critical_evidence")
    verifier = ExplanationVerifier()
    clean_verification = verifier.verify(bundle, clean_explanation)
    attacked_verification = verifier.verify(bundle, attacked_explanation)

    result = {
        "warning": (
            "Synthetic results are software smoke-test outputs, not publication evidence. "
            "Use attack-episode splits on HAI/SWaT for scientific evaluation."
        ),
        "device": str(device),
        "dataset": {
            "rows": len(frame),
            "windows": len(arrays.y),
            "nodes": list(topology.nodes),
        },
        "test_metrics": metrics,
        "analysis": analysis,
        "controlled_attack_check": {
            "attack": "omit_critical_evidence",
            "clean_status": clean_verification.status,
            "attacked_status": attacked_verification.status,
            "attacked_reasons": attacked_verification.reasons,
        },
        "training_history": history,
    }
    (output / "demo_results.json").write_text(
        json.dumps(result, indent=2, allow_nan=True), encoding="utf-8"
    )
    frame.to_csv(output / "synthetic_telemetry.csv", index=False)
    np.save(output / "adjacency.npy", adjacency.numpy())
    test.save(str(output / "test_windows.npz"))
    save_detector_checkpoint(
        model,
        output / "detector.pt",
        parameters,
        list(topology.nodes),
        list(topology.relation_names),
        history,
        physics_invariants=[invariant],
        process_settings={
            "invariant_threshold": 0.03,
            "require_process_grounding": True,
            "counterfactual_minimum": 0.05,
        },
    )
    return result
