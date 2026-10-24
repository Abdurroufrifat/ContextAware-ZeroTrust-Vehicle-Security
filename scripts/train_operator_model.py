from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from coact_cps.models.operator_state import OperatorStateTransformer
from coact_cps.study.sequence_dataset import build_operator_sequences
from coact_cps.training import seed_everything


def _participant_split(participants: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    unique = np.unique(participants)
    if len(unique) < 3:
        raise ValueError(
            "At least three participants are required for a participant-held-out split"
        )
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    test_count = max(1, int(round(0.2 * len(unique))))
    test_people = set(unique[-test_count:])
    train_mask = np.asarray([person not in test_people for person in participants])
    return train_mask, ~train_mask


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the operator belief/reliance Transformer")
    parser.add_argument("--study-csv", required=True)
    parser.add_argument("--output", default="artifacts/operator")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--max-sequence-length", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    seed_everything(args.seed)
    frame = pd.read_csv(args.study_csv)
    if "approval_reference" in frame and set(frame["approval_reference"]) == {"DEVELOPMENT_ONLY"}:
        print("WARNING: development-only records cannot support publication claims.")
    arrays = build_operator_sequences(frame, args.max_sequence_length)
    train_mask, test_mask = _participant_split(arrays.participants, args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parameters = {
        "input_size": int(arrays.interactions.shape[-1]),
        "n_diagnoses": len(arrays.diagnosis_labels),
        "max_sequence_length": args.max_sequence_length,
        "d_model": 48,
        "n_heads": 4,
        "n_layers": 2,
        "dropout": 0.15,
    }
    model = OperatorStateTransformer(**parameters).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    indices = np.where(train_mask)[0]
    rng = np.random.default_rng(args.seed)
    model.train()
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(indices)
        losses = []
        for start in range(0, len(indices), 32):
            selected = indices[start : start + 32]
            interactions = torch.from_numpy(arrays.interactions[selected]).to(device)
            mask = torch.from_numpy(arrays.padding_mask[selected]).to(device)
            diagnosis = torch.from_numpy(arrays.diagnosis[selected]).to(device)
            states = torch.from_numpy(arrays.states[selected]).to(device)
            output = model(interactions, mask)
            predicted_states = torch.stack(
                [output.confidence, output.reliance, output.workload], dim=-1
            )
            loss = nn.functional.cross_entropy(output.diagnosis_logits, diagnosis)
            loss = loss + nn.functional.mse_loss(predicted_states, states)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if epoch == 1 or epoch % 10 == 0:
            print(f"epoch={epoch} train_loss={np.mean(losses):.4f}")

    model.eval()
    selected = np.where(test_mask)[0]
    with torch.no_grad():
        output = model(
            torch.from_numpy(arrays.interactions[selected]).to(device),
            torch.from_numpy(arrays.padding_mask[selected]).to(device),
        )
    predicted = output.diagnosis_logits.argmax(dim=-1).cpu().numpy()
    predicted_states = (
        torch.stack([output.confidence, output.reliance, output.workload], dim=-1).cpu().numpy()
    )
    metrics = {
        "participant_held_out_accuracy": float(np.mean(predicted == arrays.diagnosis[selected])),
        "confidence_mae": float(
            np.mean(np.abs(predicted_states[:, 0] - arrays.states[selected, 0]))
        ),
        "reliance_mae": float(np.mean(np.abs(predicted_states[:, 1] - arrays.states[selected, 1]))),
        "workload_mae": float(np.mean(np.abs(predicted_states[:, 2] - arrays.states[selected, 2]))),
        "test_participants": sorted(np.unique(arrays.participants[selected]).tolist()),
    }
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_parameters": parameters,
            "diagnosis_labels": arrays.diagnosis_labels,
            "metrics": metrics,
        },
        destination / "operator_state.pt",
    )
    (destination / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
