from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from coact_cps.models.collaboration_policy import ACTIONS, CollaborationPolicyNet
from coact_cps.training import seed_everything

STATUSES = ("VERIFIED", "WARNING", "ABSTAIN")


def encode(
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    required = {
        "alarm_probability",
        "explanation_uncertainty",
        "verification_score",
        "verification_status",
        "evidence_count",
        "alternative_count",
        "self_confidence",
        "reliance",
        "workload",
        "collaboration_action",
        "outcome_reward",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"Missing policy-log columns: {missing}")
    action_index = {action.value: index for index, action in enumerate(ACTIONS)}
    unknown = sorted(set(frame["collaboration_action"]) - set(action_index))
    if unknown:
        raise ValueError(f"Unknown actions: {unknown}")
    ai_rows = []
    for _, row in frame.iterrows():
        status = np.zeros(len(STATUSES), dtype=np.float32)
        value = str(row["verification_status"])
        if value not in STATUSES:
            raise ValueError(f"Unknown verifier status: {value}")
        status[STATUSES.index(value)] = 1.0
        ai_rows.append(
            np.concatenate(
                [
                    np.asarray(
                        [
                            row["alarm_probability"],
                            row["explanation_uncertainty"],
                            row["verification_score"],
                        ],
                        dtype=np.float32,
                    ),
                    status,
                    np.asarray(
                        [
                            min(float(row["evidence_count"]) / 10.0, 1.0),
                            min(float(row["alternative_count"]) / 5.0, 1.0),
                        ],
                        dtype=np.float32,
                    ),
                ]
            )
        )
    operator = frame.loc[:, ["self_confidence", "reliance", "workload"]].to_numpy(dtype=np.float32)
    actions = frame["collaboration_action"].map(action_index).to_numpy(dtype=np.int64)
    rewards = frame["outcome_reward"].to_numpy(dtype=np.float32)
    if "behavior_propensity" in frame:
        propensities = frame["behavior_propensity"].to_numpy(dtype=np.float32)
    else:
        propensities = np.full(len(frame), 1.0 / len(ACTIONS), dtype=np.float32)
    return np.stack(ai_rows), operator, actions, rewards, np.clip(propensities, 0.02, 1.0)


def safety_penalty(probabilities: torch.Tensor, ai: torch.Tensor) -> torch.Tensor:
    index = {action.value: idx for idx, action in enumerate(ACTIONS)}
    verified, warning, abstain = ai[:, 3], ai[:, 4], ai[:, 5]
    uncertainty = ai[:, 1]
    penalty = probabilities.new_tensor(0.0)
    abstain_required = abstain > 0.5
    if abstain_required.any():
        penalty = penalty + (1.0 - probabilities[abstain_required, index["abstain"]]).mean()
    warning_required = warning > 0.5
    if warning_required.any():
        safe_mass = probabilities[warning_required][
            :, [index["show_evidence"], index["warn_uncertainty"], index["abstain"]]
        ].sum(dim=1)
        penalty = penalty + (1.0 - safe_mass).mean()
    uncertain = (verified > 0.5) & (uncertainty >= 0.25)
    if uncertain.any():
        safe_mass = probabilities[uncertain][
            :,
            [
                index["show_alternatives"],
                index["ask_context"],
                index["warn_uncertainty"],
                index["abstain"],
            ],
        ].sum(dim=1)
        penalty = penalty + (1.0 - safe_mass).mean()
    return penalty


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a safety-constrained offline collaboration policy"
    )
    parser.add_argument("--study-csv", required=True)
    parser.add_argument("--output", default="artifacts/policy")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--safety-weight", type=float, default=2.0)
    args = parser.parse_args()
    seed_everything(args.seed)
    frame = pd.read_csv(args.study_csv)
    if len(frame) < 30:
        print("WARNING: fewer than 30 logged trials; this is a pipeline test, not evidence.")
    ai_np, operator_np, actions_np, rewards_np, propensity_np = encode(frame)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ai = torch.from_numpy(ai_np).to(device)
    operator = torch.from_numpy(operator_np).to(device)
    actions = torch.from_numpy(actions_np).to(device)
    rewards = torch.from_numpy(rewards_np).to(device)
    propensity = torch.from_numpy(propensity_np).to(device)
    model = CollaborationPolicyNet(ai.shape[1], operator.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    for epoch in range(1, args.epochs + 1):
        logits = model(ai, operator)
        probabilities = torch.softmax(logits, dim=-1)
        chosen = probabilities.gather(1, actions[:, None]).squeeze(1).clamp_min(1e-6)
        importance = (chosen.detach() / propensity).clamp(max=10.0)
        policy_loss = -(importance * rewards * torch.log(chosen)).mean()
        imitation_loss = nn.functional.cross_entropy(logits, actions)
        constraint = safety_penalty(probabilities, ai)
        entropy = -(probabilities * probabilities.clamp_min(1e-6).log()).sum(dim=1).mean()
        loss = (
            policy_loss + 0.20 * imitation_loss + args.safety_weight * constraint - 0.01 * entropy
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if epoch == 1 or epoch % 25 == 0:
            print(f"epoch={epoch} loss={float(loss):.4f} constraint={float(constraint):.4f}")

    model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(model(ai, operator), dim=-1)
        predicted = probabilities.argmax(dim=-1)
        constraint = safety_penalty(probabilities, ai)
    metrics = {
        "logged_action_agreement": float((predicted == actions).float().mean().cpu()),
        "mean_safety_constraint_violation": float(constraint.cpu()),
        "note": (
            "In-sample diagnostics only; use participant-held-out off-policy evaluation "
            "for the paper."
        ),
    }
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "ai_state_size": int(ai.shape[1]),
            "operator_state_size": int(operator.shape[1]),
            "actions": [action.value for action in ACTIONS],
            "metrics": metrics,
        },
        destination / "collaboration_policy.pt",
    )
    (destination / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
