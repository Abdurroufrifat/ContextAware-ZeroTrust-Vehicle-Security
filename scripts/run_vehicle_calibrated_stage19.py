"""Matched fine-tuning on reused development routes; never unlocks final tests."""

import argparse
import io
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import run_vehicle_development_stage17 as prior  # noqa: E402

from coact_cps.vehicle.ai_models13 import (  # noqa: E402
    aggregate_scores,
    create_adapter,
    fit_node_calibration,
    score_nodes,
    threshold,
)
from coact_cps.vehicle.calibrated19 import (  # noqa: E402
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    fine_tune,
    make_windows,
    require_clean,
)

ARMS = {"history_control": (1.0, 0.0), "calibrated": (0.7916753765330672, -0.374991439420167)}
FIT = ("civic13_07", "civic13_08")
CALIBRATE = "civic13_09"
EVALUATE = ("civic13_10", "civic13_11", "civic13_12")


def screen(
    base_fpr,
    control_fpr,
    candidate_fpr,
    worst_route_fpr,
    base_grants,
    candidate_grants,
    loss_vs_base,
    loss_vs_control,
):
    checks = {
        "normal_fpr_at_most_5_percent": candidate_fpr <= 0.05,
        "worst_route_normal_fpr_at_most_5_percent": worst_route_fpr <= 0.05,
        "at_least_1pp_fpr_gain_vs_matched_control": control_fpr - candidate_fpr >= 0.01,
        "normal_fpr_not_worse_than_frozen_original": candidate_fpr <= base_fpr,
        "normal_grants_not_lower_than_frozen_original": candidate_grants >= base_grants,
        "each_route_case_detection_loss_at_most_5pp": max(loss_vs_base, loss_vs_control) <= 0.05,
    }
    return {
        **checks,
        "passed": all(checks.values()),
        "eligible_for_test": False,
        "fresh_validation_required": True,
        "status": "DEVELOPMENT_ONLY_NOT_PROMOTED",
    }


def run(source, output):
    output = Path(output)
    if output.exists():
        raise ValueError("output must not already exist")
    z = prior.read_source(source)
    protocol, lock, source_rows, checked = prior.audit_source(z)
    cfg = dict(protocol["config"], batch_size=BATCH_SIZE)
    models = {m["seed"]: m for m in lock["models"] if m["method"] == "coact_physics"}
    cases = protocol["config"]["cases"]
    rows = [
        r
        for r in source_rows
        if r["trip"] in EVALUATE or (r["trip"] in (*FIT, CALIBRATE) and r["case"] == "normal")
    ]
    expected = {(r, c, s) for r in EVALUATE for c in cases for s in prior.SEEDS}
    expected |= {(r, "normal", s) for r in (*FIT, CALIBRATE) for s in prior.SEEDS}
    if len(rows) != len(expected) or {(r["trip"], r["case"], r["seed"]) for r in rows} != expected:
        raise ValueError("incomplete fixed development roles")
    contexts, baseline = {}, {}
    for r in rows:
        c, b = prior.checked_block(z, r, models[r["seed"]])
        contexts[r["trip"], r["case"]] = c
        baseline[r["trip"], r["case"], r["seed"]] = b
    for r in (*FIT, CALIBRATE):
        require_clean(contexts[r, "normal"])
    windows = {key: make_windows(c, lock["scaler"]) for key, c in contexts.items()}
    xfit = np.concatenate([windows[r, "normal"][0] for r in FIT])
    dtfit = np.concatenate([windows[r, "normal"][1] for r in FIT])
    root = Path(__file__).resolve().parents[1]
    sources = [
        "scripts/run_vehicle_calibrated_stage19.py",
        "src/coact_cps/vehicle/calibrated19.py",
        "src/coact_cps/vehicle/ai_models13.py",
        "src/coact_cps/models/temporal_graph.py",
        "scripts/run_vehicle_development_stage17.py",
        "src/coact_cps/vehicle/physics17.py",
    ]
    frozen = {
        "schema": "vehicle-calibrated-development-stage19-v1",
        "status": "DEVELOPMENT_ONLY_NOT_PROMOTED",
        "eligible_for_test": False,
        "source_zip_sha256": prior.SOURCE_SHA256,
        "code_sha256": {f: prior.sha((root / f).read_bytes()) for f in sources},
        "roles": {
            "fit_and_node_scale_normal": list(FIT),
            "threshold_normal": CALIBRATE,
            "exploratory_comparison": list(EVALUATE),
        },
        "arms": ARMS,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "physics_weight": 0.05,
        "history_intervals": 5,
        "imu_lag_epochs": 1,
        "batch_size": BATCH_SIZE,
        "seeds": list(prior.SEEDS),
        "retained_anomaly_channels": ["gnss", "wheel", "imu"],
        "stage18_summary_sha256": (
            "e89f85220f2e0acce1249e62c1bae11ea51b99a29b8c0a0fbb995a6cebe155a7"
        ),
        "epoch_selection": "fixed five epochs; no early stopping or attack-based tuning",
        "screen_rules": {
            "normal_fpr_max": 0.05,
            "worst_route_fpr_max": 0.05,
            "fpr_gain_vs_control_min": 0.01,
            "each_route_case_tpr_loss_max": 0.05,
        },
        "qualification": "All routes were previously inspected; no fresh validation or final test.",
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }
    output.mkdir(parents=True)
    (output / "frozen_protocol.json").write_text(json.dumps(frozen, indent=2) + "\n")
    (output / "blocks").mkdir()
    (output / "models").mkdir()
    scores, fit_receipts, replay = {}, [], []

    def load_original(seed):
        adapter = create_adapter("coact_physics", cfg, root, seed)
        raw = z.read(prior.ROOT + f"fit/seed_{seed}/coact_physics.pt")
        if prior.sha(raw) != models[seed]["checkpoint_sha256"]:
            raise ValueError("checkpoint hash mismatch")
        adapter.model.load_state_dict(
            torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
        )
        adapter.model.eval()
        return adapter

    for seed in prior.SEEDS:
        adapter = load_original(seed)
        replayed = score_nodes(adapter, windows[CALIBRATE, "normal"][0], cfg, seed)
        difference = float(abs(replayed - baseline[CALIBRATE, "normal", seed]["nodes"]).max())
        if difference > 0.001:
            raise ValueError("baseline replay mismatch exceeds predeclared 0.001 tolerance")
        replay.append({"seed": seed, "maximum_node_absolute_error": difference})
        for arm, (slope, offset) in ARMS.items():
            adapter = load_original(seed)
            history = fine_tune(adapter, xfit, dtfit, lock["scaler"], seed, slope, offset)
            fit_nodes = score_nodes(adapter, xfit, cfg, seed)
            calibration = fit_node_calibration(fit_nodes)
            clean = score_nodes(adapter, windows[CALIBRATE, "normal"][0], cfg, seed)
            cutoff = threshold(aggregate_scores(clean, calibration, "coact_physics"))
            checkpoint = output / "models" / f"{arm}__{seed}.pt"
            torch.save(adapter.model.state_dict(), checkpoint)
            fit_receipts.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "loss": history,
                    "node_calibration": calibration,
                    "threshold": cutoff,
                    "checkpoint": str(checkpoint.relative_to(output)).replace("\\", "/"),
                    "checkpoint_sha256": prior.sha(checkpoint.read_bytes()),
                    "original_checkpoint_sha256": models[seed]["checkpoint_sha256"],
                }
            )
            for route in EVALUATE:
                for case in cases:
                    c = contexts[route, case]
                    nodes = score_nodes(adapter, windows[route, case][0], cfg, seed)
                    scalar = aggregate_scores(nodes, calibration, "coact_physics")
                    alarm = scalar > cutoff
                    grant = np.zeros(len(c["labels"]), dtype=bool)
                    grant[c["positions"]] = c["base_grant"][c["positions"]] & ~alarm
                    np.savez_compressed(
                        output / "blocks" / f"{route}__{case}__{arm}__{seed}.npz",
                        nodes=nodes,
                        score=scalar,
                        alarm=alarm,
                        positions=c["positions"],
                        grant=grant,
                    )
                    scores[route, case, seed, arm] = prior.metrics(c, alarm)
            print(f"{arm}, seed {seed}: five epochs and 39 development blocks complete", flush=True)
    result_rows = []
    for route in EVALUATE:
        for case in cases:
            for seed in prior.SEEDS:
                c = contexts[route, case]
                result_rows.append(
                    {
                        "route": route,
                        "case": case,
                        "seed": seed,
                        "frozen_original": prior.metrics(c, baseline[route, case, seed]["alarm"]),
                        **{a: scores[route, case, seed, a] for a in ARMS},
                    }
                )
    normal = [r for r in result_rows if r["case"] == "normal"]
    aggregate = {}
    for arm in ("frozen_original", *ARMS):
        aggregate[arm] = {
            "normal_fpr": float(np.mean([r[arm]["normal_endpoint_fpr"] for r in normal])),
            "normal_grants": float(np.mean([r[arm]["grant_fraction_all_epochs"] for r in normal])),
            "worst_route_normal_fpr": max(
                float(
                    np.mean([r[arm]["normal_endpoint_fpr"] for r in normal if r["route"] == route])
                )
                for route in EVALUATE
            ),
            "attack_cases": {
                case: {
                    k: float(np.mean([r[arm][k] for r in result_rows if r["case"] == case]))
                    for k in ("active_endpoint_tpr", "active_grant_fraction")
                }
                for case in cases
                if case != "normal"
            },
        }
    losses = []
    for reference in ("frozen_original", "history_control"):
        for route in EVALUATE:
            for case in cases:
                if case == "normal":
                    continue
                group = [r for r in result_rows if r["route"] == route and r["case"] == case]
                loss = float(
                    np.mean(
                        [
                            r[reference]["active_endpoint_tpr"]
                            - r["calibrated"]["active_endpoint_tpr"]
                            for r in group
                        ]
                    )
                )
                losses.append(
                    {"reference": reference, "route": route, "case": case, "tpr_loss": loss}
                )
    worst = {
        ref: max((v for v in losses if v["reference"] == ref), key=lambda v: v["tpr_loss"])
        for ref in ("frozen_original", "history_control")
    }
    a, b, c = (aggregate[k] for k in ("frozen_original", "history_control", "calibrated"))
    report = {
        "status": "DEVELOPMENT_ONLY_NOT_PROMOTED",
        "eligible_for_test": False,
        "fresh_validation_required": True,
        "source_indexed_files_verified": checked,
        "protocol_sha256": prior.sha((output / "frozen_protocol.json").read_bytes()),
        "fit_receipts": fit_receipts,
        "baseline_replay": replay,
        "aggregate": aggregate,
        "worst_route_case_losses": worst,
        "route_case_losses": losses,
        "rows": result_rows,
        "development_screen": screen(
            a["normal_fpr"],
            b["normal_fpr"],
            c["normal_fpr"],
            c["worst_route_normal_fpr"],
            a["normal_grants"],
            c["normal_grants"],
            worst["frozen_original"]["tpr_loss"],
            worst["history_control"]["tpr_loss"],
        ),
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    hashes = {
        p.relative_to(output).as_posix(): prior.sha(p.read_bytes())
        for p in sorted(output.rglob("*"))
        if p.is_file()
    }
    (output / "file_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print(
        json.dumps(
            {"aggregate": aggregate, "screen": report["development_screen"], "worst": worst},
            indent=2,
        )
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage13-zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.stage13_zip, args.output)
