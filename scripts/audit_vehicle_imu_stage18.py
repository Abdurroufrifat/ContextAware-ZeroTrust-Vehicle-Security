"""Replay-only IMU calibration diagnostics on previously inspected Stage13 data."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import run_vehicle_development_stage17 as prior  # noqa: E402

from coact_cps.vehicle.imu_audit18 import (  # noqa: E402
    FIT_ROUTES,
    causal_pairs,
    fit_routes,
    metrics,
    select_lag,
)

ROLES = {"civic13_07": "fit", "civic13_08": "fit", "civic13_09": "check"}


def status():
    return {
        "status": "DIAGNOSTIC_ONLY_NOT_PROMOTED",
        "eligible_for_test": False,
        "policy_changed": False,
        "fresh_validation_required": True,
        "evidence_role": "PREVIOUSLY_INSPECTED_DEVELOPMENT_ROUTES",
        "stage16_final_routes_evaluated": False,
    }


def require_clean(context):
    labels = context["labels"]
    if labels.ndim != 1 or labels.dtype != bool or labels.any():
        raise ValueError("only normal-labelled contexts may enter diagnostics")


def run(source, output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("output must be new or empty")
    z = prior.read_source(source)
    _, lock, rows, checked = prior.audit_source(z)
    models = {m["seed"]: m for m in lock["models"] if m["method"] == "coact_physics"}
    selected = [r for r in rows if r["trip"] in ROLES and r["case"] == "normal"]
    if {(r["trip"], r["seed"]) for r in selected} != {
        (r, s) for r in ROLES for s in prior.SEEDS
    } or len(selected) != 9:
        raise ValueError("incomplete diagnostic roles")
    pairs, old_alarms = {}, []
    for row in selected:
        c, b = prior.checked_block(z, row, models[row["seed"]])
        require_clean(c)
        route = row["trip"]
        if route not in pairs:
            pairs[route] = causal_pairs(c["features"], c["time_ms"], c["start_ms"])
        cal = models[row["seed"]]["node_calibration"]
        exceed = ((b["nodes"] - cal["median"]) / cal["scale"]) > models[row["seed"]]["threshold"]
        old_alarms.append(
            {
                "route": route,
                "seed": row["seed"],
                "scored_endpoints": len(exceed),
                "baseline_normal_fpr": float(b["alarm"].mean()),
                "imu_only_alarms": int((exceed[:, 2] & ~exceed[:, :2].any(axis=1)).sum()),
                "total_alarms": int(b["alarm"].sum()),
            }
        )
    fit = {r: pairs[r] for r in FIT_ROUTES}
    lag, table = select_lag(fit)
    beta, offset = fit_routes(fit, lag)
    comparisons = {}
    for route, p in pairs.items():
        x, y = p["x"], p["y"]
        local = np.linalg.lstsq(np.column_stack((x[:, 0], np.ones(len(y)))), y, rcond=None)[0]
        comparisons[route] = {
            "role": ROLES[route],
            "diagnostic_endpoints": len(y),
            "lag0_correlation": float(np.corrcoef(x[:, 0], y)[0, 1]),
            "lag0_descriptive_slope": float(local[0]),
            "lag0_descriptive_offset": float(local[1]),
            "identity": metrics(y, x[:, 0]),
            "sign_flip": metrics(y, -x[:, 0]),
            "zero": metrics(y, np.zeros(len(y))),
            "frozen_affine": metrics(y, beta * x[:, lag] + offset),
        }
    check = comparisons["civic13_09"]
    report = {
        **status(),
        "source_zip_sha256": prior.SOURCE_SHA256,
        "source_indexed_files_verified": checked,
        "roles": ROLES,
        "calibration": {
            "lag_epochs": lag,
            "slope": beta,
            "offset_mps2": offset,
            "fit_weighting": "equal total weight per route",
        },
        "lag_selection": table,
        "routes": comparisons,
        "baseline_alarm_channels": old_alarms,
        "check_mae_reduction_vs_identity_fraction": 1
        - check["frozen_affine"]["mae"] / check["identity"]["mae"],
        "limitations": [
            "All routes, including the coefficient-check route, were previously inspected.",
            "Lag choice is exploratory, not a calibrated sensor-clock delay estimate.",
            "Speed-derived acceleration is a proxy, not independent accelerometer ground truth.",
            "No 3D attitude, gravity, grade or raw-axis correction can be established here.",
            "Check-route local slope is descriptive only and never used for the shared fit.",
            "No attack evaluation or inference about improved detection is made.",
            "An affine transform must not replace the detector without retraining "
            "and fresh validation.",
        ],
    }
    code = [
        Path(__file__),
        Path(__file__).parents[1] / "src/coact_cps/vehicle/imu_audit18.py",
        Path(prior.__file__),
        Path(__file__).parents[1] / "src/coact_cps/vehicle/physics17.py",
    ]
    report["code_sha256"] = {
        p.relative_to(Path(__file__).parents[1]).as_posix(): prior.sha(p.read_bytes()) for p in code
    }
    output.mkdir(parents=True, exist_ok=True)
    for route, p in pairs.items():
        np.savez_compressed(output / (route + "__normal_pairs.npz"), **p)
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    manifest = {p.name: prior.sha(p.read_bytes()) for p in output.iterdir() if p.is_file()}
    (output / "file_hashes.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "calibration": report["calibration"],
                "check_route": check,
                "mae_reduction_fraction": report["check_mae_reduction_vs_identity_fraction"],
            },
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
