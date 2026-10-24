"""Temporal outcomes for fixed saved detectors on reused development data."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import run_vehicle_development_stage17 as prior  # noqa: E402
from audit_vehicle_context_stage22 import STAGE21_MANIFEST, verify_stage21  # noqa: E402

from coact_cps.vehicle.temporal23 import summarize  # noqa: E402

ROUTES = ("civic13_10", "civic13_11", "civic13_12")
GUARD = "context_bounded"


def run(source, stage21, output):
    output, stage21 = Path(output), Path(stage21)
    if output.exists():
        raise ValueError("output must not already exist")
    z = prior.read_source(source)
    protocol, lock, source_rows, checked = prior.audit_source(z)
    _, checked21 = verify_stage21(stage21)
    models = {m["seed"]: m for m in lock["models"] if m["method"] == "coact_physics"}
    chosen = [r for r in source_rows if r["trip"] in ROUTES]
    expected = {(r, c, s) for r in ROUTES for c in protocol["config"]["cases"] for s in prior.SEEDS}
    if len(chosen) != 117 or {(r["trip"], r["case"], r["seed"]) for r in chosen} != expected:
        raise ValueError("complete fixed development roles required")
    root = Path(__file__).resolve().parents[1]
    code = [
        "scripts/evaluate_vehicle_temporal_stage23.py",
        "src/coact_cps/vehicle/temporal23.py",
        "scripts/audit_vehicle_context_stage22.py",
        "scripts/run_vehicle_development_stage17.py",
        "src/coact_cps/vehicle/physics17.py",
    ]
    frozen = {
        "schema": "vehicle-temporal-outcomes-stage23-v1",
        "source_zip_sha256": prior.SOURCE_SHA256,
        "stage21_manifest_sha256": STAGE21_MANIFEST,
        "code_sha256": {n: prior.sha((root / n).read_bytes()) for n in code},
        "design_sha256": prior.sha((root / "STAGE23_FROZEN_DESIGN.txt").read_bytes()),
        "methods": ["frozen_original", GUARD],
        "routes": list(ROUTES),
        "observed_span_qualification": "Sensor interval sum; no token lifetime or safety claim.",
        **prior.status_receipt(),
    }
    rows = []
    for row in chosen:
        route, case, seed = row["trip"], row["case"], row["seed"]
        c, b = prior.checked_block(z, row, models[seed])
        name = f"{route}__{case}__{GUARD}__{seed}.npz"
        with np.load(stage21 / "blocks" / name, allow_pickle=False) as obj:
            guard = {k: obj[k] for k in obj.files}
        if not np.array_equal(guard["positions"], c["positions"]):
            raise ValueError("guard positions mismatch")
        model = models[seed]
        cal = model["node_calibration"]
        nodes = (b["nodes"] - np.asarray(cal["median"])) / np.asarray(cal["scale"])
        speed = (nodes[:, :2] > model["threshold"]).any(axis=1)
        reconstructed = speed | (nodes[:, 2] > guard["imu_boundary"])
        if (
            not np.array_equal(reconstructed, guard["alarm"])
            or np.any(guard["alarm"] & ~b["alarm"])
            or np.any(b["grant"] & ~guard["grant"])
        ):
            raise ValueError("saved guard preservation mismatch")
        outcomes = {}
        for method, block in (("frozen_original", b), (GUARD, guard)):
            outcomes[method] = summarize(
                c["time_ms"],
                c["start_ms"],
                c["labels"],
                c["positions"],
                block["alarm"],
                c["base_grant"],
                block["grant"],
            )
        old, new = outcomes["frozen_original"], outcomes[GUARD]
        delta = {
            key: new[key] - old[key]
            for key in (
                "normal_granted_epochs",
                "attack_granted_epochs",
                "attack_granted_observed_span_ms",
                "attack_longest_grant_run_epochs",
                "attack_longest_grant_run_observed_span_ms",
                "neural_missed_events",
                "fully_denied_events",
            )
        }
        rows.append(dict(route=route, case=case, seed=seed, delta=delta, **outcomes))
    grouped = []
    for route in ROUTES:
        for case in protocol["config"]["cases"]:
            group = [r for r in rows if r["route"] == route and r["case"] == case]
            grouped.append(
                dict(
                    route=route,
                    case=case,
                    mean_delta={
                        k: float(np.mean([r["delta"][k] for r in group])) for k in rows[0]["delta"]
                    },
                )
            )
    result = {
        **prior.status_receipt(),
        "policy_changed": False,
        "model_training_or_inference": False,
        "stage13_indexed_files_verified": checked,
        "stage21_indexed_files_verified": checked21,
        "original_speed_alarms_preserved": True,
        "guarded_grants_superset_of_original": True,
        "worst_extra_attack_granted_observed_span": max(
            (r for r in grouped if r["case"] != "normal"),
            key=lambda r: r["mean_delta"]["attack_granted_observed_span_ms"],
        ),
        "route_case_seed_means": grouped,
        "rows": rows,
    }
    output.mkdir(parents=True)
    (output / "frozen_protocol.json").write_text(json.dumps(frozen, indent=2) + "\n")
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    index = {p.name: prior.sha(p.read_bytes()) for p in sorted(output.iterdir()) if p.is_file()}
    (output / "file_hashes.json").write_text(json.dumps(index, indent=2) + "\n")
    print(json.dumps(result["worst_extra_attack_granted_observed_span"], indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage13-zip", type=Path, required=True)
    parser.add_argument("--stage21-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.stage13_zip, args.stage21_results, args.output)
