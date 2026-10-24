"""Predeclared IMU-only guards on reused data; no model inference or activation."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import run_vehicle_development_stage17 as prior  # noqa: E402

from coact_cps.vehicle.imu_guard21 import (  # noqa: E402
    PRIMARY_CAP,
    boundaries,
    causal_context,
    decide,
    fit_partition,
    fit_profile,
    fit_quantile,
)

FIT = ("civic13_07", "civic13_08")
CALIBRATE = "civic13_09"
EVALUATE = ("civic13_10", "civic13_11", "civic13_12")
VARIANTS = ("global_imu", "context_unbounded", "context_bounded")
PRIMARY = "context_bounded"


def screen(
    original_fpr, candidate_fpr, worst_route_fpr, original_grants, candidate_grants, worst_tpr_loss
):
    checks = {
        "normal_fpr_at_most_5_percent": candidate_fpr <= 0.05,
        "worst_route_normal_fpr_at_most_5_percent": worst_route_fpr <= 0.05,
        "at_least_1pp_fpr_reduction": original_fpr - candidate_fpr >= 0.01,
        "normal_grants_not_lower": candidate_grants >= original_grants,
        "each_route_case_tpr_loss_at_most_5pp": worst_tpr_loss <= 0.05,
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
    models = {m["seed"]: m for m in lock["models"] if m["method"] == "coact_physics"}
    cases = protocol["config"]["cases"]
    chosen = [
        r
        for r in source_rows
        if r["trip"] in EVALUATE or (r["trip"] in (*FIT, CALIBRATE) and r["case"] == "normal")
    ]
    expected = {(r, c, s) for r in EVALUATE for c in cases for s in prior.SEEDS}
    expected |= {(r, "normal", s) for r in (*FIT, CALIBRATE) for s in prior.SEEDS}
    if (
        len(chosen) != len(expected)
        or {(r["trip"], r["case"], r["seed"]) for r in chosen} != expected
    ):
        raise ValueError("incomplete fixed development roles")
    contexts, blocks = {}, {}
    for row in chosen:
        c, b = prior.checked_block(z, row, models[row["seed"]])
        contexts[row["trip"], row["case"]] = c
        blocks[row["trip"], row["case"], row["seed"]] = b
    for route in (*FIT, CALIBRATE):
        if contexts[route, "normal"]["labels"].any():
            raise ValueError("only normal labelled routes may enter guard fitting")
    context = {
        k: causal_context(c["features"], c["time_ms"], c["start_ms"])[c["positions"]]
        for k, c in contexts.items()
    }
    partition = fit_partition({r: context[r, "normal"] for r in FIT})
    root = Path(__file__).resolve().parents[1]
    files = [
        "scripts/run_vehicle_imu_guard_stage21.py",
        "src/coact_cps/vehicle/imu_guard21.py",
        "scripts/run_vehicle_development_stage17.py",
        "src/coact_cps/vehicle/physics17.py",
    ]
    frozen = {
        "schema": "vehicle-imu-guard-development-stage21-v1",
        "source_zip_sha256": prior.SOURCE_SHA256,
        "code_sha256": {n: prior.sha((root / n).read_bytes()) for n in files},
        "primary": PRIMARY,
        "primary_cap_original_score_units": PRIMARY_CAP,
        "variants": list(VARIANTS),
        "partition": partition,
        "roles": {
            "fit_normal": list(FIT),
            "threshold_normal": CALIBRATE,
            "exploratory_comparison": list(EVALUATE),
        },
        "context": "absolute speed acceleration over five intervals ending at t-1; no IMU input",
        "normalization": "per-bin median/IQR of original normalized IMU residual, IQR floor0.1",
        "minimum_bin_and_supported_calibration_count": 30,
        "unsupported_context": "retain original IMU threshold",
        "preserved": "original checkpoint, original GNSS/wheel node calibration and thresholds",
        "screen_rules": {
            "normal_fpr_max": 0.05,
            "worst_route_normal_fpr_max": 0.05,
            "fpr_gain_min": 0.01,
            "worst_route_case_tpr_loss_max": 0.05,
        },
        "qualification": "Previously inspected development data; no fresh validation.",
        "eligible_for_test": False,
        "policy_changed": False,
        "model_inference_or_training": False,
    }
    output.mkdir(parents=True)
    (output / "frozen_protocol.json").write_text(json.dumps(frozen, indent=2) + "\n")
    (output / "blocks").mkdir()
    profiles, result_rows = [], []
    invariants = {
        "original_speed_alarms_preserved": True,
        "no_new_alarms_added": True,
        "bounded_high_imu_alarms_preserved": True,
    }
    for seed in prior.SEEDS:
        model = models[seed]
        cal = model["node_calibration"]
        old_cut = model["threshold"]

        def normalized(route, case, seed=seed, cal=cal):
            return (blocks[route, case, seed]["nodes"] - np.array(cal["median"])) / np.array(
                cal["scale"]
            )

        profile = fit_profile(
            {r: context[r, "normal"] for r in FIT},
            {r: normalized(r, "normal")[:, 2] for r in FIT},
            partition,
        )
        clean_imu = normalized(CALIBRATE, "normal")[:, 2]
        q = fit_quantile(context[CALIBRATE, "normal"], clean_imu, profile)
        global_cut = max(old_cut, float(np.quantile(clean_imu, 0.95)))
        profiles.append(
            {
                "seed": seed,
                "profile": profile,
                "conditional_quantile": q,
                "global_imu_cutoff": global_cut,
                "original_threshold": old_cut,
            }
        )
        for route in EVALUATE:
            for case in cases:
                c = contexts[route, case]
                block = blocks[route, case, seed]
                nodes = normalized(route, case)
                cutoffs = {
                    "global_imu": np.full(len(nodes), global_cut),
                    "context_unbounded": boundaries(context[route, case], profile, q, old_cut),
                    "context_bounded": boundaries(
                        context[route, case], profile, q, old_cut, PRIMARY_CAP
                    ),
                }
                row = {
                    "route": route,
                    "case": case,
                    "seed": seed,
                    "frozen_original": prior.metrics(c, block["alarm"]),
                }
                speed = (nodes[:, :2] > old_cut).any(axis=1)
                for variant, cutoff in cutoffs.items():
                    alarm = decide(nodes, old_cut, cutoff)
                    if np.any(speed & ~alarm):
                        raise ValueError("original speed alarm lost")
                    if np.any(alarm & ~block["alarm"]):
                        raise ValueError("unexpected new alarm added")
                    if variant == PRIMARY and np.any(
                        (nodes[:, 2] > old_cut + PRIMARY_CAP) & ~alarm
                    ):
                        raise ValueError("bounded high IMU alarm lost")
                    grant = np.zeros(len(c["labels"]), dtype=bool)
                    grant[c["positions"]] = c["base_grant"][c["positions"]] & ~alarm
                    np.savez_compressed(
                        output / "blocks" / f"{route}__{case}__{variant}__{seed}.npz",
                        positions=c["positions"],
                        imu_boundary=cutoff,
                        alarm=alarm,
                        grant=grant,
                    )
                    row[variant] = prior.metrics(c, alarm)
                result_rows.append(row)
    aggregate = {}
    normal = [r for r in result_rows if r["case"] == "normal"]
    for method in ("frozen_original", *VARIANTS):
        aggregate[method] = {
            "normal_fpr": float(np.mean([r[method]["normal_endpoint_fpr"] for r in normal])),
            "normal_grants": float(
                np.mean([r[method]["grant_fraction_all_epochs"] for r in normal])
            ),
            "worst_route_normal_fpr": max(
                float(
                    np.mean(
                        [r[method]["normal_endpoint_fpr"] for r in normal if r["route"] == route]
                    )
                )
                for route in EVALUATE
            ),
            "attack_cases": {
                case: {
                    metric: float(
                        np.mean([r[method][metric] for r in result_rows if r["case"] == case])
                    )
                    for metric in ("active_endpoint_tpr", "active_grant_fraction")
                }
                for case in cases
                if case != "normal"
            },
        }
    losses = []
    for variant in VARIANTS:
        for route in EVALUATE:
            for case in cases:
                if case == "normal":
                    continue
                group = [r for r in result_rows if r["route"] == route and r["case"] == case]
                loss = float(
                    np.mean(
                        [
                            r["frozen_original"]["active_endpoint_tpr"]
                            - r[variant]["active_endpoint_tpr"]
                            for r in group
                        ]
                    )
                )
                losses.append({"variant": variant, "route": route, "case": case, "tpr_loss": loss})
    worst = {
        v: max((r for r in losses if r["variant"] == v), key=lambda r: r["tpr_loss"])
        for v in VARIANTS
    }
    base = aggregate["frozen_original"]
    screens = {}
    for v in VARIANTS:
        a = aggregate[v]
        screens[v] = screen(
            base["normal_fpr"],
            a["normal_fpr"],
            a["worst_route_normal_fpr"],
            base["normal_grants"],
            a["normal_grants"],
            worst[v]["tpr_loss"],
        )
    result = {
        "status": "DEVELOPMENT_ONLY_NOT_PROMOTED",
        "eligible_for_test": False,
        "policy_changed": False,
        "fresh_validation_required": True,
        "primary": PRIMARY,
        "source_indexed_files_verified": checked,
        "protocol_sha256": prior.sha((output / "frozen_protocol.json").read_bytes()),
        "profiles": profiles,
        "invariants": invariants,
        "aggregate": aggregate,
        "worst_route_case_losses": worst,
        "development_screens": screens,
        "route_case_losses": losses,
        "rows": result_rows,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    hashes = {
        p.relative_to(output).as_posix(): prior.sha(p.read_bytes())
        for p in sorted(output.rglob("*"))
        if p.is_file()
    }
    (output / "file_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print(
        json.dumps(
            {
                "normal": {
                    v: {k: x for k, x in a.items() if k != "attack_cases"}
                    for v, a in aggregate.items()
                },
                "worst": worst,
                "screens": screens,
            },
            indent=2,
        )
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage13-zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.stage13_zip, args.output)
