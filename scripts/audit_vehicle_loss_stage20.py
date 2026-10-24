"""Frozen saved-output decomposition; no model inference, training or promotion."""

import argparse
import csv
import json
import sys
from pathlib import Path, PurePosixPath

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import run_vehicle_development_stage17 as prior  # noqa: E402

from coact_cps.vehicle.loss_attribution20 import (  # noqa: E402
    FACTORS,
    factorial_alarms,
    normalized,
    shapley,
)

STAGE19_INDEX_SHA = "00070e156ceeccd33caa66fa734aad8c2f07013e6075c15ac9727a6d154df39e"
CONTRASTS = (
    ("frozen_original", "history_control"),
    ("frozen_original", "calibrated"),
    ("history_control", "calibrated"),
)
ROUTES = ("civic13_10", "civic13_11", "civic13_12")


def status():
    return {
        "status": "DIAGNOSTIC_ONLY_NOT_PROMOTED",
        "eligible_for_test": False,
        "policy_changed": False,
        "model_training_performed": False,
        "model_inference_performed": False,
        "fresh_validation_required": True,
        "stage16_final_routes_evaluated": False,
    }


def verify_results(folder):
    folder = Path(folder).resolve()
    index = folder / "file_hashes.json"
    if not index.is_file() or prior.sha(index.read_bytes()) != STAGE19_INDEX_SHA:
        raise ValueError("Stage19 input binding mismatch; use the delivered Stage19 results")
    hashes = json.loads(index.read_bytes())
    for name, expected in hashes.items():
        relative = PurePosixPath(name)
        path = folder.joinpath(*relative.parts).resolve()
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not path.is_relative_to(folder)
            or not path.is_file()
            or prior.sha(path.read_bytes()) != expected
        ):
            raise ValueError("Stage19 indexed payload hash or confinement failure")
    return json.loads((folder / "summary.json").read_bytes()), len(hashes)


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(source, stage19, output):
    output, stage19 = Path(output), Path(stage19).resolve()
    if output.exists():
        raise ValueError("output must not already exist")
    z = prior.read_source(source)
    previous, checked_results = verify_results(stage19)
    protocol, lock, _, checked_source = prior.audit_source(z)
    models = {m["seed"]: m for m in lock["models"] if m["method"] == "coact_physics"}
    receipts = {(r["arm"], r["seed"]): r for r in previous["fit_receipts"]}
    cases = protocol["config"]["cases"]
    old_keys = {(r["route"], r["case"], r["seed"]) for r in previous["rows"]}
    expected = {(r, c, s) for r in ROUTES for c in cases for s in prior.SEEDS}
    if old_keys != expected or len(previous["rows"]) != len(expected):
        raise ValueError("Stage19 comparison cohort is incomplete")
    factorial_rows, attributions, node_rows = [], [], []
    for old_row in previous["rows"]:
        route, case, seed = (old_row[k] for k in ("route", "case", "seed"))
        context = prior.load_npz(z, f"test/contexts/{route}__{case}.npz")
        old = prior.load_npz(z, f"test/blocks/{route}__{case}__coact_physics__{seed}.npz")
        blocks = {"frozen_original": old}
        settings = {
            "frozen_original": (models[seed]["node_calibration"], models[seed]["threshold"])
        }
        for arm in ("history_control", "calibrated"):
            with np.load(
                stage19 / "blocks" / f"{route}__{case}__{arm}__{seed}.npz", allow_pickle=False
            ) as obj:
                blocks[arm] = {k: obj[k] for k in obj.files}
            receipt = receipts[arm, seed]
            settings[arm] = receipt["node_calibration"], receipt["threshold"]
        for method, block in blocks.items():
            if not np.array_equal(block["positions"], context["positions"]):
                raise ValueError("endpoint alignment mismatch")
            cal, cutoff = settings[method]
            standardized = normalized(block["nodes"], cal)
            score = standardized.max(axis=1)
            alarm = score > cutoff
            np.testing.assert_allclose(score, block["score"], rtol=0, atol=1e-10)
            if not np.array_equal(alarm, block["alarm"]):
                raise ValueError("saved endpoint alarm mismatch")
            if prior.metrics(context, alarm) != old_row[method]:
                raise ValueError("Stage19 saved metric mismatch")
            selected = context["labels"][context["positions"]]
            population = "attack-labelled endpoints"
            if case == "normal":
                selected = ~selected
                population = "normal endpoints"
            for j, channel in enumerate(("gnss", "wheel", "imu")):
                raw = block["nodes"][selected, j]
                node_rows.append(
                    {
                        "route": route,
                        "case": case,
                        "seed": seed,
                        "method": method,
                        "channel": channel,
                        "population": population,
                        "count": int(selected.sum()),
                        "raw_residual_median": float(np.median(raw)),
                        "raw_residual_p90": float(np.quantile(raw, 0.9)),
                        "normalized_median": float(np.median(standardized[selected, j])),
                        "channel_alarm_fraction": float(
                            (standardized[selected, j] > cutoff).mean()
                        ),
                    }
                )
        for reference, target in CONTRASTS:
            name = reference + "__to__" + target
            old_cal, old_cutoff = settings[reference]
            new_cal, new_cutoff = settings[target]
            alarms = factorial_alarms(
                blocks[reference]["nodes"],
                blocks[target]["nodes"],
                old_cal,
                new_cal,
                old_cutoff,
                new_cutoff,
            )
            values = {}
            for mask, alarm in alarms.items():
                metric = prior.metrics(context, alarm)
                row = {
                    "contrast": name,
                    "route": route,
                    "case": case,
                    "seed": seed,
                    "state": mask,
                    "residual_source": target if mask & 1 else reference,
                    "normalizer_source": target if mask & 2 else reference,
                    "threshold_source": target if mask & 4 else reference,
                    **metric,
                }
                factorial_rows.append(row)
                values[mask] = metric
            metrics = (
                ("normal_endpoint_fpr", "grant_fraction_all_epochs")
                if case == "normal"
                else ("active_endpoint_tpr", "active_grant_fraction")
            )
            for metric in metrics:
                cube = {i: m[metric] for i, m in values.items()}
                effect = shapley(cube)
                attributions.append(
                    {
                        "contrast": name,
                        "route": route,
                        "case": case,
                        "seed": seed,
                        "metric": metric,
                        "reference_value": cube[0],
                        "target_value": cube[7],
                        "total_change": cube[7] - cube[0],
                        **effect,
                    }
                )
    grouped = []
    for reference, target in CONTRASTS:
        contrast = reference + "__to__" + target
        for route in ROUTES:
            for case in cases:
                rows = [
                    a
                    for a in attributions
                    if a["contrast"] == contrast
                    and a["route"] == route
                    and a["case"] == case
                    and a["metric"]
                    == ("normal_endpoint_fpr" if case == "normal" else "active_endpoint_tpr")
                ]
                grouped.append(
                    {
                        "contrast": contrast,
                        "route": route,
                        "case": case,
                        "metric": rows[0]["metric"],
                        **{
                            key: float(np.mean([r[key] for r in rows]))
                            for key in ("reference_value", "target_value", "total_change", *FACTORS)
                        },
                    }
                )
    worst = {}
    states = []
    for reference, target in CONTRASTS:
        contrast = reference + "__to__" + target
        loss = min(
            (v for v in grouped if v["contrast"] == contrast and v["case"] != "normal"),
            key=lambda v: v["total_change"],
        )
        worst[contrast] = {
            **loss,
            "largest_negative_component": min(FACTORS, key=lambda f: loss[f]),
        }
        for mask in range(8):
            same = [r for r in factorial_rows if r["contrast"] == contrast and r["state"] == mask]
            normal = [r for r in same if r["case"] == "normal"]
            states.append(
                {
                    "contrast": contrast,
                    "state": mask,
                    "normal_fpr": float(np.mean([r["normal_endpoint_fpr"] for r in normal])),
                    "normal_grants": float(
                        np.mean([r["grant_fraction_all_epochs"] for r in normal])
                    ),
                    "worst_route_normal_fpr": max(
                        float(
                            np.mean(
                                [r["normal_endpoint_fpr"] for r in normal if r["route"] == route]
                            )
                        )
                        for route in ROUTES
                    ),
                    "worst_loss_case_tpr": float(
                        np.mean(
                            [
                                r["active_endpoint_tpr"]
                                for r in same
                                if r["route"] == loss["route"] and r["case"] == loss["case"]
                            ]
                        )
                    ),
                }
            )
    original_tpr = {}
    for route in ROUTES:
        for case in cases:
            if case != "normal":
                original_tpr[route, case] = float(
                    np.mean(
                        [
                            r["frozen_original"]["active_endpoint_tpr"]
                            for r in previous["rows"]
                            if r["route"] == route and r["case"] == case
                        ]
                    )
                )
    limits = []
    for state in states:
        same = [
            r
            for r in factorial_rows
            if r["contrast"] == state["contrast"] and r["state"] == state["state"]
        ]
        losses = []
        for (route, case), original in original_tpr.items():
            current = float(
                np.mean(
                    [
                        r["active_endpoint_tpr"]
                        for r in same
                        if r["route"] == route and r["case"] == case
                    ]
                )
            )
            losses.append(original - current)
        worst_loss = max(losses)
        passes = (
            state["normal_fpr"] <= 0.05
            and state["worst_route_normal_fpr"] <= 0.05
            and worst_loss <= 0.05
            and state["normal_grants"] >= previous["aggregate"]["frozen_original"]["normal_grants"]
        )
        limits.append(
            {
                "contrast": state["contrast"],
                "state": state["state"],
                "worst_route_case_tpr_loss_vs_frozen_original": worst_loss,
                "joint_development_criteria_passed": passes,
            }
        )
    root = Path(__file__).resolve().parents[1]
    code_paths = [
        "scripts/audit_vehicle_loss_stage20.py",
        "src/coact_cps/vehicle/loss_attribution20.py",
        "scripts/run_vehicle_development_stage17.py",
        "src/coact_cps/vehicle/physics17.py",
    ]
    result = {
        **status(),
        "source_zip_sha256": prior.SOURCE_SHA256,
        "stage19_manifest_sha256": STAGE19_INDEX_SHA,
        "source_indexed_files_verified": checked_source,
        "stage19_indexed_files_verified": checked_results,
        "factorial_rows": len(factorial_rows),
        "attribution_rows": len(attributions),
        "factor_bit_order": {str(i): f for i, f in enumerate(FACTORS)},
        "worst_detection_losses": worst,
        "state_summaries": states,
        "joint_criteria_checks": limits,
        "states_meeting_joint_criteria": sum(
            v["joint_development_criteria_passed"] for v in limits
        ),
        "code_sha256": {p: prior.sha((root / p).read_bytes()) for p in code_paths},
        "qualification": [
            "Previously inspected development routes only; no fresh validation.",
            "Shapley contributions describe deterministic component switches; "
            "they do not establish real-world causes.",
            "Interactions are shared over all six switch orders; component definitions matter.",
            "Raw residuals use the same original feature scaling, permitting this interchange.",
            "Absolute residuals cannot reconstruct prediction direction "
            "or prove physical accuracy.",
            "Counterfactual pipeline states are diagnostics, not selected or promoted repairs.",
        ],
    }
    output.mkdir(parents=True)
    write_csv(output / "factorial_metrics.csv", factorial_rows)
    write_csv(output / "endpoint_attributions.csv", attributions)
    write_csv(output / "route_case_attributions.csv", grouped)
    write_csv(output / "channel_response.csv", node_rows)
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    hashes = {p.name: prior.sha(p.read_bytes()) for p in sorted(output.iterdir()) if p.is_file()}
    (output / "file_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print(
        json.dumps(
            {
                "worst_detection_losses": worst,
                "factorial_rows": len(factorial_rows),
                "status": result["status"],
            },
            indent=2,
        )
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage13-zip", type=Path, required=True)
    parser.add_argument("--stage19-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.stage13_zip, args.stage19_dir, args.output)
