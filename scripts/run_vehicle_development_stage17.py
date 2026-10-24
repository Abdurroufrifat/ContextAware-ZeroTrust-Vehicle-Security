"""Replay an explicitly reused Stage13 archive; never evaluates Stage16 routes."""

import argparse
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from coact_cps.vehicle.physics17 import (  # noqa: E402
    STATUS,
    fit_physical_scale,
    hybrid_score,
    physical_residual,
    threshold_from_clean,
)

SOURCE_SHA256 = "217d28bdafb0fe57c31a6c1befb630b9759b62463f43362a5a8b9f67fcb67f5c"
ROOT = "vehicle_ai_stage13r1/"
EVALUATION_ROUTES = ("civic13_10", "civic13_11", "civic13_12")
SEEDS = (7, 19, 43)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def status_receipt():
    return {
        "status": STATUS,
        "eligible_for_test": False,
        "fresh_validation_required": True,
        "evidence_role": "REUSED_STAGE13_EXPLORATORY_DEVELOPMENT",
        "stage16_final_routes_accessed": False,
    }


def read_source(path):
    raw = Path(path).read_bytes()
    if sha(raw) != SOURCE_SHA256:
        raise ValueError("source archive hash mismatch; use the original Windows Stage13 ZIP")
    z = zipfile.ZipFile(io.BytesIO(raw))
    names = z.namelist()
    if len(names) != len(set(names)) or z.testzip() is not None:
        raise ValueError("invalid archive")
    return z


def read_json(z, relative):
    return json.loads(z.read(ROOT + relative))


def load_npz(z, relative):
    with np.load(io.BytesIO(z.read(ROOT + relative)), allow_pickle=False) as obj:
        return {k: obj[k] for k in obj.files}


def audit_source(z):
    protocol = read_json(z, "protocol.json")
    if protocol["schema"] != "vehicle-ai-stage13-recovery-r1":
        raise ValueError("Stage13 source schema required")
    count = 0
    for phase in ("fit", "validation", "test"):
        index = read_json(z, phase + "/file_hashes.json")
        for relative, expected in index.items():
            if sha(z.read(ROOT + phase + "/" + relative)) != expected:
                raise ValueError("indexed payload mismatch")
            count += 1
    lock = read_json(z, "fit/model_lock.json")
    if lock["protocol_sha256"] != sha(z.read(ROOT + "protocol.json")):
        raise ValueError("protocol lock mismatch")
    rows = []
    for phase in ("validation", "test"):
        summary = read_json(z, phase + "/summary.json")
        if summary["protocol_sha256"] != lock["protocol_sha256"] or summary[
            "model_lock_sha256"
        ] != sha(z.read(ROOT + "fit/model_lock.json")):
            raise ValueError("summary lineage mismatch")
        for row in summary["rows"]:
            if row["method"] == "coact_physics":
                rows.append(dict(row, phase=phase))
    return protocol, lock, rows, count


def checked_block(z, row, model):
    phase = row["phase"]
    c = load_npz(z, phase + "/" + row["context"])
    b = load_npz(z, phase + "/" + row["block"])
    n = len(c["features"])
    p = c["positions"]
    if (
        p.ndim != 1
        or p.dtype.kind not in "iu"
        or np.any(p < 0)
        or np.any(p >= n)
        or np.any(np.diff(p) <= 0)
        or not np.array_equal(p, b["positions"])
        or any(c[k].shape != (n,) or c[k].dtype != bool for k in ("labels", "base_grant"))
        or b["nodes"].shape != (len(p), 3)
    ):
        raise ValueError("invalid context or block dimensions")
    cal = model["node_calibration"]
    expected = ((b["nodes"] - np.asarray(cal["median"])) / np.asarray(cal["scale"])).max(axis=1)
    if not np.isfinite(expected).all() or not np.allclose(b["score"], expected, rtol=0, atol=1e-10):
        raise ValueError("saved score mismatch")
    alarm = expected > model["threshold"]
    grant = np.zeros(n, dtype=bool)
    grant[p] = c["base_grant"][p] & ~alarm
    if not np.array_equal(alarm, b["alarm"]) or not np.array_equal(grant, b["grant"]):
        raise ValueError("saved alarm or grant mismatch")
    return c, b


def metrics(context, alarm):
    p = context["positions"]
    labels = context["labels"]
    active = labels[p]
    grant = np.zeros(len(labels), dtype=bool)
    grant[p] = context["base_grant"][p] & ~alarm
    return {
        "normal_endpoint_fpr": float(np.mean(alarm[~active])) if (~active).any() else None,
        "active_endpoint_tpr": float(np.mean(alarm[active])) if active.any() else None,
        "active_endpoint_count": int(active.sum()),
        "normal_endpoint_count": int((~active).sum()),
        "grant_fraction_all_epochs": float(grant.mean()),
        "active_grant_fraction": float(grant[labels].mean()) if labels.any() else None,
    }


def run(source, output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be empty")
    z = read_source(source)
    protocol, lock, source_rows, checked_files = audit_source(z)
    models = {m["seed"]: m for m in lock["models"] if m["method"] == "coact_physics"}
    cases = protocol["config"]["cases"]
    selected = {}
    for row in source_rows:
        if row["trip"] in EVALUATION_ROUTES or (
            row["trip"] in ("civic13_07", "civic13_08", "civic13_09") and row["case"] == "normal"
        ):
            key = (row["trip"], row["case"], row["seed"])
            if key in selected or row["seed"] not in SEEDS:
                raise ValueError("duplicate or unexpected row")
            selected[key] = row
    expected_keys = {(r, c, s) for r in EVALUATION_ROUTES for c in cases for s in SEEDS}
    expected_keys |= {
        (r, "normal", s) for r in ("civic13_07", "civic13_08", "civic13_09") for s in SEEDS
    }
    if set(selected) != expected_keys:
        raise ValueError("incomplete development roles")
    loaded = {k: checked_block(z, row, models[k[2]]) for k, row in selected.items()}
    # Fit once per route, not once per seed. No attack rows enter scale/threshold fit.
    residuals = {}
    for (route, case, _seed), (c, _) in loaded.items():
        key = route, case
        if key not in residuals:
            residuals[key] = physical_residual(c["features"], c["time_ms"], c["start_ms"])
    scale_inputs = {}
    for route in ("civic13_07", "civic13_08"):
        context = loaded[route, "normal", 7][0]
        if context["labels"].any():
            raise ValueError("scale route contains attack labels")
        scale_inputs[route] = residuals[route, "normal"]
    med, scale = fit_physical_scale(scale_inputs)

    def candidate(key):
        c, b = loaded[key]
        cal = models[key[2]]["node_calibration"]
        return hybrid_score(
            b["nodes"], cal["median"], cal["scale"], residuals[key[:2]][c["positions"]], med, scale
        )

    thresholds = {}
    for seed in SEEDS:
        if loaded["civic13_09", "normal", seed][0]["labels"].any():
            raise ValueError("threshold route contains attack labels")
        thresholds[seed] = threshold_from_clean(
            candidate(("civic13_09", "normal", seed)), models[seed]["threshold"]
        )
    rows = []
    output.mkdir(parents=True, exist_ok=True)
    (output / "blocks").mkdir()
    for route in EVALUATION_ROUTES:
        for case in cases:
            for seed in SEEDS:
                key = route, case, seed
                c, b = loaded[key]
                score = candidate(key)
                alarm = score > thresholds[seed]
                row = {
                    "route": route,
                    "case": case,
                    "seed": seed,
                    "baseline": metrics(c, b["alarm"]),
                    "candidate": metrics(c, alarm),
                }
                rows.append(row)
                np.savez_compressed(
                    output / "blocks" / f"{route}__{case}__{seed}.npz",
                    positions=c["positions"],
                    score=score,
                    alarm=alarm,
                )
    normal = [r for r in rows if r["case"] == "normal"]
    attack = [r for r in rows if r["case"] != "normal"]
    aggregate = {}
    for method in ("baseline", "candidate"):
        aggregate[method] = {
            "normal_fpr_macro_route_seed": float(
                np.mean([r[method]["normal_endpoint_fpr"] for r in normal])
            ),
            "normal_grants_macro_route_seed": float(
                np.mean([r[method]["grant_fraction_all_epochs"] for r in normal])
            ),
            "attack_cases": {
                case: {
                    metric: float(np.mean([r[method][metric] for r in attack if r["case"] == case]))
                    for metric in ("active_endpoint_tpr", "active_grant_fraction")
                }
                for case in cases
                if case != "normal"
            },
        }
    route_losses = []
    for route in EVALUATION_ROUTES:
        for case in cases:
            if case == "normal":
                continue
            group = [r for r in attack if r["route"] == route and r["case"] == case]
            loss = np.mean(
                [
                    r["baseline"]["active_endpoint_tpr"] - r["candidate"]["active_endpoint_tpr"]
                    for r in group
                ]
            )
            route_losses.append({"route": route, "case": case, "tpr_loss": float(loss)})
    report = {
        **status_receipt(),
        "source_zip_sha256": SOURCE_SHA256,
        "source_indexed_files_verified": checked_files,
        "blocks_compared": len(rows),
        "physical_scale": {"median": med, "iqr": scale},
        "thresholds": thresholds,
        "roles": {
            "scale_normal": ["civic13_07", "civic13_08"],
            "threshold_normal": ["civic13_09"],
            "exploratory": list(EVALUATION_ROUTES),
        },
        "aggregate": aggregate,
        "worst_route_case_tpr_loss": max(route_losses, key=lambda x: x["tpr_loss"]),
        "route_case_tpr_losses": route_losses,
        "rows": rows,
        "limitations": [
            "All evaluation routes have previously been inspected.",
            "One vehicle/logger; controlled perturbations, not naturally occurring attacks.",
            "Consistency cannot detect physically coherent common-mode spoofing.",
            "Acceleration axis, gravity, grade and timing can cause residual errors.",
            "No model inference, real-time latency or deployment validation performed.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    manifest = {
        str(p.relative_to(output)).replace("\\", "/"): sha(p.read_bytes())
        for p in sorted(output.rglob("*"))
        if p.is_file()
    }
    (output / "file_hashes.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "status",
                    "blocks_compared",
                    "physical_scale",
                    "thresholds",
                    "worst_route_case_tpr_loss",
                )
            },
            indent=2,
        )
    )
    print(
        json.dumps(
            {m: {k: v for k, v in a.items() if k != "attack_cases"} for m, a in aggregate.items()},
            indent=2,
        )
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage13-zip", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.stage13_zip, args.output)
