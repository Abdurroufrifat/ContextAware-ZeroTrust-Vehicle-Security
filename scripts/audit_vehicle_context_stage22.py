"""Audit attacker-influenced context using paired saved development histories."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import run_vehicle_development_stage17 as prior  # noqa: E402

from coact_cps.vehicle.context_integrity22 import paired_effect, verify_pair  # noqa: E402
from coact_cps.vehicle.imu_guard21 import boundaries, causal_context  # noqa: E402

STAGE21_MANIFEST = "2ad92ff21e4063db174b7c4c3fef5e7280513c52dfef65ba66e7c2abc1415ff9"
VARIANTS = ("global_imu", "context_unbounded", "context_bounded")
ROUTES = ("civic13_10", "civic13_11", "civic13_12")


def verify_stage21(directory):
    root = Path(directory).resolve()
    raw = (root / "file_hashes.json").read_bytes()
    if prior.sha(raw) != STAGE21_MANIFEST:
        raise ValueError("Stage21 manifest hash mismatch")
    index = json.loads(raw)
    if len(index) != 353:
        raise ValueError("Stage21 index count mismatch")
    for relative, digest in index.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or prior.sha(path.read_bytes()) != digest:
            raise ValueError("Stage21 indexed payload mismatch")
    summary = json.loads((root / "summary.json").read_bytes())
    frozen = json.loads((root / "frozen_protocol.json").read_bytes())
    if frozen["source_zip_sha256"] != prior.SOURCE_SHA256:
        raise ValueError("Stage21 source lineage mismatch")
    return summary, len(index)


def run(source, stage21, output):
    output, stage21 = Path(output), Path(stage21)
    if output.exists():
        raise ValueError("output must not already exist")
    z = prior.read_source(source)
    protocol, lock, source_rows, checked = prior.audit_source(z)
    saved21, checked21 = verify_stage21(stage21)
    models = {m["seed"]: m for m in lock["models"] if m["method"] == "coact_physics"}
    profiles = {p["seed"]: p for p in saved21["profiles"]}
    cases = protocol["config"]["cases"]
    chosen = [r for r in source_rows if r["trip"] in ROUTES]
    if len(chosen) != 117 or {(r["trip"], r["case"], r["seed"]) for r in chosen} != {
        (r, c, s) for r in ROUTES for c in cases for s in prior.SEEDS
    }:
        raise ValueError("complete fixed117 development blocks required")
    contexts, blocks = {}, {}
    for row in chosen:
        c, b = prior.checked_block(z, row, models[row["seed"]])
        contexts[row["trip"], row["case"]] = c
        blocks[row["trip"], row["case"], row["seed"]] = b
    qcontext = {}
    for (route, case), c in contexts.items():
        verify_pair(c, contexts[route, "normal"])
        qcontext[route, case] = causal_context(
            c["features"],
            c["time_ms"],
            c["start_ms"],
        )[c["positions"]]
    root = Path(__file__).resolve().parents[1]
    code = [
        "scripts/audit_vehicle_context_stage22.py",
        "src/coact_cps/vehicle/context_integrity22.py",
        "src/coact_cps/vehicle/imu_guard21.py",
        "scripts/run_vehicle_development_stage17.py",
        "src/coact_cps/vehicle/physics17.py",
    ]
    frozen = {
        "schema": "vehicle-context-integrity-stage22-v1",
        "source_zip_sha256": prior.SOURCE_SHA256,
        "stage21_manifest_sha256": STAGE21_MANIFEST,
        "code_sha256": {n: prior.sha((root / n).read_bytes()) for n in code},
        "design_sha256": prior.sha((root / "STAGE22_FROZEN_DESIGN.txt").read_bytes()),
        "variants": list(VARIANTS),
        "routes": list(ROUTES),
        "intervention": "Swap only context history; attack residuals, labels, base grants fixed.",
        "oracle": "Paired normal history is offline-only and unavailable at deployment.",
        "primary_cap": 1.0,
        "retuning": False,
        **prior.status_receipt(),
    }
    # All inputs and pair alignments are checked before producing results.
    output.mkdir(parents=True)
    (output / "blocks").mkdir()
    (output / "frozen_protocol.json").write_text(json.dumps(frozen, indent=2) + "\n")
    rows = []
    for route in ROUTES:
        clean_context = qcontext[route, "normal"]
        for case in cases:
            c = contexts[route, case]
            active = c["labels"][c["positions"]]
            for seed in prior.SEEDS:
                model, p = models[seed], profiles[seed]
                cal, threshold = model["node_calibration"], model["threshold"]
                if threshold != p["original_threshold"]:
                    raise ValueError("original threshold mismatch")
                b = blocks[route, case, seed]
                nodes = (b["nodes"] - np.asarray(cal["median"])) / np.asarray(cal["scale"])
                for variant in VARIANTS:
                    if variant == "global_imu":
                        reported_cut = clean_cut = np.full(len(nodes), p["global_imu_cutoff"])
                    else:
                        cap = 1.0 if variant == "context_bounded" else None
                        reported_cut = boundaries(
                            qcontext[route, case],
                            p["profile"],
                            p["conditional_quantile"],
                            threshold,
                            cap,
                        )
                        clean_cut = boundaries(
                            clean_context, p["profile"], p["conditional_quantile"], threshold, cap
                        )
                    effect, reported, clean = paired_effect(
                        nodes,
                        threshold,
                        reported_cut,
                        clean_cut,
                        active,
                    )
                    name = f"{route}__{case}__{variant}__{seed}.npz"
                    grant = np.zeros(len(c["labels"]), dtype=bool)
                    grant[c["positions"]] = c["base_grant"][c["positions"]] & ~reported
                    with np.load(stage21 / "blocks" / name, allow_pickle=False) as old:
                        for key, value in (
                            ("positions", c["positions"]),
                            ("imu_boundary", reported_cut),
                            ("alarm", reported),
                            ("grant", grant),
                        ):
                            if not np.array_equal(old[key], value):
                                raise ValueError(f"Stage21 replay mismatch: {name}/{key}")
                    speed = (nodes[:, :2] > threshold).any(axis=1)
                    if np.any(speed & ~(reported & clean)):
                        raise ValueError("original speed alarm lost under context swap")
                    if variant == "global_imu" and not np.array_equal(reported, clean):
                        raise ValueError("context-independent control changed")
                    np.savez_compressed(
                        output / "blocks" / name,
                        positions=c["positions"],
                        active=active,
                        reported_boundary=reported_cut,
                        clean_boundary=clean_cut,
                        reported_alarm=reported,
                        clean_alarm=clean,
                    )
                    rows.append(dict(route=route, case=case, seed=seed, variant=variant, **effect))
    grouped = []
    for variant in VARIANTS:
        for route in ROUTES:
            for case in cases:
                if case == "normal":
                    continue
                group = [
                    r
                    for r in rows
                    if (r["variant"], r["route"], r["case"]) == (variant, route, case)
                ]
                grouped.append(
                    dict(
                        variant=variant,
                        route=route,
                        case=case,
                        mean_net_tpr_loss=float(np.mean([r["active_net_tpr_loss"] for r in group])),
                        mean_lost_fraction=float(
                            np.mean([r["active_lost_count"] / r["active_count"] for r in group])
                        ),
                        mean_gained_fraction=float(
                            np.mean([r["active_gained_count"] / r["active_count"] for r in group])
                        ),
                    )
                )
    result = {
        **prior.status_receipt(),
        "policy_changed": False,
        "model_training_or_inference": False,
        "stage13_indexed_files_verified": checked,
        "stage21_indexed_files_verified": checked21,
        "stage21_blocks_replayed": len(rows),
        "original_speed_alarms_preserved": True,
        "global_negative_control_zero_effect": True,
        "worst_context_path_losses": {
            v: max((r for r in grouped if r["variant"] == v), key=lambda r: r["mean_net_tpr_loss"])
            for v in VARIANTS
        },
        "route_case_seed_means": grouped,
        "rows": rows,
        "interpretation": (
            "Diagnostic context-path intervention on reused injected attacks; no new candidate."
        ),
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    index = {
        p.relative_to(output).as_posix(): prior.sha(p.read_bytes())
        for p in sorted(output.rglob("*"))
        if p.is_file()
    }
    (output / "file_hashes.json").write_text(json.dumps(index, indent=2) + "\n")
    print(json.dumps(result["worst_context_path_losses"], indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage13-zip", type=Path, required=True)
    parser.add_argument("--stage21-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.stage13_zip, args.stage21_results, args.output)
