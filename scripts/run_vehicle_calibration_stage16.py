"""Prospective calibration ablation with fresh validation and locked test routes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from run_vehicle_ai_stage13 import base_grants, environment, load_adapter, score_seed
from vehicle_ai_data13 import CHANNELS, Scaling, load_trip, windows
from vehicle_native_archive import RangeArchive, validate_npy
from vehicle_recovery_scenarios10b import perturb

from coact_cps.vehicle.ai_models13 import aggregate_scores, score_nodes
from coact_cps.vehicle.calibration16 import check_reservations, fit_threshold, validation_gate
from coact_cps.vehicle.native_detectors import ModelConfig

OUT = Path("artifacts/vehicle_calibration_stage16")
OLD = Path("artifacts/vehicle_ai_stage13r1")
NEW_SOURCES = [
    "src/coact_cps/vehicle/calibration16.py",
    "scripts/run_vehicle_calibration_stage16.py",
    "scripts/audit_vehicle_calibration_stage16.py",
    "tests/test_vehicle_calibration16.py",
    "VEHICLE_STAGE16_DESIGN.txt",
    "VEHICLE_STAGE16_SELECTION.json",
    "VEHICLE_STAGE16_INDEX.json",
]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def source_hash(path):
    path = Path(path)
    data = path.read_bytes()
    return sha(data.replace(b"\r\n", b"\n") if path.suffix in {".py", ".json", ".txt"} else data)


def freeze(root):
    target = root / OUT / "protocol.json"
    if target.exists():
        return verify(root)
    plan = read(root / "VEHICLE_STAGE16_SELECTION.json")
    check_reservations(plan["trips"], plan["prior_trips"])
    old = read(root / OLD / "protocol.json")
    lock = read(root / OLD / "fit/model_lock.json")
    if read(root / OLD / "fit/independent_audit.json")["status"] != "PASS" or lock[
        "protocol_sha256"
    ] != sha((root / OLD / "protocol.json").read_bytes()):
        raise ValueError("audited_original_fit_required")
    for name, expected in old["source_sha256"].items():
        if source_hash(root / name) != expected:
            raise ValueError(f"original_source_changed:{name}")
    specs = [v for v in lock["models"] if v["method"] == "coact_physics"]
    if sorted(v["seed"] for v in specs) != [7, 19, 43]:
        raise ValueError("three_frozen_physics_models_required")
    index = read(root / "VEHICLE_STAGE16_INDEX.json")
    central = root / "data/reference/vehicle_calibration_stage16/chunk10_central.bin"
    if sha(central.read_bytes()) != index["central_sha256"]:
        raise ValueError("metadata_fingerprint_mismatch")
    protocol = dict(
        schema="coact-calibration-stage16-v1",
        trips=plan["trips"],
        index=index,
        model_specs=specs,
        scaler=lock["scaler"],
        config=old["config"],
        old_model_lock_sha256=sha((root / OLD / "fit/model_lock.json").read_bytes()),
        old_protocol_sha256=sha((root / OLD / "protocol.json").read_bytes()),
        source_sha256={**old["source_sha256"], **{n: source_hash(root / n) for n in NEW_SOURCES}},
        environment=environment(),
        quantile=0.95,
        criteria=dict(
            normal_fpr_max=0.05, normal_fpr_reduction_min=0.01, per_case_tpr_loss_max=0.05
        ),
        status="EXPERIMENTAL_NOT_PROMOTED",
    )
    for spec in specs:
        if sha((root / spec["checkpoint"]).read_bytes()) != spec["checkpoint_sha256"]:
            raise ValueError("original_checkpoint_changed")
    write(target, protocol)
    return protocol


def verify(root):
    p = read(root / OUT / "protocol.json")
    for name, expected in p["source_sha256"].items():
        if source_hash(root / name) != expected:
            raise ValueError(f"frozen_source_changed:{name}")
    if environment() != p["environment"]:
        raise ValueError("stage16_environment_changed")
    if sha((root / OLD / "fit/model_lock.json").read_bytes()) != p["old_model_lock_sha256"]:
        raise ValueError("original_model_lock_changed")
    for spec in p["model_specs"]:
        if sha((root / spec["checkpoint"]).read_bytes()) != spec["checkpoint_sha256"]:
            raise ValueError("original_checkpoint_changed")
    return p


def hash_outputs(folder):
    return {
        p.relative_to(folder).as_posix(): sha(p.read_bytes())
        for p in sorted(folder.rglob("*"))
        if p.is_file() and p.name not in {"file_hashes.json", "independent_audit.json"}
    }


def verify_phase(root, phase, audited=True):
    folder = root / OUT / phase
    files = read(folder / "file_hashes.json")
    for name, expected in files.items():
        if sha((folder / name).read_bytes()) != expected:
            raise ValueError("phase_output_changed")
    summary = read(folder / "summary.json")
    for name, expected in summary["raw_sha256"].items():
        if sha((root / name).read_bytes()) != expected:
            raise ValueError("phase_raw_data_changed")
    if summary["protocol_sha256"] != sha((root / OUT / "protocol.json").read_bytes()):
        raise ValueError("phase_protocol_binding_failed")
    if phase != "calibration" and summary["threshold_lock_sha256"] != sha(
        (root / OUT / "calibration/threshold_lock.json").read_bytes()
    ):
        raise ValueError("phase_threshold_binding_failed")
    if audited:
        audit = read(folder / "independent_audit.json")
        if audit["status"] != "PASS" or audit["phase_hashes_sha256"] != sha(
            (folder / "file_hashes.json").read_bytes()
        ):
            raise ValueError("matching_saved_output_audit_required")
    return summary


def authorize_phase(root, phase):
    if phase not in {"calibration", "validation", "test"}:
        raise ValueError("invalid_phase")
    if phase in {"validation", "test"}:
        verify_phase(root, "calibration")
    if phase == "test":
        summary = verify_phase(root, "validation")
        if not summary["gate"]["passed"]:
            raise ValueError("validation_failed_test_routes_remain_locked")


def acquire(root, protocol, phase):
    authorize_phase(root, phase)
    index = protocol["index"]
    reader = RangeArchive(
        index["archive_url"],
        index["archive_size"],
        root / "data/cache/vehicle_calibration_stage16/10",
    )
    for trip in protocol["trips"]:
        if trip["phase"] != phase:
            continue
        for segment, part in zip(trip["segments"], trip["parts"], strict=True):
            records = []
            for e in index["entries"]:
                if e["trip_id"] != trip["id"] or e["segment"] != segment:
                    continue
                data = reader.entry(e)
                name = e["name"].split("/processed_log/", 1)[1]
                channel, suffix = name.rsplit("/", 1)
                arr = validate_npy(data, None if suffix == "t" else CHANNELS[channel])
                path = root / part / "processed_log" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists() and path.read_bytes() != data:
                    raise ValueError("existing_raw_bytes_conflict")
                path.write_bytes(data)
                records.append(
                    dict(
                        path="processed_log/" + name,
                        sha256=sha(data),
                        bytes=len(data),
                        shape=list(arr.shape),
                        source_zip_entry=e["name"],
                        entry_crc32=e["crc32"],
                    )
                )
            if len(records) != 6:
                raise ValueError("six_source_arrays_per_part_required")
            write(
                root / part / "source_manifest.json",
                dict(
                    route_id=trip["route_id"],
                    segment_id=trip["route_id"] + "/" + str(segment),
                    source_url=index["archive_url"],
                    files=sorted(records, key=lambda r: r["path"]),
                ),
            )
        print("Acquired", phase, trip["id"], flush=True)


def metric_row(trip, seed, case, context, block):
    positions = block["positions"]
    active = context["labels"][positions] if case != "normal" else np.ones(len(positions), bool)
    if not active.any():
        raise ValueError("active_scored_endpoints_required")
    return dict(
        trip=trip,
        seed=seed,
        case=case,
        baseline=float(block["baseline_alarm"][active].mean()),
        candidate=float(block["candidate_alarm"][active].mean()),
        base_grant=float(context["base_grant"].mean()),
        baseline_grant=float(block["baseline_grant"].mean()),
        candidate_grant=float(block["candidate_grant"].mean()),
    )


def run_phase(root, phase):
    p = verify(root)
    authorize_phase(root, phase)
    folder = root / OUT / phase
    if (folder / "summary.json").exists():
        return verify_phase(root, phase, audited=False)
    if (folder / "started.json").exists():
        raise ValueError("incomplete_phase_requires_registered_recovery")
    acquire(root, p, phase)
    write(
        folder / "started.json",
        dict(protocol_sha256=sha((root / OUT / "protocol.json").read_bytes())),
    )
    cfg = p["config"]
    torch.set_num_threads(cfg["threads"])
    original = read(root / cfg["reference_contract"])
    model = ModelConfig(**original["model"])
    scale = Scaling(tuple(p["scaler"]["mean"]), tuple(p["scaler"]["scale"]))
    adapters = {s["seed"]: load_adapter(root, s, cfg) for s in p["model_specs"]}
    lock = read(root / OUT / "calibration/threshold_lock.json") if phase != "calibration" else None
    calibration = {s["seed"]: {} for s in p["model_specs"]}
    blocks, metrics, raw_hashes = [], [], {}
    for trip in p["trips"]:
        if trip["phase"] != phase:
            continue
        frames = load_trip(root, trip)
        for part in trip["parts"]:
            for path in sorted((root / part).rglob("*")):
                if path.is_file():
                    raw_hashes[path.relative_to(root).as_posix()] = sha(path.read_bytes())
        for case in ["normal"] if phase == "calibration" else cfg["cases"]:
            changed = perturb(frames, case, model, original["session_config"])
            w = windows(changed, cfg["window"], model)
            x = scale.transform(w.x)
            labels = np.array(
                [
                    case != "normal"
                    and max(
                        abs(a.gnss_speed - b.gnss_speed),
                        abs(a.wheel_speed - b.wheel_speed),
                        abs(a.imu_accel - b.imu_accel),
                    )
                    > 1e-9
                    for a, b in zip(changed, frames, strict=True)
                ]
            )
            base, _, _ = base_grants(changed, original)
            context = dict(labels=labels, base_grant=base, positions=w.positions)
            context_path = f"contexts/{trip['id']}__{case}.npz"
            (folder / "contexts").mkdir(exist_ok=True)
            np.savez_compressed(folder / context_path, **context)
            for spec in p["model_specs"]:
                seed = spec["seed"]
                nodes = score_nodes(adapters[seed], x, cfg, score_seed(seed, trip["id"]))
                scores = aggregate_scores(nodes, spec["node_calibration"], spec["method"])
                block = dict(nodes=nodes, score=scores, positions=w.positions)
                if phase == "calibration":
                    calibration[seed][trip["id"]] = scores
                else:
                    old_alarm = scores > spec["threshold"]
                    new_alarm = scores > lock["seeds"][str(seed)]["threshold"]
                    old_grant, new_grant = np.zeros(len(frames), bool), np.zeros(len(frames), bool)
                    old_grant[w.positions] = base[w.positions] & ~old_alarm
                    new_grant[w.positions] = base[w.positions] & ~new_alarm
                    block.update(
                        baseline_alarm=old_alarm,
                        candidate_alarm=new_alarm,
                        baseline_grant=old_grant,
                        candidate_grant=new_grant,
                    )
                    metrics.append(metric_row(trip["id"], seed, case, context, block))
                block_path = f"blocks/{trip['id']}__{case}__{seed}.npz"
                (folder / "blocks").mkdir(exist_ok=True)
                np.savez_compressed(folder / block_path, **block)
                blocks.append(
                    dict(
                        trip=trip["id"],
                        seed=seed,
                        case=case,
                        block=block_path,
                        context=context_path,
                    )
                )
            print("Scored", phase, trip["id"], case, len(w.positions), flush=True)
    summary = dict(
        phase=phase,
        protocol_sha256=sha((root / OUT / "protocol.json").read_bytes()),
        rows=blocks,
        metrics=metrics,
        raw_sha256=raw_hashes,
        fitting=False,
        status="EXPERIMENTAL_NOT_PROMOTED",
    )
    if phase == "calibration":
        write(
            folder / "threshold_lock.json",
            dict(
                seeds={
                    str(s["seed"]): fit_threshold(calibration[s["seed"]], s["threshold"])
                    for s in p["model_specs"]
                },
                protocol_sha256=summary["protocol_sha256"],
            ),
        )
    else:
        summary["threshold_lock_sha256"] = sha(
            (root / OUT / "calibration/threshold_lock.json").read_bytes()
        )
        if phase == "validation":
            summary["gate"] = validation_gate(metrics, cfg["cases"])
    write(folder / "summary.json", summary)
    write(folder / "file_hashes.json", hash_outputs(folder))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--phase", choices=["freeze", "calibration", "validation", "test", "all"], required=True
    )
    args = parser.parse_args()
    root = args.root.resolve()
    if args.phase == "freeze":
        freeze(root)
    elif args.phase == "all":
        from audit_vehicle_calibration_stage16 import audit

        freeze(root)
        for phase in ["calibration", "validation"]:
            run_phase(root, phase)
            audit(root, phase)
        gate = verify_phase(root, "validation")["gate"]
        print(json.dumps(gate, indent=2))
        if gate["passed"]:
            run_phase(root, "test")
            audit(root, "test")
        else:
            print("NOT PROMOTED. Test routes remain unopened.")
    else:
        run_phase(root, args.phase)


if __name__ == "__main__":
    main()
