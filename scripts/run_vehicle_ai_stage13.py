"""Frozen normal-only vehicle AI comparison. Existing phases cannot be overwritten."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import zlib
from pathlib import Path

import joblib
import numpy as np
import torch
from prepare_vehicle_ai_stage13 import verify_upstream
from run_vehicle_native_stage6 import packets
from run_vehicle_witness_stage12 import SOURCE_FILES as OLD_SOURCES
from vehicle_ai_data13 import Scaling, array_hash, features, load_trip, select_trips, windows
from vehicle_native_archive import central_location, parse_central
from vehicle_protected_data11 import selected_index
from vehicle_recovery_scenarios10b import perturb

from coact_cps.vehicle.ai_models13 import (
    METHODS,
    aggregate_scores,
    create_adapter,
    fit_adapter,
    fit_node_calibration,
    score_nodes,
    threshold,
)
from coact_cps.vehicle.integrity_policy import IntegrityPolicyEngine
from coact_cps.vehicle.native_detectors import ModelConfig
from coact_cps.vehicle.policy import SOURCES, VehicleRequest

OUT = "artifacts/vehicle_ai_stage13r1"
RAW = "data/raw/vehicle_native_stage13/manifest.json"
NEW_SOURCES = [
    "scripts/vehicle_ai_data13.py",
    "scripts/acquire_vehicle_ai_stage13.py",
    "scripts/prepare_vehicle_ai_stage13.py",
    "scripts/run_vehicle_ai_stage13.py",
    "scripts/audit_vehicle_ai_stage13.py",
    "src/coact_cps/vehicle/ai_models13.py",
    "src/coact_cps/vehicle/ai_policy13.py",
    "src/coact_cps/models/temporal_graph.py",
    "src/coact_cps/models/__init__.py",
    "src/coact_cps/models/operator_state.py",
    "src/coact_cps/models/collaboration_policy.py",
    "configs/vehicle_ai_stage13.json",
    "VEHICLE_STAGE13_SELECTION.json",
    "VEHICLE_STAGE13_INDEX.json",
    "VEHICLE_STAGE13_UPSTREAM.json",
    "requirements_vehicle_ai_stage13.txt",
]


def sha(b):
    return hashlib.sha256(b).hexdigest()


def source_digest(path):
    path = Path(path)
    b = path.read_bytes()
    return sha(b.replace(b"\r\n", b"\n") if path.suffix in {".py", ".json", ".txt"} else b)


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def environment():
    return dict(
        python=platform.python_version(),
        platform=platform.platform(),
        packages={
            n: importlib.metadata.version(n)
            for n in ["torch", "numpy", "scipy", "scikit-learn", "joblib", "einops", "cryptography"]
        },
    )


def source_files(root):
    paths = set(OLD_SOURCES + NEW_SOURCES)
    paths.update(
        p.relative_to(root).as_posix()
        for p in (root / "third_party/vehicle_ai_stage13/catch").rglob("*.py")
    )
    paths.update(
        p.relative_to(root).as_posix()
        for p in (root / "data/reference/vehicle_ai_stage13").rglob("*")
        if p.is_file()
    )
    return sorted(paths)


def verify_selection(root):
    plan = json.loads((root / "VEHICLE_STAGE13_SELECTION.json").read_bytes())
    index = json.loads((root / "VEHICLE_STAGE13_INDEX.json").read_bytes())
    records = []
    for k, a in index["archives"].items():
        base = root / "data/reference/vehicle_ai_stage13"
        central = (base / f"chunk{k}_central.bin").read_bytes()
        tail = (base / f"chunk{k}_tail.bin").read_bytes()
        _offset, size, count = central_location(tail, a["archive_size"])
        assert len(central) == size and sha(central) == a["central_directory_sha256"]
        records.extend(parse_central(central, count))
    assert (
        select_trips(records, plan["prior_trips"]) == plan["trips"]
        and selected_index(records, plan["trips"]) == index["entries"]
    )
    return plan


def verify_entry(path, item, entry):
    b = Path(path).read_bytes()
    if (
        item["source_zip_entry"] != entry["name"]
        or item["path"] != "processed_log/" + entry["name"].split("/processed_log/", 1)[1]
        or item["entry_crc32"] != entry["crc32"]
        or len(b) != entry["size"]
        or item["bytes"] != len(b)
        or zlib.crc32(b) != entry["crc32"]
    ):
        raise ValueError("central_entry_provenance_mismatch")


def freeze(root):
    root = Path(root).resolve()
    verify_upstream(root)
    plan = verify_selection(root)
    cfg = json.loads((root / "configs/vehicle_ai_stage13.json").read_bytes())
    if cfg["window"] != 32 or cfg["quantile"] != 0.95 or cfg["methods"] != list(METHODS):
        raise ValueError("fixed_protocol_required")
    manifest = json.loads((root / RAW).read_bytes())
    index = json.loads((root / "VEHICLE_STAGE13_INDEX.json").read_bytes())
    assert [{k: t[k] for k in plan["trips"][0]} for t in manifest["trips"]] == plan["trips"]
    data = {RAW: sha((root / RAW).read_bytes())}
    epochs = {}
    for t in manifest["trips"]:
        fs = load_trip(root, t)
        epochs[t["id"]] = len(fs)
        if len(windows(fs, 32).x) < 2:
            raise ValueError("insufficient_windows")
        assert len(t["parts"]) == len(t["segments"]) == 2
        for part, segment in zip(t["parts"], t["segments"], strict=True):
            assert part == f"data/raw/vehicle_native_stage13/{t['id']}/segment_{segment}"
            p = root / part / "source_manifest.json"
            pm = json.loads(p.read_bytes())
            data[p.relative_to(root).as_posix()] = sha(p.read_bytes())
            arc = index["archives"][str(t["archive_id"])]
            assert pm["route_id"] == t["route_id"] and pm["segment_id"] == t[
                "route_id"
            ] + "/" + str(segment)
            assert (
                pm["source_url"] == arc["archive_url"]
                and pm["central_directory_sha256"] == arc["central_directory_sha256"]
            )
            entries = {
                e["name"]: e
                for e in index["entries"]
                if e["trip_id"] == t["id"] and e["segment"] == segment
            }
            assert len(pm["files"]) == len(entries) == 6 and {
                v["source_zip_entry"] for v in pm["files"]
            } == set(entries)
            for item in pm["files"]:
                path = root / part / item["path"]
                verify_entry(path, item, entries[item["source_zip_entry"]])
                assert sha(path.read_bytes()) == item["sha256"]
                data[path.relative_to(root).as_posix()] = item["sha256"]
    value = dict(
        schema="vehicle-ai-stage13-recovery-r1",
        config=cfg,
        trips=manifest["trips"],
        epochs=epochs,
        source_sha256={n: source_digest(root / n) for n in source_files(root)},
        data_sha256=data,
        environment=environment(),
    )
    path = root / OUT / "protocol.json"
    if path.exists():
        if json.loads(path.read_bytes()) != value:
            raise ValueError("existing_protocol_mismatch")
    else:
        write_json(path, value)
    return value


def check_frozen(root):
    root = Path(root)
    f = json.loads((root / OUT / "protocol.json").read_bytes())
    for n, h in f["source_sha256"].items():
        if source_digest(root / n) != h:
            raise ValueError("frozen_source_changed:" + n)
    for n, h in f["data_sha256"].items():
        if sha((root / n).read_bytes()) != h:
            raise ValueError("frozen_data_changed:" + n)
    return f


def begin_phase(output, phase):
    folder = Path(output) / phase
    folder.mkdir(parents=True, exist_ok=False)
    write_json(folder / "started.json", dict(phase=phase))
    return folder


def fitting_trips(f):
    return [t for t in f["trips"] if t["phase"] in {"train", "scale", "calibrate"}]


def score_seed(seed, trip):
    return int.from_bytes(
        hashlib.sha256(f"{seed}:{trip}:score".encode()).digest()[:4], "little"
    ) % (2**31)


def checkpoint(root, folder, a, cfg):
    if a.method == "persistence":
        return None, None
    ext = ".joblib" if a.method in {"isolation_forest", "lof"} else ".pt"
    path = folder / (a.method + ext)
    if ext == ".joblib":
        joblib.dump(a.model, path, compress=3)
    else:
        torch.save(a.model.state_dict(), path)
    return path.relative_to(root).as_posix(), sha(path.read_bytes())


def load_adapter(root, spec, cfg):
    a = create_adapter(spec["method"], cfg, root, spec["seed"])
    if spec["checkpoint"]:
        path = root / spec["checkpoint"]
        if sha(path.read_bytes()) != spec["checkpoint_sha256"]:
            raise ValueError("checkpoint_hash_mismatch")
        if spec["method"] in {"isolation_forest", "lof"}:
            a.model = joblib.load(path)
        else:
            a.model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            a.model.eval()
    return a


def hash_phase(folder):
    write_json(
        folder / "file_hashes.json",
        {
            p.relative_to(folder).as_posix(): sha(p.read_bytes())
            for p in sorted(folder.rglob("*"))
            if p.is_file() and p.name not in {"file_hashes.json", "independent_audit.json"}
        },
    )


def fit(root):
    root = Path(root).resolve()
    f = check_frozen(root)
    cfg = f["config"]
    if environment() != f["environment"]:
        raise ValueError("fitting_environment_changed")
    old = json.loads((root / cfg["reference_contract"]).read_bytes())
    m = ModelConfig(**old["model"])
    trips = {t["id"]: (t, load_trip(root, t)) for t in fitting_trips(f)}
    train = [(t, fs) for t, fs in trips.values() if t["phase"] == "train"]
    st, sf = next((t, fs) for t, fs in trips.values() if t["phase"] == "scale")
    ct, cf = next((t, fs) for t, fs in trips.values() if t["phase"] == "calibrate")
    scaler = Scaling.fit(np.concatenate([features(fs, m) for t, fs in train]))
    tw = [windows(fs, cfg["window"], m) for t, fs in train]
    x = np.concatenate([scaler.transform(w.x) for w in tw])
    dt = np.concatenate([w.dt for w in tw])
    sx = scaler.transform(windows(sf, cfg["window"], m).x)
    cx = scaler.transform(windows(cf, cfg["window"], m).x)
    folder = begin_phase(root / OUT, "fit")
    specs = []
    for seed in cfg["seeds"]:
        for method in cfg["methods"]:
            a, history = fit_adapter(method, x, dt, scaler, cfg, seed, root)
            sn = score_nodes(a, sx, cfg, score_seed(seed, st["id"]))
            cal = fit_node_calibration(sn)
            cn = score_nodes(a, cx, cfg, score_seed(seed, ct["id"]))
            cs = aggregate_scores(cn, cal, method)
            tau = threshold(cs)
            target = folder / f"seed_{seed}"
            target.mkdir(exist_ok=True)
            path, h = checkpoint(root, target, a, cfg)
            clean = target / (method + "_clean_scores.npz")
            np.savez_compressed(clean, scale_nodes=sn, calibrate_nodes=cn, calibrate_scores=cs)
            specs.append(
                dict(
                    method=method,
                    seed=seed,
                    checkpoint=path,
                    checkpoint_sha256=h,
                    node_calibration=cal,
                    threshold=tau,
                    training_loss=history,
                    clean_scores=clean.relative_to(root).as_posix(),
                )
            )
            print("Fitted", seed, method, "threshold", tau, flush=True)
    lock = dict(
        schema="vehicle-ai-stage13-fit-r1",
        protocol_sha256=sha((root / OUT / "protocol.json").read_bytes()),
        scaler=scaler.to_dict(),
        train_routes=[t["id"] for t, fs in train],
        scale_route=st["id"],
        calibrate_route=ct["id"],
        train_windows_sha256=array_hash(x),
        scale_windows_sha256=array_hash(sx),
        calibrate_windows_sha256=array_hash(cx),
        models=specs,
        environment=environment(),
    )
    write_json(folder / "model_lock.json", lock)
    hash_phase(folder)
    return lock


def verify_fit(root):
    root = Path(root)
    f = check_frozen(root)
    folder = root / OUT / "fit"
    hashes = json.loads((folder / "file_hashes.json").read_bytes())
    for n, h in hashes.items():
        if sha((folder / n).read_bytes()) != h:
            raise ValueError("fit_artifact_changed:" + n)
    lock = json.loads((folder / "model_lock.json").read_bytes())
    if lock["protocol_sha256"] != sha((root / OUT / "protocol.json").read_bytes()):
        raise ValueError("fit_protocol_mismatch")
    return f, lock


def assert_test_ready(root):
    root = Path(root)
    folder = root / OUT / "validation"
    audit = json.loads((folder / "independent_audit.json").read_bytes())
    if audit["status"] != "PASS" or audit["phase"] != "validation":
        raise ValueError("validation_audit_required")
    bindings = {
        "summary_sha256": folder / "summary.json",
        "phase_hashes_sha256": folder / "file_hashes.json",
        "protocol_sha256": root / OUT / "protocol.json",
        "model_lock_sha256": root / OUT / "fit/model_lock.json",
        "metrics_sha256": folder / "metrics.json",
        "intervals_sha256": folder / "route_intervals.json",
    }
    if any(audit.get(k) != sha(p.read_bytes()) for k, p in bindings.items()):
        raise ValueError("validation_audit_binding_changed")
    for n, h in json.loads((folder / "file_hashes.json").read_bytes()).items():
        if sha((folder / n).read_bytes()) != h:
            raise ValueError("validation_evidence_changed")


def base_grants(frames, old):
    keys = {s: hashlib.sha256(("stage13 base " + s).encode()).digest() for s in SOURCES}
    e = IntegrityPolicyEngine(
        keys,
        b"stage13 capability key" * 2,
        {("native-reader", "native-ego", "vehicle.telemetry.longitudinal", "read")},
        ModelConfig(**old["model"]),
        old["session_config"]["civic_thresholds"],
        "horizon_dual",
        old["session_config"]["stage7_horizon"],
    )
    result = []
    issue = []
    use = []
    for f in frames:
        q = VehicleRequest(
            "r" + str(f.frame),
            "n" + str(f.frame),
            "native-reader",
            "native-ego",
            "vehicle.telemetry.longitudinal",
            "read",
        )
        d = e.evaluate(q, packets(f, 2 * f.frame + 1, keys), f.time_ms)
        granted = False
        if d.outcome == "ALLOW":
            granted = e.consume_capability(
                e.issue_capability(d), q, packets(f, 2 * f.frame + 2, keys), f.time_ms
            ).authorized
        result.append(granted)
        issue.append(d.outcome)
        use.append(e._states["native-ego"].quorum)
    return np.array(result, dtype=bool), np.array(issue, dtype="U16"), np.array(use, dtype=np.int64)


def evaluate_phase(root, phase):
    if phase not in {"validation", "test"}:
        raise ValueError("invalid_phase")
    root = Path(root).resolve()
    f, lock = verify_fit(root)
    cfg = f["config"]
    if environment() != lock["environment"]:
        raise ValueError("evaluation_environment_changed")
    if phase == "test":
        assert_test_ready(root)
    old = json.loads((root / cfg["reference_contract"]).read_bytes())
    m = ModelConfig(**old["model"])
    scaler = Scaling(tuple(lock["scaler"]["mean"]), tuple(lock["scaler"]["scale"]))
    folder = begin_phase(root / OUT, phase)
    rows = []
    adapters = {(s["method"], s["seed"]): load_adapter(root, s, cfg) for s in lock["models"]}
    for t in f["trips"]:
        if t["phase"] != phase:
            continue
        original = load_trip(root, t)
        for case in cfg["cases"]:
            changed = perturb(original, case, m, old["session_config"])
            w = windows(changed, cfg["window"], m)
            x = scaler.transform(w.x)
            labels = np.array(
                [
                    case != "normal"
                    and max(
                        abs(a.gnss_speed - b.gnss_speed),
                        abs(a.wheel_speed - b.wheel_speed),
                        abs(a.imu_accel - b.imu_accel),
                    )
                    > 1e-9
                    for a, b in zip(changed, original, strict=True)
                ]
            )
            grant, issued, quorum = base_grants(changed, old)
            context = folder / "contexts" / (t["id"] + "__" + case + ".npz")
            context.parent.mkdir(exist_ok=True)
            np.savez_compressed(
                context,
                features=features(changed, m),
                time_ms=np.array([v.time_ms for v in changed]),
                start_ms=np.array([v.interval_start_ms for v in changed]),
                labels=labels,
                base_grant=grant,
                issued=issued,
                quorum=quorum,
                positions=w.positions,
            )
            for spec in lock["models"]:
                nodes = score_nodes(
                    adapters[spec["method"], spec["seed"]],
                    x,
                    cfg,
                    score_seed(spec["seed"], t["id"]),
                )
                scores = aggregate_scores(nodes, spec["node_calibration"], spec["method"])
                alarm = scores > spec["threshold"]
                composed = np.zeros(len(original), dtype=bool)
                composed[w.positions] = grant[w.positions] & ~alarm
                block = (
                    folder
                    / "blocks"
                    / (
                        t["id"]
                        + "__"
                        + case
                        + "__"
                        + spec["method"]
                        + "__"
                        + str(spec["seed"])
                        + ".npz"
                    )
                )
                block.parent.mkdir(exist_ok=True)
                np.savez_compressed(
                    block,
                    nodes=nodes,
                    score=scores,
                    alarm=alarm,
                    positions=w.positions,
                    grant=composed,
                )
                rows.append(
                    dict(
                        trip=t["id"],
                        case=case,
                        method=spec["method"],
                        seed=spec["seed"],
                        block=block.relative_to(folder).as_posix(),
                        context=context.relative_to(folder).as_posix(),
                    )
                )
            print(phase, t["id"], case, "scored", len(w.x), "windows", flush=True)
    summary = dict(
        schema="vehicle-ai-stage13-scores-r1",
        phase=phase,
        rows=rows,
        protocol_sha256=sha((root / OUT / "protocol.json").read_bytes()),
        model_lock_sha256=sha((root / OUT / "fit/model_lock.json").read_bytes()),
        status="EXPERIMENTAL_NOT_PROMOTED",
        environment=environment(),
    )
    write_json(folder / "summary.json", summary)
    hash_phase(folder)
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--phase", choices=["freeze", "fit", "validation", "test"], required=True)
    a = p.parse_args()
    root = a.root.resolve()
    freeze(root) if a.phase == "freeze" else fit(root) if a.phase == "fit" else evaluate_phase(
        root, a.phase
    )
    print("Completed", a.phase, "without overwriting prior phases.")
