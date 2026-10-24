"""Frozen session-reference comparison on unused native routes."""

import argparse
import json
import os
import platform
import shutil
import sys
import time
import zipfile
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import run_vehicle_native_stage6 as prior
import run_vehicle_native_stage7 as frozen7
import scipy
from run_vehicle_native_stage6 import csv_write, drive_quantile, fingerprints, packets, write_json
from vehicle_integrity_data import load_integrity_trip as load_trip
from vehicle_native_data import inside, sha
from vehicle_native_scenarios import bounds
from vehicle_session_scenarios import CASES, active, period, perturb

from coact_cps.vehicle.explanation import decision_card, verify_decision_card
from coact_cps.vehicle.integrity_policy import IntegrityPolicyEngine
from coact_cps.vehicle.native_detectors import ModelConfig
from coact_cps.vehicle.native_policy import METHODS as BASE_METHODS
from coact_cps.vehicle.native_policy import NativePolicyEngine
from coact_cps.vehicle.policy import VehicleRequest, canonical_json
from coact_cps.vehicle.session_anchor import SessionAnchor
from coact_cps.vehicle.session_policy import SessionPolicyEngine

ROOT = Path(__file__).resolve().parents[1]
METHODS = BASE_METHODS + (
    "gated_dual_matched",
    "horizon_recovery",
    "anchor_dual",
    "anchor_recovery",
)
DATA = "data/raw/vehicle_native_stage8/manifest.json"
CONFIG = "configs/vehicle_native_stage8.json"
CALIBRATION = "artifacts/vehicle_native_stage8/frozen_calibration.json"
RECEIPT = "artifacts/vehicle_native_stage8/freeze_receipt.json"
SOURCE_FILES = frozen7.SOURCE_FILES + [
    "src/coact_cps/vehicle/session_anchor.py",
    "src/coact_cps/vehicle/session_policy.py",
    "scripts/vehicle_session_scenarios.py",
    "scripts/run_vehicle_native_stage8.py",
]


def validate_split(manifest):
    ids = set()
    routes = {}
    counts = defaultdict(int)
    for e in manifest["trips"]:
        phase = e["phase"]
        route = e["route_id"]
        if (
            e["id"] in ids
            or phase not in ("calibration", "validation", "test")
            or (route in routes and routes[route] != phase)
            or (phase != "calibration" and route in manifest["prior_routes"])
        ):
            raise ValueError("session route overlap/leakage")
        ids.add(e["id"])
        routes[route] = phase
        counts[phase] += 1
    if (
        dict(counts) != {"calibration": 4, "validation": 1, "test": 6}
        or sum(p == "test" for p in routes.values()) != 2
    ):
        raise ValueError("fixed episode/route counts changed")


def inputs(root):
    cfg = json.loads(inside(root, CONFIG).read_bytes())
    m = json.loads(inside(root, DATA).read_bytes())
    if (
        cfg["methods"] != list(METHODS)
        or cfg["cases"] != list(CASES)
        or cfg["candidate_factors"] != [1, 2, 4, 8, 16, 32]
        or cfg["alarm_budget"] != 0.01
    ):
        raise ValueError("fixed Stage8 protocol changed")
    if sha(inside(root, DATA).read_bytes()) != cfg["manifest_sha256"]:
        raise ValueError("session manifest changed")
    validate_split(m)
    for phase, ids in [
        ("calibration", {"hf_0", "hf_7", "hf_14", "hf_21"}),
        ("validation", {"hf_52"}),
        ("test", {"hf_58", "hf_59", "hf_60", "hf_61", "hf_62", "hf_63"}),
    ]:
        if {e["id"] for e in m["trips"] if e["phase"] == phase} != ids:
            raise ValueError("fixed session episode IDs changed")
    if fingerprints(root, list(cfg["chunk_manifest_sha256"])) != cfg["chunk_manifest_sha256"]:
        raise ValueError("session native manifests changed")
    old = inside(root, "data/reference/vehicle_native_stage7/linux_frozen_calibration.json")
    if sha(old.read_bytes()) != cfg["stage7_calibration_sha256"]:
        raise ValueError("Stage7 reference changed")
    baseline = json.loads(old.read_bytes())
    if baseline["source_sha256"] != fingerprints(root, frozen7.SOURCE_FILES):
        raise ValueError("Stage7 source changed")
    if (
        sha(
            inside(
                root, "data/reference/vehicle_native_stage7/windows_stage7_review.zip"
            ).read_bytes()
        )
        != cfg["stage7_windows_sha256"]
    ):
        raise ValueError("verified Windows Stage7 archive changed")
    if (
        not Path(sys.modules[SessionPolicyEngine.__module__].__file__)
        .resolve()
        .is_relative_to(root / "src")
    ):
        raise ValueError("session gate import outside project; set PYTHONPATH")
    return m, cfg, baseline


def check_frozen(root):
    p = inside(root, CALIBRATION)
    receipt = json.loads(inside(root, RECEIPT).read_bytes())
    if receipt["calibration_sha256"] != sha(p.read_bytes()):
        raise ValueError("Stage8 calibration changed")
    f = json.loads(p.read_bytes())
    if (
        f["source_sha256"] != fingerprints(root, SOURCE_FILES)
        or f["config_sha256"] != sha(inside(root, CONFIG).read_bytes().replace(b"\r\n", b"\n"))
        or f["manifest_sha256"] != sha(inside(root, DATA).read_bytes())
    ):
        raise ValueError("session source/config/manifest changed after freeze")
    return f


def new_engine(model, method, thresholds, config):
    keys = {s: os.urandom(32) for s in prior.SOURCES}
    cap = os.urandom(32)
    acl = {("native-reader", "native-ego", "vehicle.telemetry.longitudinal", "read")}
    if method in BASE_METHODS:
        return NativePolicyEngine(keys, cap, acl, model, method, thresholds["baseline"]), keys
    if method == "gated_dual_matched":
        return NativePolicyEngine(
            keys, cap, acl, model, "gated_dual", thresholds["candidate"]
        ), keys
    if method == "horizon_recovery":
        return IntegrityPolicyEngine(
            keys, cap, acl, model, thresholds["candidate"], method, config["stage7_horizon"]
        ), keys
    return SessionPolicyEngine(
        keys, cap, acl, model, thresholds["candidate"], method, config["anchor"]
    ), keys


def anchor_features(frames, model, scale=1.0, threshold=1e8):
    monitor = SessionAnchor(model.gnss_offset, scale, threshold)
    out = []
    for f in frames:
        monitor.ingest(f.imu_accel, f.interval_start_ms, f.time_ms)
        out.append(monitor.sample(f.gnss_speed, f.wheel_speed, f.time_ms))
    return out


def clean_rates(trips, model, method, thresholds, config):
    out = {}
    for e, frames in trips:
        engine, keys = new_engine(model, method, thresholds, config)
        alarms = blocked = unready = 0
        for f in frames:
            q = VehicleRequest(
                f"r{f.frame}",
                f"n{f.frame}",
                "native-reader",
                "native-ego",
                "vehicle.telemetry.longitudinal",
                "read",
            )
            d = engine.evaluate(q, packets(f, 2 * f.frame + 1, keys), f.time_ms)
            diag = engine.diagnostics("native-ego")
            alarms += diag.get("primary_alarm", diag["alarm"]) or diag.get(
                "anchor_alarm", diag.get("horizon_alarm", False)
            )
            blocked += d.outcome != "ALLOW"
            unready += method.startswith("anchor") and not diag["anchor_ready"]
        out[e["id"]] = dict(
            detector_alarm_fraction=alarms / len(frames),
            blocked_fraction=blocked / len(frames),
            anchor_unready_fraction=unready / len(frames),
        )
    return out


def calibrate(root, manifest, cfg, baseline):
    if inside(root, CALIBRATION).exists():
        return check_frozen(root)
    trips = [(e, load_trip(root, e)) for e in manifest["trips"] if e["phase"] == "calibration"]
    model = ModelConfig(**baseline["model"])
    raw = [anchor_features(f, model) for _, f in trips]
    scale = max(
        0.2,
        drive_quantile([[r.score for r in rows if r.ready and not r.lost] for rows in raw], 0.9),
    )
    normalized = [anchor_features(f, model, scale) for _, f in trips]
    threshold = drive_quantile(
        [[r.score for r in rows if r.ready and not r.lost] for rows in normalized], 0.995
    )
    threshold = max(threshold, 1e-8)
    search = []
    selected = None
    for factor in cfg["candidate_factors"]:
        config = dict(
            stage7_horizon=baseline["horizon"],
            anchor=dict(scale=scale, threshold=threshold * factor),
        )
        rates = clean_rates(trips, model, "anchor_recovery", baseline["thresholds"], config)
        mean = float(np.mean([r["detector_alarm_fraction"] for r in rates.values()]))
        search.append(dict(factor=factor, equal_trip_detector_alarm_fraction=mean, per_trip=rates))
        if mean <= cfg["alarm_budget"]:
            selected = config
            break
    if selected is None:
        raise ValueError("clean session grid exhausted; evaluation blocked")
    reports = {m: clean_rates(trips, model, m, baseline["thresholds"], selected) for m in METHODS}
    frozen = dict(
        schema="native-stage8-calibration-v1",
        model=baseline["model"],
        thresholds=baseline["thresholds"],
        session_config=selected,
        unscaled_anchor_threshold=threshold,
        candidate_search=search,
        calibration_epochs={e["id"]: len(f) for e, f in trips},
        clean_rates=reports,
        source_sha256=fingerprints(root, SOURCE_FILES),
        config_sha256=sha(inside(root, CONFIG).read_bytes().replace(b"\r\n", b"\n")),
        manifest_sha256=cfg["manifest_sha256"],
        limitations=[
            "Baseline3epoch grants during5s enrollment do not have anchored assurance",
            "Enrollment bias/speed can be poisoned",
            "Lost IMU reference blocks honest postgap traffic",
            "Only2new route clusters",
        ],
    )
    write_json(inside(root, CALIBRATION), frozen)
    with inside(root, RECEIPT).open("x", encoding="utf-8") as stream:
        json.dump(
            dict(
                schema="native-stage8-freeze-v1",
                calibration_sha256=sha(inside(root, CALIBRATION).read_bytes()),
            ),
            stream,
            indent=2,
        )
        stream.write("\n")
    print("Frozen Stage8 calibration", inside(root, CALIBRATION), flush=True)
    return frozen


def pool_route_metrics(metrics, manifest):
    route = {e["id"]: e["route_id"] for e in manifest["trips"]}
    groups = defaultdict(list)
    for m in metrics:
        groups[route[m["trip"]], m["case"], m["method"], m["period"]].append(m)
    out = []
    for (r, c, m, p), values in sorted(groups.items()):
        row = dict(route_id=r, case=c, method=m, period=p, episodes=len(values))
        for field in (
            "requests",
            "grants",
            "motion_alarms",
            "blocked",
            "quarantine_requests",
            "recovery_releases",
            "current_detector_alarms",
            "challenges",
        ):
            if field in values[0]:
                row[field] = sum(v[field] for v in values)
        row["grant_fraction"] = row["grants"] / row["requests"]
        out.append(row)
    return out


def paired_intervals(metrics, seed):
    lookup = {
        (r["route_id"], r["case"], r["method"], r["period"]): r["grant_fraction"] for r in metrics
    }
    groups = defaultdict(list)
    for r in metrics:
        if r["method"] == "anchor_recovery":
            for m in METHODS[:-1]:
                groups[r["case"], m, r["period"]].append(
                    r["grant_fraction"] - lookup[r["route_id"], r["case"], m, r["period"]]
                )
    rng = np.random.default_rng(seed)
    out = []
    for (c, m, p), values in sorted(groups.items()):
        v = np.asarray(values)
        samples = v[rng.integers(0, len(v), (10000, len(v)))].mean(axis=1)
        low, high = np.quantile(samples, [0.025, 0.975])
        out.append(
            dict(
                case=c,
                reference=m,
                period=p,
                route_clusters=len(v),
                candidate_minus_reference=float(v.mean()),
                bootstrap_lower=float(low),
                bootstrap_upper=float(high),
                interpretation="exploratory whole-route pooling/resampling;2test routes",
            )
        )
    return out


def evaluate_phase(root, manifest, cfg, phase):
    frozen = check_frozen(root)
    model = ModelConfig(**frozen["model"])
    thresholds = frozen["thresholds"]
    horizon = frozen["session_config"]
    if phase == "test":
        validation = json.loads(
            inside(
                root, "artifacts/vehicle_native_stage8/latest_validation_summary.json"
            ).read_bytes()
        )
        if validation["calibration_sha256"] != sha(inside(root, CALIBRATION).read_bytes()):
            raise ValueError("validation freeze mismatch")
    trips = [(e, load_trip(root, e)) for e in manifest["trips"] if e["phase"] == phase]
    destination = inside(root, f"artifacts/vehicle_native_stage8/{phase}_{time.time_ns()}")
    destination.mkdir(parents=True)
    rows = []
    events = []
    with (destination / "contexts.jsonl").open("w", encoding="utf-8") as contexts:
        for entry, frames in trips:
            for case in CASES:
                changed = perturb(frames, case, model, horizon)
                r, e = run_case(entry, frames, changed, case, model, thresholds, contexts, horizon)
                rows.extend(r)
                events.extend(e)
            print(phase, entry["id"], len(frames), "epochs", flush=True)
    expected = sum(len(f) for _, f in trips) * len(CASES) * len(METHODS)
    if len(rows) != expected or any(
        not r["nonce_rejected"]
        or not r["expiry_bounded"]
        or r["display_mutated"] == r["display_accepted"]
        for r in rows
    ):
        raise ValueError("paired/card/capability invariant failed")
    metrics = aggregate(rows)
    routes = pool_route_metrics(metrics, manifest)
    csv_write(destination / "per_route_metrics.csv", routes)
    csv_write(destination / "requests.csv", rows)
    csv_write(destination / "per_trip_metrics.csv", metrics)
    csv_write(
        destination / "paired_trip_intervals.csv", paired_intervals(routes, cfg["bootstrap_seed"])
    )
    write_json(destination / "alarm_events.json", events)
    summary = dict(
        schema="native-stage8-study-v1",
        phase=phase,
        requests=len(rows),
        epochs={e["id"]: len(f) for e, f in trips},
        cases=list(CASES),
        methods=list(METHODS),
        calibration_sha256=sha(inside(root, CALIBRATION).read_bytes()),
        source_sha256=fingerprints(root, SOURCE_FILES),
        input_manifest_sha256=cfg["manifest_sha256"],
        stage7_windows_sha256=cfg["stage7_windows_sha256"],
        route_by_episode={e["id"]: e["route_id"] for e, _ in trips},
        route_clusters=len({e["route_id"] for e, _ in trips}),
        capability_invariants=True,
        exact_card_invariant=True,
        environment=dict(
            python=sys.version,
            numpy=np.__version__,
            scipy=scipy.__version__,
            platform=platform.platform(),
        ),
        limitations=manifest["limitations"]
        + frozen["limitations"]
        + [
            (
                "Coherent/biased-startup alternatives are negative controls, "
                "not solved physical-truth attestation"
            ),
            (
                "21 controlled families x10 methods are paired repeated measurements, "
                "not independent rows"
            ),
        ],
    )
    write_json(destination / "summary.json", summary)
    for name, target in [
        (CALIBRATION, "frozen_calibration.json"),
        (RECEIPT, "freeze_receipt.json"),
        (CONFIG, "benchmark_config.json"),
        (DATA, "native_input_manifest.json"),
    ]:
        shutil.copy2(inside(root, name), destination / target)
    write_json(
        destination / "native_chunk_manifests.json",
        {
            name: json.loads(inside(root, name).read_bytes())
            for name in cfg["chunk_manifest_sha256"]
        },
    )
    for name in SOURCE_FILES:
        target = destination / "frozen_sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(inside(root, name), target)
    hashes = {
        p.relative_to(destination).as_posix(): sha(p.read_bytes())
        for p in destination.rglob("*")
        if p.is_file()
    }
    write_json(destination / "file_hashes.json", hashes)
    archive = destination.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(destination.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(destination).as_posix())
    label = "study" if phase == "test" else "validation"
    latest = inside(root, f"artifacts/vehicle_native_stage8/latest_{label}_review.zip")
    temp = latest.with_suffix(".zip.tmp")
    shutil.copy2(archive, temp)
    os.replace(temp, latest)
    write_json(
        inside(
            root,
            f"artifacts/vehicle_native_stage8/latest_{label}_summary.json",
        ),
        summary,
    )
    print("Review ZIP", latest, "SHA256", sha(latest.read_bytes()), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=["calibrate", "validate", "study"])
    args = parser.parse_args()
    root = ROOT.resolve()
    manifest, cfg, baseline = inputs(root)
    if args.phase == "calibrate":
        calibrate(root, manifest, cfg, baseline)
    else:
        evaluate_phase(root, manifest, cfg, "validation" if args.phase == "validate" else "test")


def run_case(entry, frames, changed, case, model, thresholds, contexts, horizon_config):
    rows = []
    events = []
    begin, end = bounds(frames)
    for method in METHODS:
        engine, keys = new_engine(model, method, thresholds, horizon_config)
        previous_gnss = None
        last_alarm = False
        last_quarantine = False
        last_recovery = 0
        quarantine_start = None
        first_alarm = None
        for original, f in zip(frames, changed, strict=True):
            on = active(original, frames, case)
            q = VehicleRequest(
                f"r{f.frame}",
                f"n{f.frame}",
                "native-reader",
                "native-ego",
                "vehicle.telemetry.longitudinal",
                "read",
            )
            wire_f = (
                replace(f, gnss_time_ms=f.gnss_time_ms - 1000) if case == "stale_gnss" and on else f
            )
            wires = packets(
                wire_f, 2 * f.frame + 1, keys, revoked=case == "identity_revoked" and on
            )
            fresh_gnss = wires["gnss"]
            if case == "gnss_replay" and on and previous_gnss is not None:
                wires["gnss"] = previous_gnss
            previous_gnss = fresh_gnss
            if (case == "imu_dropout" and on) or (
                case == "gap_reenrollment" and begin + 500 <= f.time_ms < begin + 1500
            ):
                wires.pop("imu")
            start = time.perf_counter_ns()
            decision = engine.evaluate(q, wires, f.time_ms)
            evaluation_us = (time.perf_counter_ns() - start) / 1000
            diagnostics = engine.diagnostics("native-ego")
            scored = (
                engine._states.get("native-ego") is not None
                and engine._states["native-ego"].cache_end == f.time_ms
            )
            current_primary = scored and diagnostics.get("primary_alarm", diagnostics["alarm"])
            current_horizon = scored and diagnostics.get("horizon_alarm", False)
            current_anchor = scored and diagnostics.get("anchor_alarm", False)
            motion_alarm = scored and diagnostics["alarm"]
            quarantined = diagnostics.get("quarantined", False)
            recovery = diagnostics.get("recovery_resets", 0)
            released = recovery > last_recovery
            if quarantined and not last_quarantine:
                quarantine_start = f.time_ms
                events.append(
                    dict(
                        trip=entry["id"],
                        case=case,
                        method=method,
                        time_ms=f.time_ms,
                        event="quarantine_enter",
                        active=on,
                    )
                )
            if not quarantined and last_quarantine:
                events.append(
                    dict(
                        trip=entry["id"],
                        case=case,
                        method=method,
                        time_ms=f.time_ms,
                        event="quarantine_exit",
                        active=on,
                        elapsed_ms=f.time_ms - quarantine_start,
                    )
                )
            last_quarantine = quarantined
            last_recovery = recovery
            card = decision_card(decision)
            mutated = case == "display_downplay" and on
            if mutated:
                card["mode"] = "snapshot" if decision.mode != "snapshot" else "stream"
            card_ok = verify_decision_card(decision, canonical_json(card))
            authorized = False
            nonce_rejected = True
            reasons = ""
            expiry_ok = True
            use_us = 0.0
            if decision.outcome == "ALLOW":
                capability = engine.issue_capability(decision)
                current = packets(
                    wire_f,
                    2 * f.frame + 2,
                    keys,
                    revoked=(case in ("postissue_revoked", "identity_revoked") and on),
                )
                if case == "gnss_replay" and on:
                    current["gnss"] = wires["gnss"]
                if (case == "imu_dropout" and on) or (
                    case == "gap_reenrollment" and begin + 500 <= f.time_ms < begin + 1500
                ):
                    current.pop("imu")
                start = time.perf_counter_ns()
                result = engine.consume_capability(capability, q, current, f.time_ms + 1)
                use_us = (time.perf_counter_ns() - start) / 1000
                authorized = result.authorized
                reasons = "|".join(result.reasons)
                expiry_ok = not result.authorized or result.expires_ms <= decision.expires_ms
                nonce_rejected = not engine.consume_capability(
                    capability, q, current, f.time_ms + 2
                ).authorized
            if motion_alarm and on and first_alarm is None:
                first_alarm = f.time_ms
            if motion_alarm and not last_alarm:
                events.append(
                    dict(
                        trip=entry["id"],
                        case=case,
                        method=method,
                        time_ms=f.time_ms,
                        event="motion_alarm_episode_start",
                        active=on,
                    )
                )
            last_alarm = motion_alarm
            rows.append(
                dict(
                    trip=entry["id"],
                    case=case,
                    method=method,
                    frame=f.frame,
                    time_ms=f.time_ms,
                    period=period(original, frames, case),
                    perturbed=on,
                    issued_outcome=decision.outcome,
                    capability_authorized=authorized,
                    final_grant=authorized and card_ok,
                    motion_alarm=motion_alarm,
                    motion_scored=scored,
                    current_detector_alarm=current_primary or current_horizon or current_anchor,
                    primary_alarm=current_primary,
                    horizon_alarm=current_horizon,
                    horizon_score=diagnostics.get("horizon_score", 0.0),
                    anchor_score=diagnostics.get("anchor_score", 0.0),
                    anchor_alarm=current_anchor,
                    anchor_ready=diagnostics.get("anchor_ready", False),
                    anchor_lost=diagnostics.get("anchor_lost", False),
                    anchor_bias=diagnostics.get("anchor_bias"),
                    quarantined=quarantined,
                    confirmation_epochs=diagnostics.get("confirmation_epochs", 0),
                    recovery_resets=recovery,
                    recovery_release=released,
                    shadow_updates=diagnostics.get("shadow_updates", 0),
                    z=diagnostics["z"],
                    statistic=diagnostics["statistic"],
                    observer_resets=diagnostics["resets"],
                    measurement_updates=diagnostics["measurement_updates"],
                    display_mutated=mutated,
                    display_accepted=card_ok,
                    nonce_rejected=nonce_rejected,
                    expiry_bounded=expiry_ok,
                    evaluation_us=evaluation_us,
                    use_us=use_us,
                    reasons="|".join(decision.reasons),
                    use_reasons=reasons,
                )
            )
        active_rows = [r for r in rows if r["method"] == method and r["perturbed"]]
        if active_rows:
            events.append(
                dict(
                    trip=entry["id"],
                    case=case,
                    method=method,
                    time_ms=first_alarm if first_alarm is not None else -1,
                    event="first_motion_alarm_delay_ms",
                    active=True,
                    delay_ms=first_alarm - active_rows[0]["time_ms"]
                    if first_alarm is not None
                    else -1,
                )
            )
    # Store each common stream once, with evaluation labels in separate fields.
    for original, f in zip(frames, changed, strict=True):
        contexts.write(
            json.dumps(
                dict(
                    trip=entry["id"],
                    case=case,
                    native=asdict(original),
                    observed=asdict(f),
                    evaluation_only=dict(
                        active=active(original, frames, case), period=period(original, frames, case)
                    ),
                ),
                allow_nan=False,
            )
            + "\n"
        )
    return rows, events


def aggregate(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[r["trip"], r["case"], r["method"], r["period"]].append(r)
    result = []
    for (trip, case, method, p), rs in sorted(groups.items()):
        result.append(
            dict(
                trip=trip,
                case=case,
                method=method,
                period=p,
                requests=len(rs),
                grants=sum(r["final_grant"] for r in rs),
                grant_fraction=float(np.mean([r["final_grant"] for r in rs])),
                motion_alarms=sum(r["motion_alarm"] for r in rs),
                blocked=sum(not r["final_grant"] for r in rs),
                quarantine_requests=sum(r["quarantined"] for r in rs),
                recovery_releases=sum(r["recovery_release"] for r in rs),
                current_detector_alarms=sum(r["current_detector_alarm"] for r in rs),
                challenges=sum(r["issued_outcome"] == "CHALLENGE" for r in rs),
                evaluation_us_median=float(np.median([r["evaluation_us"] for r in rs])),
            )
        )
    return result


if __name__ == "__main__":
    main()
