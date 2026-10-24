"""New-route native integrity/recovery benchmark; preserves frozen Stage6."""

from __future__ import annotations

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
import scipy
from run_vehicle_native_stage6 import csv_write, drive_quantile, fingerprints, packets, write_json
from vehicle_integrity_data import load_integrity_trip as load_trip
from vehicle_integrity_scenarios import CASES, perturb
from vehicle_native_data import inside, sha
from vehicle_native_scenarios import active, bounds, period

from coact_cps.vehicle.explanation import decision_card, verify_decision_card
from coact_cps.vehicle.integrity_policy import IntegrityPolicyEngine
from coact_cps.vehicle.integrity_window import HorizonMonitor
from coact_cps.vehicle.native_detectors import ModelConfig
from coact_cps.vehicle.native_policy import METHODS as BASE_METHODS
from coact_cps.vehicle.native_policy import NativePolicyEngine
from coact_cps.vehicle.policy import VehicleRequest, canonical_json

ROOT = Path(__file__).resolve().parents[1]
METHODS = BASE_METHODS + (
    "dual_ungated_matched",
    "gated_dual_matched",
    "recovery_dual",
    "horizon_dual",
    "horizon_recovery",
)
DATA = "data/raw/vehicle_native_stage7/manifest.json"
CONFIG = "configs/vehicle_native_stage7.json"
CALIBRATION = "artifacts/vehicle_native_stage7/frozen_calibration.json"
RECEIPT = "artifacts/vehicle_native_stage7/freeze_receipt.json"
SOURCE_FILES = prior.SOURCE_FILES + [
    "src/coact_cps/vehicle/integrity_window.py",
    "src/coact_cps/vehicle/integrity_policy.py",
    "scripts/vehicle_integrity_scenarios.py",
    "scripts/run_vehicle_native_stage7.py",
    "scripts/vehicle_integrity_data.py",
]


def validate_split(manifest):
    seen = set()
    counts = defaultdict(int)
    old = set(manifest["prior_stage6_routes"])
    for entry in manifest["trips"]:
        route = entry["route_id"]
        phase = entry["phase"]
        if (
            route in seen
            or phase not in ("calibration", "validation", "test")
            or (phase != "calibration" and route in old)
        ):
            raise ValueError("native route overlap/leakage")
        seen.add(route)
        counts[phase] += 1
    if dict(counts) != {"calibration": 4, "validation": 2, "test": 5}:
        raise ValueError("fixed Stage7 route counts changed")


def inputs(root):
    cfg = json.loads(inside(root, CONFIG).read_bytes())
    if (
        cfg["methods"] != list(METHODS)
        or cfg["cases"] != list(CASES)
        or cfg["candidate_factors"] != [1, 2, 4, 8, 16, 32]
        or cfg["horizons_ms"] != [2000, 5000, 10000, 20000]
        or cfg["alarm_budget"] != 0.01
    ):
        raise ValueError("fixed Stage7 protocol changed")
    if cfg["manifest_sha256"] != sha(inside(root, DATA).read_bytes()):
        raise ValueError("Stage7 native manifest changed")
    manifest = json.loads(inside(root, DATA).read_bytes())
    validate_split(manifest)
    for phase, expected in [
        ("calibration", {"hf_0", "hf_7", "hf_14", "hf_21"}),
        ("validation", {"hf_3", "hf_8"}),
        ("test", {"hf_11", "hf_17", "hf_31", "hf_38", "hf_45"}),
    ]:
        if {e["id"] for e in manifest["trips"] if e["phase"] == phase} != expected:
            raise ValueError("fixed Stage7 trip IDs changed")
    if cfg["chunk_manifest_sha256"] != fingerprints(root, list(cfg["chunk_manifest_sha256"])):
        raise ValueError("native source manifests changed")
    old = inside(root, "data/reference/vehicle_native_stage6/linux_frozen_calibration.json")
    if cfg["stage6_calibration_sha256"] != sha(old.read_bytes()):
        raise ValueError("Stage6 baseline calibration changed")
    baseline = json.loads(old.read_bytes())
    if baseline["source_sha256"] != fingerprints(root, prior.SOURCE_FILES):
        raise ValueError("Stage6 implementation modified")
    if cfg["stage6_windows_sha256"] != sha(
        inside(root, "data/reference/vehicle_native_stage6/windows_stage6_review.zip").read_bytes()
    ):
        raise ValueError("verified Windows baseline archive changed")
    import coact_cps.vehicle.integrity_policy as module

    if not Path(module.__file__).resolve().is_relative_to(root / "src"):
        raise ValueError("imported integrity gate outside this project; set PYTHONPATH")
    return manifest, cfg, baseline


def candidate_config(baseline, scales, threshold, factor):
    dual = dict(
        baseline, dual_chi=baseline["dual_chi"] * factor, dual_cusum=baseline["dual_cusum"] * factor
    )
    return dual, dict(
        horizons=(2000, 5000, 10000, 20000), scales=tuple(scales), threshold=threshold * factor
    )


def horizon_features(frames, model, scales=(1.0, 1.0, 1.0, 1.0), threshold=1e8):
    monitor = HorizonMonitor(model.imu_offset, (2000, 5000, 10000, 20000), tuple(scales), threshold)
    results = []
    for f in frames:
        monitor.ingest(f.imu_accel, f.interval_start_ms, f.time_ms)
        results.append(monitor.sample(f.gnss_speed, f.wheel_speed, f.time_ms))
    return results


def clean_rates(trips, model, method, thresholds, horizon):
    result = {}
    for entry, frames in trips:
        engine, keys = new_engine(model, method, thresholds, horizon)
        alarms = blocked = 0
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
            alarms += diag.get("primary_alarm", diag["alarm"]) or diag.get("horizon_alarm", False)
            blocked += d.outcome != "ALLOW"
        result[entry["id"]] = dict(
            detector_alarm_fraction=alarms / len(frames), blocked_fraction=blocked / len(frames)
        )
    return result


def check_frozen(root):
    path = inside(root, CALIBRATION)
    receipt = json.loads(inside(root, RECEIPT).read_bytes())
    if receipt["calibration_sha256"] != sha(path.read_bytes()):
        raise ValueError("Stage7 frozen calibration changed")
    frozen = json.loads(path.read_bytes())
    if (
        frozen["source_sha256"] != fingerprints(root, SOURCE_FILES)
        or frozen["config_sha256"] != sha(inside(root, CONFIG).read_bytes().replace(b"\r\n", b"\n"))
        or frozen["manifest_sha256"] != sha(inside(root, DATA).read_bytes())
    ):
        raise ValueError("source/config/manifest changed after clean freeze")
    return frozen


def calibrate(root, manifest, cfg, baseline):
    if inside(root, CALIBRATION).exists():
        print("Retained unchanged calibration", inside(root, CALIBRATION))
        return check_frozen(root)
    trips = [(e, load_trip(root, e)) for e in manifest["trips"] if e["phase"] == "calibration"]
    model = ModelConfig(**baseline["model"])
    raw = [horizon_features(frames, model) for _, frames in trips]
    scales = [
        max(
            0.2,
            drive_quantile(
                [[r.residuals[i] for r in group if r.residuals[i] is not None] for group in raw],
                0.9,
            ),
        )
        for i in range(4)
    ]
    scores = [
        [r.score for r in horizon_features(frames, model, scales) if r.ready] for _, frames in trips
    ]
    threshold = drive_quantile(scores, 0.995)
    search = []
    selected = None
    for factor in cfg["candidate_factors"]:
        dual, horizon = candidate_config(baseline["thresholds"], scales, threshold, factor)
        thresholds = dict(baseline=baseline["thresholds"], candidate=dual)
        rates = clean_rates(trips, model, "horizon_recovery", thresholds, horizon)
        mean = float(np.mean([r["detector_alarm_fraction"] for r in rates.values()]))
        search.append(dict(factor=factor, equal_trip_detector_alarm_fraction=mean, per_trip=rates))
        if mean <= cfg["alarm_budget"]:
            selected = (thresholds, horizon)
            break
    if selected is None:
        raise ValueError("predeclared clean calibration grid exhausted; held-out study blocked")
    thresholds, horizon = selected
    reports = {m: clean_rates(trips, model, m, thresholds, horizon) for m in METHODS}
    frozen = dict(
        schema="native-stage7-calibration-v1",
        model=baseline["model"],
        thresholds=thresholds,
        horizon=horizon,
        unscaled_horizon_threshold=threshold,
        candidate_search=search,
        calibration_trip_ids=[e["id"] for e, _ in trips],
        calibration_epochs={e["id"]: len(f) for e, f in trips},
        clean_rates=reports,
        source_sha256=fingerprints(root, SOURCE_FILES),
        config_sha256=sha(inside(root, CONFIG).read_bytes().replace(b"\r\n", b"\n")),
        manifest_sha256=cfg["manifest_sha256"],
        limitations=[
            "Alarm budget counts current detector alarms; quarantine/startup blocked fraction is separately reported",
            "Same-vehicle clean calibration; uncertainty is empirical, not physical attestation",
            "Recovery resets older horizons conditionally; long horizons warm up again",
        ],
    )
    write_json(inside(root, CALIBRATION), frozen)
    with inside(root, RECEIPT).open("x", encoding="utf-8") as out:
        json.dump(
            dict(
                schema="native-stage7-freeze-receipt-v1",
                calibration_sha256=sha(inside(root, CALIBRATION).read_bytes()),
            ),
            out,
            indent=2,
        )
        out.write("\n")
    print("Frozen Stage7 calibration", inside(root, CALIBRATION))
    return frozen


def evaluate_phase(root, manifest, cfg, phase):
    frozen = check_frozen(root)
    model = ModelConfig(**frozen["model"])
    thresholds = frozen["thresholds"]
    horizon = frozen["horizon"]
    if phase == "test":
        validation = json.loads(
            inside(
                root, "artifacts/vehicle_native_stage7/latest_validation_summary.json"
            ).read_bytes()
        )
        if validation["calibration_sha256"] != sha(inside(root, CALIBRATION).read_bytes()):
            raise ValueError("validation freeze mismatch")
    trips = [(e, load_trip(root, e)) for e in manifest["trips"] if e["phase"] == phase]
    destination = inside(root, f"artifacts/vehicle_native_stage7/{phase}_{time.time_ns()}")
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
    csv_write(destination / "requests.csv", rows)
    csv_write(destination / "per_trip_metrics.csv", metrics)
    csv_write(
        destination / "paired_trip_intervals.csv", paired_intervals(metrics, cfg["bootstrap_seed"])
    )
    write_json(destination / "alarm_events.json", events)
    summary = dict(
        schema="native-stage7-study-v1",
        phase=phase,
        requests=len(rows),
        epochs={e["id"]: len(f) for e, f in trips},
        cases=list(CASES),
        methods=list(METHODS),
        calibration_sha256=sha(inside(root, CALIBRATION).read_bytes()),
        source_sha256=fingerprints(root, SOURCE_FILES),
        input_manifest_sha256=cfg["manifest_sha256"],
        stage6_windows_sha256=cfg["stage6_windows_sha256"],
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
            "Coherent/biased-startup alternatives are negative controls, not solved physical-truth attestation",
            "19 controlled families x11 methods are paired repeated measurements, not independent rows",
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
    latest = inside(
        root,
        f"artifacts/vehicle_native_stage7/latest_{'study' if phase == 'test' else 'validation'}_review.zip",
    )
    temp = latest.with_suffix(".zip.tmp")
    shutil.copy2(archive, temp)
    os.replace(temp, latest)
    write_json(
        inside(
            root,
            f"artifacts/vehicle_native_stage7/latest_{'study' if phase == 'test' else 'validation'}_summary.json",
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


def new_engine(model, method, thresholds, horizon_config=None):
    keys = {s: os.urandom(32) for s in prior.SOURCES}
    capkey = os.urandom(32)
    permission = {("native-reader", "native-ego", "vehicle.telemetry.longitudinal", "read")}
    if method in BASE_METHODS:
        return NativePolicyEngine(
            keys, capkey, permission, model, method, thresholds["baseline"]
        ), keys
    if method in ("dual_ungated_matched", "gated_dual_matched"):
        return NativePolicyEngine(
            keys,
            capkey,
            permission,
            model,
            "dual_ungated" if method == "dual_ungated_matched" else "gated_dual",
            thresholds["candidate"],
        ), keys
    return IntegrityPolicyEngine(
        keys, capkey, permission, model, thresholds["candidate"], method, horizon_config
    ), keys


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
                    current_detector_alarm=current_primary or current_horizon,
                    primary_alarm=current_primary,
                    horizon_alarm=current_horizon,
                    horizon_score=diagnostics.get("horizon_score", 0.0),
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


def paired_intervals(metrics, seed):
    lookup = {
        (r["trip"], r["case"], r["method"], r["period"]): r["grant_fraction"] for r in metrics
    }
    groups = defaultdict(list)
    for r in metrics:
        if r["method"] == "horizon_recovery":
            for method in METHODS[:-1]:
                groups[r["case"], method, r["period"]].append(
                    r["grant_fraction"] - lookup[r["trip"], r["case"], method, r["period"]]
                )
    rng = np.random.default_rng(seed)
    result = []
    for (case, method, p), values in sorted(groups.items()):
        v = np.asarray(values)
        samples = np.mean(v[rng.integers(0, len(v), (10000, len(v)))], axis=1)
        low, high = np.quantile(samples, [0.025, 0.975])
        result.append(
            dict(
                case=case,
                reference=method,
                period=p,
                trip_clusters=len(v),
                candidate_minus_reference=float(np.mean(v)),
                bootstrap_lower=float(low),
                bootstrap_upper=float(high),
                interpretation="exploratory whole-trip resampling; same vehicle",
            )
        )
    return result


if __name__ == "__main__":
    main()
