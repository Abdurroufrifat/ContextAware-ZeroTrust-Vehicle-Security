"""Route-heldout native benchmark. Run calibrate, validate, study in that order."""

from __future__ import annotations

import argparse
import csv
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
import scipy
from scipy.stats import chi2
from vehicle_native_data import inside, load_trip, sha
from vehicle_native_scenarios import CASES, VALIDATION_CASES, active, bounds, period, perturb

from coact_cps.vehicle.explanation import decision_card, verify_decision_card
from coact_cps.vehicle.native_detectors import ModelConfig, Observer, Statistic, innovation_energy
from coact_cps.vehicle.native_policy import METHODS, NativePolicyEngine
from coact_cps.vehicle.policy import SOURCES, VehicleRequest, canonical_json, seal_source

ROOT = Path(__file__).resolve().parents[1]
DATA = "data/raw/vehicle_native_stage6/manifest.json"
CONFIG = "configs/vehicle_native_stage6.json"
CALIBRATION = "artifacts/vehicle_native_stage6/frozen_calibration.json"
RECEIPT = "artifacts/vehicle_native_stage6/freeze_receipt.json"
SOURCE_FILES = [
    "scripts/run_vehicle_native_stage6.py",
    "scripts/vehicle_native_scenarios.py",
    "scripts/vehicle_native_data.py",
    "src/coact_cps/vehicle/native_detectors.py",
    "src/coact_cps/vehicle/native_policy.py",
    "src/coact_cps/vehicle/policy.py",
    "src/coact_cps/vehicle/explanation.py",
]
EXPECTED_MANIFEST = "8c154228973343516d3636b29fbcefeb8312c86fcc735036fcafca0fcc3f2205"
EXPECTED_SPLIT = {
    "calibration": {"hf_0", "hf_7", "hf_14", "hf_21"},
    "validation": {"hf_28", "github_example"},
    "test": {"hf_35", "hf_42", "hf_49", "hf_56", "hf_9_10"},
}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def fingerprints(root, paths):
    return {name: sha(inside(root, name).read_bytes().replace(b"\r\n", b"\n")) for name in paths}


def validate_split(manifest):
    routes = set()
    ids = set()
    counts = defaultdict(int)
    for entry in manifest["trips"]:
        if (
            entry["route_id"] in routes
            or entry["id"] in ids
            or entry["phase"] not in EXPECTED_SPLIT
        ):
            raise ValueError("trip split overlap or invalid phase")
        routes.add(entry["route_id"])
        ids.add(entry["id"])
        counts[entry["phase"]] += 1
    if dict(counts) != {"calibration": 4, "validation": 2, "test": 5}:
        raise ValueError("fixed route counts changed")


def inputs(root):
    if sha(inside(root, DATA).read_bytes()) != EXPECTED_MANIFEST:
        raise ValueError("frozen native manifest changed")
    manifest = json.loads(inside(root, DATA).read_bytes())
    validate_split(manifest)
    for phase, names in EXPECTED_SPLIT.items():
        if {e["id"] for e in manifest["trips"] if e["phase"] == phase} != names:
            raise ValueError("frozen route split changed")
    cfg = json.loads(inside(root, CONFIG).read_bytes())
    for key in ("baseline_source_canonical_sha256", "native_chunk_manifest_sha256"):
        if cfg[key] != fingerprints(root, list(cfg[key])):
            raise ValueError("frozen baseline source or native chunk manifest changed")
    if (
        tuple(cfg["cases"]) != CASES
        or tuple(cfg["methods"]) != METHODS
        or cfg["candidate_factors"] != [1, 2, 4, 8, 16, 32]
        or cfg["alarm_budget"] != 0.01
    ):
        raise ValueError("benchmark protocol changed")
    # Enforce import provenance: no installed stale copy silently executes.
    import coact_cps.vehicle.native_policy as policy_module

    if not Path(policy_module.__file__).resolve().is_relative_to(root / "src"):
        raise ValueError("policy imports from outside this project; install pip -e .")
    baseline = inside(root, "data/reference/vehicle_native_stage6/windows_stage5_review.zip")
    if sha(baseline.read_bytes()) != cfg["stage5_windows_sha256"]:
        raise ValueError("Stage5 frozen Windows evidence changed")
    return manifest, cfg


def phase_trips(root, manifest, phase):
    return [(e, load_trip(root, e)) for e in manifest["trips"] if e["phase"] == phase]


def packets(f, sequence, keys, revoked=False):
    signals = {
        "identity": dict(
            subject_id="native-reader", credential_valid=True, revoked=revoked, attestation_ok=True
        ),
        "gnss": dict(speed_mps=f.gnss_speed),
        "wheel": dict(speed_mps=f.wheel_speed),
        "imu": dict(
            acceleration_mps2=f.imu_accel,
            interval_start_ms=f.interval_start_ms,
            interval_end_ms=f.time_ms,
            maximum_gap_ms=f.maximum_gap_ms,
        ),
    }
    timestamps = {
        "identity": f.time_ms,
        "gnss": f.gnss_time_ms,
        "wheel": f.wheel_time_ms,
        "imu": f.imu_time_ms,
    }
    return {
        s: seal_source(s, "native-ego", f"epoch-{f.frame}", timestamps[s], sequence, v, keys[s])
        for s, v in signals.items()
    }


def new_engine(model, method, thresholds):
    keys = {s: os.urandom(32) for s in SOURCES}
    permission = {("native-reader", "native-ego", "vehicle.telemetry.longitudinal", "read")}
    return NativePolicyEngine(keys, os.urandom(32), permission, model, method, thresholds), keys


def drive_quantile(groups, q):
    if not 0 < q <= 1 or any(not g for g in groups):
        raise ValueError("nonempty equal-drive score groups required")
    values = np.concatenate([np.asarray(g, dtype=float) for g in groups])
    weights = np.concatenate([np.full(len(g), 1 / len(g) / len(groups)) for g in groups])
    order = np.argsort(values, kind="stable")
    index = min(len(order) - 1, int(np.searchsorted(np.cumsum(weights[order]), q, side="left")))
    return max(1e-8, float(values[order[index]]))


def robust_sd(values):
    v = np.asarray(values)
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


def fit_model(trips):
    difference = []
    acceleration = []
    for _, frames in trips:
        difference.append(np.array([f.gnss_speed - f.wheel_speed for f in frames]))
        acceleration.append(
            np.array(
                [
                    f.imu_accel
                    - (f.wheel_speed - p.wheel_speed) / ((f.time_ms - f.interval_start_ms) / 1000)
                    for p, f in zip(frames[:-1], frames[1:], strict=True)
                ]
            )
        )
    variance = max(0.15, float(np.median([robust_sd(d) for d in difference]))) ** 2
    return ModelConfig(
        gnss_offset=float(np.median([np.median(d) for d in difference])),
        imu_offset=float(np.median([np.median(a) for a in acceleration])),
        r_g=max(0.05**2, 0.8 * variance),
        r_w=max(0.02**2, 0.2 * variance),
        sigma_a=max(0.2, float(np.median([robust_sd(a) for a in acceleration]))),
    )


def clean_scores(frames, model):
    o = Observer(model)
    scores = []
    for f in frames:
        o.predict(f.imu_accel, f.interval_start_ms, f.time_ms)
        if o.x is None:
            o.initialize(f.gnss_speed, f.wheel_speed, f.time_ms)
        r, s = o.innovation(f.gnss_speed, f.wheel_speed)
        scores.append(innovation_energy(r, s))
        o.update(f.gnss_speed, f.wheel_speed)
    return scores


def cusum_threshold(groups, budget):
    for threshold in np.geomspace(0.01, 1048576, 100):
        fractions = []
        for group in groups:
            stat = Statistic("cusum", float(threshold))
            fractions.append(sum(stat.step(z).alarm for z in group) / len(group))
        if np.mean(fractions) <= budget:
            return float(threshold)
    raise ValueError("declared CUSUM threshold search exhausted")


def clean_alarm_fraction(trips, model, method, thresholds):
    result = {}
    for entry, frames in trips:
        engine, keys = new_engine(model, method, thresholds)
        alarms = 0
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
            alarms += "native_motion_statistic_alarm" in d.reasons
        result[entry["id"]] = alarms / len(frames)
    return result


def calibrate(root, manifest, cfg):
    destination = inside(root, CALIBRATION)
    if destination.exists():
        frozen = check_frozen(root)
        print("Existing unchanged calibration retained:", destination)
        return frozen
    trips = phase_trips(root, manifest, "calibration")
    model = fit_model(trips)
    groups = [clean_scores(frames, model) for _, frames in trips]
    windows = [[sum(g[i - 9 : i + 1]) for i in range(9, len(g))] for g in groups]
    thresholds = dict(
        chi=drive_quantile(groups, 0.99),
        window=drive_quantile(windows, 0.99),
        cusum=cusum_threshold(groups, 0.01),
        dual_chi=drive_quantile(groups, 0.995),
        dual_cusum=cusum_threshold(groups, 0.005),
    )
    base = dict(thresholds)
    search = []
    selected = None
    for factor in cfg["candidate_factors"]:
        candidate = dict(
            base, dual_chi=base["dual_chi"] * factor, dual_cusum=base["dual_cusum"] * factor
        )
        fractions = clean_alarm_fraction(trips, model, "gated_dual", candidate)
        mean = float(np.mean(list(fractions.values())))
        search.append(dict(factor=factor, alarm_fraction=mean, per_trip=fractions))
        if mean <= cfg["alarm_budget"]:
            selected = candidate
            break
    if selected is None:
        raise ValueError(
            "candidate calibration budget unmet on predefined grid; test evaluation blocked"
        )
    thresholds = selected
    reports = {method: clean_alarm_fraction(trips, model, method, thresholds) for method in METHODS}
    frozen = dict(
        schema="native-stage6-calibration-v1",
        manifest_sha256=EXPECTED_MANIFEST,
        config_sha256=sha(inside(root, CONFIG).read_bytes().replace(b"\r\n", b"\n")),
        source_sha256=fingerprints(root, SOURCE_FILES),
        model=asdict(model),
        thresholds=thresholds,
        candidate_search=search,
        clean_alarm_fractions=reports,
        nominal_gaussian_thresholds={
            "chi_df2": float(chi2.ppf(0.99, 2)),
            "window_df20": float(chi2.ppf(0.99, 20)),
        },
        calibration_trip_ids=[e["id"] for e, _ in trips],
        calibration_epochs={e["id"]: len(f) for e, f in trips},
        limitations=[
            "Empirical calibration only; native innovations not proven iid/Gaussian",
            "4:1 GNSS/wheel variance allocation is a declared assumption",
            "Effective forward IMU offset is not full attitude calibration",
        ],
    )
    write_json(destination, frozen)
    receipt = inside(root, RECEIPT)
    with receipt.open("x", encoding="utf-8") as out:
        json.dump(
            dict(
                schema="native-stage6-freeze-receipt-v1",
                calibration_sha256=sha(destination.read_bytes()),
            ),
            out,
            indent=2,
        )
        out.write("\n")
    print("Frozen calibration:", destination)
    return frozen


def check_frozen(root):
    receipt = json.loads(inside(root, RECEIPT).read_bytes())
    if receipt["calibration_sha256"] != sha(inside(root, CALIBRATION).read_bytes()):
        raise ValueError("frozen calibration content changed; thresholds cannot be retuned")
    frozen = json.loads(inside(root, CALIBRATION).read_bytes())
    if (
        frozen["source_sha256"] != fingerprints(root, SOURCE_FILES)
        or frozen["config_sha256"] != sha(inside(root, CONFIG).read_bytes().replace(b"\r\n", b"\n"))
        or frozen["manifest_sha256"] != sha(inside(root, DATA).read_bytes())
    ):
        raise ValueError(
            "source/config/input changed after calibration; do not retune on test results"
        )
    return frozen


def csv_write(path, rows):
    if not rows:
        raise ValueError("empty result table")
    with path.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_case(entry, frames, changed, case, model, thresholds, contexts):
    rows = []
    events = []
    begin, end = bounds(frames)
    for method in METHODS:
        engine, keys = new_engine(model, method, thresholds)
        previous_gnss = None
        last_alarm = False
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
            if case == "imu_dropout" and on:
                wires.pop("imu")
            start = time.perf_counter_ns()
            decision = engine.evaluate(q, wires, f.time_ms)
            evaluation_us = (time.perf_counter_ns() - start) / 1000
            diagnostics = engine.diagnostics("native-ego")
            motion_alarm = "native_motion_statistic_alarm" in decision.reasons
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
                if case == "imu_dropout" and on:
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
        if r["method"] == "gated_dual":
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


def evaluate_phase(root, manifest, cfg, phase):
    frozen = check_frozen(root)
    model = ModelConfig(**frozen["model"])
    thresholds = frozen["thresholds"]
    if phase == "test":
        validation = inside(root, "artifacts/vehicle_native_stage6/latest_validation_summary.json")
        v = json.loads(validation.read_bytes())
        if v["calibration_sha256"] != sha(inside(root, CALIBRATION).read_bytes()):
            raise ValueError("validation calibration does not match study")
    trips = phase_trips(root, manifest, phase)
    cases = CASES if phase == "test" else VALIDATION_CASES
    destination = inside(root, f"artifacts/vehicle_native_stage6/{phase}_{time.time_ns()}")
    destination.mkdir(parents=True)
    rows = []
    events = []
    with (destination / "contexts.jsonl").open("w", encoding="utf-8") as contexts:
        for entry, frames in trips:
            for case in cases:
                changed = perturb(frames, case, model)
                r, e = run_case(entry, frames, changed, case, model, thresholds, contexts)
                rows.extend(r)
                events.extend(e)
            print(phase, entry["id"], len(frames), "epochs", flush=True)
    expected = sum(len(f) for _, f in trips) * len(cases) * len(METHODS)
    if len(rows) != expected or any(
        not r["nonce_rejected"] or not r["expiry_bounded"] for r in rows
    ):
        raise ValueError("paired count/capability invariant failed")
    if any(r["display_mutated"] == r["display_accepted"] for r in rows):
        raise ValueError("exact card invariant failed")
    csv_write(destination / "requests.csv", rows)
    metrics = aggregate(rows)
    csv_write(destination / "per_trip_metrics.csv", metrics)
    csv_write(
        destination / "paired_trip_intervals.csv", paired_intervals(metrics, cfg["bootstrap_seed"])
    )
    write_json(destination / "alarm_events.json", events)
    normal = {
        method: float(
            np.mean(
                [
                    r["grant_fraction"]
                    for r in metrics
                    if r["method"] == method and r["case"] == "normal" and r["period"] != "active"
                ]
            )
        )
        for method in METHODS
    }
    summary = dict(
        schema="native-stage6-study-v1",
        phase=phase,
        requests=len(rows),
        epochs={e["id"]: len(f) for e, f in trips},
        cases=list(cases),
        methods=list(METHODS),
        calibration_sha256=sha(inside(root, CALIBRATION).read_bytes()),
        source_sha256=fingerprints(root, SOURCE_FILES),
        input_manifest_sha256=EXPECTED_MANIFEST,
        normal_equal_trip_period_grant_fraction=normal,
        capability_invariants=True,
        exact_card_invariant=True,
        environment=dict(
            python=sys.version,
            numpy=np.__version__,
            scipy=scipy.__version__,
            platform=platform.platform(),
        ),
        limitations=manifest["limitations"]
        + [
            "Five held-out same-vehicle trips; intervals are exploratory",
            "Native sensor recordings with controlled cyberattack perturbations and synthetic signed identity",
            "Published detector mechanisms adapted to shared longitudinal observer; no full-architecture reproduction",
            "No LLM/learned-model novelty or human usability study established",
        ],
    )
    write_json(destination / "summary.json", summary)
    shutil.copy2(inside(root, CALIBRATION), destination / "frozen_calibration.json")
    shutil.copy2(inside(root, CONFIG), destination / "benchmark_config.json")
    hashes = {p.name: sha(p.read_bytes()) for p in destination.iterdir() if p.is_file()}
    write_json(destination / "file_hashes.json", hashes)
    archive = destination.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(destination.iterdir()):
            z.write(p, p.name)
    latest = inside(
        root,
        f"artifacts/vehicle_native_stage6/latest_{'study' if phase == 'test' else 'validation'}_review.zip",
    )
    temporary = latest.with_suffix(".zip.tmp")
    shutil.copy2(archive, temporary)
    os.replace(temporary, latest)
    write_json(
        inside(
            root,
            f"artifacts/vehicle_native_stage6/latest_{'study' if phase == 'test' else 'validation'}_summary.json",
        ),
        summary,
    )
    print("Review ZIP:", latest, "SHA256", sha(latest.read_bytes()), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["calibrate", "validate", "study"], required=True)
    args = parser.parse_args()
    root = ROOT.resolve()
    manifest, cfg = inputs(root)
    if args.phase == "calibrate":
        calibrate(root, manifest, cfg)
    else:
        evaluate_phase(root, manifest, cfg, "validation" if args.phase == "validate" else "test")


if __name__ == "__main__":
    main()
