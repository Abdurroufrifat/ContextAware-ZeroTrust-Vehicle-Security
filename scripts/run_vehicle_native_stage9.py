"""Frozen RAV4 policy transfer to metadata-selected native Civic routes."""

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
import run_vehicle_native_stage8 as frozen8
import scipy
from run_vehicle_native_stage6 import csv_write, fingerprints, packets, write_json
from vehicle_native_data import inside, sha
from vehicle_native_scenarios import bounds
from vehicle_session_scenarios import CASES, active, period, perturb
from vehicle_transfer_data import load_transfer_trip as load_trip

from coact_cps.vehicle.explanation import decision_card, verify_decision_card
from coact_cps.vehicle.native_detectors import ModelConfig
from coact_cps.vehicle.native_policy import NativePolicyEngine
from coact_cps.vehicle.policy import VehicleRequest, canonical_json

ROOT = Path(__file__).resolve().parents[1]
METHODS = tuple(m for m in frozen8.METHODS if m != "anchor_dual")
DATA = "data/raw/vehicle_native_stage9/manifest.json"
CONFIG = "configs/vehicle_native_stage9.json"
CONTRACT = "artifacts/vehicle_native_stage9/frozen_transfer.json"
RECEIPT = "artifacts/vehicle_native_stage9/freeze_receipt.json"
REFERENCE = "data/reference/vehicle_native_stage9"
SOURCE_FILES = frozen8.SOURCE_FILES + [
    "scripts/vehicle_native_archive.py",
    "scripts/acquire_vehicle_native_stage9.py",
    "scripts/vehicle_transfer_data.py",
    "scripts/run_vehicle_native_stage9.py",
]
new_engine = frozen8.new_engine


def validate_split(manifest):
    ids = set()
    routes = set()
    days = set()
    counts = defaultdict(int)
    for e in manifest["trips"]:
        if (
            e["id"] in ids
            or e["route_id"] in routes
            or e["day"] in days
            or not e["route_id"].startswith("99c94dc769b5d96e|")
            or e["route_id"].split("|")[1][:10] != e["day"]
            or e["phase"] not in ("validation", "test")
            or len(e["segments"]) != 2
            or e["segments"][1] != e["segments"][0] + 1
            or len(e["parts"]) != 2
        ):
            raise ValueError("Civic route/date/continuity split changed")
        ids.add(e["id"])
        routes.add(e["route_id"])
        days.add(e["day"])
        counts[e["phase"]] += 1
    if dict(counts) != {"validation": 2, "test": 5}:
        raise ValueError("fixed Civic phase counts changed")


def verify_reference_bytes(calibration, receipt, expected):
    if sha(calibration) != expected or json.loads(receipt)["calibration_sha256"] != expected:
        raise ValueError("imported reference calibration changed")
    return json.loads(calibration)


def inputs(root):
    cfg = json.loads(inside(root, CONFIG).read_bytes())
    if (
        cfg["methods"] != list(METHODS)
        or cfg["cases"] != list(CASES)
        or cfg["fit_on_civic"] is not False
    ):
        raise ValueError("fixed transfer protocol changed")
    if sha(inside(root, DATA).read_bytes()) != cfg["manifest_sha256"]:
        raise ValueError("Civic manifest changed")
    manifest = json.loads(inside(root, DATA).read_bytes())
    validate_split(manifest)
    for n in cfg["metadata_sha256"]:
        if sha(inside(root, n).read_bytes()) != cfg["metadata_sha256"][n]:
            raise ValueError("metadata-only selection changed")
    selection = json.loads(inside(root, "VEHICLE_STAGE9_SELECTION.json").read_bytes())
    from vehicle_native_archive import verify_metadata

    verify_metadata(
        root, selection, json.loads(inside(root, "VEHICLE_STAGE9_ARCHIVE_INDEX.json").read_bytes())
    )
    if [
        {k: e[k] for k in ("id", "route_id", "day", "segments", "phase")} for e in manifest["trips"]
    ] != selection["trips"]:
        raise ValueError("manifest differs from predeclared selection")
    if fingerprints(root, list(cfg["chunk_manifest_sha256"])) != cfg["chunk_manifest_sha256"]:
        raise ValueError("Civic array manifests changed")
    if {n: sha(inside(root, n).read_bytes()) for n in cfg["reference_sha256"]} != cfg[
        "reference_sha256"
    ]:
        raise ValueError("frozen reference evidence changed")
    refs = []
    for stage in (7, 8):
        n = f"{REFERENCE}/stage{stage}_linux_frozen_calibration.json"
        r = f"{REFERENCE}/stage{stage}_linux_freeze_receipt.json"
        reference = verify_reference_bytes(
            inside(root, n).read_bytes(), inside(root, r).read_bytes(), cfg["reference_sha256"][n]
        )
        refs.append(reference)
        if reference["source_sha256"] != fingerprints(root, list(reference["source_sha256"])):
            raise ValueError("inherited policy source changed")
        audit = json.loads(
            inside(root, f"{REFERENCE}/stage{stage}_windows_verification.json").read_bytes()
        )
        if audit["status"] != "VERIFIED" or not audit["all_discrete_request_fields_identical"]:
            raise ValueError("baseline Windows reproduction not verified")
    f7, f8 = refs
    if (
        f7["model"] != f8["model"]
        or f7["thresholds"] != f8["thresholds"]
        or f7["horizon"] != f8["session_config"]["stage7_horizon"]
    ):
        raise ValueError("inherited baseline mismatch")
    if (
        not Path(sys.modules[NativePolicyEngine.__module__].__file__)
        .resolve()
        .is_relative_to(root / "src")
    ):
        raise ValueError("policy import outside project; set PYTHONPATH")
    return manifest, cfg, f7, f8


def freeze(root):
    manifest, cfg, f7, f8 = inputs(root)
    if inside(root, CONTRACT).exists():
        return check_frozen(root)
    if inside(root, RECEIPT).exists():
        raise ValueError("orphan freeze receipt; do not overwrite")
    f = dict(
        schema="native-stage9-frozen-transfer-v1",
        model=f7["model"],
        thresholds=f7["thresholds"],
        session_config=f8["session_config"],
        source_sha256=fingerprints(root, SOURCE_FILES),
        config_sha256=sha(inside(root, CONFIG).read_bytes().replace(b"\r\n", b"\n")),
        manifest_sha256=cfg["manifest_sha256"],
        reference_sha256=cfg["reference_sha256"],
        metadata_sha256=cfg["metadata_sha256"],
        fit_on_civic=False,
        online_adaptation="Existing observer and Stage8 enrollment dynamics remain unchanged",
    )
    write_json(inside(root, CONTRACT), f)
    with inside(root, RECEIPT).open("x", encoding="utf-8") as stream:
        json.dump(
            dict(
                schema="native-stage9-receipt-v1",
                contract_sha256=sha(inside(root, CONTRACT).read_bytes()),
            ),
            stream,
            indent=2,
        )
        stream.write("\n")
    print(
        "Frozen imported RAV4 policies; no Civic offline fitting",
        inside(root, CONTRACT),
        flush=True,
    )
    return f


def check_frozen(root):
    path = inside(root, CONTRACT)
    f = json.loads(path.read_bytes())
    receipt = json.loads(inside(root, RECEIPT).read_bytes())
    if receipt["contract_sha256"] != sha(path.read_bytes()):
        raise ValueError("transfer contract changed")
    if (
        f["source_sha256"] != fingerprints(root, SOURCE_FILES)
        or f["config_sha256"] != sha(inside(root, CONFIG).read_bytes().replace(b"\r\n", b"\n"))
        or f["manifest_sha256"] != sha(inside(root, DATA).read_bytes())
    ):
        raise ValueError("source/config/manifest changed after transfer freeze")
    if {n: sha(inside(root, n).read_bytes()) for n in f["reference_sha256"]} != f[
        "reference_sha256"
    ]:
        raise ValueError("imported baseline changed after freeze")
    return f


def ensure_phase_unrun(root, phase):
    if phase not in ("validation", "test"):
        raise ValueError("unknown transfer phase")
    label = "validation" if phase == "validation" else "study"
    if (
        inside(root, f"artifacts/vehicle_native_stage9/{phase}").exists()
        or inside(root, f"artifacts/vehicle_native_stage9/latest_{label}_summary.json").exists()
        or inside(root, f"artifacts/vehicle_native_stage9/latest_{label}_review.zip").exists()
    ):
        raise ValueError("phase already started/completed; no silent recomputation")


def normal_overall_metrics(metrics):
    groups = defaultdict(list)
    for m in metrics:
        if m["case"] == "normal":
            groups[m["trip"], m["method"]].append(m)
    out = []
    for (trip, method), rows in sorted(groups.items()):
        n = sum(r["requests"] for r in rows)
        g = sum(r["grants"] for r in rows)
        out.append(dict(trip=trip, method=method, requests=n, grants=g, grant_fraction=g / n))
    return out


def label_comparison(row, original, observed, normal_grants):
    normal = normal_grants[row["method"], row["frame"]]
    grant = row["final_grant"]
    on = row["perturbed"]
    row.update(
        motion_changed=any(
            getattr(original, k) != getattr(observed, k)
            for k in ("gnss_speed", "wheel_speed", "imu_accel")
        ),
        normal_grant=normal,
        attack_only_denial=on and normal and not grant,
        also_denied_normal=on and not normal and not grant,
        attack_enabled_grant=on and not normal and grant,
    )
    return row


def paired_intervals(metrics, seed):
    metrics = list(metrics) + [
        dict(r, case="normal", period="overall") for r in normal_overall_metrics(metrics)
    ]
    lookup = {
        (r["trip"], r["case"], r["method"], r["period"]): r["grant_fraction"] for r in metrics
    }
    groups = defaultdict(list)
    for r in metrics:
        if r["method"] == "horizon_recovery":
            for m in METHODS:
                if m != "horizon_recovery":
                    groups[r["case"], m, r["period"]].append(
                        r["grant_fraction"] - lookup[r["trip"], r["case"], m, r["period"]]
                    )
    rng = np.random.default_rng(seed)
    out = []
    for (case, method, phase_period), values in sorted(groups.items()):
        v = np.array(values)
        samples = v[rng.integers(0, len(v), (10000, len(v)))].mean(axis=1)
        lo, hi = np.quantile(samples, [0.025, 0.975])
        out.append(
            dict(
                case=case,
                candidate="horizon_recovery",
                reference=method,
                period=phase_period,
                route_clusters=len(v),
                candidate_minus_reference=float(v.mean()),
                bootstrap_lower=float(lo),
                bootstrap_upper=float(hi),
                interpretation="exploratory whole-route resampling; one Civic",
            )
        )
    return out


def evaluate_phase(root, manifest, cfg, phase):
    ensure_phase_unrun(root, phase)
    f = check_frozen(root)
    if phase == "test":
        v = json.loads(
            inside(
                root, "artifacts/vehicle_native_stage9/latest_validation_summary.json"
            ).read_bytes()
        )
        if v["contract_sha256"] != sha(inside(root, CONTRACT).read_bytes()):
            raise ValueError("validation contract mismatch")
    model = ModelConfig(**f["model"])
    thresholds = f["thresholds"]
    config = f["session_config"]
    trips = [(e, load_trip(root, e)) for e in manifest["trips"] if e["phase"] == phase]
    destination = inside(root, f"artifacts/vehicle_native_stage9/{phase}")
    destination.mkdir(parents=True)
    metrics = []
    events = []
    rows_count = 0
    writer = None
    with (
        (destination / "requests.csv").open("w", encoding="utf-8", newline="") as request_stream,
        (destination / "contexts.jsonl").open("w", encoding="utf-8") as contexts,
    ):
        for entry, frames in trips:
            normal = {}
            original_by_frame = {x.frame: x for x in frames}
            for case in CASES:
                changed = perturb(frames, case, model, config)
                changed_by_frame = {x.frame: x for x in changed}
                rows, ev = run_case(
                    entry, frames, changed, case, model, thresholds, contexts, config
                )
                if len(rows) != len(frames) * len(METHODS):
                    raise ValueError("paired method count changed")
                for row in rows:
                    if (
                        not row["nonce_rejected"]
                        or not row["expiry_bounded"]
                        or row["display_mutated"] == row["display_accepted"]
                    ):
                        raise ValueError("capability/card invariant failed")
                    if case == "normal":
                        normal[row["method"], row["frame"]] = row["final_grant"]
                    label_comparison(
                        row, original_by_frame[row["frame"]], changed_by_frame[row["frame"]], normal
                    )
                if writer is None:
                    writer = csv.DictWriter(request_stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                writer.writerows(rows)
                rows_count += len(rows)
                metrics.extend(aggregate(rows))
                events.extend(ev)
            print(phase, entry["id"], len(frames), "epochs", flush=True)
    if rows_count != sum(len(frames) for _, frames in trips) * len(CASES) * len(METHODS):
        raise ValueError("study row count changed")
    csv_write(destination / "per_trip_metrics.csv", metrics)
    csv_write(destination / "normal_overall_metrics.csv", normal_overall_metrics(metrics))
    csv_write(
        destination / "paired_trip_intervals.csv", paired_intervals(metrics, cfg["bootstrap_seed"])
    )
    write_json(destination / "alarm_events.json", events)
    summary = dict(
        schema="native-stage9-study-v1",
        phase=phase,
        requests=rows_count,
        epochs={e["id"]: len(frames) for e, frames in trips},
        route_by_episode={e["id"]: e["route_id"] for e, _ in trips},
        route_clusters=len(trips),
        cases=list(CASES),
        methods=list(METHODS),
        contract_sha256=sha(inside(root, CONTRACT).read_bytes()),
        source_sha256=f["source_sha256"],
        input_manifest_sha256=f["manifest_sha256"],
        reference_sha256=f["reference_sha256"],
        fit_on_civic=False,
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
            "One Civic, same dataset/logger/highway, not independent-dataset validation",
            "Built-in online observer/session enrollment remains; hyperparameters frozen",
            (
                "Coherent/enrollment attacks retained; "
                "normal blocks separated from attack-only denials"
            ),
        ],
    )
    write_json(destination / "summary.json", summary)
    for source, target in [
        (CONTRACT, "frozen_transfer.json"),
        (RECEIPT, "freeze_receipt.json"),
        (CONFIG, "benchmark_config.json"),
        (DATA, "native_input_manifest.json"),
        ("VEHICLE_STAGE9_SELECTION.json", "selection_plan.json"),
        ("VEHICLE_STAGE9_ARCHIVE_INDEX.json", "archive_entry_index.json"),
    ]:
        shutil.copy2(inside(root, source), destination / target)
    write_json(
        destination / "native_chunk_manifests.json",
        {n: json.loads(inside(root, n).read_bytes()) for n in cfg["chunk_manifest_sha256"]},
    )
    for n in list(cfg["reference_sha256"]) + [
        x for x in cfg["metadata_sha256"] if x.startswith("data/")
    ]:
        target = destination / "reference_evidence" / Path(n).name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(inside(root, n), target)
    for n in SOURCE_FILES:
        target = destination / "frozen_sources" / n
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(inside(root, n), target)
    write_json(
        destination / "file_hashes.json",
        {
            p.relative_to(destination).as_posix(): sha(p.read_bytes())
            for p in destination.rglob("*")
            if p.is_file()
        },
    )
    label = "validation" if phase == "validation" else "study"
    latest = inside(root, f"artifacts/vehicle_native_stage9/latest_{label}_review.zip")
    temp = latest.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(destination.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(destination).as_posix())
    os.replace(temp, latest)
    write_json(
        inside(root, f"artifacts/vehicle_native_stage9/latest_{label}_summary.json"), summary
    )
    print("Review ZIP", latest, "SHA256", sha(latest.read_bytes()), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", required=True, choices=["freeze", "validate", "study"])
    args = p.parse_args()
    root = ROOT.resolve()
    if args.phase == "freeze":
        freeze(root)
    else:
        manifest, cfg, _, _ = inputs(root)
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
                motion_changed_requests=sum(r["motion_changed"] for r in rs),
                attack_only_denials=sum(r["attack_only_denial"] for r in rs),
                also_denied_normally=sum(r["also_denied_normal"] for r in rs),
                attack_enabled_grants=sum(r["attack_enabled_grant"] for r in rs),
                evaluation_us_median=float(np.median([r["evaluation_us"] for r in rs])),
            )
        )
    return result


if __name__ == "__main__":
    main()
