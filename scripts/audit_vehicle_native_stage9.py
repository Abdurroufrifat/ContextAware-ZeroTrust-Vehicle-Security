"""Audit saved outcomes without running or tuning policies."""

import argparse
import csv
import hashlib
import io
import json
import statistics
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from vehicle_native_archive import verify_metadata

p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
p.add_argument("--phase", choices=["validation", "test"], required=True)
args = p.parse_args()
root = args.root.resolve()
out = root / "artifacts/vehicle_native_stage9"
archive = out / (
    "latest_validation_review.zip" if args.phase == "validation" else "latest_study_review.zip"
)


def sha(b):
    return hashlib.sha256(b).hexdigest()


counts = defaultdict(Counter)
normal = defaultdict(Counter)
normal_flags = {}
paired_exposure = defaultdict(Counter)
groups = {}
transitions = {}
expected_events = []
active_delay = {}
issued = mutated = releases = lost_requests = unready_grants = 0
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    hashes = json.loads(z.read("file_hashes.json"))
    assert set(hashes) == set(z.namelist()) - {"file_hashes.json"}
    for n, h in hashes.items():
        assert sha(z.read(n)) == h, n
    s = json.loads(z.read("summary.json"))
    cfg = json.loads(z.read("benchmark_config.json"))
    f = json.loads(z.read("frozen_transfer.json"))
    receipt = json.loads(z.read("freeze_receipt.json"))
    assert receipt["contract_sha256"] == s["contract_sha256"] == sha(z.read("frozen_transfer.json"))
    assert z.read("frozen_transfer.json") == (out / "frozen_transfer.json").read_bytes()
    assert s["source_sha256"] == f["source_sha256"]
    for n, h in s["source_sha256"].items():
        assert sha((root / n).read_bytes().replace(b"\r\n", b"\n")) == h, n
        assert sha(z.read("frozen_sources/" + n).replace(b"\r\n", b"\n")) == h, n
    assert sha(z.read("benchmark_config.json")) == f["config_sha256"]
    assert (
        sha(z.read("native_input_manifest.json"))
        == f["manifest_sha256"]
        == s["input_manifest_sha256"]
    )
    native = json.loads(z.read("native_input_manifest.json"))
    arrays = 0
    for n, m in json.loads(z.read("native_chunk_manifests.json")).items():
        p = root / n
        assert (
            json.loads(p.read_bytes()) == m
            and sha(p.read_bytes()) == cfg["chunk_manifest_sha256"][n]
        )
        for item in m["files"]:
            path = p.parent / item["path"]
            assert sha(path.read_bytes()) == item["sha256"]
            assert list(np.load(path, allow_pickle=False).shape) == item["shape"]
            arrays += 1
    with z.open("requests.csv") as stream:
        for total, r in enumerate(csv.DictReader(io.TextIOWrapper(stream)), 1):  # noqa: B007
            key = (r["trip"], r["case"], r["frame"])
            label = (r["time_ms"], r["period"], r["perturbed"], r["motion_changed"])
            if key not in groups:
                groups[key] = (label, set())
            assert groups[key][0] == label and r["method"] not in groups[key][1]
            groups[key][1].add(r["method"])
            assert r["nonce_rejected"] == "True" and r["expiry_bounded"] == "True"
            assert (r["display_mutated"] == "True") != (r["display_accepted"] == "True")
            assert (r["final_grant"] == "True") == (
                (r["capability_authorized"] == "True") and (r["display_accepted"] == "True")
            )
            event_key = r["trip"], r["case"], r["method"]
            prior = transitions.get(event_key, (False, False, None, False))
            quarantined = r["quarantined"] == "True"
            alarm = r["motion_alarm"] == "True"
            now = int(r["time_ms"])
            start = prior[2]
            base = dict(trip=r["trip"], case=r["case"], method=r["method"], time_ms=now)
            if quarantined and not prior[0]:
                start = now
                expected_events.append(
                    dict(base, event="quarantine_enter", active=r["perturbed"] == "True")
                )
            if not quarantined and prior[0]:
                expected_events.append(
                    dict(
                        base,
                        event="quarantine_exit",
                        active=r["perturbed"] == "True",
                        elapsed_ms=now - start,
                    )
                )
            if r["perturbed"] == "True":
                delay = active_delay.setdefault(event_key, [now, None])
                if alarm and delay[1] is None:
                    delay[1] = now
            if alarm and not prior[1]:
                expected_events.append(
                    dict(base, event="motion_alarm_episode_start", active=r["perturbed"] == "True")
                )
            if prior[3]:
                assert r["anchor_lost"] == "True"
            transitions[event_key] = quarantined, alarm, start, r["anchor_lost"] == "True"
            issued += r["issued_outcome"] == "ALLOW"
            mutated += r["display_mutated"] == "True"
            if r["recovery_release"] == "True":
                releases += 1
                assert r["issued_outcome"] == "CHALLENGE" and r["final_grant"] == "False"
            if r["anchor_lost"] == "True":
                lost_requests += 1
                assert r["issued_outcome"] == "DENY" and r["final_grant"] == "False"
            if (
                r["method"].startswith("anchor")
                and r["anchor_ready"] == "False"
                and r["final_grant"] == "True"
            ):
                unready_grants += 1
            c = counts[r["trip"], r["case"], r["method"], r["period"]]
            c["requests"] += 1
            for field, k in [
                ("grants", "final_grant"),
                ("motion_alarms", "motion_alarm"),
                ("quarantine_requests", "quarantined"),
                ("recovery_releases", "recovery_release"),
                ("current_detector_alarms", "current_detector_alarm"),
                ("motion_changed_requests", "motion_changed"),
                ("attack_only_denials", "attack_only_denial"),
                ("also_denied_normally", "also_denied_normal"),
                ("attack_enabled_grants", "attack_enabled_grant"),
            ]:
                c[field] += r[k] == "True"
            c["blocked"] += r["final_grant"] == "False"
            c["challenges"] += r["issued_outcome"] == "CHALLENGE"
            if r["case"] == "normal":
                normal_flags[r["trip"], r["frame"], r["method"]] = r["final_grant"] == "True"
            clean = normal_flags[r["trip"], r["frame"], r["method"]]
            grant = r["final_grant"] == "True"
            on = r["perturbed"] == "True"
            assert (r["normal_grant"] == "True") == clean
            assert (r["attack_only_denial"] == "True") == (on and clean and not grant)
            assert (r["also_denied_normal"] == "True") == (on and not clean and not grant)
            assert (r["attack_enabled_grant"] == "True") == (on and not clean and grant)
            if r["case"] != "normal" and on:
                clean = normal_flags[r["trip"], r["frame"], r["method"]]
                grant = r["final_grant"] == "True"
                paired_exposure[r["case"], r["method"]]["active_requests"] += 1
                paired_exposure[r["case"], r["method"]]["grants"] += grant
                paired_exposure[r["case"], r["method"]]["attack_only_denials"] += (
                    clean and not grant
                )
                paired_exposure[r["case"], r["method"]]["also_denied_normally"] += (
                    not clean and not grant
                )
                paired_exposure[r["case"], r["method"]]["attack_enabled_grants"] += (
                    not clean and grant
                )
            if r["case"] == "normal":
                normal[s["route_by_episode"][r["trip"]], r["method"]]["requests"] += 1
                normal[s["route_by_episode"][r["trip"]], r["method"]]["grants"] += (
                    r["final_grant"] == "True"
                )
    epochs = sum(s["epochs"].values())
    assert (
        total
        == s["requests"]
        == epochs * 21 * 9
        == (217539 if args.phase == "validation" else 523341)
    )
    assert len(groups) == epochs * 21 and all(g[1] == set(s["methods"]) for g in groups.values())
    seen = set()
    with z.open("contexts.jsonl") as stream:
        for line in stream:
            c = json.loads(line)
            key = (c["trip"], c["case"], str(c["native"]["frame"]))
            assert key not in seen
            seen.add(key)
            label = groups[key][0]
            assert (
                str(c["observed"]["time_ms"]) == label[0]
                and c["evaluation_only"]["period"] == label[1]
            )
            assert c["evaluation_only"]["active"] == (label[2] == "True")
            assert (label[3] == "True") == any(
                c["native"][k] != c["observed"][k]
                for k in ("gnss_speed", "wheel_speed", "imu_accel")
            )
            assert not set(c["native"]).intersection(
                {"normal_grant", "attack_only_denial", "perturbed"}
            )
            for k in (
                "time_ms",
                "interval_start_ms",
                "gnss_time_ms",
                "imu_time_ms",
                "wheel_time_ms",
                "maximum_gap_ms",
            ):
                assert c["native"][k] == c["observed"][k]
    assert seen == set(groups)
    metrics = list(csv.DictReader(io.StringIO(z.read("per_trip_metrics.csv").decode())))
    assert len(metrics) == len(counts)
    routecounts = defaultdict(Counter)
    episode_sets = defaultdict(set)
    for m in metrics:
        key = (m["trip"], m["case"], m["method"], m["period"])
        c = counts[key]
        for k, v in c.items():
            assert int(m[k]) == v
        assert abs(float(m["grant_fraction"]) - c["grants"] / c["requests"]) < 1e-14
        key2 = (s["route_by_episode"][m["trip"]], m["case"], m["method"], m["period"])
        episode_sets[key2].add(m["trip"])
        routecounts[key2].update(c)
    overall = list(csv.DictReader(io.StringIO(z.read("normal_overall_metrics.csv").decode())))
    assert len(overall) == len(normal)
    for r in overall:
        c = normal[s["route_by_episode"][r["trip"]], r["method"]]
        assert int(r["requests"]) == c["requests"] and int(r["grants"]) == c["grants"]
        assert abs(float(r["grant_fraction"]) - c["grants"] / c["requests"]) < 1e-14
    route_ids = sorted(set(s["route_by_episode"].values()))
    n = len(route_ids)
    assert n == s["route_clusters"] == (2 if args.phase == "validation" else 5)
    intervals = list(csv.DictReader(io.StringIO(z.read("paired_trip_intervals.csv").decode())))
    rng = np.random.default_rng(cfg["bootstrap_seed"])
    for i in intervals:
        assert i["candidate"] == "horizon_recovery"

        def fraction(t, m, i=i):
            c = (
                normal[t, m]
                if i["period"] == "overall"
                else routecounts[t, i["case"], m, i["period"]]
            )
            return c["grants"] / c["requests"]

        v = np.array(
            [fraction(t, "horizon_recovery") - fraction(t, i["reference"]) for t in route_ids]
        )
        assert int(i["route_clusters"]) == n
        low, high = np.quantile(v[rng.integers(0, n, (10000, n))].mean(axis=1), [0.025, 0.975])
        for k, vv in [
            ("candidate_minus_reference", v.mean()),
            ("bootstrap_lower", low),
            ("bootstrap_upper", high),
        ]:
            assert abs(float(i[k]) - vv) < 1e-14
    actual_events = json.loads(z.read("alarm_events.json"))
    assert [
        e for e in actual_events if e["event"] != "first_motion_alarm_delay_ms"
    ] == expected_events
    delay_events = [e for e in actual_events if e["event"] == "first_motion_alarm_delay_ms"]
    assert len(delay_events) == len(active_delay)
    for e in delay_events:
        begin, first = active_delay[e["trip"], e["case"], e["method"]]
        assert e["time_ms"] == (first if first is not None else -1)
        assert e["delay_ms"] == (first - begin if first is not None else -1)
        assert e["active"] is True
    for name, h in f["reference_sha256"].items():
        assert sha(z.read("reference_evidence/" + Path(name).name)) == h
    for name, h in f["metadata_sha256"].items():
        target = {
            "VEHICLE_STAGE9_SELECTION.json": "selection_plan.json",
            "VEHICLE_STAGE9_ARCHIVE_INDEX.json": "archive_entry_index.json",
        }.get(name, "reference_evidence/" + Path(name).name)
        assert sha(z.read(target)) == h
old = json.loads(
    (
        root / "data/reference/vehicle_native_stage9/stage7_linux_frozen_calibration.json"
    ).read_bytes()
)
assert (
    f["model"] == old["model"]
    and f["thresholds"] == old["thresholds"]
    and f["session_config"]["stage7_horizon"] == old["horizon"]
)
metadata = verify_metadata(
    root,
    json.loads((root / "VEHICLE_STAGE9_SELECTION.json").read_bytes()),
    json.loads((root / "VEHICLE_STAGE9_ARCHIVE_INDEX.json").read_bytes()),
)
audit = dict(
    status="PASS",
    phase=args.phase,
    requests=total,
    native_epochs=epochs,
    route_clusters=n,
    cases=21,
    methods=9,
    paired_contexts=len(groups),
    metric_rows=len(metrics),
    normal_overall_rows=len(overall),
    paired_route_interval_rows=len(intervals),
    raw_arrays_verified=arrays,
    source_snapshots_verified=len(s["source_sha256"]),
    archive_content_hashes_verified=len(hashes),
    issued_capabilities=issued,
    altered_cards=mutated,
    recovery_releases=releases,
    all_releases_challenge=True,
    alarm_events_recomputed=len(actual_events),
    lost_reference_latch_verified=True,
    reference_lost_requests=lost_requests,
    all_lost_reference_requests_denied=True,
    grants_while_anchor_unready=unready_grants,
    all_denominators_and_matched_normal_flags_verified=True,
    all_route_intervals_recomputed=True,
    inherited_parameters_unchanged=True,
    metadata_verification=metadata,
    archive_sha256=sha(archive.read_bytes()),
    contract_sha256=s["contract_sha256"],
)
audit["active_comparison"] = {
    case: {method: dict(paired_exposure[case, method]) for method in s["methods"]}
    for case in s["cases"]
    if case != "normal"
}
(out / ("independent_" + args.phase + "_audit.json")).write_text(json.dumps(audit, indent=2) + "\n")
if args.phase == "test":
    lines = [
        "CO-ACT-CPS Stage9 Civic transfer results; project evidence, no manuscript",
        "",
        "255 full-project tests;21Stage9 tests. One existing legacy warning.",
        "One pre-outcome code review;2Important/1Minor findings resolved/tested.",
        "Source/config/reference/data frozen before validation/test; no Civic offline fitting.",
        "Existing online observer and enrollment remain unchanged. No outcome-based tuning.",
        f"{epochs}native epochs;5test routes/dates,21cases,9methods;{total}requests.",
        "Audit PASS: hashes/receipts/source/native arrays/shared contexts/matched-normal",
        "flags/all metric denominator counts/every whole-route bootstrap interval verified.",
        "",
        "Normal availability, all periods pooled within route before equal-route mean:",
    ]
    for m in s["methods"]:
        cs = [normal[t, m] for t in route_ids]
        lines.append(
            f"{m:22s} {sum(c['grants'] for c in cs)}/{sum(c['requests'] for c in cs)} grants; "
            f"equal-route%={100 * statistics.mean(c['grants'] / c['requests'] for c in cs):.2f}"
        )
    lines += ["", "Active perturbed-window grants/requests; ordinary normal blocks separated:"]
    for case in s["cases"]:
        if case == "normal":
            continue
        for m in ("gated_dual", "gated_dual_matched", "horizon_recovery", "anchor_recovery"):
            c = paired_exposure[case, m]
            lines.append(
                f"{case:22s} {m:22s} {c['grants']}/{c['active_requests']}; "
                f"attack-only-denials={c['attack_only_denials']}; "
                f"also-normal-denial={c['also_denied_normally']}; "
                f"attack-enabled-grants={c['attack_enabled_grants']}"
            )
    lines += ["", "Honest after-withdrawal grants/requests (availability cost):"]
    for case in (
        "joint_jump",
        "joint_ramp",
        "coherent_withdrawal",
        "observer_aware",
        "imu_dropout",
        "horizon_budget",
        "gap_reenrollment",
        "anchor_budget",
    ):
        for m in ("gated_dual", "gated_dual_matched", "horizon_recovery", "anchor_recovery"):
            cs = [routecounts[t, case, m, "after"] for t in route_ids]
            lines.append(
                f"{case:22s} {m:22s} {sum(c['grants'] for c in cs)}/"
                f"{sum(c['requests'] for c in cs)}"
            )
    lines += [
        "",
        "Five route clusters from one Civic, same comma2k19 logger/highway.",
        "Cross-vehicle transfer, not independent-dataset or population validation.",
        "Presumed benign recordings plus controlled perturbations, not real attacks.",
        "Empirical consistency is not hardware/physical truth. Coherent/startup attacks",
        "must use matched-normal evidence; normal denials are not attack detection.",
        f"{unready_grants}grants before anchor readiness;"
        f"{lost_requests}lost-reference requests denied.",
        "Reference loss can sacrifice honest availability. No human/actuation study.",
        "No journal acceptance/universal superiority claim. Windows reproduction pending.",
        "",
        "Study SHA256:" + audit["archive_sha256"],
        "Contract SHA256:" + audit["contract_sha256"],
    ]
    (root / "VEHICLE_STAGE9_AUDIT_RESULTS.txt").write_text("\n".join(lines) + "\n")
print(json.dumps({k: v for k, v in audit.items() if k != "active_comparison"}, indent=2))
