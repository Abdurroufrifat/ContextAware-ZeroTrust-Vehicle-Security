"""Fresh SUMO robustness traces with a frozen Stage 4 policy and explicit faults."""

from __future__ import annotations

import argparse
import csv
import io
import json
import platform
import random
import subprocess
import sys
import uuid
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_vehicle_sumo_pilot as analysis  # noqa: E402
import run_vehicle_stage4_replay as stage4  # noqa: E402
import run_vehicle_sumo_pilot as collector  # noqa: E402
import vehicle_stage5_protocol as protocol  # noqa: E402
import vehicle_sumo_pilot_core as core  # noqa: E402


class StudyHarness(stage4.TrialHarness):
    def packets(self, observation, sequence, frame_id, timestamp_ms, transport):
        packets = super().packets(observation, sequence, frame_id, timestamp_ms, transport)
        if "source_delays_ms" in transport:
            for source, wire in list(packets.items()):
                body = core.strict_loads(wire)["body"]
                packets[source] = core.seal_source(
                    source,
                    body["vehicle_id"],
                    body["frame_id"],
                    body["timestamp_ms"] - transport["source_delays_ms"][source],
                    body["sequence"],
                    body["signals"],
                    self.keys[source],
                )
        return packets


def verify_frozen_policy(root, path, expected):
    path = analysis.inside(root, path)
    data = path.read_bytes()
    analysis.require_digest(expected)
    if analysis.digest(data) != expected:
        raise ValueError("Uploaded Stage 4 bundle differs from frozen input")
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        entries = z.infolist()
        stage4.check_archive_names([i.filename for i in entries])
        if (
            len(entries) > 50
            or sum(i.file_size for i in entries) > 100 * 1024 * 1024
            or z.testzip() is not None
        ):
            raise ValueError("Invalid Stage 4 archive")
        metrics = json.loads(z.read("metrics.json"))
        for name, sha in metrics["data_sha256"].items():
            analysis.require_digest(sha)
            if analysis.digest(z.read(name)) != sha:
                raise ValueError("Stage 4 data digest differs")
        sources = {}
        for name, sha in metrics["source_sha256"].items():
            analysis.require_digest(sha)
            frozen = z.read("source/" + name)
            if analysis.digest(frozen) != sha:
                raise ValueError("Stage 4 source digest differs")
            p = analysis.inside(root, root / name)
            current = p.read_bytes()
            if current.replace(b"\r\n", b"\n") != frozen.replace(b"\r\n", b"\n"):
                raise ValueError(f"Frozen policy/helper changed: {name}")
            sources[name] = current
        # Collector was frozen in the nested Stage 3 evidence, not the Stage 4 source map.
        with zipfile.ZipFile(io.BytesIO(z.read("frozen_windows_baseline.zip"))) as inner:
            frozen = inner.read("input/source/scripts/run_vehicle_sumo_pilot.py")
            current = (root / "scripts/run_vehicle_sumo_pilot.py").read_bytes()
            if current.replace(b"\r\n", b"\n") != frozen.replace(b"\r\n", b"\n"):
                raise ValueError("Original SUMO collector changed")
            sources["scripts/run_vehicle_sumo_pilot.py"] = current
    for name in (
        "coact_cps.vehicle.policy",
        "coact_cps.vehicle.temporal_policy",
        "coact_cps.vehicle.explanation",
        "coact_cps.security.operator_surface",
        "coact_cps.security.verifier",
        "coact_cps.genai.grounded_generator",
        "coact_cps.types",
    ):
        relative = "src/" + name.replace(".", "/") + ".py"
        if Path(sys.modules[name].__file__).resolve() != (root / relative).resolve():
            raise ValueError("Loaded policy source is outside the project")
    for name in (
        "analyze_vehicle_sumo_pilot",
        "run_vehicle_stage4_replay",
        "run_vehicle_sumo_pilot",
        "vehicle_sumo_pilot_core",
        "vehicle_stage5_protocol",
    ):
        if Path(sys.modules[name].__file__).resolve().parent != (root / "scripts").resolve():
            raise ValueError("Loaded study helper is outside the project")
    config = stage4.TemporalConfig(**metrics["temporal_config"])
    if asdict(config) != metrics["config"]["temporal_config"]:
        raise ValueError("Candidate configuration mismatch")
    return config, sources


def create_profile_network(profile, raw, binary):
    path = raw / "traffic.net.xml"
    command = [
        str(binary),
        "--grid",
        "--grid.number",
        str(profile["grid"]),
        "--grid.length",
        str(profile["length_m"]),
        "--default.speed",
        str(profile["speed_mps"]),
        "--default.lanenumber",
        "1",
        "--output-file",
        str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=60)
    (raw / "netgenerate_command.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
    (raw / "netgenerate.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    return path


def create_profile_routes(network, path, seed, profile):
    collector.create_routes(network, path, seed)
    tree = ET.parse(path)
    root = tree.getroot()
    root.find("vType").set("maxSpeed", str(profile["speed_mps"]))
    prototypes = [
        r.attrib.copy() for r in root.findall("route") if r.get("id", "").startswith("bg-route-")
    ]
    for child in list(root):
        if child.get("id", "").startswith(("bg-route-", "bg-")):
            root.remove(child)
    rng = random.Random(seed + 900001)
    for i in range(profile["background"]):
        route = prototypes[i] if i < len(prototypes) else rng.choice(prototypes)
        ET.SubElement(root, "route", id=f"bg-route-{i}", edges=route["edges"])
        ET.SubElement(
            root,
            "vehicle",
            id=f"bg-{i}",
            type="passenger",
            route=f"bg-route-{i}",
            depart=f"{1 + i * 0.8:.1f}",
            departSpeed="0",
            departLane="best",
        )
    tree.write(path, encoding="utf-8", xml_declaration=True)


def update_stats(stats, row):
    key = (row["seed"], row["policy"], row["scenario"], row["phase"])
    counts = stats.setdefault(key, [0, 0, 0, 0, 0])
    counts[0] += 1
    counts[1] += bool(row["any_grant"])
    counts[2] += bool(row["full_grant"])
    counts[3] += bool(row["display_mutated"])
    counts[4] += bool(row["display_mutated"] and row["display_accepted"])


def summarize_stats(stats, profile):
    grouped = {}
    seed_rows = []
    records = []
    deltas = []
    for (seed, policy, case, phase), c in sorted(stats.items()):
        for measure, index, denom in [
            ("any_grant", 1, c[0]),
            ("full_grant", 2, c[0]),
            ("altered_display_acceptance", 4, c[3]),
        ]:
            if not denom:
                continue
            value = c[index] / denom
            key = (policy, case, phase, measure)
            grouped.setdefault(key, {})[seed] = value
            seed_rows.append(
                dict(
                    profile=profile,
                    seed=seed,
                    policy=policy,
                    scenario=case,
                    phase=phase,
                    measure=measure,
                    value=value,
                    requests=denom,
                )
            )

    def interval(values, label):
        ordered = [values[s] for s in sorted(values)]
        if len(ordered) == 1:
            return dict(
                mean=ordered[0],
                percentile_95=None,
                traffic_seed_clusters=1,
                interpretation="smoke setup check; no interval",
            )
        return analysis.seed_interval(ordered, label)

    for key, values in sorted(grouped.items()):
        policy, case, phase, measure = key
        result = interval(values, f"stage5/{profile}/{key}")
        total = sum(
            r["requests"]
            for r in seed_rows
            if (r["policy"], r["scenario"], r["phase"], r["measure"]) == key
        )
        records.append(
            dict(
                profile=profile,
                policy=policy,
                scenario=case,
                phase=phase,
                measure=measure,
                requests=total,
                **result,
            )
        )
        baseline = grouped.get(("stage1", case, phase, measure))
        if policy != "stage1" and baseline is not None:
            if set(values) != set(baseline):
                raise ValueError("Unpaired profile seeds")
            difference = {s: values[s] - baseline[s] for s in values}
            deltas.append(
                dict(
                    profile=profile,
                    candidate=policy,
                    reference="stage1",
                    scenario=case,
                    phase=phase,
                    measure=measure,
                    difference="candidate minus stage1",
                    **interval(difference, f"stage5/delta/{profile}/{key}"),
                )
            )
    return records, deltas, seed_rows


def trial_frames(trace, seed, profile, study, case):
    start = profile["duration_s"] * study["window_fraction"][0]
    end = profile["duration_s"] * study["window_fraction"][1]
    for truth, observed in zip(trace, core.make_observations(trace, seed), strict=True):
        t = truth["time_s"]
        if t < study["warmup_s"]:
            continue
        active = start <= t < end
        if case in protocol.FAULTS or case == "coherent_motion_forgery":
            changed, transport, active = protocol.fault(
                observed, case, t, start, end, profile["name"], seed
            )
        elif case in stage4.STRESS:
            changed, transport, active = stage4.stress_observation(observed, case, t, start, end)
        else:
            changed, transport = (
                core.inject_scenario(observed, case, (t - start) / (end - start))
                if active
                else (observed, {})
            )
        phase = "active" if active else "before" if t < start else "after"
        yield truth, changed, transport, active, phase


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "study"), default="smoke")
    parser.add_argument("--root", default=str(ROOT))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    try:
        cfg_path = analysis.inside(root, root / "configs/vehicle_stage5.json")
        study = json.loads(cfg_path.read_bytes())
        protocol.validate_config(study)
        baseline = root / "artifacts/vehicle_zero_trust_stage4/latest_stage4_review.zip"
        fixed, sources = verify_frozen_policy(root, baseline, study["stage4_zip_sha256"])
        for path in [Path(__file__), Path(protocol.__file__), cfg_path]:
            path = analysis.inside(root, path)
            sources[path.relative_to(root).as_posix()] = path.read_bytes()
        versions, binaries = collector.check_versions()
        run_id = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8]
        raw = analysis.inside(root, root / "data/raw/vehicle_zero_trust_stage5" / run_id)
        out = analysis.inside(root, root / "artifacts/vehicle_zero_trust_stage5" / run_id)
        latest = out.parent / ("latest_" + args.mode + "_review.zip")
        if latest.is_symlink():
            raise ValueError("Latest output is a symlink")
        analysis.inside(root, latest)
        raw.mkdir(parents=True, exist_ok=False)
        out.mkdir(parents=True, exist_ok=False)
        variants = {
            "stage1": None,
            "temporal_only": replace(fixed, allow_degraded_v2x=False),
            "degraded_only": replace(fixed, enable_temporal=False),
            "temporal_degraded": fixed,
        }
        cases = (
            list(analysis.SCENARIOS)
            + [(s, True) for s in stage4.STRESS]
            + [(s, False) for s in protocol.FAULTS]
        )
        interval_rows = []
        differences = []
        seed_rates = []
        drives = []
        row_count = 0
        with (
            (out / "request_comparisons.csv").open("w", newline="", encoding="utf-8") as requests,
            (raw / "labels.csv").open("w", newline="", encoding="utf-8") as labels,
            (raw / "context_observations.jsonl").open("w", encoding="utf-8") as observations,
        ):
            request_writer = None
            label_writer = csv.DictWriter(
                labels,
                fieldnames=[
                    "profile",
                    "seed",
                    "scenario",
                    "frame",
                    "time_s",
                    "active",
                    "is_attack",
                    "phase",
                ],
            )
            label_writer.writeheader()
            for profile in study["profiles"]:
                stats = {}
                profile_raw = raw / profile["name"]
                profile_raw.mkdir()
                network = create_profile_network(profile, profile_raw, binaries["netgenerate"])
                seeds = [profile["smoke_seed"]] if args.mode == "smoke" else profile["seeds"]
                for seed in seeds:
                    routes = profile_raw / f"routes_seed_{seed}.rou.xml"
                    create_profile_routes(network, routes, seed, profile)
                    trace, version = collector.collect_trace(
                        network, routes, seed, profile, profile_raw, binaries["sumo"]
                    )
                    drives.append(
                        dict(
                            profile=profile["name"],
                            seed=seed,
                            frames=len(trace),
                            simulator=version,
                            mean_speed_mps=sum(r["speed_mps"] for r in trace) / len(trace),
                            max_neighbors_50m=max(r["neighbors_50m"] for r in trace),
                        )
                    )
                    for case, attack in cases:
                        harnesses = {name: StudyHarness(cfg) for name, cfg in variants.items()}
                        for truth, observed, transport, active, phase in trial_frames(
                            trace, seed, profile, study, case
                        ):
                            label = dict(
                                profile=profile["name"],
                                seed=seed,
                                scenario=case,
                                frame=truth["frame"],
                                time_s=truth["time_s"],
                                active=active,
                                is_attack=attack,
                                phase=phase,
                            )
                            label_writer.writerow(label)
                            observations.write(
                                json.dumps(
                                    dict(
                                        profile=profile["name"],
                                        seed=seed,
                                        scenario=case,
                                        observation=observed,
                                        transport_events=transport,
                                    ),
                                    sort_keys=True,
                                )
                                + "\n"
                            )
                            for name, harness in harnesses.items():
                                result = harness.step(observed, transport, active, case)["full"]
                                row = dict(label, policy=name, **result)
                                if request_writer is None:
                                    request_writer = csv.DictWriter(requests, fieldnames=list(row))
                                    request_writer.writeheader()
                                request_writer.writerow(row)
                                update_stats(stats, row)
                                row_count += 1
                    print(f"Completed {args.mode}: {profile['name']}, seed {seed}", flush=True)
                intervals, deltas, seed_rows = summarize_stats(stats, profile["name"])
                interval_rows.extend(intervals)
                differences.extend(deltas)
                seed_rates.extend(seed_rows)
        for name, content in sources.items():
            if (root / name).read_bytes() != content:
                raise ValueError("Frozen source changed during study; result not accepted")
        report = dict(
            stage="Stage 5 fixed-policy simulation robustness study",
            mode=args.mode,
            config=study,
            temporal_config=asdict(fixed),
            execution_platform=platform.platform(),
            python=sys.version.split()[0],
            package_versions=versions,
            source_sha256={n: analysis.digest(b) for n, b in sources.items()},
            input_stage4_sha256=study["stage4_zip_sha256"],
            drives=drives,
            distinct_drives=len(drives),
            scenarios=len(cases),
            variants=list(variants),
            paired_request_rows=row_count,
            profile_intervals=interval_rows,
            paired_differences=differences,
            raw_directory=str(raw),
            result_directory=str(out),
            scope=study["scope"],
            fault_parameters={
                "imu_bias_mps2": [-0.06, 0.06],
                "wheel_scale": 1.06,
                "imu_dropout_s": 2,
                "source_timestamp_delays_ms": [0, 20],
                "coherent_forgery_acceleration_mps2": 3,
            },
            limitations=[
                "simulator proxies share SUMO origin; no independent physical-observability proof",
                "synthetic identity/CAN/V2X; no trained vehicle IDS or actuator outcome",
                "fresh software acquisition at virtual +1 ms on same frame",
                "all initialization, expiry, bias/dropout and post-forgery recovery costs retained",
                "six seeds per profile give exploratory intervals only; no global pooled accuracy or held-out guarantee",
                "smoke uses separate setup seeds and provides no uncertainty intervals",
            ],
        )
        lines = [
            "Stage 5 fixed-policy robustness study",
            f"Mode: {args.mode}; drives: {len(drives)}; paired rows: {row_count}",
            "Per-profile candidate phase rates; no policy threshold changes.",
        ]
        for item in interval_rows:
            if (
                item["policy"] == "temporal_degraded"
                and item["measure"] == "any_grant"
                and item["scenario"]
                in [
                    "normal",
                    "benign_imu_bias_positive",
                    "benign_imu_bias_negative",
                    "benign_imu_dropout",
                    "coordinated_speed_spoof",
                ]
            ):
                lines.append(
                    f"{item['profile']} / {item['scenario']} / {item['phase']}: {item['mean']:.6f}"
                )
        lines.extend(["", "LIMITATIONS", *report["limitations"]])
        review = "\n".join(lines) + "\n"
        files = {
            "review.txt": review.encode(),
            "seed_phase_rates.csv": analysis.csv_bytes(seed_rates),
        }
        for name, value in files.items():
            (out / name).write_bytes(value)
        files["request_comparisons.csv"] = (out / "request_comparisons.csv").read_bytes()
        for p in raw.rglob("*"):
            if p.is_file():
                files["raw/" + p.relative_to(raw).as_posix()] = p.read_bytes()
        files.update({"source/" + name: value for name, value in sources.items()})
        report["data_sha256"] = {n: analysis.digest(b) for n, b in files.items()}
        files["metrics.json"] = json.dumps(report, indent=2, allow_nan=False).encode()
        (out / "metrics.json").write_bytes(files["metrics.json"])
        archive = out / "study_review.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for name, value in files.items():
                z.writestr(name, value)
        analysis.latest_copy(root, latest, archive.read_bytes())
        print(review)
        print(f"Upload this file: {latest}")
    except (
        ValueError,
        KeyError,
        TypeError,
        OSError,
        UnicodeError,
        zipfile.BadZipFile,
        subprocess.SubprocessError,
    ) as exc:
        parser.exit(1, f"Stage 5 stopped: {exc}\nNo policy settings were changed.\n")


if __name__ == "__main__":
    main()
