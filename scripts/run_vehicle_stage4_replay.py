"""Paired development replay of a frozen Windows pilot; no fresh simulation or fitting."""

from __future__ import annotations

import argparse
import copy
import io
import json
import platform
import secrets
import sys
import uuid
import zipfile
from collections import defaultdict
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_vehicle_sumo_pilot as analysis  # noqa: E402
from coact_cps.vehicle.policy import SOURCES  # noqa: E402
from coact_cps.vehicle.temporal_policy import (  # noqa: E402
    TemporalConfig,
    TemporalVehiclePolicyEngine,
)
from vehicle_sumo_pilot_core import PolicyHarness  # noqa: E402

VARIANTS = ("stage1", "temporal_only", "degraded_only", "temporal_degraded")
STRESS = ("startup_speed_bias", "coherent_motion_forgery", "adversarial_v2x_suppression")


class TrialHarness(PolicyHarness):
    def __init__(self, config):
        super().__init__("ego")
        if config is not None:
            self.engine = TemporalVehiclePolicyEngine(
                self.keys, secrets.token_bytes(32), self.permissions, temporal_config=config
            )


def check_archive_names(names):
    if len(set(names)) != len(names):
        raise ValueError("Duplicate archive entries")
    for name in names:
        path = PurePosixPath(name)
        if (
            path.is_absolute()
            or ".." in path.parts
            or "\\" in name
            or ":" in name
            or not name
            or name.endswith("/")
        ):
            raise ValueError("Unsafe archive name")


def load_bundle(path, expected_sha, root):
    path = analysis.inside(root, path)
    data = path.read_bytes()
    analysis.require_digest(expected_sha)
    if analysis.digest(data) != expected_sha:
        raise ValueError("Baseline ZIP differs from the frozen uploaded bundle")
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        entries = z.infolist()
        check_archive_names([i.filename for i in entries])
        if len(entries) > 100 or sum(i.file_size for i in entries) > 100 * 1024 * 1024:
            raise ValueError("Evidence archive exceeds bounds")
        if z.testzip() is not None:
            raise ValueError("Archive CRC failed")
        payloads = {i.filename: z.read(i) for i in entries}
    summary = json.loads(payloads["analysis/summary.json"])
    metrics = json.loads(payloads["input/metrics.json"])
    expected = set(payloads) - {
        "analysis/summary.json",
        "analysis/review.txt",
        "analysis/seed_scenario_rates.csv",
    }
    if set(summary["snapshot_input_sha256"]) != expected:
        raise ValueError("Snapshot manifest entry coverage differs")
    for name, sha in summary["snapshot_input_sha256"].items():
        analysis.require_digest(sha)
        if analysis.digest(payloads[name]) != sha:
            raise ValueError("Snapshot input digest differs")
    if (
        metrics["mode"] != "pilot"
        or metrics["config"]["protocol_version"] != "coact-sumo-development-pilot-v1"
    ):
        raise ValueError("Unsupported baseline protocol")
    for relative, sha in metrics["data_sha256"].items():
        analysis.require_digest(sha)
        if analysis.digest(payloads["input/raw/" + relative]) != sha:
            raise ValueError("Original raw manifest differs")
    source_paths = {}
    for name, sha in metrics["source_sha256"].items():
        p = PureWindowsPath(name)
        if (
            not p.is_absolute()
            or len(p.parts) < 3
            or p.parts[0].lower() != "d:\\"
            or p.parts[1].lower() != "co-act-cps"
        ):
            raise ValueError("Frozen source paths must belong to D:\\CO-ACT-CPS")
        relative = PurePosixPath(*p.parts[2:]).as_posix()
        if relative in source_paths:
            raise ValueError("Duplicate source path")
        analysis.require_digest(sha)
        frozen = payloads["input/source/" + relative]
        if analysis.digest(frozen) != sha:
            raise ValueError("Original source digest differs")
        current = analysis.inside(root, root / relative)
        if current.read_bytes().replace(b"\r\n", b"\n") != frozen.replace(b"\r\n", b"\n"):
            raise ValueError(f"Current baseline source differs: {relative}")
        source_paths[relative] = str(current)
    if set(source_paths) != set(analysis.SOURCES):
        raise ValueError("Incomplete baseline source manifest")
    if Path(analysis.__file__).read_bytes().replace(b"\r\n", b"\n") != payloads[
        "analysis/analyze_vehicle_sumo_pilot.py"
    ].replace(b"\r\n", b"\n"):
        raise ValueError("Loaded analysis helper differs from frozen helper")
    # Validate actually imported modules, not a different source tree's files.
    for name in (
        "coact_cps.vehicle.policy",
        "coact_cps.vehicle.explanation",
        "coact_cps.security.operator_surface",
        "coact_cps.security.verifier",
        "coact_cps.genai.grounded_generator",
        "coact_cps.types",
    ):
        relative = "src/" + name.replace(".", "/") + ".py"
        if Path(sys.modules[name].__file__).resolve() != Path(source_paths[relative]):
            raise ValueError("Loaded baseline module is outside this project")
    for name in ("vehicle_sumo_pilot_core", "analyze_vehicle_sumo_pilot"):
        if Path(sys.modules[name].__file__).resolve().parent != (root / "scripts").resolve():
            raise ValueError("Loaded helper is outside this project")
    rows = analysis.read_csv(payloads["input/request_comparisons.csv"], "requests")
    labels = analysis.read_csv(payloads["input/raw/labels.csv"], "labels")
    analysis.check_pairing(rows, labels, metrics["traffic_seeds"])
    analysis.check_aggregate(rows, metrics["aggregate_policy_summary"])
    return data, metrics, rows, labels, payloads


def stress_observation(observed, scenario, time_s, start, end):
    result = copy.deepcopy(observed)
    s = result["signals"]
    transport = {}
    active = True if scenario == "startup_speed_bias" else start <= time_s < end
    if active:
        if scenario == "startup_speed_bias":
            for source in ("gnss", "wheel"):
                s[source]["speed_mps"] += 8.0
        elif scenario == "coherent_motion_forgery":
            offset = 5.0 * (time_s - start)
            for source in ("gnss", "wheel"):
                s[source]["speed_mps"] += offset
            for source in ("imu", "wheel"):
                s[source]["acceleration_mps2"] += 5.0
        elif scenario == "adversarial_v2x_suppression":
            transport["missing"] = "v2x"
        else:
            raise ValueError("Unknown stress scenario")
    return result, transport, active


def replay(metrics, original_rows, labels, payloads, config):
    original = {
        (r["seed"], r["episode_id"], r["frame"]): r for r in original_rows if r["policy"] == "full"
    }
    label_map = {(r["seed"], r["episode_id"], r["frame"]): r for r in labels}
    episodes = defaultdict(list)
    seen = set()
    for line in payloads["input/raw/context_observations.jsonl"].decode("utf-8").splitlines():
        item = json.loads(line)
        observed = item["observation"]
        key = (item["seed"], item["episode_id"], observed["frame"])
        if key in seen or key not in label_map:
            raise ValueError("Observation is duplicate or has no label")
        seen.add(key)
        label = label_map[key]
        if set(observed) != {"frame", "timestamp_ms", "signals"} or set(observed["signals"]) != set(
            SOURCES
        ):
            raise ValueError("Observation schema differs")
        if observed["timestamp_ms"] != round(label["time_s"] * 1000) + 1000000:
            raise ValueError("Observation time differs from label")
        episodes[(item["seed"], label["scenario"])].append(
            (label, observed, item["transport_events"])
        )
    if seen != set(label_map):
        raise ValueError("Observation coverage differs from labels")
    settings = metrics["config"]["pilot"]
    start = settings["duration_s"] * metrics["config"]["window_fraction"][0]
    end = settings["duration_s"] * metrics["config"]["window_fraction"][1]
    for seed in metrics["traffic_seeds"]:
        normal = episodes[(seed, "normal")]
        for scenario in STRESS:
            generated = []
            for label, observed, _ in normal:
                modified, transport, active = stress_observation(
                    observed, scenario, label["time_s"], start, end
                )
                generated.append(
                    (
                        dict(
                            label,
                            scenario=scenario,
                            episode_id=f"seed-{seed}-{scenario}",
                            active=active,
                            is_attack=True,
                        ),
                        modified,
                        transport,
                    )
                )
            episodes[(seed, scenario)] = generated
    variants = {
        "stage1": None,
        "temporal_only": replace(config, allow_degraded_v2x=False),
        "degraded_only": replace(config, enable_temporal=False),
        "temporal_degraded": config,
    }
    results = []
    matched = 0
    for (seed, scenario), frames in sorted(episodes.items()):
        harnesses = {name: TrialHarness(cfg) for name, cfg in variants.items()}
        for label, observed, transport in sorted(frames, key=lambda x: x[0]["frame"]):
            for name, harness in harnesses.items():
                result = harness.step(observed, transport, label["active"], scenario)["full"]
                if name == "stage1" and scenario not in STRESS:
                    expected = original[(seed, label["episode_id"], label["frame"])]
                    for field in (
                        "outcome",
                        "mode",
                        "risk",
                        "any_grant",
                        "full_grant",
                        "display_accepted",
                        "display_mutated",
                        "explanation_status",
                    ):
                        if not analysis.equal(result[field], expected[field]):
                            raise ValueError(
                                f"Baseline replay differs at {seed}/{scenario}/{label['frame']}/{field}"
                            )
                    matched += 1
                results.append(dict(label, policy=name, **result))
        print(f"Replayed seed {seed}, scenario {scenario}", flush=True)
    if matched != len(original):
        raise ValueError("Incomplete original baseline reproduction")
    return results, matched


def summarize(rows):
    rates = analysis.group_rates(rows)
    records = []
    for (policy, scenario, measure), seeds in sorted(rates.items()):
        summary = analysis.seed_interval(
            [seeds[s] for s in sorted(seeds)], f"stage4/{policy}/{scenario}/{measure}"
        )
        records.append(dict(policy=policy, scenario=scenario, measure=measure, **summary))
    return records


def seed_comparisons(rows):
    rates = analysis.group_rates(rows)
    seed_rows = []
    deltas = []
    for (policy, scenario, measure), values in sorted(rates.items()):
        seed_rows.extend(
            dict(policy=policy, scenario=scenario, measure=measure, seed=seed, value=value)
            for seed, value in sorted(values.items())
        )
        baseline = rates.get(("stage1", scenario, measure))
        if policy != "stage1" and baseline is not None:
            interval = analysis.paired_interval(
                values, baseline, f"stage4/delta/{policy}/{scenario}/{measure}"
            )
            deltas.append(
                dict(
                    candidate=policy,
                    reference="stage1",
                    scenario=scenario,
                    measure=measure,
                    difference="candidate minus stage1",
                    **interval,
                )
            )
    return seed_rows, deltas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    try:
        config_path = analysis.inside(root, root / "configs/vehicle_stage4.json")
        protocol = json.loads(config_path.read_bytes())
        if protocol["version"] != "coact-vehicle-stage4-development-replay-v1":
            raise ValueError("Unsupported Stage 4 config")
        config = TemporalConfig(**protocol["temporal_config"])
        baseline = root / "artifacts/vehicle_zero_trust_pilot_analysis/latest_baseline_review.zip"
        data, metrics, original_rows, labels, payloads = load_bundle(
            baseline, protocol["baseline_zip_sha256"], root
        )
        rows, matched = replay(metrics, original_rows, labels, payloads, config)
        intervals = summarize(rows)
        seed_rows, deltas = seed_comparisons(rows)
        parent = analysis.inside(root, root / "artifacts/vehicle_zero_trust_stage4")
        latest = parent / "latest_stage4_review.zip"
        if latest.is_symlink():
            raise ValueError("Latest output cannot be a symlink")
        analysis.inside(root, latest)
        out = parent / (
            datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8]
        )
        out.mkdir(parents=True, exist_ok=False)
        source_paths = [
            Path(__file__).resolve(),
            config_path,
            Path(sys.modules["coact_cps.vehicle.temporal_policy"].__file__).resolve(),
            Path(analysis.__file__).resolve(),
            Path(sys.modules["vehicle_sumo_pilot_core"].__file__).resolve(),
        ]
        source_paths.extend(
            Path(sys.modules[name].__file__).resolve()
            for name in (
                "coact_cps.vehicle.policy",
                "coact_cps.vehicle.explanation",
                "coact_cps.security.operator_surface",
                "coact_cps.security.verifier",
                "coact_cps.genai.grounded_generator",
                "coact_cps.types",
            )
        )
        sources = {}
        for p in source_paths:
            p = analysis.inside(root, p)
            sources[p.relative_to(root).as_posix()] = p.read_bytes()
        report = dict(
            stage="Stage 4 development candidate replay",
            config=protocol,
            temporal_config=asdict(config),
            source_traffic_platform=metrics["platform"],
            execution_platform=platform.platform(),
            python=sys.version.split()[0],
            original_stage1_requests_reproduced=matched,
            paired_rows=len(rows),
            traffic_seeds=metrics["traffic_seeds"],
            variants=list(VARIANTS),
            source_sha256={name: analysis.digest(value) for name, value in sources.items()},
            baseline_zip_sha256=analysis.digest(data),
            scenario_intervals=intervals,
            paired_differences=deltas,
            scope=protocol["scope"],
            limitations=[
                "development reuse of observed pilot; no held-out claim or fitting",
                "startup bias and joint motion-source forgery remain potential blind spots",
                "IMU error allowance is illustrative, not a calibrated bound",
                "IMU gaps can permanently block this process-local reference",
                "cold enrollment depends on truthful initial wheel speed",
                "V2X suppression, benign or adversarial, can yield an authorized snapshot; never a stream",
                "same-frame software recheck at virtual +1 ms; no actuator or human-participant study",
            ],
        )
        lines = [
            "Stage 4 development candidate replay",
            f"Original Stage 1 rows exactly reproduced: {matched}",
            f"Paired rows across four variants: {len(rows)}",
            "FULL CANDIDATE: active-window any/full grant fractions",
        ]
        for item in intervals:
            if item["policy"] == "temporal_degraded" and item["measure"] in (
                "active_any_grant",
                "active_full_grant",
            ):
                lines.append(f"{item['scenario']} {item['measure']}: {item['mean']:.6f}")
        lines.extend(["", "LIMITATIONS", *report["limitations"]])
        text = "\n".join(lines) + "\n"
        files = {
            "metrics.json": json.dumps(report, indent=2, allow_nan=False).encode(),
            "request_comparisons.csv": analysis.csv_bytes(rows),
            "seed_scenario_rates.csv": analysis.csv_bytes(seed_rows),
            "review.txt": text.encode(),
            "frozen_windows_baseline.zip": data,
        }
        files.update({"source/" + name: value for name, value in sources.items()})
        report["data_sha256"] = {
            name: analysis.digest(value) for name, value in files.items() if name != "metrics.json"
        }
        files["metrics.json"] = json.dumps(report, indent=2, allow_nan=False).encode()
        for name in (
            "metrics.json",
            "request_comparisons.csv",
            "seed_scenario_rates.csv",
            "review.txt",
        ):
            (out / name).write_bytes(files[name])
        archive = out / "stage4_review.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
            for name, value in files.items():
                z.writestr(name, value)
        analysis.latest_copy(root, latest, archive.read_bytes())
        print(text)
        print(f"Upload this file: {latest}")
    except (ValueError, KeyError, TypeError, OSError, UnicodeError, zipfile.BadZipFile) as exc:
        parser.exit(1, f"Stage 4 replay stopped: {exc}\n")


if __name__ == "__main__":
    main()
