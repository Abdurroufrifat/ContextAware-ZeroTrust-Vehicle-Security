"""Collect a real SUMO traffic run, then evaluate synthetic context perturbations.

The result is a development simulation pilot, not native vehicle sensor data,
not a trained vehicle IDS, and not a final held-out scientific confirmation.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import uuid
import xml.etree.ElementTree as ET
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from coact_cps.vehicle.policy import PolicyConfig  # noqa: E402
from vehicle_sumo_pilot_core import (  # noqa: E402
    POLICIES,
    SCENARIOS,
    PolicyHarness,
    inject_scenario,
    make_observations,
    summarize,
)


def validate_config(config, mode):
    if (
        config.get("protocol_version") != "coact-sumo-development-pilot-v1"
        or config.get("sumo_version") != "1.28.0"
    ):
        raise ValueError("Unsupported protocol or simulator version")
    settings = config[mode]
    seeds = settings.get("seeds")
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(type(s) is not int or not 0 <= s < 2**31 for s in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("Seeds must be a nonempty list of unique nonnegative integers")

    def finite(value):
        return type(value) in (int, float) and math.isfinite(value)

    duration, step = settings.get("duration_s"), settings.get("step_s")
    if not finite(duration) or not finite(step) or duration < 4 or not 0.05 <= step <= 1:
        raise ValueError("Duration must be >=4 seconds and step must be between .05 and 1 second")
    if not math.isclose(duration / step, round(duration / step), abs_tol=1e-8):
        raise ValueError("Duration must contain a whole number of simulation steps")
    window = config.get("window_fraction")
    if (
        not isinstance(window, list)
        or len(window) != 2
        or not all(finite(v) for v in window)
        or not 0 < window[0] < window[1] < 1
    ):
        raise ValueError("Window fractions must satisfy 0 < start < end < 1")
    warmup = config.get("warmup_s")
    if not finite(warmup) or not 0 <= warmup < duration * window[0]:
        raise ValueError("Warmup must precede the perturbation window")
    if not any(
        duration * window[0] <= i * step < duration * window[1]
        for i in range(1, round(duration / step) + 1)
    ):
        raise ValueError("Perturbation window must contain a simulation frame")
    return settings


def check_source_root(root, module_paths):
    root = Path(root).resolve()
    for relative, loaded in module_paths.items():
        expected = (root / relative).resolve()
        if (
            not expected.is_relative_to(root)
            or not expected.is_file()
            or Path(loaded).resolve() != expected
        ):
            raise ValueError(f"Loaded Stage 1 source differs from --root: {relative}")


def latest_destination(path, root):
    root = Path(root).resolve()
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError("Latest metrics destination must be a regular project file")


def write_latest(path, content, root):
    latest_destination(path, root)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(content)
    try:
        latest_destination(path, root)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def sha256_file(path):
    import hashlib

    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def find_binary(name):
    suffix = ".exe" if os.name == "nt" else ""
    spec = importlib.util.find_spec("sumo")
    candidates = []
    if spec and spec.origin:
        candidates.append(Path(spec.origin).parent / "bin" / (name + suffix))
    found = shutil.which(name)
    if found:
        candidates.append(Path(found))
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise RuntimeError("SUMO binary unavailable. Install the pinned requirements-vehicle-sumo.txt.")


def check_versions():
    versions = {}
    for package in ("eclipse-sumo", "sumolib", "traci"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                "Install requirements-vehicle-sumo.txt in the project venv."
            ) from exc
        if versions[package] != "1.28.0":
            raise RuntimeError(f"{package} must be 1.28.0; observed {versions[package]}")
    binaries = {name: find_binary(name) for name in ("sumo", "netgenerate")}
    for name, path in binaries.items():
        version = subprocess.run(
            [str(path), "--version"], capture_output=True, text=True, check=True, timeout=15
        ).stdout
        if "1.28.0" not in version:
            raise RuntimeError(f"{name} binary does not match pinned version 1.28.0")
        versions[name + "_binary_sha256"] = sha256_file(path)
    return versions, binaries


def create_network(raw, binary):
    network = raw / "traffic.net.xml"
    command = [
        str(binary),
        "--grid",
        "--grid.number",
        "3",
        "--grid.length",
        "150",
        "--default.speed",
        "13.9",
        "--default.lanenumber",
        "1",
        "--output-file",
        str(network),
    ]
    process = subprocess.run(command, capture_output=True, text=True, check=True, timeout=60)
    (raw / "netgenerate_command.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
    (raw / "netgenerate.log").write_text(process.stdout + process.stderr, encoding="utf-8")
    return network


def create_routes(network, path, seed):
    import sumolib

    net = sumolib.net.readNet(str(network))
    edges = sorted(net.getEdges(), key=lambda edge: edge.getID())
    # Select a reachable loop using only SUMO's connectivity; archive the XML.
    route = None
    for start in edges:
        for end in reversed(edges):
            forward, cost = net.getShortestPath(start, end, vClass="passenger")
            backward, back_cost = net.getShortestPath(end, start, vClass="passenger")
            if forward and backward and cost + back_cost > 600:
                cycle = [edge.getID() for edge in forward] + [edge.getID() for edge in backward[1:]]
                route = cycle + cycle[1:] + cycle[1:] + cycle[1:]
                break
        if route:
            break
    if not route:
        raise RuntimeError("No persistent ego route found in generated network")
    root = ET.Element("routes")
    ET.SubElement(
        root,
        "vType",
        id="passenger",
        vClass="passenger",
        accel="2.0",
        decel="4.5",
        sigma="0.5",
        length="4.5",
        maxSpeed="13.9",
    )
    ET.SubElement(root, "route", id="ego-route", edges=" ".join(route))
    ET.SubElement(
        root,
        "vehicle",
        id="ego",
        type="passenger",
        route="ego-route",
        depart="0",
        departSpeed="0",
        departLane="best",
    )
    import random

    rng = random.Random(seed)
    for i in range(12):
        # Shift a closed loop to vary background departure position.
        simple = route[: len(cycle) - 1]
        offset = rng.randrange(len(simple))
        rotated = simple[offset:] + simple[:offset]
        repeated = rotated + rotated + rotated
        ET.SubElement(root, "route", id=f"bg-route-{i}", edges=" ".join(repeated))
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
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def collect_trace(network, routes, seed, config, raw, binary):
    import traci

    label = "coact-" + uuid.uuid4().hex[:10]
    command = [
        str(binary),
        "--net-file",
        str(network),
        "--route-files",
        str(routes),
        "--seed",
        str(seed),
        "--step-length",
        str(config["step_s"]),
        "--no-step-log",
        "true",
        "--duration-log.disable",
        "true",
        "--time-to-teleport",
        "-1",
        "--error-log",
        str(raw / f"sumo_seed_{seed}_errors.log"),
    ]
    trace, connection = [], None
    log_path = raw / f"sumo_seed_{seed}.log"
    (raw / f"sumo_seed_{seed}_command.json").write_text(
        json.dumps(command, indent=2), encoding="utf-8"
    )
    with log_path.open("w", encoding="utf-8") as log:
        try:
            traci.start(command, label=label, numRetries=5, stdout=log)
            connection = traci.getConnection(label)
            version = connection.getVersion()
            steps = int(round(config["duration_s"] / config["step_s"]))
            for frame in range(steps):
                connection.simulationStep()
                ids = connection.vehicle.getIDList()
                if "ego" not in ids:
                    raise RuntimeError(f"ego missing at frame {frame}; inspect SUMO logs")
                position = connection.vehicle.getPosition("ego")
                neighbors = sum(
                    math.dist(position, connection.vehicle.getPosition(v)) < 50
                    for v in ids
                    if v != "ego"
                )
                trace.append(
                    {
                        "frame": frame,
                        "time_s": connection.simulation.getTime(),
                        "x_m": position[0],
                        "y_m": position[1],
                        "speed_mps": connection.vehicle.getSpeed("ego"),
                        "acceleration_mps2": connection.vehicle.getAcceleration("ego"),
                        "angle_deg": connection.vehicle.getAngle("ego"),
                        "neighbors_50m": neighbors,
                    }
                )
        finally:
            if connection is not None:
                connection.close()
    with (raw / f"traffic_truth_seed_{seed}.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(trace[0]))
        writer.writeheader()
        writer.writerows(trace)
    return trace, list(version)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "pilot"), default="smoke")
    parser.add_argument("--root", default=str(PROJECT_ROOT))
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/vehicle_sumo_stage3.json"))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    config_path = Path(args.config).resolve()
    if not root.is_dir() or not config_path.is_file():
        parser.error("Run this script from the CO-ACT-CPS project with its packaged config")
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    try:
        settings = validate_config(config, args.mode)
        module_names = (
            "coact_cps.vehicle.policy",
            "coact_cps.vehicle.explanation",
            "coact_cps.security.operator_surface",
            "coact_cps.security.verifier",
            "coact_cps.genai.grounded_generator",
            "coact_cps.types",
        )
        loaded_sources = {
            "src/" + name.replace(".", "/") + ".py": sys.modules[name].__file__
            for name in module_names
        }
        check_source_root(root, loaded_sources)
        latest = (
            root / "artifacts/vehicle_zero_trust_stage3" / ("latest_" + args.mode + "_metrics.json")
        )
        latest_destination(latest, root)
    except (ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    versions, binaries = check_versions()
    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8]
    raw = root / "data/raw/vehicle_sumo_stage3" / run_id
    result = root / "artifacts/vehicle_zero_trust_stage3" / run_id
    for path in (raw, result):
        if not path.resolve().is_relative_to(root):
            parser.error("Output paths must remain inside the project root")
        path.mkdir(parents=True, exist_ok=False)
    network = create_network(raw, binaries["netgenerate"])
    rows, episode_rows, simulator_versions = [], [], []
    labels_stream = (raw / "labels.csv").open("w", newline="", encoding="utf-8")
    label_writer = csv.DictWriter(
        labels_stream,
        fieldnames=["seed", "episode_id", "frame", "time_s", "scenario", "active", "is_attack"],
    )
    label_writer.writeheader()
    observations_stream = (raw / "context_observations.jsonl").open("w", encoding="utf-8")
    try:
        for seed in settings["seeds"]:
            routes = raw / f"routes_seed_{seed}.rou.xml"
            create_routes(network, routes, seed)
            trace, simulator_version = collect_trace(
                network, routes, seed, settings, raw, binaries["sumo"]
            )
            simulator_versions.append(simulator_version)
            base = make_observations(trace, seed)
            start_s = settings["duration_s"] * config["window_fraction"][0]
            end_s = settings["duration_s"] * config["window_fraction"][1]
            for scenario_index, (scenario, is_attack) in enumerate(SCENARIOS):
                episode = f"seed-{seed}-episode-{scenario_index:02d}"
                harness = PolicyHarness("ego")
                episode_start = len(rows)
                for truth, observed in zip(trace, base, strict=True):
                    if truth["time_s"] < config["warmup_s"]:
                        continue
                    active = start_s <= truth["time_s"] < end_s
                    modified, transport = (
                        inject_scenario(
                            observed, scenario, (truth["time_s"] - start_s) / (end_s - start_s)
                        )
                        if active
                        else (observed, {})
                    )
                    label_writer.writerow(
                        dict(
                            seed=seed,
                            episode_id=episode,
                            frame=truth["frame"],
                            time_s=truth["time_s"],
                            scenario=scenario,
                            active=active,
                            is_attack=is_attack,
                        )
                    )
                    # Episode ID and perturbation instructions are experiment-control
                    # metadata. Only signals/packets enter the policy functions.
                    observations_stream.write(
                        json.dumps(
                            {
                                "seed": seed,
                                "episode_id": episode,
                                "observation": modified,
                                "transport_events": transport,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    outputs = harness.step(modified, transport, active, scenario)
                    for policy in POLICIES:
                        rows.append(
                            dict(
                                seed=seed,
                                episode_id=episode,
                                scenario=scenario,
                                frame=truth["frame"],
                                time_s=truth["time_s"],
                                active=active,
                                is_attack=is_attack,
                                policy=policy,
                                **outputs[policy],
                            )
                        )
                subset = rows[episode_start:]
                for policy in POLICIES:
                    policy_rows = [r for r in subset if r["policy"] == policy]
                    summary = summarize(policy_rows)[policy]
                    active_rows = [r for r in policy_rows if r["active"]]
                    denied = [r["time_s"] for r in active_rows if not r["any_grant"]]
                    episode_rows.append(
                        dict(
                            seed=seed,
                            scenario=scenario,
                            policy=policy,
                            active_requests=len(active_rows),
                            active_any_grant_fraction=sum(r["any_grant"] for r in active_rows)
                            / len(active_rows),
                            first_block_delay_s=min(denied) - start_s if denied else None,
                            benign_any_grant_fraction=summary["benign_any_grant_fraction"],
                        )
                    )
            print(f"Collected and evaluated development traffic seed {seed}", flush=True)
    finally:
        labels_stream.close()
        observations_stream.close()
    fields = sorted({key for row in rows for key in row})
    with (result / "request_comparisons.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (result / "episode_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(episode_rows[0]))
        writer.writeheader()
        writer.writerows(episode_rows)
    provenance = {
        "sumo_state": "native simulator position, speed, acceleration, heading and proximity queries; not real vehicle ground truth",
        "gnss": "position finite-difference proxy plus independent seeded 0.005 m Gaussian noise; first frame uses simulator speed; capped at 99 m/s",
        "imu": "SUMO longitudinal acceleration plus 0.03 m/s2 noise; three-frame heading-rate mean clipped to +/-9 rad/s plus noise",
        "wheel": "simulator speed with 0.2% bias and 0.02 m/s noise; acceleration and heading-rate proxies; no wheel encoder or slip model",
        "can": "controlled scalar anomaly proxy from simulator acceleration plus noise; no CAN packets or trained detector",
        "identity": "experiment subject/credential/attestation booleans; no native certificates or hardware attestation",
        "v2x": "controlled scalar inconsistency/uncertainty proxy and traffic-neighbor count; no native V2X protocol stack",
        "authentication": "ephemeral distinct HMAC source keys and capability key; post-ingestion experiment authentication only; keys not saved",
        "use_time_recheck": "second software-collector acquisition at virtual +1 ms using the same traffic frame; identity may change; no intervening SUMO step",
        "labels": "separate labels.csv; benchmark episode IDs/transport-event instructions are not policy/model features",
        "independence": "all observation families share a simulator state origin; independent noise/keys do not prove independent physical observability",
        "split": "all runs are development/pilot; no detector training, fitted thresholds or final held-out evaluation",
        "effect": "permissions and displays only; no policy output changes SUMO driving or actuator behavior",
    }
    (raw / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    source_paths = [
        Path(__file__),
        Path(__file__).with_name("vehicle_sumo_pilot_core.py"),
        config_path,
    ]
    source_paths += [Path(path) for path in loaded_sources.values()]
    metrics = {
        "policy_config": asdict(PolicyConfig()),
        "stage": "SUMO traffic-grounded development pilot",
        "mode": args.mode,
        "evidence_scope": "simulator-derived observation proxies and synthetic security contexts; no native vehicle-performance claim",
        "traffic_seeds": settings["seeds"],
        "distinct_traffic_traces": len(settings["seeds"]),
        "paired_scenarios_per_trace": len(SCENARIOS),
        "policies": list(POLICIES),
        "policy_request_rows": len(rows),
        "aggregate_policy_summary": summarize(rows),
        "provenance": provenance,
        "config": config,
        "simulator_versions": simulator_versions,
        "package_versions": versions,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "source_sha256": {str(path): sha256_file(path) for path in source_paths},
        "raw_data_directory": str(raw),
        "result_directory": str(result),
        "scenario_window_protection_claim": "none; all failures and benign restrictions retained",
        "licensing_sources": [
            "https://eclipse.dev/sumo/docs/Downloads.html",
            "https://eclipse.dev/sumo/docs/TraCI/Interfacing_TraCI_from_Python.html",
        ],
    }
    metrics["data_sha256"] = {
        str(path.relative_to(raw)): sha256_file(path) for path in raw.rglob("*") if path.is_file()
    }
    metrics_path = result / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_latest(latest, json.dumps(metrics, indent=2), root)
    print(
        json.dumps(
            {
                "mode": args.mode,
                "traffic_traces": len(settings["seeds"]),
                "paired_scenarios": len(SCENARIOS),
                "request_rows": len(rows),
                "metrics": str(latest),
                "full_policy": metrics["aggregate_policy_summary"]["full"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
