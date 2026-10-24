"""Small fully paired, consistent file set for evidence-validation regressions."""

import csv
import hashlib
import io
import json


def csv_data(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def fixture(root, module):
    seeds = [11, 23]
    config = {
        "protocol_version": "coact-sumo-development-pilot-v1",
        "pilot": {"seeds": seeds, "duration_s": 4.0, "step_s": 0.2},
        "warmup_s": 0.6,
        "window_fraction": [0.35, 0.65],
    }
    raw = root / "data/raw/vehicle_sumo_stage3/run_example"
    result = root / "artifacts/vehicle_zero_trust_stage3/run_example"
    raw.mkdir(parents=True)
    result.mkdir(parents=True)
    sources = {}
    for relative in module.SOURCES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        data = (
            json.dumps(config).encode()
            if relative.startswith("configs/")
            else b"# synthetic fixture source"
        )
        path.write_bytes(data)
        sources[str(path)] = hashlib.sha256(data).hexdigest()
    labels = []
    rows = []
    episodes = []
    rawfiles = {}
    for seed in seeds:
        truth = [dict(frame=i, time_s=round((i + 1) * 0.2, 8)) for i in range(20)]
        rawfiles[f"traffic_truth_seed_{seed}.csv"] = csv_data(truth)
        for name in [
            f"routes_seed_{seed}.rou.xml",
            f"sumo_seed_{seed}.log",
            f"sumo_seed_{seed}_errors.log",
            f"sumo_seed_{seed}_command.json",
        ]:
            rawfiles[name] = b"fixture"
        for index, (scenario, attack) in enumerate(module.SCENARIOS):
            eligible = [t for t in truth if t["time_s"] >= 0.6]
            for item in eligible:
                label = dict(
                    seed=seed,
                    scenario=scenario,
                    frame=item["frame"],
                    episode_id=f"seed-{seed}-episode-{index:02d}",
                    time_s=item["time_s"],
                    active=1.4 <= item["time_s"] < 2.6,
                    is_attack=attack,
                )
                labels.append(label)
                for policy in module.POLICIES:
                    rows.append(
                        dict(
                            label,
                            policy=policy,
                            any_grant=True,
                            full_grant=True,
                            outcome="ALLOW",
                            mode="stream",
                            risk=0.1,
                            latency_ms=0.5,
                            display_mutated=False,
                            display_accepted="",
                        )
                    )
            for policy in module.POLICIES:
                episodes.append(
                    dict(
                        seed=seed,
                        scenario=scenario,
                        policy=policy,
                        active_requests=6,
                        active_any_grant_fraction=1.0,
                        first_block_delay_s="",
                        benign_any_grant_fraction=1.0,
                    )
                )
    rawfiles["labels.csv"] = csv_data(labels)
    for name in [
        "context_observations.jsonl",
        "traffic.net.xml",
        "netgenerate.log",
        "netgenerate_command.json",
        "provenance.json",
    ]:
        rawfiles[name] = b"fixture"
    for name, data in rawfiles.items():
        (raw / name).write_bytes(data)
    (result / "request_comparisons.csv").write_bytes(csv_data(rows))
    (result / "episode_summary.csv").write_bytes(csv_data(episodes))
    metrics = dict(
        mode="pilot",
        config=config,
        traffic_seeds=seeds,
        distinct_traffic_traces=2,
        paired_scenarios_per_trace=len(module.SCENARIOS),
        policies=list(module.POLICIES),
        raw_data_directory=str(raw),
        result_directory=str(result),
        source_sha256=sources,
        data_sha256={name: hashlib.sha256(data).hexdigest() for name, data in rawfiles.items()},
        policy_request_rows=len(rows),
        aggregate_policy_summary=module.aggregate(rows),
    )
    path = result / "metrics.json"
    path.write_text(json.dumps(metrics))
    return path, raw, result
