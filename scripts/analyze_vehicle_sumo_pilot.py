"""Verify, summarize and snapshot the Stage 3 development pilot; no policy changes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import random
import statistics
import tempfile
import uuid
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

SCENARIOS = (
    ("normal", False),
    ("benign_gnss_bias", False),
    ("benign_v2x_loss", False),
    ("gnss_spoof", True),
    ("slow_gnss_drift", True),
    ("can_anomaly", True),
    ("identity_revocation", True),
    ("stale_gnss", True),
    ("source_replay", True),
    ("coordinated_speed_spoof", True),
    ("display_downplay", True),
    ("post_issue_revocation", True),
)
POLICIES = (
    "identity_only",
    "can_only",
    "multisource_no_kinematics",
    "full_unbound_display",
    "full_no_use_recheck",
    "full",
)
SOURCES = (
    "scripts/run_vehicle_sumo_pilot.py",
    "scripts/vehicle_sumo_pilot_core.py",
    "configs/vehicle_sumo_stage3.json",
    "src/coact_cps/vehicle/policy.py",
    "src/coact_cps/vehicle/explanation.py",
    "src/coact_cps/security/operator_surface.py",
    "src/coact_cps/security/verifier.py",
    "src/coact_cps/genai/grounded_generator.py",
    "src/coact_cps/types.py",
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def require_digest(value):
    if (
        type(value) is not str
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError("Declared hashes must be lowercase hexadecimal SHA-256 digests")


def inside(root, value):
    path = Path(value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Path leaves project: {value}")
    return path


def number(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Nonfinite numeric evidence")
    return result


def boolean(value):
    if value not in ("True", "False"):
        raise ValueError(f"Invalid boolean: {value!r}")
    return value == "True"


def read_csv(data, kind):
    rows = list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"), newline="")))
    for row in rows:
        row["seed"] = int(row["seed"])
        if kind in ("requests", "labels", "truth"):
            row["frame"] = int(row["frame"])
            row["time_s"] = number(row["time_s"])
        if kind in ("requests", "labels"):
            for field in ("active", "is_attack"):
                row[field] = boolean(row[field])
        if kind == "requests":
            for field in ("any_grant", "full_grant", "display_mutated"):
                row[field] = boolean(row[field])
            row["display_accepted"] = (
                boolean(row["display_accepted"]) if row["display_accepted"] else None
            )
            for field in ("risk", "latency_ms"):
                row[field] = number(row[field])
            if row["latency_ms"] < 0 or not 0 <= row["risk"] <= 1:
                raise ValueError("Invalid risk or latency")
            if row["outcome"] not in {"ALLOW", "RESTRICT", "CHALLENGE", "DENY"}:
                raise ValueError("Invalid outcome")
            if row["full_grant"] and not row["any_grant"]:
                raise ValueError("Full grant cannot exceed any grant")
        if kind == "episodes":
            row["active_requests"] = int(row["active_requests"])
            for field in (
                "active_any_grant_fraction",
                "first_block_delay_s",
                "benign_any_grant_fraction",
            ):
                row[field] = number(row[field]) if row[field] else None
    return rows


def check_pairing(rows, labels, seeds):
    label_map = {}
    scenario_map = dict(SCENARIOS)
    seed_set = set(seeds)
    for row in labels:
        key = (row["seed"], row["scenario"], row["frame"])
        if key in label_map:
            raise ValueError("Duplicate scenario/frame label")
        if row["seed"] not in seed_set or row["scenario"] not in scenario_map:
            raise ValueError("Unexpected label seed/scenario")
        index = [s for s, _ in SCENARIOS].index(row["scenario"])
        if (
            row["is_attack"] != scenario_map[row["scenario"]]
            or row["episode_id"] != f"seed-{row['seed']}-episode-{index:02d}"
        ):
            raise ValueError("Scenario label or episode ID differs from protocol")
        label_map[key] = row
    coverage = {(r["seed"], r["scenario"]) for r in labels}
    if coverage != {(s, name) for s in seeds for name, _ in SCENARIOS}:
        raise ValueError("Incomplete seed/scenario coverage")
    seen = set()
    for row in rows:
        label_key = (row["seed"], row["scenario"], row["frame"])
        key = (*label_key, row["policy"])
        if key in seen or row["policy"] not in POLICIES or label_key not in label_map:
            raise ValueError("Duplicate or unpaired policy request")
        seen.add(key)
        label = label_map[label_key]
        for field in ("episode_id", "active", "is_attack", "time_s"):
            if row[field] != label[field]:
                raise ValueError(f"Policy row and label disagree: {field}")
    if len(seen) != len(labels) * len(POLICIES):
        raise ValueError("Missing policy/frame comparison")


def fraction(rows, field):
    return sum(bool(r[field]) for r in rows) / len(rows) if rows else None


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["policy"]].append(row)
    result = {}
    for policy, group in groups.items():
        attacks = [r for r in group if r["active"] and r["is_attack"]]
        benign = [r for r in group if not (r["active"] and r["is_attack"])]
        altered = [r for r in group if r["display_mutated"]]
        result[policy] = dict(
            requests=len(group),
            traffic_seed_clusters=len({r["seed"] for r in group}),
            attack_window_requests=len(attacks),
            benign_requests=len(benign),
            attack_window_any_grant_fraction=fraction(attacks, "any_grant"),
            attack_window_full_grant_fraction=fraction(attacks, "full_grant"),
            benign_any_grant_fraction=fraction(benign, "any_grant"),
            benign_full_grant_fraction=fraction(benign, "full_grant"),
            altered_display_cards=len(altered),
            altered_display_acceptance_fraction=fraction(altered, "display_accepted"),
            outcomes={
                outcome: sum(r["outcome"] == outcome for r in group)
                for outcome in ("ALLOW", "RESTRICT", "CHALLENGE", "DENY")
            },
            latency_median_ms=statistics.median(r["latency_ms"] for r in group),
        )
    return result


def equal(actual, expected):
    if isinstance(actual, dict):
        return isinstance(expected, dict) and all(
            k in expected and equal(v, expected[k]) for k, v in actual.items()
        )
    if type(actual) is float:
        return (
            type(expected) in (int, float)
            and math.isfinite(expected)
            and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10)
        )
    return actual == expected


def check_aggregate(rows, expected):
    if set(expected) != set(POLICIES) or not equal(aggregate(rows), expected):
        raise ValueError("Request CSV does not reproduce the stored aggregate metrics")


def group_rates(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["policy"], row["scenario"], row["seed"])].append(row)
    rates = defaultdict(dict)
    for (policy, scenario, seed), group in groups.items():
        active = [r for r in group if r["active"]]
        benign = [r for r in group if not (r["active"] and r["is_attack"])]
        changed = [r for r in group if r["display_mutated"]]
        for scope, population, field in (
            ("active_any_grant", active, "any_grant"),
            ("active_full_grant", active, "full_grant"),
            ("benign_any_grant", benign, "any_grant"),
            ("altered_display_acceptance", changed, "display_accepted"),
        ):
            value = fraction(population, field)
            if value is not None:
                rates[(policy, scenario, scope)][seed] = value
    return rates


def seed_interval(values, label, repeats=10000):
    values = list(values)
    if len(values) < 2 or any(not math.isfinite(v) for v in values):
        raise ValueError("Exploratory interval needs at least two finite traffic-seed values")
    rng = random.Random(int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "big"))
    n = len(values)
    samples = sorted(sum(values[rng.randrange(n)] for _ in values) / n for _ in range(repeats))

    def quantile(p):
        index = (len(samples) - 1) * p
        lo = math.floor(index)
        hi = math.ceil(index)
        return samples[lo] + (samples[hi] - samples[lo]) * (index - lo)

    return dict(
        mean=statistics.mean(values),
        percentile_95=[quantile(0.025), quantile(0.975)],
        traffic_seed_clusters=n,
        resamples=repeats,
        interpretation="exploratory seed-cluster percentile interval for this scripted development protocol",
    )


def paired_interval(baseline, full, label, repeats=10000):
    if set(baseline) != set(full):
        raise ValueError("Paired comparison needs identical traffic seeds")
    return seed_interval([baseline[s] - full[s] for s in sorted(full)], label, repeats)


def csv_bytes(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def verify_evidence(root, metrics_path):
    root = Path(root).resolve()
    metrics_path = inside(root, metrics_path)
    metric_bytes = metrics_path.read_bytes()
    metrics = json.loads(metric_bytes)
    if (
        metrics["mode"] != "pilot"
        or metrics["config"]["protocol_version"] != "coact-sumo-development-pilot-v1"
    ):
        raise ValueError("Only completed Stage 3 pilot-v1 runs are supported")
    seeds = metrics["traffic_seeds"]
    if (
        not isinstance(seeds, list)
        or len(seeds) < 2
        or any(type(s) is not int for s in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("Need at least two distinct integer traffic seeds")
    if (
        metrics["distinct_traffic_traces"] != len(seeds)
        or metrics["paired_scenarios_per_trace"] != len(SCENARIOS)
        or metrics["policies"] != list(POLICIES)
        or metrics["config"]["pilot"]["seeds"] != seeds
    ):
        raise ValueError("Pilot design metadata does not match protocol")
    raw = inside(root, metrics["raw_data_directory"])
    result = inside(root, metrics["result_directory"])
    if (
        raw.parent != inside(root, root / "data/raw/vehicle_sumo_stage3")
        or result.parent != inside(root, root / "artifacts/vehicle_zero_trust_stage3")
        or raw.name != result.name
    ):
        raise ValueError("Unexpected pilot directories")
    if json.loads((result / "metrics.json").read_bytes()) != metrics:
        raise ValueError("Latest and run metrics differ")
    payloads = {"input/metrics.json": metric_bytes}
    expected_raw = {
        "context_observations.jsonl",
        "labels.csv",
        "traffic.net.xml",
        "netgenerate.log",
        "netgenerate_command.json",
        "provenance.json",
    }
    for seed in seeds:
        expected_raw.update(
            {
                f"routes_seed_{seed}.rou.xml",
                f"traffic_truth_seed_{seed}.csv",
                f"sumo_seed_{seed}.log",
                f"sumo_seed_{seed}_errors.log",
                f"sumo_seed_{seed}_command.json",
            }
        )
    if set(metrics["data_sha256"]) != expected_raw:
        raise ValueError("Raw data manifest is incomplete or contains unexpected filenames")

    def retain(path, archive_name, expected_hash=None):
        path = inside(root, path)
        if not path.is_file() or path.stat().st_size > 100 * 1024 * 1024:
            raise ValueError("Evidence file unavailable or larger than 100 MiB")
        data = path.read_bytes()
        if expected_hash and digest(data) != expected_hash:
            raise ValueError(f"Evidence hash mismatch: {archive_name}")
        payloads[archive_name] = data

    for relative, sha in metrics["data_sha256"].items():
        require_digest(sha)
        retain(raw / relative, "input/raw/" + relative, sha)
    declared = {}
    for path, sha in metrics["source_sha256"].items():
        require_digest(sha)
        relative = inside(root, path).relative_to(root).as_posix()
        if relative in declared:
            raise ValueError("Duplicate source declaration")
        declared[relative] = sha
    if set(declared) != set(SOURCES):
        raise ValueError("Source manifest differs from Stage 3 protocol")
    for relative, sha in declared.items():
        retain(root / relative, "input/source/" + relative, sha)
    if json.loads(payloads["input/source/configs/vehicle_sumo_stage3.json"]) != metrics["config"]:
        raise ValueError("Recorded and source config differ")
    for name in ["request_comparisons.csv", "episode_summary.csv"]:
        retain(result / name, "input/" + name)
    rows = read_csv(payloads["input/request_comparisons.csv"], "requests")
    labels = read_csv(payloads["input/raw/labels.csv"], "labels")
    check_pairing(rows, labels, seeds)
    if metrics["policy_request_rows"] != len(rows):
        raise ValueError("Request-row total differs")
    check_aggregate(rows, metrics["aggregate_policy_summary"])
    settings = metrics["config"]["pilot"]
    start = settings["duration_s"] * metrics["config"]["window_fraction"][0]
    end = settings["duration_s"] * metrics["config"]["window_fraction"][1]
    label_by_seed_scenario = defaultdict(list)
    for label in labels:
        label_by_seed_scenario[(label["seed"], label["scenario"])].append(label)
    for seed in seeds:
        truth = list(
            csv.DictReader(
                io.StringIO(
                    payloads[f"input/raw/traffic_truth_seed_{seed}.csv"].decode("utf-8-sig")
                )
            )
        )
        if len(truth) != round(settings["duration_s"] / settings["step_s"]):
            raise ValueError("Incomplete native traffic trace")
        expected = []
        for i, row in enumerate(truth):
            t = number(row["time_s"])
            if int(row["frame"]) != i or not math.isclose(
                t, (i + 1) * settings["step_s"], abs_tol=1e-8
            ):
                raise ValueError("Trace frames/times do not match simulation step")
            if t >= metrics["config"]["warmup_s"]:
                expected.append((i, t, start <= t < end))
        for scenario, _ in SCENARIOS:
            actual = sorted(
                (r["frame"], r["time_s"], r["active"])
                for r in label_by_seed_scenario[(seed, scenario)]
            )
            if actual != expected:
                raise ValueError("Labels do not cover the eligible native trace frames")
    episodes = read_csv(payloads["input/episode_summary.csv"], "episodes")
    groups = defaultdict(list)
    for row in rows:
        groups[(row["seed"], row["scenario"], row["policy"])].append(row)
    seen = set()
    for episode in episodes:
        key = (episode["seed"], episode["scenario"], episode["policy"])
        if key in seen or key not in groups:
            raise ValueError("Duplicate/unexpected episode summary")
        seen.add(key)
        group = groups[key]
        active = [r for r in group if r["active"]]
        denied = [r["time_s"] for r in active if not r["any_grant"]]
        benign = [r for r in group if not (r["active"] and r["is_attack"])]
        expected = dict(
            active_requests=len(active),
            active_any_grant_fraction=fraction(active, "any_grant"),
            first_block_delay_s=min(denied) - start if denied else None,
            benign_any_grant_fraction=fraction(benign, "any_grant"),
        )
        if not equal(expected, episode):
            raise ValueError("Episode summary does not match request CSV")
    if seen != set(groups):
        raise ValueError("Missing episode summary")
    return metrics, rows, payloads


def create_analysis(metrics, rows):
    rates = group_rates(rows)
    seed_rows = []
    intervals = []
    paired = []
    for (policy, scenario, measure), by_seed in sorted(rates.items()):
        for seed, value in sorted(by_seed.items()):
            seed_rows.append(
                dict(policy=policy, scenario=scenario, measure=measure, seed=seed, value=value)
            )
        summary = seed_interval(
            [by_seed[s] for s in sorted(by_seed)], f"{policy}/{scenario}/{measure}"
        )
        intervals.append(dict(policy=policy, scenario=scenario, measure=measure, **summary))
        if policy != "full" and (full := rates.get(("full", scenario, measure))):
            delta = paired_interval(by_seed, full, f"delta/{policy}/{scenario}/{measure}")
            paired.append(
                dict(
                    baseline=policy,
                    reference="full",
                    scenario=scenario,
                    measure=measure,
                    difference="baseline minus full",
                    **delta,
                )
            )
    summary = dict(
        status="verified_development_pilot_evidence",
        traffic_seeds=metrics["traffic_seeds"],
        policy_request_rows=len(rows),
        source_platform=metrics["platform"],
        scope=metrics["evidence_scope"],
        aggregate=aggregate(rows),
        scenario_intervals=intervals,
        paired_differences=paired,
        uncertainty="10,000 deterministic whole-traffic-seed bootstrap resamples, equal seed weights; exploratory 95% percentile intervals; six seeds do not establish population-wide coverage",
        integrity="source/raw hashes checked against original manifest; labels, pairing, trace coverage, episode summaries and aggregates checked; result CSV hashes are retrospective",
        limitations=[
            "simulator proxies and synthetic security contexts",
            "development seeds and fixed scripted attacks; no held-out confirmation",
            "coordinated-speed spoof and benign V2X-loss outcomes retained",
            "constant seed results yield degenerate intervals, not universal certainty",
            "no-use-recheck control is a counterfactual, with shared full-pipeline latency",
            "display acceptance is not actuator security or human-subject evidence",
        ],
    )
    lines = [
        "CO-ACT-CPS vehicle pilot evidence review",
        f"Platform: {metrics['platform']}",
        f"Traffic seeds: {metrics['traffic_seeds']}; request rows: {len(rows)}",
        "Integrity checks passed for the stored manifest and tabular consistency.",
        "This is a development simulator study; integrity checks do not independently attest execution.",
        "",
        "FULL POLICY: ACTIVE-WINDOW ANY-GRANT FRACTION BY SCENARIO",
        "Mean across traffic seeds; exploratory 95% seed-bootstrap interval.",
    ]
    for item in intervals:
        if item["policy"] == "full" and item["measure"] == "active_any_grant":
            low, high = item["percentile_95"]
            lines.append(f"{item['scenario']}: {item['mean']:.6f} [{low:.6f}, {high:.6f}]")
    lines.extend(
        [
            "",
            "INTERPRETATION",
            summary["uncertainty"],
            "A grant during a scripted attack window is a permission outcome, not IDS detection accuracy.",
            "The coordinated-speed spoof blind spot and benign-source-loss denial must remain visible.",
            "The frozen ZIP is an evidence snapshot, not a complete project or trusted immutable timestamp.",
            "Native vehicle data and a frozen broader evaluation are still needed before strong vehicle claims.",
        ]
    )
    return summary, seed_rows, "\n".join(lines) + "\n"


def latest_copy(root, path, data):
    if path.is_symlink():
        raise ValueError("Latest destination cannot be a symlink")
    inside(root, path)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(data)
    try:
        if path.is_symlink():
            raise ValueError("Latest destination cannot be a symlink")
        inside(root, path)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--metrics")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    metrics_path = (
        Path(args.metrics)
        if args.metrics
        else root / "artifacts/vehicle_zero_trust_stage3/latest_pilot_metrics.json"
    )
    try:
        metrics, rows, payloads = verify_evidence(root, metrics_path)
        print("Manifest, trace coverage, labels and paired results verified.", flush=True)
        summary, seed_rows, report = create_analysis(metrics, rows)
        parent = inside(root, root / "artifacts/vehicle_zero_trust_pilot_analysis")
        for name in ["latest_baseline_review.zip", "latest_summary.json"]:
            path = parent / name
            if path.is_symlink():
                raise ValueError("Latest destination cannot be a symlink")
            inside(root, path)
        out = parent / (
            datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8]
        )
        out.mkdir(parents=True, exist_ok=False)
        payloads["analysis/analyze_vehicle_sumo_pilot.py"] = Path(__file__).read_bytes()
        summary["snapshot_input_sha256"] = {name: digest(data) for name, data in payloads.items()}
        payloads["analysis/summary.json"] = json.dumps(summary, indent=2, allow_nan=False).encode()
        payloads["analysis/seed_scenario_rates.csv"] = csv_bytes(seed_rows)
        payloads["analysis/review.txt"] = report.encode()
        for name in ["summary.json", "seed_scenario_rates.csv", "review.txt"]:
            (out / name).write_bytes(payloads["analysis/" + name])
        archive = out / "baseline_review.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for name, data in sorted(payloads.items()):
                z.writestr(name, data)
        latest_copy(root, parent / "latest_baseline_review.zip", archive.read_bytes())
        latest_copy(root, parent / "latest_summary.json", payloads["analysis/summary.json"])
        print(report)
        print(f"Upload this file: {parent / 'latest_baseline_review.zip'}")
    except (
        ValueError,
        KeyError,
        TypeError,
        OSError,
        UnicodeError,
        csv.Error,
        OverflowError,
    ) as exc:
        parser.exit(1, f"Evidence analysis stopped: {exc}\nNo original pilot data was changed.\n")


if __name__ == "__main__":
    main()
