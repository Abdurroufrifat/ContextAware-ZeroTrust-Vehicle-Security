"""Saved-output availability diagnostics; never train, infer or select thresholds."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RATE_FIELDS = (
    "alarm_rate_scored",
    "grant_rate_all_epochs",
    "base_grant_rate_all_epochs",
    "AI_added_denied_rate_all_epochs",
    "unscored_base_granted_rate_all_epochs",
)


def score_channels(nodes, calibration, method):
    x = np.asarray(nodes, dtype=float)
    median = np.asarray(calibration["median"], dtype=float)
    scale = np.asarray(calibration["scale"], dtype=float)
    width = 1 if method in {"isolation_forest", "lof"} else 3
    if (
        x.ndim != 2
        or x.shape[1] != width
        or median.shape != (width,)
        or scale.shape != (width,)
        or not np.isfinite(x).all()
        or not np.isfinite(median).all()
        or not np.isfinite(scale).all()
        or np.any(scale <= 0)
    ):
        raise ValueError("invalid_node_calibration")
    x = (x - median) / scale
    if method == "gdn_adapted":
        total = np.vstack([np.zeros((1, 3)), np.cumsum(x, axis=0)])
        x = np.stack([(total[i + 1] - total[max(0, i - 3)]) / min(4, i + 1) for i in range(len(x))])
    return x, x.mean(-1) if method == "catch_adapted" else x.max(-1)


def decompose(context, block, spec):
    base, labels, pos = context["base_grant"], context["labels"], block["positions"]
    grant, alarm = block["grant"], block["alarm"]
    n, m = len(base), len(pos)
    if (
        n == 0
        or m == 0
        or pos.dtype.kind not in "iu"
        or pos.ndim != 1
        or pos[0] < 0
        or pos[-1] >= n
        or np.any(np.diff(pos) <= 0)
        or not np.array_equal(pos, context["positions"])
        or any(a.dtype != np.dtype(bool) for a in (base, labels, grant, alarm))
        or base.shape != (n,)
        or labels.shape != (n,)
        or grant.shape != (n,)
        or alarm.shape != (m,)
        or labels.any()
    ):
        raise ValueError("invalid_normal_context_or_alignment")
    channels, score = score_channels(block["nodes"], spec["node_calibration"], spec["method"])
    threshold = spec["threshold"]
    if (
        not np.isfinite(threshold)
        or score.shape != (m,)
        or not np.array_equal(score, block["score"])
        or not np.array_equal(score > threshold, alarm)
    ):
        raise ValueError("saved_score_or_alarm_mismatch")
    expected = np.zeros(n, dtype=bool)
    expected[pos] = base[pos] & ~alarm
    if not np.array_equal(grant, expected):
        raise ValueError("saved_grant_mismatch")
    scored = np.zeros(n, dtype=bool)
    scored[pos] = True
    warm = int((base & ~scored).sum())
    added = int((base[pos] & alarm).sum())
    overlap = int((~base[pos] & alarm).sum())
    if int(base.sum() - grant.sum()) != warm + added:
        raise ValueError("availability_accounting_failed")
    patterns = Counter()
    if spec["method"] in {"coact_physics", "coact_no_graph", "coact_no_physics"}:
        for flags in channels[alarm] > threshold:
            patterns[
                "+".join(
                    name for name, flag in zip(("gnss", "wheel", "imu"), flags, strict=True) if flag
                )
            ] += 1
        if sum(patterns.values()) != int(alarm.sum()) or "" in patterns:
            raise ValueError("channel_accounting_failed")
    return dict(
        epochs=n,
        scored_windows=m,
        base_denied_epochs=int((~base).sum()),
        base_granted_epochs=int(base.sum()),
        unscored_base_granted_epochs=warm,
        AI_added_denied_epochs=added,
        AI_alarm_on_base_denied_epochs=overlap,
        composed_granted_epochs=int(grant.sum()),
        alarm_windows=int(alarm.sum()),
        at_threshold_windows=int((score == threshold).sum()),
        threshold=float(threshold),
        score_q95=float(np.quantile(score, 0.95)),
        alarm_rate_scored=float(alarm.mean()),
        grant_rate_all_epochs=float(grant.mean()),
        base_grant_rate_all_epochs=float(base.mean()),
        AI_added_denied_rate_all_epochs=added / n,
        unscored_base_granted_rate_all_epochs=warm / n,
        exceedance_patterns=dict(sorted(patterns.items())),
    )


def summarize(rows):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["phase"] == "test":
            grouped[row["method"]][row["trip"]].append(row)
    result = {}
    for method, routes in sorted(grouped.items()):
        means = [
            {key: float(np.mean([r[key] for r in values])) for key in RATE_FIELDS}
            for values in routes.values()
        ]
        result[method] = {key: float(np.mean([r[key] for r in means])) for key in RATE_FIELDS}
        result[method]["route_count"] = len(routes)
    return result


class SavedStudy:
    def __init__(self, path):
        self.path = Path(path)
        self.archive = None
        if self.path.is_file():
            self.archive = zipfile.ZipFile(self.path)
            names = self.archive.namelist()
            if len(names) != len(set(names)):
                raise ValueError("duplicate_archive_members")
            roots = [n[: -len("protocol.json")] for n in names if n.endswith("/protocol.json")]
            if "protocol.json" in names:
                roots.append("")
            if len(roots) != 1:
                raise ValueError("single_study_root_required")
            self.prefix = roots[0]

    def read(self, relative):
        p = PurePosixPath(relative)
        if p.is_absolute() or ".." in p.parts or "\\" in relative:
            raise ValueError("unsafe_relative_path")
        return (
            self.archive.read(self.prefix + relative)
            if self.archive
            else (self.path / relative).read_bytes()
        )

    def json(self, relative):
        return json.loads(self.read(relative))

    def arrays(self, relative):
        with np.load(io.BytesIO(self.read(relative)), allow_pickle=False) as data:
            return {key: data[key] for key in data.files}


def audit_saved(path, manifest):
    source = SavedStudy(path)
    try:
        for relative, expected in manifest["input_files_sha256"].items():
            if hashlib.sha256(source.read(relative)).hexdigest() != expected:
                raise ValueError(f"input_binding_mismatch:{relative}")
        verified_files = 0
        for phase in ("fit", "validation", "test"):
            if source.json(phase + "/independent_audit.json")["status"] != "PASS":
                raise ValueError("prior_audit_not_passed")
            for relative, expected in source.json(phase + "/file_hashes.json").items():
                if hashlib.sha256(source.read(phase + "/" + relative)).hexdigest() != expected:
                    raise ValueError(f"phase_file_hash_mismatch:{phase}:{relative}")
                verified_files += 1
        lock = source.json("fit/model_lock.json")
        specs = {(v["method"], v["seed"]): v for v in lock["models"]}
        if len(specs) != 24:
            raise ValueError("24_frozen_method_seed_specs_required")
        calibration = []
        for (method, seed), spec in specs.items():
            nodes = source.arrays(f"fit/seed_{seed}/{method}_clean_scores.npz")
            _, scores = score_channels(nodes["calibrate_nodes"], spec["node_calibration"], method)
            if (
                not np.array_equal(scores, nodes["calibrate_scores"])
                or float(np.quantile(scores, 0.95)) != spec["threshold"]
            ):
                raise ValueError("frozen_calibration_mismatch")
            calibration.append(
                dict(
                    method=method,
                    seed=seed,
                    alarm_rate_scored=float((scores > spec["threshold"]).mean()),
                    scored_windows=len(scores),
                )
            )
        rows = []
        for phase, route_count in (("validation", 1), ("test", 5)):
            summary = source.json(phase + "/summary.json")
            selected = [r for r in summary["rows"] if r["case"] == "normal"]
            unique = {(r["trip"], r["method"], r["seed"]) for r in selected}
            if len(selected) != 24 * route_count or len(unique) != len(selected):
                raise ValueError("incomplete_normal_blocks")
            for row in selected:
                out = decompose(
                    source.arrays(phase + "/" + row["context"]),
                    source.arrays(phase + "/" + row["block"]),
                    specs[row["method"], row["seed"]],
                )
                rows.append(
                    dict(
                        phase=phase, trip=row["trip"], method=row["method"], seed=row["seed"], **out
                    )
                )
        return dict(
            status="PASS",
            study_kind="posthoc_saved_output_diagnostic",
            verified_phase_files=verified_files,
            normal_blocks=len(rows),
            calibration_blocks=len(calibration),
            methods=summarize(rows),
            calibration=calibration,
            rows=rows,
            fitting=False,
            model_inference=False,
            thresholds_changed=False,
            limits=[
                "Post hoc accounting is not causal evidence or new held-out validation.",
                "Score channels are model residual channels, not physical attack causes.",
                "Five routes from one vehicle; fit seeds are averaged within routes.",
                "Any proposed remedy needs a new protocol and untouched validation data.",
            ],
        )
    finally:
        if source.archive:
            source.archive.close()


def text_report(result):
    lines = [
        "CO-ACT-CPS STAGE 15 - SAVED-OUTPUT FALSE-ALARM DIAGNOSTIC",
        "",
        "No fitting, model inference, threshold change or policy change.",
        "Rates below average fit seeds within route, then average the five test routes.",
        "Alarm rate uses scored windows. Grant/loss rates use all normal epochs.",
        "",
        "method | alarm % | base grant % | composed grant % | AI added loss pp | unscored loss pp",
    ]
    for method, row in result["methods"].items():
        values = [row[k] * 100 for k in RATE_FIELDS]
        lines.append(
            f"{method} | {values[0]:.4f} | {values[2]:.4f} | {values[1]:.4f}"
            f" | {values[3]:.4f} | {values[4]:.4f}"
        )
    calibrate = [
        r["alarm_rate_scored"] for r in result["calibration"] if r["method"] == "coact_physics"
    ]
    lines += [
        "",
        f"Physics variant calibration-route exceedance: {np.mean(calibrate) * 100:.4f}%.",
        "",
        "Physics variant by route:",
        "route | alarm % | composed grant % | AI added loss pp | unscored loss pp",
    ]
    candidate = [
        r for r in result["rows"] if r["phase"] == "test" and r["method"] == "coact_physics"
    ]
    for trip in sorted({r["trip"] for r in candidate}):
        rows = [r for r in candidate if r["trip"] == trip]
        vals = [
            np.mean([r[k] for r in rows]) * 100
            for k in (
                "alarm_rate_scored",
                "grant_rate_all_epochs",
                "AI_added_denied_rate_all_epochs",
                "unscored_base_granted_rate_all_epochs",
            )
        ]
        lines.append(trip + " | " + " | ".join(f"{v:.4f}" for v in vals))
    lines += [
        "",
        "Exact accounting: base grants minus composed grants equals",
        "AI alarms blocking base grants plus base grants outside the scored positions.",
        "An AI alarm on an already-denied epoch adds no availability loss.",
        "The initial unscored period is the 32-epoch model window startup in these data.",
        "Do not remove the startup restriction without a separate security study.",
        "",
        "Interpretation: the fixed clean calibration-route threshold does not maintain",
        "the nominal 5% exceedance rate across every recorded route. This demonstrates",
        "a route-dependent operating-point problem; it does not establish its cause.",
        "Channel-pattern counts in diagnostic.json describe residual-score exceedances.",
        "They must not be presented as proof that a named sensor was attacked.",
        "",
        "Next remedy must be designed using development/calibration data and then",
        "tested on newly reserved routes. Existing test routes have now been inspected.",
        "No new operating point or superiority claim is selected from this diagnostic.",
        "",
    ]
    lines.extend(result["limits"])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "artifacts/vehicle_ai_stage13r1")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts/vehicle_false_alarm_stage15"
    )
    args = parser.parse_args()
    manifest = json.loads((ROOT / "VEHICLE_STAGE15_MANIFEST.json").read_text())
    for relative, expected in manifest["source_files_sha256"].items():
        if (
            hashlib.sha256((ROOT / relative).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            != expected
        ):
            raise ValueError(f"diagnostic_source_mismatch:{relative}")
    result = audit_saved(args.input, manifest)
    result["environment"] = dict(python=platform.python_version(), platform=platform.platform())
    result["manifest_sha256"] = hashlib.sha256(
        (ROOT / "VEHICLE_STAGE15_MANIFEST.json").read_bytes().replace(b"\r\n", b"\n")
    ).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "diagnostic.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    report = text_report(result)
    (args.output / "DIAGNOSTIC.txt").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
