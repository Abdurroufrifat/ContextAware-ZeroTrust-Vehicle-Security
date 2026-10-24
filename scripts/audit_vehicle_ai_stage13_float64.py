"""Reconstruct calibration/metrics from saved evidence, without model inference."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def node_count(method):
    return 1 if method in {"isolation_forest", "lof"} else 3


def verify_calibration_shape(cal, nodes):
    assert np.asarray(cal["median"]).shape == np.asarray(cal["scale"]).shape == (nodes,)
    assert (
        np.isfinite(cal["median"]).all()
        and np.isfinite(cal["scale"]).all()
        and np.all(np.asarray(cal["scale"]) >= 0.01)
    )


def aggregate(nodes, cal, method):
    x = (np.asarray(nodes, dtype=float) - np.asarray(cal["median"])) / np.asarray(cal["scale"])
    assert np.isfinite(x).all() and np.all(np.asarray(cal["scale"]) >= 0.01)
    if method == "gdn_adapted":
        x = np.stack([x[max(0, i - 3) : i + 1].mean(0) for i in range(len(x))])
    return x.mean(-1) if method == "catch_adapted" else x.max(-1)


def verify_score_block(block, spec, method, base):
    assert np.isfinite(block["nodes"]).all() and np.isfinite(block["score"]).all()
    pos = block["positions"]
    assert pos.ndim == 1 and pos.dtype.kind in "iu" and len(np.unique(pos)) == len(pos)
    assert np.all(np.diff(pos) > 0) and np.all((pos >= 0) & (pos < len(base)))
    nodes = node_count(method)
    assert block["nodes"].shape == (len(pos), nodes)
    verify_calibration_shape(spec["node_calibration"], nodes)
    expected = aggregate(block["nodes"], spec["node_calibration"], method)
    assert expected.shape == block["score"].shape == (len(pos),) and np.allclose(
        expected, block["score"], rtol=1e-10, atol=1e-12
    )
    assert block["alarm"].dtype.kind == "b" and np.array_equal(
        block["alarm"], block["score"] > spec["threshold"]
    )
    grant = np.zeros(len(base), dtype=bool)
    grant[pos] = base[pos] & ~block["alarm"]
    assert (
        block["grant"].dtype.kind == "b"
        and np.array_equal(block["grant"], grant)
        and not np.any(block["grant"] & ~base)
    )


def background_metrics(labels, positions, alarms, grants, periods, case):
    after = (np.asarray(periods) == "after") & ~labels
    if case == "normal":
        after[:] = False
    scored = after[positions]
    n = int(scored.sum())
    requests = int(after.sum())
    hits = int(alarms[scored].sum())
    allowed = int(grants[after].sum())
    return dict(
        post_withdrawal_windows=n,
        post_withdrawal_alarms=hits,
        post_withdrawal_alarm_fraction=hits / n if n else None,
        post_withdrawal_requests=requests,
        post_withdrawal_grants=allowed,
        post_withdrawal_grant_fraction=allowed / requests if requests else None,
    )


def route_intervals(rows, seed=10152026, resamples=10000):
    rng = np.random.default_rng(seed)
    out = []
    for case in sorted({r["case"] for r in rows}):
        for ref in sorted({r["method"] for r in rows}):
            if ref == "coact_physics":
                continue
            for metric in ["normal_grant_fraction", "active_grant_fraction", "active_tpr"]:
                values = []
                for trip in sorted({r["trip"] for r in rows if r["case"] == case}):
                    a = [
                        r[metric]
                        for r in rows
                        if r["trip"] == trip
                        and r["case"] == case
                        and r["method"] == "coact_physics"
                        and r.get(metric) is not None
                    ]
                    b = [
                        r[metric]
                        for r in rows
                        if r["trip"] == trip
                        and r["case"] == case
                        and r["method"] == ref
                        and r.get(metric) is not None
                    ]
                    if a and b:
                        values.append(float(np.mean(a) - np.mean(b)))
                if not values:
                    continue
                v = np.array(values)
                n = len(v)
                if n >= 2:
                    low, high = np.quantile(
                        v[rng.integers(0, n, (resamples, n))].mean(1), [0.025, 0.975]
                    )
                else:
                    low = high = None
                out.append(
                    dict(
                        case=case,
                        reference=ref,
                        metric=metric,
                        route_clusters=n,
                        candidate_minus_reference=float(v.mean()),
                        lower=None if low is None else float(low),
                        upper=None if high is None else float(high),
                        interpretation=(
                            "exploratory whole-route bootstrap; fit seeds averaged within routes"
                        )
                        if n >= 2
                        else "one route: no confidence interval",
                    )
                )
    return out


def verify_fit_audit(root):
    from run_vehicle_ai_stage13 import array_hash, load_trip, sha, verify_fit
    from vehicle_ai_data13 import Scaling, features, windows

    from coact_cps.vehicle.native_detectors import ModelConfig

    root = Path(root)
    f, lock = verify_fit(root)
    cfg = f["config"]
    old = json.loads((root / cfg["reference_contract"]).read_bytes())
    m = ModelConfig(**old["model"])
    train = [t for t in f["trips"] if t["phase"] == "train"]
    st = next(t for t in f["trips"] if t["phase"] == "scale")
    ct = next(t for t in f["trips"] if t["phase"] == "calibrate")
    frames = [load_trip(root, t) for t in train]
    scaling = Scaling.fit(np.concatenate([features(fs, m) for fs in frames]))
    assert scaling.to_dict() == lock["scaler"]
    assert (
        lock["train_routes"] == [t["id"] for t in train]
        and lock["scale_route"] == st["id"]
        and lock["calibrate_route"] == ct["id"]
    )
    x = np.concatenate([scaling.transform(windows(fs, cfg["window"], m).x) for fs in frames])
    sx = scaling.transform(windows(load_trip(root, st), cfg["window"], m).x)
    cx = scaling.transform(windows(load_trip(root, ct), cfg["window"], m).x)
    for key, a in [
        ("train_windows_sha256", x),
        ("scale_windows_sha256", sx),
        ("calibrate_windows_sha256", cx),
    ]:
        assert lock[key] == array_hash(a)
    assert {(s["method"], s["seed"]) for s in lock["models"]} == {
        (m, s) for m in cfg["methods"] for s in cfg["seeds"]
    } and len(lock["models"]) == len(cfg["methods"]) * len(cfg["seeds"])
    for spec in lock["models"]:
        data = np.load(root / spec["clean_scores"], allow_pickle=False)
        raw = np.asarray(data["scale_nodes"], dtype=float)
        nodes = node_count(spec["method"])
        assert raw.shape == (len(sx), nodes) and np.isfinite(raw).all()
        assert (
            data["calibrate_nodes"].shape == (len(cx), nodes)
            and np.isfinite(data["calibrate_nodes"]).all()
        )
        assert data["calibrate_scores"].shape == (len(cx),)
        verify_calibration_shape(spec["node_calibration"], nodes)
        cal = dict(
            median=np.median(raw, 0).tolist(),
            scale=np.maximum(
                np.quantile(raw, 0.75, axis=0) - np.quantile(raw, 0.25, axis=0), 0.01
            ).tolist(),
        )
        assert spec["node_calibration"] == cal
        scores = aggregate(data["calibrate_nodes"], cal, spec["method"])
        assert (
            np.allclose(scores, data["calibrate_scores"], rtol=1e-10, atol=1e-12)
            and abs(spec["threshold"] - float(np.quantile(scores, 0.95))) < 1e-10
        )
        if spec["checkpoint"]:
            assert sha((root / spec["checkpoint"]).read_bytes()) == spec["checkpoint_sha256"]
            if spec["method"] not in {"isolation_forest", "lof"}:
                assert (
                    len(spec["training_loss"]) == cfg["epochs"]
                    and np.isfinite(spec["training_loss"]).all()
                )
    return f, lock


def audit(root, phase):
    from run_vehicle_ai_stage13 import OUT, sha, write_json
    from vehicle_ai_data13 import features, load_trip, windows
    from vehicle_recovery_scenarios10b import period, perturb

    from coact_cps.vehicle.native_detectors import ModelConfig

    root = Path(root).resolve()
    from vehicle_ai_audit_repair13 import REPAIR, check_repair, verify_recovered_copies

    repair = check_repair(root)
    f, lock = verify_fit_audit(root)
    cfg = f["config"]
    folder = root / OUT / phase
    for n, h in json.loads((folder / "file_hashes.json").read_bytes()).items():
        assert sha((folder / n).read_bytes()) == h, n
    common = dict(
        status="PASS",
        phase=phase,
        source_files=len(f["source_sha256"]),
        raw_files=len(f["data_sha256"]),
        protocol_sha256=sha((root / OUT / "protocol.json").read_bytes()),
        model_lock_sha256=sha((root / OUT / "fit/model_lock.json").read_bytes()),
        phase_hashes_sha256=sha((folder / "file_hashes.json").read_bytes()),
        model_inference_repeated=False,
        fitting_repeated=False,
        limits=(
            "hash/denominator/calibration audit does not prove training chronology "
            "or model output mathematics"
        ),
        candidate_status="EXPERIMENTAL_NOT_PROMOTED",
        audit_repair_sha256=sha((root / OUT / REPAIR).read_bytes()),
        audit_repair="float64 clean calibration; original frozen model/thresholds unchanged",
    )
    if phase == "fit":
        write_json(folder / "independent_audit.json", common)
        return common
    summary = json.loads((folder / "summary.json").read_bytes())
    assert (
        summary["phase"] == phase
        and summary["protocol_sha256"] == common["protocol_sha256"]
        and summary["model_lock_sha256"] == common["model_lock_sha256"]
    )
    if phase == "validation" and repair["interrupted_validation_sha256"]:
        common["recovery_copy_counts"] = verify_recovered_copies(
            folder, summary, repair, sha((root / OUT / REPAIR).read_bytes())
        )
    expected = {
        (t["id"], c, m, s)
        for t in f["trips"]
        if t["phase"] == phase
        for c in cfg["cases"]
        for m in cfg["methods"]
        for s in cfg["seeds"]
    }
    actual = [(r["trip"], r["case"], r["method"], r["seed"]) for r in summary["rows"]]
    assert set(actual) == expected and len(actual) == len(expected)
    old = json.loads((root / cfg["reference_contract"]).read_bytes())
    m = ModelConfig(**old["model"])
    contexts = {}
    periods = {}
    normal = {}
    blocks = {}
    metrics = []
    for t in f["trips"]:
        if t["phase"] != phase:
            continue
        original = load_trip(root, t)
        for case in cfg["cases"]:
            fs = perturb(original, case, m, old["session_config"])
            w = windows(fs, cfg["window"], m)
            data = np.load(
                folder / "contexts" / (t["id"] + "__" + case + ".npz"), allow_pickle=False
            )
            assert np.allclose(
                data["features"], features(fs, m), rtol=1e-10, atol=1e-12
            ) and np.array_equal(data["positions"], w.positions)
            assert np.array_equal(data["time_ms"], [v.time_ms for v in fs]) and np.array_equal(
                data["start_ms"], [v.interval_start_ms for v in fs]
            )
            labels = np.array(
                [
                    case != "normal"
                    and max(
                        abs(a.gnss_speed - b.gnss_speed),
                        abs(a.wheel_speed - b.wheel_speed),
                        abs(a.imu_accel - b.imu_accel),
                    )
                    > 1e-9
                    for a, b in zip(fs, original, strict=True)
                ]
            )
            assert (
                np.array_equal(data["labels"], labels)
                and data["base_grant"].shape == labels.shape
                and data["base_grant"].dtype.kind == "b"
            )
            assert not np.any(data["base_grant"] & (data["issued"] != "ALLOW")) and np.all(
                data["quorum"][data["base_grant"]] >= 3
            )
            contexts[t["id"], case] = data
            periods[t["id"], case] = np.array([period(v, fs, case) for v in fs])
    specs = {(s["method"], s["seed"]): s for s in lock["models"]}
    for r in summary["rows"]:
        assert (
            r["context"] == f"contexts/{r['trip']}__{r['case']}.npz"
            and r["block"] == f"blocks/{r['trip']}__{r['case']}__{r['method']}__{r['seed']}.npz"
        )
        b = np.load(folder / r["block"], allow_pickle=False)
        c = contexts[r["trip"], r["case"]]
        assert np.array_equal(b["positions"], c["positions"])
        verify_score_block(b, specs[r["method"], r["seed"]], r["method"], c["base_grant"])
        blocks[r["trip"], r["case"], r["method"], r["seed"]] = b
        if r["case"] == "normal":
            normal[r["trip"], r["method"], r["seed"]] = b
    for r in summary["rows"]:
        b = blocks[r["trip"], r["case"], r["method"], r["seed"]]
        c = contexts[r["trip"], r["case"]]
        n = normal[r["trip"], r["method"], r["seed"]]
        nc = contexts[r["trip"], "normal"]
        pos = b["positions"]
        y = c["labels"][pos]
        alarm = b["alarm"]
        g = b["grant"]
        active = c["labels"]
        tp = int(np.sum(alarm & y))
        fp = int(np.sum(alarm & ~y))
        fn = int(np.sum(~alarm & y))
        den = 2 * tp + fp + fn
        entry = dict(
            trip=r["trip"],
            case=r["case"],
            method=r["method"],
            seed=r["seed"],
            scoreable_windows=len(pos),
            full_route_epochs=len(g),
            active_windows=int(y.sum()),
            normal_fpr=float(n["alarm"].mean()),
            normal_grant_fraction=float(n["grant"].mean()),
            normal_base_grant_fraction=float(nc["base_grant"].mean()),
            active_tpr=float(alarm[y].mean()) if y.any() else None,
            active_grants=int(g[active].sum()),
            active_requests=int(active.sum()),
            active_base_grants=int(c["base_grant"][active].sum()),
            active_grant_fraction=float(g[active].mean()) if active.any() else None,
            endpoint_f1=2 * tp / den if den else 0.0,
            point_adjustment=False,
        )
        entry.update(
            background_metrics(active, pos, alarm, g, periods[r["trip"], r["case"]], r["case"])
        )
        if y.any():
            yy = np.r_[np.zeros(len(n["score"]), dtype=int), np.ones(int(y.sum()), dtype=int)]
            ss = np.r_[n["score"], b["score"][y]]
            entry["paired_auc"] = float(roc_auc_score(yy, ss))
            entry["paired_ap"] = float(average_precision_score(yy, ss))
        else:
            entry["paired_auc"] = entry["paired_ap"] = None
        metrics.append(entry)
    intervals = route_intervals(metrics, cfg["bootstrap_seed"], cfg["bootstrap_resamples"])
    for name, obj in [("metrics.json", metrics), ("route_intervals.json", intervals)]:
        path = folder / name
        if path.exists():
            assert json.loads(path.read_bytes()) == obj, "existing_audit_metric_mismatch"
        else:
            write_json(path, obj)
    common.update(
        score_blocks=len(summary["rows"]),
        common_streams=len(contexts),
        metric_rows=len(metrics),
        split_and_window_causality_checked=True,
        thresholds_recomputed_from_clean_calibration=True,
        all_AI_grants_subset_of_saved_base_grants=True,
        model_seed_averaged_before_route_bootstrap=True,
        baseline_limit=(
            "saved base grants checked for issued outcome/quorum; "
            "no independent reexecution or safety certification"
        ),
        summary_sha256=sha((folder / "summary.json").read_bytes()),
        metrics_sha256=sha((folder / "metrics.json").read_bytes()),
        intervals_sha256=sha((folder / "route_intervals.json").read_bytes()),
    )
    write_json(folder / "independent_audit.json", common)
    return common


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--phase", choices=["fit", "validation", "test"], required=True)
    a = p.parse_args()
    print(json.dumps(audit(a.root, a.phase), indent=2))
