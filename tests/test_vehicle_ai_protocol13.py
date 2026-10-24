import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest


def test_fit_roles_and_phase_overwrite_guard(tmp_path):
    from run_vehicle_ai_stage13 import begin_phase, fitting_trips

    trips = [
        dict(id=str(i), phase=p)
        for i, p in enumerate(["train", "scale", "calibrate", "validation", "test"])
    ]
    assert [t["phase"] for t in fitting_trips(dict(trips=trips))] == ["train", "scale", "calibrate"]
    begin_phase(tmp_path, "fit")
    with pytest.raises(FileExistsError):
        begin_phase(tmp_path, "fit")


def test_freeze_requires_complete_input(tmp_path):
    from run_vehicle_ai_stage13 import OUT, freeze

    with pytest.raises((FileNotFoundError, ValueError)):
        freeze(tmp_path)
    assert not (tmp_path / OUT / "protocol.json").exists()


def test_binary_hashes_exact_text_crlf_portable(tmp_path):
    from run_vehicle_ai_stage13 import sha, source_digest

    p = tmp_path / "x.py"
    q = tmp_path / "x.bin"
    p.write_bytes(b"a\r\nb")
    q.write_bytes(b"a\r\nb")
    assert source_digest(p) == sha(b"a\nb") and source_digest(q) == sha(b"a\r\nb")


def test_central_entry_identity_and_crc(tmp_path):
    import zlib

    from run_vehicle_ai_stage13 import verify_entry

    b = b"native"
    p = tmp_path / "value"
    p.write_bytes(b)
    e = dict(
        name="Chunk_8/route/5/processed_log/CAN/wheel_speed/value", size=len(b), crc32=zlib.crc32(b)
    )
    i = dict(
        path="processed_log/CAN/wheel_speed/value",
        source_zip_entry=e["name"],
        entry_crc32=e["crc32"],
        bytes=len(b),
    )
    verify_entry(p, i, e)
    p.write_bytes(b"bad")
    with pytest.raises(ValueError):
        verify_entry(p, i, e)
    p.write_bytes(b)
    i["source_zip_entry"] = "other route"
    with pytest.raises(ValueError):
        verify_entry(p, i, e)


def test_score_block_rejects_wrong_nodes_false_warmup_grants_and_thresholds():
    from audit_vehicle_ai_stage13 import verify_score_block

    b = dict(
        nodes=np.repeat([[1.0], [3.0], [2.0]], 3, axis=1),
        score=np.array([1.0, 3.0, 2.0]),
        alarm=np.array([False, True, False]),
        positions=np.array([2, 3, 4]),
        grant=np.array([False, False, True, False, True]),
    )
    s = dict(node_calibration={"median": [0.0] * 3, "scale": [1.0] * 3}, threshold=2.0)
    base = np.ones(5, dtype=bool)
    verify_score_block(b, s, "persistence", base)
    b["grant"][0] = True
    with pytest.raises(AssertionError):
        verify_score_block(b, s, "persistence", base)
    b["grant"][0] = False
    b["nodes"] = b["nodes"][:, :1]
    with pytest.raises(AssertionError):
        verify_score_block(b, s, "persistence", base)
    b["nodes"] = np.repeat(b["nodes"], 3, axis=1)
    s["threshold"] = 4.0
    with pytest.raises(AssertionError):
        verify_score_block(b, s, "persistence", base)


def test_postwithdrawal_carryover_counts_exclude_preattack_alarms():
    from audit_vehicle_ai_stage13 import background_metrics

    y = np.array([False, True, True, False, False, False])
    p = np.array(["before", "active", "active", "after", "after", "after"])
    a = np.array([True, True, True, True, True, False])
    g = ~a
    r = background_metrics(y, np.arange(6), a, g, p, "coherent_withdrawal")
    assert (
        r["post_withdrawal_alarms"] == 2
        and r["post_withdrawal_windows"] == 3
        and r["post_withdrawal_alarm_fraction"] == pytest.approx(2 / 3)
    )
    assert r["post_withdrawal_grants"] == 1 and r["post_withdrawal_requests"] == 3
    assert (
        background_metrics(y, np.arange(6), a, g, p, "normal")["post_withdrawal_alarm_fraction"]
        is None
    )


def test_seed_replicates_averaged_within_route():
    from audit_vehicle_ai_stage13 import route_intervals

    rows = [
        dict(
            trip=t,
            seed=s,
            method=m,
            case="normal",
            normal_grant_fraction=0.8 if m == "coact_physics" else 0.6,
        )
        for t in ["a", "b", "c", "d", "e"]
        for s in [7, 19, 43]
        for m in ["coact_physics", "reference"]
    ]
    r = route_intervals(rows, resamples=100)[0]
    assert r["route_clusters"] == 5 and r["candidate_minus_reference"] == pytest.approx(0.2)


def test_synthetic_full_pipeline_and_rehashed_truncated_calibration(tmp_path, monkeypatch):
    import run_vehicle_ai_stage13 as runner
    import vehicle_ai_data13 as data
    from audit_vehicle_ai_stage13 import aggregate, audit, verify_fit_audit
    from test_vehicle_window_recovery import WindowRecoveryTests

    import coact_cps.vehicle.ai_models13 as models

    project = Path(__file__).resolve().parents[1]
    cfg = json.loads((project / "configs/vehicle_ai_stage13.json").read_bytes())
    cfg.update(seeds=[7], epochs=1, cases=["normal", "joint_jump"], bootstrap_resamples=20)
    target = tmp_path / cfg["reference_contract"]
    target.parent.mkdir(parents=True)
    target.write_bytes((project / cfg["reference_contract"]).read_bytes())
    trips = [
        dict(id=f"synthetic{i}", phase=p)
        for i, p in enumerate(["train", "scale", "calibrate", "validation", "test"])
    ]
    frames = WindowRecoveryTests().frames()[:100]

    def loader(root, t):
        return [
            replace(
                f,
                trip_id=t["id"],
                gnss_speed=f.gnss_speed + 0.1 * np.sin(i / 9),
                wheel_speed=f.wheel_speed + 0.1 * np.sin(i / 9),
                imu_accel=f.imu_accel + 0.05 * np.cos(i / 9),
            )
            for i, f in enumerate(frames)
        ]

    monkeypatch.setattr(runner, "load_trip", loader)
    monkeypatch.setattr(data, "load_trip", loader)
    original = models.catch_modules
    monkeypatch.setattr(models, "catch_modules", lambda root: original(project))
    f = dict(
        schema="synthetic-only",
        config=cfg,
        trips=trips,
        source_sha256={},
        data_sha256={},
        environment=runner.environment(),
    )
    runner.write_json(tmp_path / runner.OUT / "protocol.json", f)
    runner.fit(tmp_path)
    assert audit(tmp_path, "fit")["status"] == "PASS"
    runner.evaluate_phase(tmp_path, "validation")
    assert audit(tmp_path, "validation")["status"] == "PASS"
    runner.evaluate_phase(tmp_path, "test")
    result = audit(tmp_path, "test")
    assert result["score_blocks"] == 16 and result["status"] == "PASS"
    assert audit(tmp_path, "test") == result
    with pytest.raises(FileExistsError):
        runner.evaluate_phase(tmp_path, "test")
    # Even a self-consistent, rehashed one-row/one-node calibration is rejected.
    path = tmp_path / runner.OUT / "fit/model_lock.json"
    lock = json.loads(path.read_bytes())
    s = next(s for s in lock["models"] if s["method"] == "persistence")
    cp = tmp_path / s["clean_scores"]
    with np.load(cp, allow_pickle=False) as clean:
        sn = clean["scale_nodes"][:1, :1]
        cn = clean["calibrate_nodes"][:1, :1]
    cal = models.fit_node_calibration(np.repeat(sn, 2, axis=0))
    scores = aggregate(cn, cal, "persistence")
    np.savez_compressed(cp, scale_nodes=sn, calibrate_nodes=cn, calibrate_scores=scores)
    s["node_calibration"] = cal
    s["threshold"] = float(scores[0])
    runner.write_json(path, lock)
    runner.hash_phase(path.parent)
    with pytest.raises(AssertionError):
        verify_fit_audit(tmp_path)


def test_validation_gate_rejects_rehashed_manifest(tmp_path):
    from run_vehicle_ai_stage13 import OUT, assert_test_ready, sha, write_json

    folder = tmp_path / OUT / "validation"
    folder.mkdir(parents=True)
    fit = tmp_path / OUT / "fit"
    fit.mkdir()
    write_json(tmp_path / OUT / "protocol.json", {})
    write_json(fit / "model_lock.json", {})
    for n in ["summary.json", "metrics.json", "route_intervals.json"]:
        write_json(folder / n, {})
    write_json(
        folder / "file_hashes.json", {"summary.json": sha((folder / "summary.json").read_bytes())}
    )
    bindings = {
        "summary_sha256": folder / "summary.json",
        "phase_hashes_sha256": folder / "file_hashes.json",
        "protocol_sha256": tmp_path / OUT / "protocol.json",
        "model_lock_sha256": fit / "model_lock.json",
        "metrics_sha256": folder / "metrics.json",
        "intervals_sha256": folder / "route_intervals.json",
    }
    write_json(
        folder / "independent_audit.json",
        dict(
            status="PASS",
            phase="validation",
            **{k: sha(p.read_bytes()) for k, p in bindings.items()},
        ),
    )
    assert_test_ready(tmp_path)
    write_json(folder / "file_hashes.json", {})
    with pytest.raises(ValueError):
        assert_test_ready(tmp_path)


def test_delivery_manifest_and_path_guards(tmp_path):
    from run_vehicle_ai_stage13 import sha, write_json
    from verify_vehicle_stage13_delivery import verify

    (tmp_path / "new.py").write_bytes(b"new\n")
    (tmp_path / "old.py").write_bytes(b"old\n")
    m = {"files": {"new.py": sha(b"new\n")}, "base_files": {"old.py": sha(b"old\n")}}
    write_json(tmp_path / "STAGE13_DELIVERY_MANIFEST.json", m)
    assert verify(tmp_path)["status"] == "PASS"
    (tmp_path / "old.py").write_bytes(b"changed")
    with pytest.raises(ValueError):
        verify(tmp_path)
    m["files"] = {"../outside": "bad"}
    write_json(tmp_path / "STAGE13_DELIVERY_MANIFEST.json", m)
    with pytest.raises(ValueError):
        verify(tmp_path)
