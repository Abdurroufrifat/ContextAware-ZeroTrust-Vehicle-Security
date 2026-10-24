"""Regression for clean float32 LOF evidence; no native model/test execution."""

import json

import numpy as np
import pytest


def test_float32_clean_scores_recompute_in_the_fitting_precision(tmp_path, monkeypatch):
    import audit_vehicle_ai_stage13 as original
    import audit_vehicle_ai_stage13_float64 as repaired
    import run_vehicle_ai_stage13 as r
    from test_vehicle_window_recovery import WindowRecoveryTests
    from vehicle_ai_data13 import Scaling, array_hash, features, windows

    from coact_cps.vehicle.ai_models13 import fit_node_calibration
    from coact_cps.vehicle.native_detectors import ModelConfig

    c = WindowRecoveryTests().config()
    (tmp_path / "contract.json").write_text(json.dumps({"model": c["model"]}))
    model = ModelConfig(**c["model"])
    fs = WindowRecoveryTests().frames()[:101]
    cfg = dict(reference_contract="contract.json", window=32, methods=["lof"], seeds=[7], epochs=1)
    trips = [dict(id=p, phase=p) for p in ["train", "scale", "calibrate"]]
    scaling = Scaling.fit(features(fs, model))
    x = scaling.transform(windows(fs, 32, model).x)
    nodes = np.linspace(1.1, 3.7, len(x), dtype=np.float32)[:, None]
    cal = fit_node_calibration(nodes)
    scores = original.aggregate(nodes, cal, "lof")
    np.savez(
        tmp_path / "clean.npz", scale_nodes=nodes, calibrate_nodes=nodes, calibrate_scores=scores
    )
    spec = dict(
        method="lof",
        seed=7,
        checkpoint=None,
        clean_scores="clean.npz",
        node_calibration=cal,
        threshold=float(np.quantile(scores, 0.95)),
        training_loss=[],
    )
    f = dict(config=cfg, trips=trips)
    lock = dict(
        scaler=scaling.to_dict(),
        train_routes=["train"],
        scale_route="scale",
        calibrate_route="calibrate",
        train_windows_sha256=array_hash(x),
        scale_windows_sha256=array_hash(x),
        calibrate_windows_sha256=array_hash(x),
        models=[spec],
    )
    monkeypatch.setattr(r, "verify_fit", lambda root: (f, lock))
    monkeypatch.setattr(r, "load_trip", lambda root, t: fs)
    with pytest.raises(AssertionError):
        original.verify_fit_audit(tmp_path)
    assert repaired.verify_fit_audit(tmp_path) == (f, lock)


def test_guarded_validation_requires_completed_fit_audit(tmp_path):
    from vehicle_ai_audit_repair13 import assert_fit_audited

    with pytest.raises((ValueError, FileNotFoundError)):
        assert_fit_audited(tmp_path)


@pytest.mark.parametrize("corrupt_context", [False, True])
def test_validation_completion_reuses_intact_scores_and_preserves_original(
    tmp_path, monkeypatch, corrupt_context
):
    from pathlib import Path

    import resume_vehicle_ai_validation13 as resume
    from run_vehicle_ai_stage13 import OUT, environment, sha, write_json
    from test_vehicle_window_recovery import WindowRecoveryTests
    from vehicle_ai_data13 import features, windows

    from coact_cps.vehicle.native_detectors import ModelConfig

    project = Path(__file__).resolve().parents[1]
    ref = json.loads(
        (
            project / "data/reference/vehicle_native_stage10b/stage10b_linux_frozen_transfer.json"
        ).read_bytes()
    )
    write_json(tmp_path / "contract.json", ref)
    m = ModelConfig(**ref["model"])
    frames = WindowRecoveryTests().frames()[:100]
    w = windows(frames, 32, m)
    cfg = dict(
        reference_contract="contract.json", window=32, cases=["normal", "joint_jump"], batch_size=8
    )
    spec = dict(
        method="persistence",
        seed=7,
        checkpoint=None,
        checkpoint_sha256=None,
        node_calibration={"median": [0.0] * 3, "scale": [1.0] * 3},
        threshold=0.5,
    )
    f = dict(config=cfg, trips=[dict(id="synthetic", phase="validation")])
    lock = dict(
        scaler={"mean": [0.0] * 3, "scale": [1.0] * 3}, models=[spec], environment=environment()
    )
    base = tmp_path / OUT
    write_json(base / "protocol.json", {})
    write_json(base / "fit/model_lock.json", {})
    write_json(base / "audit_precision_addendum.json", {})
    old = base / "validation"
    (old / "contexts").mkdir(parents=True)
    (old / "blocks").mkdir()
    grant = np.ones(len(frames), dtype=bool)
    np.savez(
        old / "contexts/synthetic__normal.npz",
        features=features(frames, m),
        time_ms=np.array([v.time_ms for v in frames]),
        start_ms=np.array([v.interval_start_ms for v in frames]),
        labels=np.zeros(len(frames), dtype=bool),
        base_grant=grant,
        issued=np.full(len(frames), "ALLOW"),
        quorum=np.full(len(frames), 3),
        positions=w.positions,
    )
    nodes = np.abs(w.x[:, -1] - w.x[:, -2]).astype(float)
    score = nodes.max(-1)
    alarm = score > 0.5
    composed = np.zeros(len(frames), dtype=bool)
    composed[w.positions] = ~alarm
    np.savez(
        old / "blocks/synthetic__normal__persistence__7.npz",
        nodes=nodes,
        score=score,
        alarm=alarm,
        positions=w.positions,
        grant=composed,
    )
    if corrupt_context:
        path = old / "contexts/synthetic__normal.npz"
        with np.load(path, allow_pickle=False) as c:
            payload = {k: c[k].copy() for k in c.files}
        payload["features"] += 100
        np.savez(path, **payload)
    hashes = {
        p.relative_to(old).as_posix(): sha(p.read_bytes()) for p in old.rglob("*") if p.is_file()
    }
    monkeypatch.setattr(
        resume, "check_repair", lambda root: {"interrupted_validation_sha256": hashes}
    )
    monkeypatch.setattr(resume, "assert_fit_audited", lambda root: (f, lock))
    monkeypatch.setattr(resume, "load_trip", lambda root, t: frames)
    calls = []

    def fake_base(fs, contract):
        calls.append(1)
        return grant, np.full(len(fs), "ALLOW"), np.full(len(fs), 3)

    monkeypatch.setattr(resume, "base_grants", fake_base)
    if corrupt_context:
        with pytest.raises(ValueError):
            resume.resume(tmp_path)
        assert calls == []
        return
    result = resume.resume(tmp_path)
    assert (
        result["recovery"]["reused_score_blocks"] == 1
        and result["recovery"]["new_score_blocks"] == 1
    )
    assert result["recovery"]["reused_contexts"] == 1 and calls == [1]
    assert {
        p.relative_to(base / "validation_interrupted").as_posix(): sha(p.read_bytes())
        for p in (base / "validation_interrupted").rglob("*")
        if p.is_file()
    } == hashes
    assert (
        sha((base / "validation/blocks/synthetic__normal__persistence__7.npz").read_bytes())
        == hashes["blocks/synthetic__normal__persistence__7.npz"]
    )


def test_recovery_audit_rejects_rehashed_replacement_and_false_counts(tmp_path):
    import copy

    from run_vehicle_ai_stage13 import sha
    from vehicle_ai_audit_repair13 import verify_recovered_copies

    (tmp_path / "contexts").mkdir()
    (tmp_path / "blocks").mkdir()
    (tmp_path / "contexts/context.npz").write_bytes(b"original context")
    (tmp_path / "blocks/block.npz").write_bytes(b"original scores")
    hashes = {
        p.relative_to(tmp_path).as_posix(): sha(p.read_bytes()) for p in tmp_path.rglob("*.npz")
    }
    addendum = {"interrupted_validation_sha256": hashes}
    summary = {
        "rows": [{"context": "contexts/context.npz", "block": "blocks/block.npz"}],
        "recovery": {
            "addendum_sha256": "bound",
            "retained_folder": "validation_interrupted",
            "reused_contexts": 1,
            "reused_score_blocks": 1,
            "new_score_blocks": 0,
            "invalid_partial_files": [],
            "no_model_refit": True,
            "no_threshold_change": True,
        },
    }
    verify_recovered_copies(tmp_path, summary, addendum, "bound")
    bad = copy.deepcopy(summary)
    bad["recovery"]["reused_score_blocks"] = 0
    with pytest.raises(ValueError):
        verify_recovered_copies(tmp_path, bad, addendum, "bound")
    (tmp_path / "blocks/block.npz").write_bytes(b"regenerated replacement")
    with pytest.raises(ValueError):
        verify_recovered_copies(tmp_path, summary, addendum, "bound")
