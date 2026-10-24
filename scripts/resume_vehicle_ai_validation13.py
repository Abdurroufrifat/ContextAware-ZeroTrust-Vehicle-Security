"""Complete interrupted validation without rescoring any intact saved block.

The original folder is retained byte-for-byte under validation_interrupted.
No model fitting, threshold selection or outcome-based policy change occurs.
"""

import argparse
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
from audit_vehicle_ai_stage13_float64 import verify_score_block
from run_vehicle_ai_stage13 import (
    OUT,
    ModelConfig,
    Scaling,
    aggregate_scores,
    base_grants,
    begin_phase,
    environment,
    features,
    hash_phase,
    load_adapter,
    load_trip,
    perturb,
    score_nodes,
    score_seed,
    sha,
    windows,
    write_json,
)
from vehicle_ai_audit_repair13 import REPAIR, assert_fit_audited, check_repair


def resume(root):
    root = Path(root).resolve()
    addendum = check_repair(root)
    f, lock = assert_fit_audited(root)
    cfg = f["config"]
    if environment() != lock["environment"]:
        raise ValueError("resume_environment_changed")
    if not addendum["interrupted_validation_sha256"]:
        raise ValueError("no_registered_interruption")
    old = root / OUT / "validation"
    retained = root / OUT / "validation_interrupted"
    if retained.exists():
        raise ValueError("validation_completion_already_started")
    if (old / "summary.json").exists():
        raise ValueError("completed_validation_must_not_repeat")
    old.rename(retained)
    folder = begin_phase(root / OUT, "validation")
    model_contract = json.loads((root / cfg["reference_contract"]).read_bytes())
    m = ModelConfig(**model_contract["model"])
    scaler = Scaling(tuple(lock["scaler"]["mean"]), tuple(lock["scaler"]["scale"]))
    rows = []
    adapters = {}
    reused = 0
    generated = 0
    reused_contexts = 0
    invalid = []
    for t in f["trips"]:
        if t["phase"] != "validation":
            continue
        original = load_trip(root, t)
        for case in cfg["cases"]:
            fs = perturb(original, case, m, model_contract["session_config"])
            w = windows(fs, cfg["window"], m)
            x = scaler.transform(w.x)
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
            name = f"contexts/{t['id']}__{case}.npz"
            src = retained / name
            dst = folder / name
            dst.parent.mkdir(exist_ok=True)
            context_ok = False
            if src.exists():
                try:
                    with np.load(src, allow_pickle=False) as c:
                        assert np.array_equal(c["positions"], w.positions) and np.array_equal(
                            c["labels"], labels
                        )
                        assert np.array_equal(
                            c["time_ms"], [v.time_ms for v in fs]
                        ) and np.array_equal(c["start_ms"], [v.interval_start_ms for v in fs])
                        assert np.allclose(c["features"], features(fs, m), rtol=1e-10, atol=1e-12)
                        grant = c["base_grant"].copy()
                        assert grant.shape == (len(fs),) and grant.dtype.kind == "b"
                        assert not np.any(grant & (c["issued"] != "ALLOW")) and np.all(
                            c["quorum"][grant] >= 3
                        )
                    context_ok = True
                except (OSError, ValueError, KeyError, AssertionError, zipfile.BadZipFile) as exc:
                    raise ValueError(
                        "invalid_existing_context_cannot_be_regenerated:" + name
                    ) from exc
            if not context_ok and any((retained / "blocks").glob(f"{t['id']}__{case}__*.npz")):
                raise ValueError("existing_scores_require_their_retained_context")
            if context_ok:
                shutil.copyfile(src, dst)
                reused_contexts += 1
            else:
                grant, issued, quorum = base_grants(fs, model_contract)
                np.savez_compressed(
                    dst,
                    features=features(fs, m),
                    time_ms=np.array([v.time_ms for v in fs]),
                    start_ms=np.array([v.interval_start_ms for v in fs]),
                    labels=labels,
                    base_grant=grant,
                    issued=issued,
                    quorum=quorum,
                    positions=w.positions,
                )
            for spec in lock["models"]:
                block = f"blocks/{t['id']}__{case}__{spec['method']}__{spec['seed']}.npz"
                src = retained / block
                dst = folder / block
                dst.parent.mkdir(exist_ok=True)
                ok = False
                if context_ok and src.exists():
                    try:
                        with np.load(src, allow_pickle=False) as b:
                            assert np.array_equal(b["positions"], w.positions)
                            verify_score_block(b, spec, spec["method"], grant)
                        ok = True
                    except (
                        OSError,
                        ValueError,
                        KeyError,
                        AssertionError,
                        zipfile.BadZipFile,
                    ) as exc:
                        raise ValueError(
                            "invalid_existing_score_cannot_be_regenerated:" + block
                        ) from exc
                if ok:
                    shutil.copyfile(src, dst)
                    reused += 1
                else:
                    key = (spec["method"], spec["seed"])
                    if key not in adapters:
                        adapters[key] = load_adapter(root, spec, cfg)
                    nodes = score_nodes(adapters[key], x, cfg, score_seed(spec["seed"], t["id"]))
                    scores = aggregate_scores(nodes, spec["node_calibration"], spec["method"])
                    alarm = scores > spec["threshold"]
                    composed = np.zeros(len(fs), dtype=bool)
                    composed[w.positions] = grant[w.positions] & ~alarm
                    np.savez_compressed(
                        dst,
                        nodes=nodes,
                        score=scores,
                        alarm=alarm,
                        positions=w.positions,
                        grant=composed,
                    )
                    generated += 1
                rows.append(
                    dict(
                        trip=t["id"],
                        case=case,
                        method=spec["method"],
                        seed=spec["seed"],
                        block=block,
                        context=name,
                    )
                )
            print("completed validation", case, "retained", reused, "new", generated, flush=True)
    summary = dict(
        schema="vehicle-ai-stage13-scores-r1-audit-repair",
        phase="validation",
        rows=rows,
        protocol_sha256=sha((root / OUT / "protocol.json").read_bytes()),
        model_lock_sha256=sha((root / OUT / "fit/model_lock.json").read_bytes()),
        status="EXPERIMENTAL_NOT_PROMOTED",
        environment=environment(),
        recovery=dict(
            addendum_sha256=sha((root / OUT / REPAIR).read_bytes()),
            retained_folder="validation_interrupted",
            reused_score_blocks=reused,
            new_score_blocks=generated,
            reused_contexts=reused_contexts,
            invalid_partial_files=invalid,
            no_model_refit=True,
            no_threshold_change=True,
        ),
    )
    write_json(folder / "summary.json", summary)
    hash_phase(folder)
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    print(resume(p.parse_args().root)["recovery"])
