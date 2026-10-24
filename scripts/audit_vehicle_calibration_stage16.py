"""Audit saved calibration/alarms/grants without rerunning model inference."""

import argparse

import numpy as np
from run_vehicle_calibration_stage16 import OUT, read, sha, verify, verify_phase, write

from coact_cps.vehicle.calibration16 import fit_threshold, validation_gate


def audit(root, phase):
    p = verify(root)
    summary = verify_phase(root, phase, audited=False)
    folder = root / OUT / phase
    trips = {t["id"] for t in p["trips"] if t["phase"] == phase}
    specs = {s["seed"]: s for s in p["model_specs"]}
    cases = ["normal"] if phase == "calibration" else p["config"]["cases"]
    actual = {(r["trip"], r["seed"], r["case"]) for r in summary["rows"]}
    expected = {(t, s, c) for t in trips for s in specs for c in cases}
    if actual != expected or len(summary["rows"]) != len(expected):
        raise ValueError("complete_unique_frozen_block_set_required")
    lock = read(root / OUT / "calibration/threshold_lock.json")
    grouped = {seed: {} for seed in specs}
    metrics = []
    for row in summary["rows"]:
        with np.load(folder / row["context"], allow_pickle=False) as a:
            context = {k: a[k] for k in a.files}
        with np.load(folder / row["block"], allow_pickle=False) as a:
            block = {k: a[k] for k in a.files}
        spec = specs[row["seed"]]
        pos = block["positions"]
        base, labels = context["base_grant"], context["labels"]
        nodes = np.asarray(block["nodes"], dtype=float)
        if (
            pos.ndim != 1
            or pos.dtype.kind not in "iu"
            or len(pos) < 32
            or pos[0] < 0
            or pos[-1] >= len(base)
            or np.any(np.diff(pos) <= 0)
            or not np.array_equal(pos, context["positions"])
            or nodes.shape != (len(pos), 3)
            or not np.isfinite(nodes).all()
            or base.dtype != np.dtype(bool)
            or labels.dtype != np.dtype(bool)
            or labels.shape != base.shape
            or base.ndim != 1
            or (row["case"] == "normal" and labels.any())
        ):
            raise ValueError("invalid_saved_context_or_nodes")
        scores = (
            (nodes - np.asarray(spec["node_calibration"]["median"]))
            / np.asarray(spec["node_calibration"]["scale"])
        ).max(axis=1)
        if not np.array_equal(scores, block["score"]):
            raise ValueError("reconstructed_score_mismatch")
        if phase == "calibration":
            grouped[row["seed"]][row["trip"]] = scores
            continue
        values = {}
        for label, threshold in [
            ("baseline", spec["threshold"]),
            ("candidate", lock["seeds"][str(row["seed"])]["threshold"]),
        ]:
            alarm = scores > threshold
            grant = np.zeros(len(base), dtype=bool)
            grant[pos] = base[pos] & ~alarm
            if not np.array_equal(alarm, block[label + "_alarm"]) or not np.array_equal(
                grant, block[label + "_grant"]
            ):
                raise ValueError("saved_alarm_or_grant_mismatch")
            active = labels[pos] if row["case"] != "normal" else np.ones(len(pos), bool)
            if not active.any():
                raise ValueError("active_scored_endpoints_required")
            values[label] = float(alarm[active].mean())
            values[label + "_grant"] = float(grant.mean())
        metrics.append(
            dict(
                trip=row["trip"],
                seed=row["seed"],
                case=row["case"],
                base_grant=float(base.mean()),
                **values,
            )
        )
    if phase == "calibration":
        for seed, spec in specs.items():
            if fit_threshold(grouped[seed], spec["threshold"]) != lock["seeds"][str(seed)]:
                raise ValueError("clean_route_threshold_lock_mismatch")
        if lock["protocol_sha256"] != summary["protocol_sha256"]:
            raise ValueError("calibration_protocol_binding_failed")
    elif metrics != summary["metrics"]:
        raise ValueError("metric_reconstruction_failed")
    if phase == "validation" and validation_gate(metrics, cases) != summary["gate"]:
        raise ValueError("validation_gate_mismatch")
    result = dict(
        status="PASS",
        phase=phase,
        blocks=len(summary["rows"]),
        protocol_sha256=summary["protocol_sha256"],
        phase_hashes_sha256=sha((folder / "file_hashes.json").read_bytes()),
        model_inference_repeated=False,
        fitting_repeated=False,
        limits=(
            "Saved-output algebra audit; no independent model inference or physical truth claim."
        ),
    )
    write(folder / "independent_audit.json", result)
    return result


if __name__ == "__main__":
    from pathlib import Path

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--phase", choices=["calibration", "validation", "test"], required=True)
    a = p.parse_args()
    print(audit(a.root.resolve(), a.phase))
