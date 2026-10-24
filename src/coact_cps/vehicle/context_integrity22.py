"""Paired offline context intervention; never a deployable clean-context oracle."""

import numpy as np

from coact_cps.vehicle.imu_guard21 import decide


def verify_pair(reported, normal):
    if np.asarray(normal["labels"]).any():
        raise ValueError("paired normal route contains attack labels")
    for key in ("time_ms", "start_ms", "positions"):
        if not np.array_equal(reported[key], normal[key]):
            raise ValueError(f"paired route mismatch: {key}")
    if np.asarray(reported["labels"]).shape != np.asarray(normal["labels"]).shape:
        raise ValueError("paired label shape mismatch")


def paired_effect(nodes, threshold, reported_boundary, clean_boundary, active):
    active = np.asarray(active)
    if active.dtype.kind != "b" or active.shape != (len(nodes),):
        raise ValueError("aligned boolean active labels required")
    reported = decide(nodes, threshold, reported_boundary)
    clean = decide(nodes, threshold, clean_boundary)
    lost, gained = clean & ~reported, reported & ~clean
    raised = np.asarray(reported_boundary) > np.asarray(clean_boundary)
    lowered = np.asarray(reported_boundary) < np.asarray(clean_boundary)
    result = {}
    for name, mask in (("active", active), ("inactive", ~active)):
        count = int(mask.sum())
        result[name + "_count"] = count
        for key, event in (
            ("lost", lost),
            ("gained", gained),
            ("boundary_raised", raised),
            ("boundary_lowered", lowered),
            ("reported_alarm", reported),
            ("clean_alarm", clean),
        ):
            result[f"{name}_{key}_count"] = int((mask & event).sum())
        result[name + "_net_tpr_loss" if name == "active" else name + "_net_alarm_loss"] = (
            float((int((mask & lost).sum()) - int((mask & gained).sum())) / count)
            if count
            else None
        )
    return result, reported, clean
