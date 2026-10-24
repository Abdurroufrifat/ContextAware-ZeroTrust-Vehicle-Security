"""Shared legacy and recovery-adaptive evaluation streams; labels stay offline."""

from dataclasses import replace

import numpy as np
from vehicle_native_scenarios import bounds
from vehicle_session_scenarios import CASES as OLD_CASES
from vehicle_session_scenarios import active as old_active
from vehicle_session_scenarios import perturb as old_perturb

from coact_cps.vehicle.native_detectors import Observer

CASES = OLD_CASES + ("v2_observer_aware", "recovery_cycling", "persistent_joint_step")


def active(frame, frames, case):
    start, end = bounds(frames)
    if case == "persistent_joint_step":
        return frame.time_ms >= start
    if case == "recovery_cycling":
        return frame.time_ms >= start and (frame.time_ms - start) % 30000 < 5000
    return old_active(frame, frames, "observer_aware" if case == "v2_observer_aware" else case)


def period(frame, frames, case):
    if active(frame, frames, case):
        return "active"
    return "before" if frame.time_ms < bounds(frames)[0] else "after"


def perturb(frames, case, model, config):
    if case == "observer_aware":
        observer = Observer(model)
        output = []
        for f in frames:
            g, w = f.gnss_speed, f.wheel_speed
            observer.predict(f.imu_accel, f.interval_start_ms, f.time_ms)
            if observer.x is None:
                observer.initialize(g, w, f.time_ms)
            if active(f, frames, case):
                _, cov = observer.innovation(g, w)
                residual = np.linalg.cholesky(cov) @ np.array([np.sqrt(3 * 0.99), 0.0])
                g = float(np.clip(observer.x[0] + model.gnss_offset + residual[0], 0.0, 100.0))
                w = float(np.clip(observer.x[0] + residual[1], 0.0, 100.0))
            observer.update(g, w)
            output.append(replace(f, gnss_speed=g, wheel_speed=w))
        return output
    if case in OLD_CASES:
        return old_perturb(frames, case, model, config)
    if case not in CASES:
        raise ValueError("unknown recovery scenario")
    observer = Observer(model) if case == "v2_observer_aware" else None
    out = []
    for f in frames:
        g, w = f.gnss_speed, f.wheel_speed
        on = active(f, frames, case)
        if observer is not None:
            observer.predict(f.imu_accel, f.interval_start_ms, f.time_ms)
            if observer.x is None:
                observer.initialize(g, w, f.time_ms)
            if on:
                _, cov = observer.innovation(g, w)
                residual = np.linalg.cholesky(cov) @ np.array(
                    [np.sqrt(0.99 * config["civic_thresholds"]["dual_chi"]), 0.0]
                )
                g = float(
                    np.clip(
                        observer.x[0] + model.gnss_offset + residual[0],
                        f.gnss_speed,
                        min(100.0, f.gnss_speed + 8.0),
                    )
                )
                w = float(
                    np.clip(
                        observer.x[0] + residual[1], f.wheel_speed, min(100.0, f.wheel_speed + 8.0)
                    )
                )
            observer.update(g, w)
        elif on:
            g += 5
            w += 5
        if not 0 <= g <= 100 or not 0 <= w <= 100:
            raise ValueError("attack exceeds frozen physical domain")
        out.append(replace(f, gnss_speed=g, wheel_speed=w))
    return out
