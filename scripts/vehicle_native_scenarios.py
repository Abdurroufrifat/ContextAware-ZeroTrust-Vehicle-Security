"""Fixed perturbations of native benign recordings; labels never reach policy."""

from dataclasses import replace

import numpy as np

from coact_cps.vehicle.native_detectors import Observer

CASES = (
    "normal",
    "gnss_bias",
    "gnss_ramp",
    "wheel_scale",
    "imu_bias",
    "imu_dropout",
    "stale_gnss",
    "gnss_replay",
    "identity_revoked",
    "postissue_revoked",
    "display_downplay",
    "joint_jump",
    "joint_ramp",
    "coherent_hold",
    "coherent_withdrawal",
    "startup_common_bias",
    "observer_aware",
)
VALIDATION_CASES = ("normal", "wheel_scale", "imu_bias", "imu_dropout")


def bounds(frames):
    first, last = frames[0].time_ms, frames[-1].time_ms
    return first + (last - first) / 3, first + 2 * (last - first) / 3


def active(frame, frames, case):
    start, end = bounds(frames)
    if case == "normal":
        return False
    if case in ("imu_bias", "startup_common_bias"):
        return True
    if case == "imu_dropout":
        return start <= frame.time_ms < start + 2000
    if case == "coherent_hold":
        return frame.time_ms >= start
    return start <= frame.time_ms <= end


def period(frame, frames, case):
    if active(frame, frames, case):
        return "active"
    return "before" if frame.time_ms < bounds(frames)[0] else "after"


def perturb(frames, case, model):
    if case not in CASES:
        raise ValueError("unknown frozen scenario")
    start, end = bounds(frames)
    output = []
    offset = 0.0
    attacker = Observer(model) if case == "observer_aware" else None
    for f in frames:
        g, w, a = f.gnss_speed, f.wheel_speed, f.imu_accel
        on = active(f, frames, case)
        window = start <= f.time_ms <= end
        if case == "observer_aware":
            attacker.predict(a, f.interval_start_ms, f.time_ms)
            if attacker.x is None:
                attacker.initialize(g, w, f.time_ms)
            if window:
                _, cov = attacker.innovation(g, w)
                residual = np.linalg.cholesky(cov) @ np.array([np.sqrt(3 * 0.99), 0.0])
                g = float(attacker.x[0] + model.gnss_offset + residual[0])
                w = float(attacker.x[0] + residual[1])
            attacker.update(g, w)
        elif case == "startup_common_bias":
            g += 8
            w += 8
        elif case == "imu_bias":
            a += 0.12
        elif case in ("coherent_hold", "coherent_withdrawal"):
            if window:
                a += 0.4
                offset += 0.4 * (f.time_ms - f.interval_start_ms) / 1000
            elif f.time_ms > end and case == "coherent_withdrawal":
                offset = 0.0
            g += offset
            w += offset
        elif on:
            ramp = 4 * (f.time_ms - start) / (end - start)
            if case == "gnss_bias":
                g += 3
            elif case == "gnss_ramp":
                g += ramp
            elif case == "wheel_scale":
                w *= 1.06
            elif case == "joint_jump":
                g += 8
                w += 8
            elif case == "joint_ramp":
                g += ramp
                w += ramp
        if not 0 <= g <= 100 or not 0 <= w <= 100:
            raise ValueError("perturbation outside frozen policy domain")
        output.append(replace(f, gnss_speed=g, wheel_speed=w, imu_accel=a))
    return output
