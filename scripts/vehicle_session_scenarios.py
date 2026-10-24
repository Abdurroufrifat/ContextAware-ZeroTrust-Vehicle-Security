"""Predeclared offline attacks; never expose evaluation fields to policy."""

import math
from dataclasses import replace

from vehicle_integrity_scenarios import CASES as OLD_CASES
from vehicle_integrity_scenarios import perturb as old_perturb
from vehicle_native_scenarios import active as old_active
from vehicle_native_scenarios import bounds

CASES = OLD_CASES + ("anchor_budget", "enrollment_ramp")


def active(frame, frames, case):
    return True if case == "enrollment_ramp" else old_active(frame, frames, case)


def period(frame, frames, case):
    if active(frame, frames, case):
        return "active"
    return "before" if frame.time_ms < bounds(frames)[0] else "after"


def perturb(frames, case, model, config):
    if case not in CASES:
        raise ValueError("unknown session scenario")
    if case in OLD_CASES:
        return old_perturb(frames, case, model, config["stage7_horizon"])
    start, end = bounds(frames)
    first = frames[0].time_ms
    out = []
    for f in frames:
        age = max(0.0, (f.time_ms - first) / 1000)
        if case == "enrollment_ramp":
            offset = 0.12 * age
        else:
            c = config["anchor"]
            offset = (
                min(
                    4.0,
                    0.5
                    * c["threshold"]
                    * c["scale"]
                    * (math.sqrt(age) - math.sqrt((start - first) / 1000)),
                )
                if start <= f.time_ms <= end
                else 0.0
            )
        g = f.gnss_speed + offset
        w = f.wheel_speed + offset
        if not 0 <= g <= 100 or not 0 <= w <= 100:
            raise ValueError("session attack exceeds frozen domain")
        out.append(replace(f, gnss_speed=g, wheel_speed=w))
    return out
