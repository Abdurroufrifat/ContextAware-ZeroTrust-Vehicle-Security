"""Stage7 controlled perturbations, generated once and shared by all methods."""

from dataclasses import replace

from vehicle_native_scenarios import (
    CASES as BASE_CASES,
)
from vehicle_native_scenarios import (
    bounds,
)
from vehicle_native_scenarios import (
    perturb as base_perturb,
)

CASES = BASE_CASES + ("horizon_budget", "gap_reenrollment")


def perturb(frames, case, model, horizon_config):
    if case == "gap_reenrollment":
        return base_perturb(frames, "joint_jump", model)
    if case != "horizon_budget":
        return base_perturb(frames, case, model)
    begin, end = bounds(frames)
    rate = 0.5 * horizon_config["threshold"] * horizon_config["scales"][-1] / 20
    output = []
    for f in frames:
        offset = min(4.0, rate * (f.time_ms - begin) / 1000) if begin <= f.time_ms <= end else 0.0
        output.append(
            replace(f, gnss_speed=f.gnss_speed + offset, wheel_speed=f.wheel_speed + offset)
        )
    return output
