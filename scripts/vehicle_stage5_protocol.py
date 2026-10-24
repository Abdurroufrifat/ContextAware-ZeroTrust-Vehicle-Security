"""Fixed study inputs; these control faults, not features supplied to a policy."""

import copy
import hashlib
import math
import re

SOURCES = ("identity", "gnss", "imu", "wheel", "can", "v2x")
FAULTS = (
    "benign_imu_bias_positive",
    "benign_imu_bias_negative",
    "benign_wheel_scale",
    "benign_imu_dropout",
    "benign_timestamp_delay_jitter",
)


def validate_config(config):
    if (
        config.get("version") != "coact-vehicle-stage5-fixed-policy-robustness-v1"
        or config.get("sumo_version") != "1.28.0"
    ):
        raise ValueError("Unsupported study protocol")

    def finite(v):
        return type(v) in (int, float) and math.isfinite(v)

    window = config.get("window_fraction")
    if (
        not isinstance(window, list)
        or len(window) != 2
        or not all(finite(v) for v in window)
        or not 0 < window[0] < window[1] < 1
    ):
        raise ValueError("Invalid window fractions")
    warmup = config.get("warmup_s")
    if not finite(warmup) or warmup < 0:
        raise ValueError("Invalid warmup")
    names = set()
    seeds = set()
    smoke = set()
    old = {11, 23, 37, 41, 53, 67}
    profiles = config.get("profiles")
    if not isinstance(profiles, list) or len(profiles) != 3:
        raise ValueError("Three fixed study profiles required")
    for p in profiles:
        name = p["name"]
        if type(name) is not str or not re.fullmatch("[a-z][a-z0-9_]{0,31}", name) or name in names:
            raise ValueError("Invalid or duplicate profile name")
        names.add(name)
        for key, low, high in [("grid", 3, 6), ("background", 0, 60)]:
            if type(p[key]) is not int or not low <= p[key] <= high:
                raise ValueError("Invalid integer profile value")
        for key, low, high in [
            ("length_m", 80, 300),
            ("speed_mps", 1, 25),
            ("step_s", 0.05, 1),
            ("duration_s", 5, 90),
        ]:
            if not finite(p[key]) or not low <= p[key] <= high:
                raise ValueError("Invalid numeric profile value")
        if not math.isclose(
            p["duration_s"] / p["step_s"], round(p["duration_s"] / p["step_s"]), abs_tol=1e-8
        ):
            raise ValueError("Partial simulation step")
        if warmup >= p["duration_s"] * window[0] or 2 >= p["duration_s"] * (window[1] - window[0]):
            raise ValueError("Event/warmup interval too short")
        if not isinstance(p["seeds"], list) or len(p["seeds"]) != 6:
            raise ValueError("Six seeds per study profile required")
        for seed in p["seeds"]:
            if type(seed) is not int or not 0 <= seed < 2**31 or seed in seeds | old:
                raise ValueError("Invalid, repeated or old seed")
            seeds.add(seed)
        s = p["smoke_seed"]
        if type(s) is not int or not 0 <= s < 2**31 or s in smoke | old:
            raise ValueError("Invalid smoke seed")
        smoke.add(s)
    if smoke & seeds:
        raise ValueError("Smoke and study seeds overlap")


def fault(observation, case, t, start, end, profile, seed):
    result = copy.deepcopy(observation)
    transport = {}
    active = start <= t < end
    s = result["signals"]
    if case in ("benign_imu_bias_positive", "benign_imu_bias_negative"):
        active = True
        s["imu"]["acceleration_mps2"] += 0.06 if case.endswith("positive") else -0.06
    elif case == "benign_imu_dropout":
        active = start <= t < start + 2
        if active:
            transport["missing"] = "imu"
    elif case == "benign_wheel_scale":
        if active:
            s["wheel"]["speed_mps"] *= 1.06
    elif case == "benign_timestamp_delay_jitter":
        if active:
            transport["source_delays_ms"] = {
                source: int.from_bytes(
                    hashlib.sha256(
                        f"stage5-delay/{profile}/{seed}/{observation['frame']}/{source}".encode()
                    ).digest()[:4],
                    "big",
                )
                % 21
                for source in SOURCES
            }
    elif case == "coherent_motion_forgery":
        if active:
            # 3 m/s2, rather than Stage 4's 5, stays within the source domain for 75 s.
            for source in ("gnss", "wheel"):
                s[source]["speed_mps"] += 3.0 * (t - start)
            for source in ("imu", "wheel"):
                s[source]["acceleration_mps2"] += 3.0
    else:
        raise ValueError("Unknown controlled fault")
    return result, transport, active
