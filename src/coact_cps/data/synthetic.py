from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .topology import Topology

SYNTHETIC_NODES = (
    "MV101",
    "FIT101",
    "LIT101",
    "P101",
    "FIT201",
    "PIT201",
)


@dataclass(frozen=True)
class SyntheticCPSConfig:
    n_steps: int = 2400
    seed: int = 7
    noise: float = 0.025


def synthetic_topology() -> Topology:
    return Topology(
        nodes=SYNTHETIC_NODES,
        relations={
            "control": (("LIT101", "MV101"), ("LIT101", "P101")),
            "physical": (
                ("MV101", "FIT101"),
                ("FIT101", "LIT101"),
                ("P101", "FIT201"),
                ("FIT201", "LIT101"),
                ("FIT201", "PIT201"),
            ),
        },
    )


def _attack_schedule(n_steps: int) -> list[tuple[int, int, str, str]]:
    candidates = [
        (int(n_steps * 0.25), int(n_steps * 0.31), "valve_forcing", "MV101"),
        (int(n_steps * 0.50), int(n_steps * 0.56), "sensor_bias", "LIT101"),
        (int(n_steps * 0.75), int(n_steps * 0.81), "pump_forcing", "P101"),
    ]
    return [(start, max(start + 5, end), kind, root) for start, end, kind, root in candidates]


def generate_synthetic_cps(
    config: SyntheticCPSConfig | None = None,
) -> tuple[pd.DataFrame, Topology]:
    """Generate development-only CPS telemetry with three labelled attacks.

    This dataset is intentionally simple. It exists to verify the software and must
    never be reported as evidence that the research hypotheses are supported.
    """

    config = config or SyntheticCPSConfig()
    if config.n_steps < 300:
        raise ValueError("n_steps must be at least 300")
    rng = np.random.default_rng(config.seed)
    schedule = _attack_schedule(config.n_steps)

    level = 50.0
    valve_command = 1.0
    pump_command = 1.0
    rows: list[dict[str, float | int | str]] = []

    for step in range(config.n_steps):
        attack_type = "normal"
        root_cause = "normal"
        for start, end, kind, root in schedule:
            if start <= step < end:
                attack_type, root_cause = kind, root
                break

        if level > 62.0:
            valve_command = 0.0
        elif level < 47.0:
            valve_command = 1.0
        if level < 37.0:
            pump_command = 0.0
        elif level > 44.0:
            pump_command = 1.0

        actual_valve = valve_command
        actual_pump = pump_command
        if attack_type == "valve_forcing":
            actual_valve = 1.0
        elif attack_type == "pump_forcing":
            actual_pump = 0.0

        inlet_flow = 1.15 * actual_valve + rng.normal(0.0, config.noise)
        outlet_flow = 0.92 * actual_pump + rng.normal(0.0, config.noise)
        level = np.clip(
            level + 0.18 * (inlet_flow - outlet_flow) + rng.normal(0.0, config.noise),
            0.0,
            100.0,
        )
        observed_level = level
        if attack_type == "sensor_bias":
            observed_level = np.clip(level + 18.0, 0.0, 100.0)
        pressure = 1.8 + 0.75 * outlet_flow + rng.normal(0.0, config.noise)

        rows.append(
            {
                "timestamp": pd.Timestamp("2026-01-01") + pd.Timedelta(step, unit="s"),
                "MV101": actual_valve,
                "FIT101": inlet_flow,
                "LIT101": observed_level / 100.0,
                "P101": actual_pump,
                "FIT201": outlet_flow,
                "PIT201": pressure / 3.0,
                "attack": int(attack_type != "normal"),
                "attack_type": attack_type,
                "root_cause": root_cause,
            }
        )

    return pd.DataFrame(rows), synthetic_topology()
