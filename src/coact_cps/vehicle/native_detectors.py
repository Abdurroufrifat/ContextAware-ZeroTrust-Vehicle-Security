"""Published innovation statistics on a declared custom longitudinal observer.

The variable-time model and empirical native calibration are adaptations, not
reproductions of a paper's original plant or Gaussian false-alarm guarantee.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass

import numpy as np


def finite(value, low=-math.inf, high=math.inf):
    return type(value) in (int, float) and math.isfinite(value) and low <= value <= high


def innovation_energy(residual, covariance):
    r = np.asarray(residual, dtype=float)
    s = np.asarray(covariance, dtype=float)
    if (
        r.ndim != 1
        or s.shape != (len(r), len(r))
        or not np.isfinite(r).all()
        or not np.isfinite(s).all()
        or not np.allclose(s, s.T, atol=1e-12)
    ):
        raise ValueError("finite symmetric innovation covariance required")
    np.linalg.cholesky(s)
    return float(r @ np.linalg.solve(s, r))


@dataclass(frozen=True)
class ModelConfig:
    gnss_offset: float = 0.0
    imu_offset: float = 0.0
    r_g: float = 0.04
    r_w: float = 0.01
    sigma_a: float = 0.2
    sigma_b: float = 0.03
    speed_initial_variance: float = 1.0
    bias_initial_variance: float = 0.25

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not finite(value, -30 if name.endswith("offset") else 1e-9, 100):
                raise ValueError("finite bounded positive model variances/scales required")


class Observer:
    def __init__(self, config):
        if not isinstance(config, ModelConfig):
            raise ValueError("typed model config required")
        self.config = config
        self.x = None
        self.p = None
        self.time_ms = None
        self.c = np.array([[1.0, 0.0], [1.0, 0.0]])
        self.r = np.diag([config.r_g, config.r_w])

    def initialize(self, gnss, wheel, time_ms):
        if (
            not finite(gnss, 0, 100)
            or not finite(wheel, 0, 100)
            or type(time_ms) is not int
            or time_ms < 0
        ):
            raise ValueError("valid native initial epoch required")
        g = gnss - self.config.gnss_offset
        speed = (g / self.config.r_g + wheel / self.config.r_w) / (
            1 / self.config.r_g + 1 / self.config.r_w
        )
        self.x = np.array([speed, self.config.imu_offset])
        self.p = np.diag([self.config.speed_initial_variance, self.config.bias_initial_variance])
        self.time_ms = time_ms

    def predict(self, acceleration, start_ms, end_ms):
        if (
            not finite(acceleration, -30, 30)
            or type(start_ms) is not int
            or type(end_ms) is not int
            or not 0 <= start_ms < end_ms
        ):
            raise ValueError("valid native input interval required")
        if self.x is None or self.time_ms != start_ms:
            self.x = self.p = self.time_ms = None
            return False
        dt = (end_ms - start_ms) / 1000
        f = np.array([[1.0, -dt], [0.0, 1.0]])
        self.x = f @ self.x + np.array([dt * acceleration, 0.0])
        q = np.diag([(self.config.sigma_a * dt) ** 2, self.config.sigma_b**2 * dt])
        self.p = f @ self.p @ f.T + q
        self.p = (self.p + self.p.T) / 2
        self.time_ms = end_ms
        return True

    def innovation(self, gnss, wheel):
        if self.x is None or not finite(gnss, 0, 100) or not finite(wheel, 0, 100):
            raise ValueError("initialized observer and finite native speeds required")
        y = np.array([gnss - self.config.gnss_offset, wheel])
        return y - self.c @ self.x, self.c @ self.p @ self.c.T + self.r

    def update(self, gnss, wheel):
        residual, s = self.innovation(gnss, wheel)
        k = np.linalg.solve(s, (self.p @ self.c.T).T).T
        self.x = self.x + k @ residual
        a = np.eye(2) - k @ self.c
        self.p = a @ self.p @ a.T + k @ self.r @ k.T
        self.p = (self.p + self.p.T) / 2


@dataclass(frozen=True)
class StatisticResult:
    alarm: bool
    statistic: float
    scored: bool


class Statistic:
    def __init__(self, kind, threshold, window=10, bias=3.0):
        if (
            kind not in ("chi", "window", "cusum")
            or not finite(threshold, 1e-9, 1e12)
            or type(window) is not int
            or not 1 <= window <= 1000
            or not finite(bias, 2.000000001, 1e6)
        ):
            raise ValueError("bounded statistic configuration required; CUSUM b>m=2")
        self.kind = kind
        self.threshold = threshold
        self.window = window
        self.bias = bias
        self.history = deque(maxlen=window)
        self.s = 0.0

    def reset(self):
        self.history.clear()
        self.s = 0.0

    def step(self, z):
        if not finite(z, 0, 1e15):
            raise ValueError("finite nonnegative innovation energy required")
        if self.kind == "chi":
            return StatisticResult(z > self.threshold, float(z), True)
        if self.kind == "window":
            self.history.append(float(z))
            value = sum(self.history)
            ready = len(self.history) == self.window
            return StatisticResult(ready and value > self.threshold, value, ready)
        # Published recurrence: an earlier crossing is reported on this call;
        # this call's observation is discarded while the statistic resets.
        if self.s > self.threshold:
            self.s = 0.0
            return StatisticResult(True, 0.0, False)
        self.s = max(0.0, self.s + float(z) - self.bias)
        return StatisticResult(False, self.s, True)
