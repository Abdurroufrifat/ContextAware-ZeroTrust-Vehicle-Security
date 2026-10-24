"""Nonadaptive finite-horizon native speed/inertial consistency checks."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from .native_detectors import finite


@dataclass(frozen=True)
class HorizonResult:
    ready: bool
    alarm: bool
    score: float
    residuals: tuple[float | None, ...]
    spans_ms: tuple[int | None, ...]
    instantaneous_error: float


class HorizonMonitor:
    def __init__(self, bias, horizons, scales, threshold):
        if (
            not finite(bias, -30, 30)
            or not isinstance(horizons, tuple)
            or not horizons
            or tuple(sorted(set(horizons))) != horizons
            or any(type(h) is not int or not 100 <= h <= 30000 for h in horizons)
            or len(scales) != len(horizons)
            or any(not finite(x, 0.001, 100) for x in scales)
            or not finite(threshold, 1e-8, 1e8)
        ):
            raise ValueError("bounded frozen horizon configuration required")
        self.bias = bias
        self.horizons = horizons
        self.scales = tuple(scales)
        self.threshold = threshold
        self.history = deque(maxlen=4096)
        self.integral = 0.0
        self.end = None
        self.last_input = None
        self.resets = 0
        self.cached = None

    def reset(self):
        self.history.clear()
        self.integral = 0.0
        self.end = None
        self.last_input = None
        self.cached = None
        self.resets += 1

    def ingest(self, acceleration, start, end):
        if (
            not finite(acceleration, -30, 30)
            or type(start) is not int
            or type(end) is not int
            or not 0 <= start < end
        ):
            raise ValueError("positive typed causal IMU interval required")
        key = (float(acceleration), start, end)
        if self.end == end:
            if key != self.last_input:
                raise ValueError("conflicting same-time horizon input")
            return True
        if self.end is not None and end < self.end:
            raise ValueError("horizon input rollback")
        covered = self.end is None or self.end == start
        if not covered:
            self.reset()
        self.integral += (float(acceleration) - self.bias) * (end - start) / 1000
        self.end = end
        self.last_input = key
        return covered

    def sample(self, gnss, wheel, end):
        if not finite(gnss, 0, 100) or not finite(wheel, 0, 100) or type(end) is not int or end < 0:
            raise ValueError("bounded native horizon measurement required")
        if self.end is not None and self.end != end:
            raise ValueError("horizon measurement lacks current IMU")
        if self.history and end <= self.history[-1][0]:
            last = self.history[-1]
            if end != last[0] or (gnss, wheel) != (last[2], last[3]):
                raise ValueError("conflicting or old horizon measurement")
            return self.cached
        if self.end is None:
            self.end = end
        while len(self.history) > 1 and self.history[1][0] < end - self.horizons[-1] - 2000:
            self.history.popleft()
        residuals = []
        spans = []
        for horizon in self.horizons:
            anchor = next((p for p in reversed(self.history) if p[0] <= end - horizon), None)
            if anchor is None:
                residuals.append(None)
                spans.append(None)
                continue
            span = end - anchor[0]
            if span > horizon + 2000:
                residuals.append(None)
                spans.append(None)
                continue
            motion = self.integral - anchor[1]
            residuals.append(max(abs(gnss - anchor[2] - motion), abs(wheel - anchor[3] - motion)))
            spans.append(span)
        score = max(
            (r / s for r, s in zip(residuals, self.scales, strict=True) if r is not None),
            default=0.0,
        )
        if not math.isfinite(score):
            raise ValueError("nonfinite horizon score")
        ready = residuals[0] is not None
        previous = self.history[-1] if self.history else None
        instantaneous = (
            0.0
            if previous is None
            else max(
                abs(gnss - previous[2] - (self.integral - previous[1])),
                abs(wheel - previous[3] - (self.integral - previous[1])),
            )
        )
        result = HorizonResult(
            ready,
            ready and score > self.threshold,
            float(score),
            tuple(residuals),
            tuple(spans),
            instantaneous,
        )
        self.history.append((end, self.integral, float(gnss), float(wheel)))
        self.cached = result
        return result
