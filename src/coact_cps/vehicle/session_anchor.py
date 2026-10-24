"""An enrolled motion reference; does not attest authentic enrollment."""

import math
from collections import deque
from dataclasses import dataclass

from .native_detectors import finite


@dataclass(frozen=True)
class AnchorResult:
    ready: bool
    lost: bool
    alarm: bool
    score: float
    bias: float | None


class SessionAnchor:
    def __init__(self, gnss_offset, scale, threshold):
        if not all(
            (finite(gnss_offset, -10, 10), finite(scale, 0.001, 100), finite(threshold, 1e-8, 1e8))
        ):
            raise ValueError("bounded anchor calibration required")
        self.gnss_offset = float(gnss_offset)
        self.scale = float(scale)
        self.threshold = float(threshold)
        self.samples = deque(maxlen=4096)
        self.integral = 0.0
        self.end = None
        self.last_input = None
        self.first = None
        self.bias = None
        self.intercept = None
        self.lost = False
        self.sample_key = None
        self.cached = None

    def invalidate_context(self):
        if self.bias is None:
            self.samples.clear()
            self.first = None

    def ingest(self, acceleration, start, end):
        if (
            not finite(acceleration, -30, 30)
            or type(start) is not int
            or type(end) is not int
            or not 0 <= start < end
            or end - start > 2000
        ):
            raise ValueError("bounded causal anchor interval required")
        key = (float(acceleration), start, end)
        if end == self.end:
            if key != self.last_input:
                raise ValueError("conflicting same-time anchor input")
            return True
        if self.end is not None and end < self.end:
            raise ValueError("anchor input rollback")
        covered = self.end is None or start == self.end
        if not covered:
            if self.bias is not None:
                self.lost = True
            else:
                self.invalidate_context()
        self.integral += float(acceleration) * (end - start) / 1000
        self.end = end
        self.last_input = key
        return covered

    def sample(self, gnss, wheel, end):
        if (
            not finite(gnss, 0, 100)
            or not finite(wheel, 0, 100)
            or type(end) is not int
            or self.end != end
        ):
            raise ValueError("current bounded anchor sample required")
        key = (float(gnss), float(wheel), end)
        if self.sample_key is not None and end <= self.sample_key[2]:
            if key != self.sample_key:
                raise ValueError("old or conflicting anchor sample")
            return self.cached
        if self.first is None:
            self.first = end
        t = (end - self.first) / 1000
        g = float(gnss) - self.gnss_offset
        w = float(wheel)
        if self.bias is None:
            self.samples.append((t, self.integral - (g + w) / 2))
            if t >= 5 and len(self.samples) >= 10:
                mt = sum(p[0] for p in self.samples) / len(self.samples)
                my = sum(p[1] for p in self.samples) / len(self.samples)
                variance = sum((p[0] - mt) ** 2 for p in self.samples)
                covariance = sum((p[0] - mt) * (p[1] - my) for p in self.samples)
                self.bias = max(-3.0, min(3.0, covariance / variance))
                self.intercept = my - self.bias * mt
        ready = self.bias is not None
        error = 0.0
        if ready:
            prediction = self.integral - self.bias * t - self.intercept
            error = max(abs(g - prediction), abs(w - prediction)) / math.sqrt(max(1.0, t))
        score = error / self.scale
        if not math.isfinite(score):
            raise ValueError("nonfinite session score")
        result = AnchorResult(
            ready, self.lost, self.lost or (ready and score > self.threshold), score, self.bias
        )
        self.sample_key = key
        self.cached = result
        return result
