"""Protected causal history is a conditional reference, not physical attestation."""

import math
from collections import deque
from dataclasses import dataclass

from .native_detectors import finite


@dataclass(frozen=True)
class ReferenceCheck:
    eligible: bool
    present: bool
    lost: bool
    expired: bool
    age_ms: int
    prediction: float | None
    gnss_error: float | None
    wheel_error: float | None
    anchor_time_ms: int = -1
    snapshot_speed: float | None = None
    snapshot_bias: float | None = None


class ProtectedReference:
    def __init__(self, error_bound, lag_ms=2000, maximum_age_ms=90000):
        if (
            not finite(error_bound, 0.001, 100)
            or type(lag_ms) is not int
            or type(maximum_age_ms) is not int
            or not 0 < lag_ms < maximum_age_ms <= 120000
        ):
            raise ValueError("bounded protected reference parameters required")
        self.error_bound = float(error_bound)
        self.lag_ms = lag_ms
        self.maximum_age_ms = maximum_age_ms
        self.history = deque(maxlen=512)
        self.end = None
        self.last_input = None
        self.integral = 0.0
        self.anchor = None
        self.active = False
        self.lost = False

    def ingest(self, acceleration, start, end):
        if (
            not finite(acceleration, -30, 30)
            or type(start) is not int
            or type(end) is not int
            or not 0 <= start < end
            or end - start > 2000
        ):
            raise ValueError("bounded causal protected interval required")
        key = float(acceleration), start, end
        if self.end is not None:
            if end == self.end:
                if key != self.last_input:
                    raise ValueError("conflicting same-time protected input")
                return
            if end < self.end:
                raise ValueError("protected input rollback")
            if start != self.end:
                self.history.clear()
                if self.active:
                    self.lost = True
        self.integral += float(acceleration) * (end - start) / 1000
        self.end, self.last_input = end, key

    def observe(self, speed, bias, end, quiet):
        if (
            not finite(speed, -100, 100)
            or not finite(bias, -30, 30)
            or type(end) is not int
            or end != self.end
            or type(quiet) is not bool
        ):
            raise ValueError("current finite protected snapshot required")
        if quiet and not self.active:
            snapshot = end, float(speed), float(bias), self.integral
            if self.history and self.history[-1][0] == end:
                if snapshot != self.history[-1]:
                    raise ValueError("conflicting same-time protected snapshot")
                return
            self.history.append(snapshot)

    def freeze(self, end):
        if type(end) is not int or end != self.end:
            raise ValueError("current quarantine onset required")
        if not self.active:
            self.anchor = next(
                (s for s in reversed(self.history) if s[0] <= end - self.lag_ms), None
            )
            self.active = True
            self.lost = False

    def check(self, gnss, wheel, end):
        if (
            not finite(gnss, -10, 110)
            or not finite(wheel, 0, 100)
            or type(end) is not int
            or end != self.end
        ):
            raise ValueError("current finite protected speeds required")
        if not self.active or self.anchor is None:
            return ReferenceCheck(False, False, self.lost, False, -1, None, None, None)
        time, speed, bias, integral = self.anchor
        age = end - time
        prediction = speed + self.integral - integral - bias * age / 1000
        eg, ew = abs(float(gnss) - prediction), abs(float(wheel) - prediction)
        expired = age > self.maximum_age_ms
        return ReferenceCheck(
            not self.lost
            and not expired
            and (
                max(eg, ew) <= self.error_bound
                or math.isclose(max(eg, ew), self.error_bound, rel_tol=0.0, abs_tol=1e-12)
            ),
            True,
            self.lost,
            expired,
            age,
            prediction,
            eg,
            ew,
            time,
            speed,
            bias,
        )

    def release(self):
        self.history.clear()
        self.anchor = None
        self.active = self.lost = False
