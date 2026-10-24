"""Causal development pairing. Neither covariance nor agreement certifies physical truth."""
import math, statistics
def causal_sample(samples, receipt_ns, sensor_ns, max_age_ns):
    if max_age_ns < 0:
        raise ValueError("Negative age limit")
    previous = None
    chosen = None
    for row in samples:
        receipt, sensor, value = row
        if previous is not None and receipt < previous:
            raise ValueError("Samples must be sorted by receipt timestamp")
        previous = receipt
        if receipt <= receipt_ns and sensor <= sensor_ns:
            if 0 <= sensor_ns - sensor <= max_age_ns:
                chosen = row
    return chosen
def planar_speed(vx, vy):
    if not all(math.isfinite(v) for v in (vx, vy)):
        raise ValueError("Nonfinite velocity")
    return math.hypot(vx, vy)
def wheel_median(values):
    if len(values) != 4 or not all(math.isfinite(v) for v in values):
        raise ValueError("Exactly four finite wheel values required")
    return statistics.median(values)
