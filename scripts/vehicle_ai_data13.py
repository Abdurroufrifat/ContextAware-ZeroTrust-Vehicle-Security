"""Reserved unused-route data and causal normal-only AI windows."""

import hashlib
from dataclasses import dataclass

import numpy as np
from vehicle_transfer_data import load_transfer_trip

from coact_cps.vehicle.native_detectors import ModelConfig

CHANNELS = {"GNSS/live_gnss_ublox": 6, "IMU/accelerometer": 3, "CAN/wheel_speed": 4}


def select_trips(records, prior):
    names = {r["name"] for r in records}
    routes = {}
    for name in names:
        p = name.split("/")
        if (
            len(p) > 4
            and p[0] in {"Chunk_8", "Chunk_9"}
            and p[1].startswith("99c94dc769b5d96e|")
            and p[2].isdigit()
        ):
            routes.setdefault((p[1], int(p[0].split("_")[1])), set()).add(int(p[2]))
    days = {t.get("day", t.get("route_id", "").split("|")[-1][:10]) for t in prior}
    prior_routes = {t.get("route_id") for t in prior}
    result = []
    for (route, arc), segments in sorted(routes.items()):
        day = route.split("|")[1][:10]
        if day in days or route in prior_routes:
            continue
        pair = next(
            (
                s
                for s in sorted(segments)
                if s + 1 in segments
                and all(
                    f"Chunk_{arc}/{route}/{j}/processed_log/{c}/{v}" in names
                    for j in (s, s + 1)
                    for c in CHANNELS
                    for v in ("t", "value")
                )
            ),
            None,
        )
        if pair is None:
            continue
        i = len(result)
        phase = (
            "train" if i < 4 else ["scale", "calibrate", "validation"][i - 4] if i < 7 else "test"
        )
        result.append(
            dict(
                id=f"civic13_{i + 1:02d}",
                archive_id=arc,
                route_id=route,
                day=day,
                segments=[pair, pair + 1],
                phase=phase,
            )
        )
        days.add(day)
        if len(result) == 12:
            break
    if len(result) != 12:
        raise ValueError("twelve_unused_dates_required")
    return result


def load_trip(root, trip):
    return load_transfer_trip(root, trip)


def features(frames, model=None):
    model = model or ModelConfig()
    x = np.array(
        [
            [f.gnss_speed - model.gnss_offset, f.wheel_speed, f.imu_accel - model.imu_offset]
            for f in frames
        ],
        dtype=np.float64,
    )
    if x.ndim != 2 or x.shape[1] != 3 or not np.isfinite(x).all():
        raise ValueError("finite_three_channel_frames_required")
    return x


@dataclass
class Windows:
    x: np.ndarray
    positions: np.ndarray
    dt: np.ndarray


def windows(frames, window=32, model=None):
    if type(window) is not int or not 2 <= window <= 256:
        raise ValueError("bounded_window_required")
    values = features(frames, model)
    start = 0
    xs = []
    positions = []
    dt = []
    for i, f in enumerate(frames):
        if type(f.time_ms) is not int or not 0 < f.time_ms - f.interval_start_ms <= 2000:
            raise ValueError("invalid_interval")
        if i and (
            f.trip_id != frames[i - 1].trip_id or f.interval_start_ms != frames[i - 1].time_ms
        ):
            start = i
        if i - start + 1 >= window:
            xs.append(values[i - window + 1 : i + 1])
            positions.append(i)
            dt.append((f.time_ms - f.interval_start_ms) / 1000)
    return Windows(
        np.asarray(xs, dtype=np.float32).reshape(-1, window, 3),
        np.asarray(positions, dtype=int),
        np.asarray(dt, dtype=np.float32),
    )


@dataclass(frozen=True)
class Scaling:
    mean: tuple
    scale: tuple

    @classmethod
    def fit(cls, x):
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != 3 or len(x) < 2 or not np.isfinite(x).all():
            raise ValueError("finite_training_rows_required")
        return cls(tuple(x.mean(0)), tuple(np.maximum(x.std(0), 1e-6)))

    def transform(self, x):
        x = np.asarray(x, dtype=float)
        if x.shape[-1] != 3 or not np.isfinite(x).all():
            raise ValueError("finite_features_required")
        return ((x - np.asarray(self.mean)) / np.asarray(self.scale)).astype(np.float32)

    def to_dict(self):
        return dict(mean=list(self.mean), scale=list(self.scale))


def array_hash(x):
    x = np.ascontiguousarray(x)
    return hashlib.sha256(str(x.dtype).encode() + str(x.shape).encode() + x.tobytes()).hexdigest()
