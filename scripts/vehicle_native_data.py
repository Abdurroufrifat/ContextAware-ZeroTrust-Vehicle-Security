"""Hash-checked native comma2k19 channels with causal, receipt-clock alignment."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REQUIRED = {
    "gnss": ("GNSS/live_gnss_ublox", 6),
    "imu": ("IMU/accelerometer", 3),
    "wheel": ("CAN/wheel_speed", 4),
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def inside(root, path):
    root = Path(root).resolve()
    path = Path(path)
    if not path.is_absolute():
        path = root / path
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("native path escapes project")
    if path.is_symlink():
        raise ValueError("native symlink rejected")
    return resolved


def validate_chunk(root, manifest):
    files = manifest["files"]
    if not isinstance(files, list) or not 1 <= len(files) <= 64:
        raise ValueError("bounded file list required")
    seen = {}
    total = 0
    for item in files:
        name = item["path"]
        if (
            type(name) is not str
            or name.startswith("/")
            or "\\" in name
            or ".." in name.split("/")
            or name in seen
        ):
            raise ValueError("invalid native file path")
        p = inside(root, name)
        size = p.stat().st_size
        total += size
        if size > 64 * 1024 * 1024 or total > 128 * 1024 * 1024:
            raise ValueError("native chunk exceeds bound")
        h = item["sha256"]
        if type(h) is not str or len(h) != 64 or sha(p.read_bytes()) != h:
            raise ValueError("native array hash mismatch")
        arr = np.load(p, allow_pickle=False, mmap_mode="r")
        if (
            arr.dtype.kind not in "fiu"
            or arr.ndim not in (1, 2)
            or list(arr.shape) != item["shape"]
        ):
            raise ValueError("native array shape/type mismatch")
        seen[name] = h
    return seen


def merge_channel(chunks):
    for t, v in chunks:
        if (
            t.ndim != 1
            or v.ndim != 2
            or len(t) != len(v)
            or len(t) < 2
            or not np.isfinite(t).all()
            or not np.isfinite(v).all()
        ):
            raise ValueError("finite native channel rows required")
        if not np.all(np.diff(t) > 0):
            raise ValueError("native chunk timestamps must increase")
    t = np.concatenate([x[0] for x in chunks])
    v = np.concatenate([x[1] for x in chunks])
    order = np.argsort(t, kind="stable")
    t = t[order]
    v = v[order]
    keep = np.ones(len(t), dtype=bool)
    for i in np.flatnonzero(np.diff(t) == 0) + 1:
        if not np.array_equal(v[i], v[i - 1]):
            raise ValueError("conflicting native event at equal time")
        keep[i] = False
    return t[keep], v[keep]


def causal_hold(t, v, start, end):
    if not math.isfinite(start) or not math.isfinite(end) or not start < end:
        raise ValueError("positive finite aggregation interval required")
    if t.ndim != 1 or v.ndim != 1 or len(t) != len(v) or not np.all(np.diff(t) > 0):
        raise ValueError("ordered scalar channel required")
    first = int(np.searchsorted(t, start, side="right")) - 1
    last = int(np.searchsorted(t, end, side="right")) - 1
    if first < 0 or last < first:
        raise ValueError("missing causal interval coverage")
    boundaries = np.concatenate(([start], t[first + 1 : last + 1], [end]))
    values = v[first : last + 1]
    widths = np.diff(boundaries)
    mean = float(np.dot(widths, values) / (end - start))
    raw_gaps = np.diff(t[first : last + 1])
    maxgap = float(
        max(np.max(widths), start - t[first], np.max(raw_gaps) if len(raw_gaps) else 0.0)
    )
    if not math.isfinite(mean):
        raise ValueError("nonfinite aggregation")
    return mean, float(t[last]), maxgap


@dataclass(frozen=True)
class NativeFrame:
    trip_id: str
    frame: int
    time_ms: int
    interval_start_ms: int
    gnss_time_ms: int
    imu_time_ms: int
    wheel_time_ms: int
    gnss_speed: float
    wheel_speed: float
    imu_accel: float
    maximum_gap_ms: int


def align_channels(trip_id, gnss, imu, wheel, stride=2, edge_s=2.0):
    if type(stride) is not int or not 1 <= stride <= 10 or not 0 <= edge_s <= 10:
        raise ValueError("bounded alignment settings required")
    channels = [merge_channel([c]) for c in (gnss, imu, wheel)]
    (gt, gv), (it, iv), (wt, wv) = channels
    if gv.shape[1] != 6 or iv.shape[1] != 3 or wv.shape[1] != 4:
        raise ValueError("wrong native channel schema")
    # UTC field is retained in raw input; decision time uses the common receipt clock.
    if np.any(gv[:, 3] < 1e11) or np.any(np.diff(gv[:, 3]) < 0):
        raise ValueError("live GNSS UTC milliseconds required")
    origin = min(gt[0], it[0], wt[0])
    start = max(gt[0], it[0], wt[0]) + edge_s
    end = min(gt[-1], it[-1], wt[-1]) - edge_s
    if start >= end:
        raise ValueError("no native interior overlap")

    def ms(t):
        return int(math.floor((t - origin) * 1000 + 1e-7))

    chosen = np.arange(0, len(gt), stride)
    frames = []
    for previous, index in zip(chosen[:-1], chosen[1:], strict=True):
        now = float(gt[index])
        if not start <= now <= end:
            continue
        a, last, gap = causal_hold(it, iv[:, 0], float(gt[previous]), now)
        wi = int(np.searchsorted(wt, now, side="right")) - 1
        if wi < 0:
            raise ValueError("missing native wheel history")
        g = float(gv[index, 2])
        w = float(np.mean(wv[wi]))
        if not 0 <= g <= 100 or not 0 <= w <= 100 or not -30 <= a <= 30:
            raise ValueError("native longitudinal value outside declared domain")
        frames.append(
            NativeFrame(
                trip_id,
                len(frames),
                ms(now),
                ms(gt[previous]),
                ms(now),
                ms(last),
                ms(wt[wi]),
                g,
                w,
                a,
                int(math.ceil(gap * 1000 - 1e-7)),
            )
        )
    if not frames:
        raise ValueError("insufficient native epoch coverage")
    return frames


def load_trip(root, entry, stride=2, edge_s=2.0):
    parts = entry["parts"]
    if not isinstance(parts, list) or not 1 <= len(parts) <= 2:
        raise ValueError("one or two contiguous parts required")
    collected = {k: [] for k in REQUIRED}
    previous_bounds = None
    for part in parts:
        folder = inside(root, part)
        manifest = json.loads((folder / "source_manifest.json").read_bytes())
        if manifest["route_id"] != entry["route_id"]:
            raise ValueError("native route binding mismatch")
        hashes = validate_chunk(folder, manifest)
        for k, (channel, columns) in REQUIRED.items():
            tname = f"processed_log/{channel}/t"
            vname = f"processed_log/{channel}/value"
            if tname not in hashes or vname not in hashes:
                raise ValueError("required native channel absent from manifest")
            t = np.load(folder / tname, allow_pickle=False)
            v = np.load(folder / vname, allow_pickle=False)
            if v.ndim != 2 or v.shape[1] != columns:
                raise ValueError("wrong required native shape")
            collected[k].append((t, v))
        gt, gv = collected["gnss"][-1]
        bounds = (float(gt[0]), float(gt[-1]), float(gv[0, 3]), float(gv[-1, 3]))
        if previous_bounds is not None:
            bootgap = bounds[0] - previous_bounds[1]
            utcgap = (bounds[2] - previous_bounds[3]) / 1000
            if not 0 <= bootgap <= 0.5 or not 0 <= utcgap <= 0.5 or abs(bootgap - utcgap) > 0.1:
                raise ValueError("native part clock continuity failed")
        previous_bounds = bounds
    merged = [merge_channel(collected[k]) for k in ("gnss", "imu", "wheel")]
    frames = align_channels(entry["route_id"], *merged, stride=stride, edge_s=edge_s)
    if len(frames) < 10:
        raise ValueError("insufficient native trip coverage")
    return frames
