"""Shared causal receipt view for native CAN updates with identical timestamps.

Raw arrays are unchanged. Stable native log order resolves bounded same-receipt
wheel updates. This does not establish physical capture order or trusted truth.
GNSS/IMU remain subject to the frozen strict timestamp checks.
"""

import json

import numpy as np
from vehicle_native_data import REQUIRED, align_channels, inside, merge_channel, validate_chunk


def wheel_receipt_view(t, v):
    if (
        t.ndim != 1
        or v.shape != (len(t), 4)
        or len(t) < 2
        or not np.isfinite(t).all()
        or not np.isfinite(v).all()
        or np.any(np.diff(t) < 0)
    ):
        raise ValueError("finite nondecreasing native wheel receipt stream required")
    keep = []
    duplicates = 0
    maximum = 0.0
    i = 0
    while i < len(t):
        j = i + 1
        while j < len(t) and t[j] == t[i]:
            j += 1
        spread = float(np.max(np.ptp(v[i:j], axis=0)))
        if j - i > 4 or spread > 1.0:
            raise ValueError("ambiguous native wheel receipt burst exceeds declared bound")
        duplicates += j - i - 1
        maximum = max(maximum, spread)
        keep.append(j - 1)
        i = j
    return (
        t[keep],
        v[keep],
        dict(
            duplicate_updates=duplicates,
            maximum_component_spread_mps=maximum,
            rule="last native log update at same receipt timestamp",
        ),
    )


def load_integrity_trip(root, entry, stride=2, edge_s=2.0):
    if len(entry["parts"]) != 1:
        raise ValueError("Stage7 requires one native segment per trip")
    folder = inside(root, entry["parts"][0])
    manifest = json.loads((folder / "source_manifest.json").read_bytes())
    if manifest["route_id"] != entry["route_id"]:
        raise ValueError("native route binding mismatch")
    hashes = validate_chunk(folder, manifest)
    channels = []
    for source, (name, columns) in REQUIRED.items():
        tname = f"processed_log/{name}/t"
        vname = f"processed_log/{name}/value"
        if tname not in hashes or vname not in hashes:
            raise ValueError("required native source missing")
        t = np.load(folder / tname, allow_pickle=False)
        v = np.load(folder / vname, allow_pickle=False)
        if v.ndim != 2 or v.shape[1] != columns:
            raise ValueError("native source shape mismatch")
        if source == "wheel":
            t, v, _ = wheel_receipt_view(t, v)
        channels.append(merge_channel([(t, v)]))
    frames = align_channels(entry["route_id"], *channels, stride=stride, edge_s=edge_s)
    if len(frames) < 10:
        raise ValueError("insufficient native trip coverage")
    return frames
