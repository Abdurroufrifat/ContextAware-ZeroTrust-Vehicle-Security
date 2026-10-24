"""Two-part native continuity with unchanged bounded wheel receipt interpretation."""

import json

import numpy as np
from vehicle_integrity_data import wheel_receipt_view
from vehicle_native_data import REQUIRED, align_channels, inside, merge_channel, validate_chunk


def load_transfer_trip(root, entry, stride=2, edge_s=2.0):
    parts = entry["parts"]
    if not isinstance(parts, list) or len(parts) != 2:
        raise ValueError("exactly two native segments required")
    collected = {k: [] for k in REQUIRED}
    previous = None
    for part in parts:
        folder = inside(root, part)
        manifest = json.loads((folder / "source_manifest.json").read_bytes())
        if manifest["route_id"] != entry["route_id"]:
            raise ValueError("native route binding mismatch")
        hashes = validate_chunk(folder, manifest)
        for k, (channel, columns) in REQUIRED.items():
            tn = "processed_log/" + channel + "/t"
            vn = "processed_log/" + channel + "/value"
            if tn not in hashes or vn not in hashes:
                raise ValueError("native channel missing")
            t = np.load(folder / tn, allow_pickle=False)
            v = np.load(folder / vn, allow_pickle=False)
            if v.ndim != 2 or v.shape[1] != columns:
                raise ValueError("wrong native shape")
            if k == "wheel":
                t, v, _ = wheel_receipt_view(t, v)
            collected[k].append((t, v))
        gt, gv = collected["gnss"][-1]
        bounds = (float(gt[0]), float(gt[-1]), float(gv[0, 3]), float(gv[-1, 3]))
        if previous is not None:
            boot = bounds[0] - previous[1]
            utc = (bounds[2] - previous[3]) / 1000
            if not 0 <= boot <= 0.5 or not 0 <= utc <= 0.5 or abs(boot - utc) > 0.1:
                raise ValueError("native part clock continuity failed")
        previous = bounds
    merged = [merge_channel(collected[k]) for k in ("gnss", "imu", "wheel")]
    frames = align_channels(entry["route_id"], *merged, stride=stride, edge_s=edge_s)
    if len(frames) < 10:
        raise ValueError("insufficient native trip coverage")
    return frames
