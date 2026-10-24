"""Acquire only preselected pinned native entries; no outcome evaluation."""

import argparse
import concurrent.futures
import hashlib
import json
import time
import urllib.error
from pathlib import Path

from vehicle_ai_data13 import CHANNELS
from vehicle_native_archive import RangeArchive, validate_npy


def acquire(root):
    root = Path(root)
    plan = json.loads((root / "VEHICLE_STAGE13_SELECTION.json").read_bytes())
    index = json.loads((root / "VEHICLE_STAGE13_INDEX.json").read_bytes())
    readers = {
        int(k): RangeArchive(
            a["archive_url"], a["archive_size"], root / "data/cache/vehicle_native_stage13" / k
        )
        for k, a in index["archives"].items()
    }

    def capture(e):
        for attempt in range(3):
            try:
                data = readers[e["archive_id"]].entry(e)
                break
            except urllib.error.URLError:
                if attempt == 2:
                    raise
                time.sleep(0.5 * (attempt + 1))
        name = e["name"].split("/processed_log/", 1)[1]
        channel, suffix = name.rsplit("/", 1)
        if channel not in CHANNELS or suffix not in {"t", "value"}:
            raise ValueError("unexpected_channel")
        arr = validate_npy(data, None if suffix == "t" else CHANNELS[channel])
        part = root / "data/raw/vehicle_native_stage13" / e["trip_id"] / f"segment_{e['segment']}"
        dest = part / "processed_log" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return (e["trip_id"], e["segment"]), dict(
            path="processed_log/" + name,
            sha256=hashlib.sha256(data).hexdigest(),
            bytes=len(data),
            shape=list(arr.shape),
            source_zip_entry=e["name"],
            entry_crc32=e["crc32"],
        )

    grouped = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for key, record in pool.map(capture, index["entries"]):
            grouped.setdefault(key, []).append(record)
    trips = []
    for t in plan["trips"]:
        parts = []
        arc = index["archives"][str(t["archive_id"])]
        for segment in t["segments"]:
            part = root / "data/raw/vehicle_native_stage13" / t["id"] / f"segment_{segment}"
            value = dict(
                route_id=t["route_id"],
                segment_id=t["route_id"] + "/" + str(segment),
                source_url=arc["archive_url"],
                central_directory_sha256=arc["central_directory_sha256"],
                archive_lfs_sha256_recorded_not_fully_verified=arc[
                    "archive_lfs_sha256_recorded_not_fully_verified"
                ],
                files=sorted(grouped[t["id"], segment], key=lambda r: r["path"]),
            )
            (part / "source_manifest.json").write_text(json.dumps(value, indent=2) + "\n")
            parts.append(part.relative_to(root).as_posix())
        trips.append(dict(t, parts=parts))
        print("Acquired", t["id"], flush=True)
    (root / "data/raw/vehicle_native_stage13/manifest.json").write_text(
        json.dumps(
            dict(
                dataset="comma2k19",
                trips=trips,
                limitations=[
                    "same vehicle/logger/highway",
                    "controlled perturbations only",
                    "whole ZIP LFS hashes not fully verified",
                ],
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    acquire(p.parse_args().root)
