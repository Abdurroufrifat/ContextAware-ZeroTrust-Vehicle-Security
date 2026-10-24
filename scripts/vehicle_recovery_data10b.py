"""Metadata-only future-date selection; native arrays use unchanged Stage9 alignment."""

import hashlib
import json
from pathlib import Path

from vehicle_native_archive import central_location, parse_central, validate_range
from vehicle_transfer_data import load_transfer_trip

CHANNELS = {"GNSS/live_gnss_ublox": 6, "IMU/accelerometer": 3, "CAN/wheel_speed": 4}


def select_routes(records, prior_dates, prior_routes):
    routes = {}
    for e in records:
        p = e["name"].split("/")
        if (
            len(p) >= 4
            and p[0] in ("Chunk_5", "Chunk_6", "Chunk_7")
            and p[1].startswith("99c94dc769b5d96e|")
        ):
            try:
                segment = int(p[2])
            except ValueError:
                continue
            routes.setdefault((p[1], int(p[0].split("_")[1])), set()).add(segment)
    result = []
    days = set(prior_dates)
    for (route, archive_id), segs in sorted(routes.items()):
        day = route.split("|", 1)[1][:10]
        pair = next((s for s in sorted(segs) if s + 1 in segs), None)
        if day in days or route in prior_routes or pair is None:
            continue
        days.add(day)
        i = len(result)
        result.append(
            dict(
                id=f"civic10b_{i + 1:02d}",
                archive_id=archive_id,
                route_id=route,
                day=day,
                segments=[pair, pair + 1],
                phase="calibration" if i < 3 else "validation" if i == 3 else "test",
            )
        )
        if len(result) == 9:
            break
    if len(result) != 9:
        raise ValueError("nine distinct new eligible dates required")
    return result


def selected_index(records, trips):
    lookup = {e["name"]: e for e in records}
    out = []
    for trip in trips:
        for segment in trip["segments"]:
            for channel in CHANNELS:
                for suffix in ("t", "value"):
                    name = (
                        f"Chunk_{trip['archive_id']}/{trip['route_id']}/{segment}/"
                        f"processed_log/{channel}/{suffix}"
                    )
                    if name not in lookup:
                        raise ValueError("predeclared native core entry missing")
                    out.append(
                        dict(
                            lookup[name],
                            trip_id=trip["id"],
                            segment=segment,
                            archive_id=trip["archive_id"],
                        )
                    )
    return out


def check_index(trips, entries):
    expected = {
        f"Chunk_{t['archive_id']}/{t['route_id']}/{s}/processed_log/{c}/{n}": (t["id"], s)
        for t in trips
        for s in t["segments"]
        for c in CHANNELS
        for n in ("t", "value")
    }
    seen = set()
    for e in entries:
        name = e["name"]
        if name not in expected or name in seen or (e["trip_id"], e["segment"]) != expected[name]:
            raise ValueError("native entry/route/segment binding mismatch")
        seen.add(name)
    if seen != set(expected):
        raise ValueError("fixed native entry missing")


def verify_metadata(root):
    root = Path(root)
    base = root / "data/reference/vehicle_native_stage10b"
    index = json.loads((root / "VEHICLE_STAGE10B_INDEX.json").read_bytes())
    plan = json.loads((root / "VEHICLE_STAGE10B_SELECTION.json").read_bytes())
    receipt = json.loads((base / "archive_metadata_receipt.json").read_bytes())
    all_records = []
    counts = 0
    for key, archive in index["archives"].items():
        rec = receipt["archives"][key]
        if rec["url"] != archive["archive_url"] or rec["total"] != archive["archive_size"]:
            raise ValueError("archive identity changed")
        data = {}
        for name, r in rec["ranges"].items():
            b = (base / name).read_bytes()
            if hashlib.sha256(b).hexdigest() != r["sha256"]:
                raise ValueError("metadata bytes changed")
            validate_range(b, 206, r["content_range"], r["start"], r["end"], rec["total"])
            data[name] = b
        tail = f"chunk{key}_tail.bin"
        central_name = f"chunk{key}_central.bin"
        off, size, count = central_location(data[tail], rec["total"])
        central = data[central_name]
        if (
            rec["ranges"][central_name]["start"] != off
            or len(central) != size
            or hashlib.sha256(central).hexdigest() != archive["central_directory_sha256"]
        ):
            raise ValueError("central evidence mismatch")
        all_records.extend(parse_central(central, count))
        counts += count
    prior = []
    for name in ("VEHICLE_STAGE9_SELECTION.json", "VEHICLE_STAGE10_SELECTION.json"):
        prior.extend(json.loads((root / name).read_bytes())["trips"])
    selected = select_routes(all_records, {t["day"] for t in prior}, {t["route_id"] for t in prior})
    if selected != plan["trips"] or selected_index(all_records, selected) != index["entries"]:
        raise ValueError("selection/index changed")
    check_index(selected, index["entries"])
    return dict(
        inventory_entries=counts,
        selected_entries=len(index["entries"]),
        selected_routes=len(selected),
    )


def load_trip(root, entry):
    return load_transfer_trip(root, entry)
