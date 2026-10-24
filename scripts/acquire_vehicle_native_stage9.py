"""Resume fixed metadata-selected Civic NPY entry downloads; no detector evaluation."""

import concurrent.futures
import hashlib
import json
import re
import shutil
from pathlib import Path

from vehicle_native_archive import RangeArchive, validate_npy, verify_metadata

CHANNELS = {"GNSS/live_gnss_ublox": 6, "IMU/accelerometer": 3, "CAN/wheel_speed": 4}
ROOT = Path(__file__).resolve().parents[1]


def check_index(plan, index):
    expected = {
        f"Chunk_3/{t['route_id']}/{seg}/processed_log/{c}/{s}": (t["id"], seg)
        for t in plan["trips"]
        for seg in t["segments"]
        for c in CHANNELS
        for s in ("t", "value")
    }
    seen = set()
    for entry in index["entries"]:
        n = entry["name"]
        if n not in expected or n in seen or (entry["trip_id"], entry["segment"]) != expected[n]:
            raise ValueError("archive index route/entry binding changed")
        seen.add(n)
    if seen != set(expected):
        raise ValueError("fixed native entries missing")


def materialize(root, entry, data):
    if (
        not re.fullmatch(r"civic_[0-9]{2}", entry["trip_id"])
        or type(entry["segment"]) is not int
        or not 0 <= entry["segment"] <= 10000
    ):
        raise ValueError("invalid Windows-safe segment identity")
    name = entry["name"].split("/processed_log/", 1)[-1]
    if name not in {c + "/" + s for c in CHANNELS for s in ("t", "value")}:
        raise ValueError("source outside native core channels")
    channel, suffix = name.rsplit("/", 1)
    array = validate_npy(data, None if suffix == "t" else CHANNELS[channel])
    path = (
        Path(root)
        / "data/raw/vehicle_native_stage9"
        / entry["trip_id"]
        / f"segment_{entry['segment']}"
        / "processed_log"
        / name
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return dict(
        path="processed_log/" + name,
        bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        shape=list(array.shape),
        source_zip_entry=entry["name"],
        entry_crc32=entry["crc32"],
        local_header_offset=entry["header_offset"],
        compressed_bytes=entry["compressed_size"],
    )


def main():
    plan = json.loads((ROOT / "VEHICLE_STAGE9_SELECTION.json").read_bytes())
    index = json.loads((ROOT / "VEHICLE_STAGE9_ARCHIVE_INDEX.json").read_bytes())
    check_index(plan, index)
    verify_metadata(ROOT, plan, index)
    reader = RangeArchive(
        index["archive_url"], index["archive_size"], ROOT / "data/cache/vehicle_native_stage9"
    )

    def capture(entry):
        data = reader.entry(entry)
        record = materialize(ROOT, entry, data)
        return entry["trip_id"], entry["segment"], record

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        result = list(pool.map(capture, index["entries"]))
    grouped = {}
    for trip, segment, record in result:
        grouped.setdefault((trip, segment), []).append(record)
    entries = []
    for trip in plan["trips"]:
        parts = []
        for segment in trip["segments"]:
            folder = ROOT / "data/raw/vehicle_native_stage9" / trip["id"] / f"segment_{segment}"
            parts.append(folder.relative_to(ROOT).as_posix())
            manifest = dict(
                source_url=index["archive_url"],
                retrieved_utc_date="2026-10-14",
                route_id=trip["route_id"],
                segment_id=trip["route_id"] + "/" + str(segment),
                central_directory_sha256=index["central_directory_sha256"],
                archive_lfs_sha256_recorded_not_fully_verified=index[
                    "archive_lfs_sha256_recorded_not_fully_verified"
                ],
                native_values_note=(
                    "Original NPY entry bytes; ZIP entry CRC and local SHA verified; "
                    "full archive LFS hash not verified"
                ),
                files=sorted(grouped[trip["id"], segment], key=lambda r: r["path"]),
            )
            (folder / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        entries.append(dict(trip, parts=parts))
        print("Acquired", trip["id"], trip["route_id"], trip["segments"], flush=True)
    base = ROOT / "data/raw/vehicle_native_stage9"
    shutil.copy2(
        ROOT / "data/reference/vehicle_native_stage9/LICENSE_COMMA2K19.txt",
        base / "LICENSE_COMMA2K19.txt",
    )
    manifest = dict(
        dataset="comma2k19 full release Civic subset",
        vehicle="Honda Civic",
        vehicle_id="99c94dc769b5d96e",
        trips=entries,
        selection_rule=plan["selection_rule"],
        limitations=[
            "Second vehicle in same dataset/logger/highway family, not independent dataset",
            "One Civic cannot establish population generalization",
            "Presumed benign recorded motion plus controlled perturbations",
            "Source receipt clocks are not physical/hardware attestation",
            "Original RAV4 policies transferred without Civic fitting",
        ],
    )
    (base / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("84native arrays acquired and hashed; no policy outcomes evaluated.", flush=True)


if __name__ == "__main__":
    main()
