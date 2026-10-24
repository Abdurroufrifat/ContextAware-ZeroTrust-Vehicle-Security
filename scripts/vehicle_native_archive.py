"""Bounded original-entry extraction; partial downloads do not verify whole ZIPs."""

import hashlib
import io
import json
import math
import struct
import urllib.parse
import urllib.request
import zipfile
import zlib
from pathlib import Path, PurePosixPath

import numpy as np

MAX_RANGE = 8 * 1024 * 1024
MAX_ARRAY = 64 * 1024 * 1024


def validate_range(data, status, header, start, end, total):
    if (
        any(type(x) is not int for x in (start, end, total))
        or not 0 <= start <= end < total
        or end - start + 1 > MAX_RANGE
    ):
        raise ValueError("bounded archive range required")
    if status != 206 or header != f"bytes {start}-{end}/{total}" or len(data) != end - start + 1:
        raise ValueError("archive range status/offset/length mismatch")
    return data


def extract_entry(blob, entry):
    name = entry["name"]
    p = PurePosixPath(name)
    if (
        type(name) is not str
        or p.is_absolute()
        or ".." in p.parts
        or "\\" in name
        or len(name) > 2048
    ):
        raise ValueError("unsafe archive entry name")
    if (
        any(type(entry[k]) is not int for k in ("size", "compressed_size", "method", "crc32"))
        or not 0 < entry["size"] <= MAX_ARRAY
        or not 0 < entry["compressed_size"] <= MAX_RANGE
    ):
        raise ValueError("bounded archive entry required")
    if entry["method"] not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED) or len(blob) < 30:
        raise ValueError("unsupported or truncated ZIP entry")
    signature, version, flags, method, mtime, mdate, crc, compressed, size, nlen, xlen = (
        struct.unpack_from("<4s5H3L2H", blob)
    )
    if (
        signature != b"PK\x03\x04"
        or flags & (1 | 32 | 64 | 8192)
        or method != entry["method"]
        or xlen > 4096
    ):
        raise ValueError("invalid/encrypted ZIP local header")
    offset = 30 + nlen + xlen
    if len(blob) < offset + entry["compressed_size"]:
        raise ValueError("truncated ZIP payload")
    try:
        actual = blob[30 : 30 + nlen].decode("utf-8" if flags & 2048 else "cp437")
    except UnicodeError as ex:
        raise ValueError("invalid ZIP filename encoding") from ex
    if actual != name:
        raise ValueError("central/local filename mismatch")
    if not flags & 8 and (
        crc != entry["crc32"] or compressed != entry["compressed_size"] or size != entry["size"]
    ):
        raise ValueError("central/local size or CRC mismatch")
    data = blob[offset : offset + entry["compressed_size"]]
    if method == zipfile.ZIP_DEFLATED:
        try:
            d = zlib.decompressobj(-15)
            data = d.decompress(data, entry["size"] + 1)
            if not d.eof or d.unconsumed_tail or d.unused_data:
                raise ValueError("invalid deflate stream")
        except zlib.error as ex:
            raise ValueError("corrupt deflate payload") from ex
    if len(data) != entry["size"] or zlib.crc32(data) & 0xFFFFFFFF != entry["crc32"]:
        raise ValueError("archive entry size/CRC mismatch")
    return data


def validate_npy(data, columns):
    if len(data) > MAX_ARRAY:
        raise ValueError("native array too large")
    stream = io.BytesIO(data)
    try:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, order, dtype = np.lib.format.read_array_header_1_0(stream, max_header_size=4096)
        elif version == (2, 0):
            shape, order, dtype = np.lib.format.read_array_header_2_0(stream, max_header_size=4096)
        else:
            raise ValueError("unsupported NPY version")
        shape_ok = len(shape) == 1 if columns is None else len(shape) == 2 and shape[1] == columns
        if (
            not shape_ok
            or any(type(n) is not int or n < 1 for n in shape)
            or dtype.kind not in "fiu"
            or dtype.itemsize > 8
        ):
            raise ValueError("invalid NPY header schema")
        size = math.prod(shape) * dtype.itemsize
        if size > MAX_ARRAY or size != len(data) - stream.tell():
            raise ValueError("NPY declared payload outside bounds")
        stream.seek(0)
        arr = np.load(stream, allow_pickle=False)
    except Exception as ex:
        raise ValueError("invalid numeric NPY") from ex
    shape_ok = arr.ndim == 1 if columns is None else arr.ndim == 2 and arr.shape[1] == columns
    if (
        not shape_ok
        or arr.dtype.kind not in "fiu"
        or arr.dtype.itemsize > 8
        or len(arr) < 2
        or not np.isfinite(arr).all()
        or stream.tell() != len(data)
    ):
        raise ValueError("invalid native array schema/values")
    return arr


class RangeArchive:
    def __init__(self, url, total, cache):
        parsed = urllib.parse.urlsplit(url)
        if (
            parsed.scheme != "https"
            or parsed.hostname != "huggingface.co"
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or type(total) is not int
            or not 0 < total < 128 * 1024**3
        ):
            raise ValueError("bounded official archive URL required")
        self.url = url
        self.total = total
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)

    def cache_paths(self, start, end):
        key = hashlib.sha256(f"{self.url}:{self.total}:{start}:{end}".encode()).hexdigest()
        return self.cache / (key + ".bin"), self.cache / (key + ".json")

    def read_cached(self, start, end):
        path, receipt = self.cache_paths(start, end)
        if not path.is_file() or not receipt.is_file():
            return None
        try:
            data = path.read_bytes()
            r = json.loads(receipt.read_bytes())
            expected = dict(
                url=self.url,
                total=self.total,
                start=start,
                end=end,
                sha256=hashlib.sha256(data).hexdigest(),
            )
            if r != expected:
                return None
            return validate_range(
                data, 206, f"bytes {start}-{end}/{self.total}", start, end, self.total
            )
        except (OSError, ValueError):
            return None

    def fetch(self, start, end):
        if (
            any(type(x) is not int for x in (start, end))
            or not 0 <= start <= end < self.total
            or end - start + 1 > MAX_RANGE
        ):
            raise ValueError("archive fetch outside bounds")
        cached = self.read_cached(start, end)
        if cached is not None:
            return cached
        query = f"?download=true&range_id={start}_{end}"
        request = urllib.request.Request(
            self.url + query, headers={"Range": f"bytes={start}-{end}"}
        )
        with urllib.request.urlopen(request, timeout=40) as response:
            data = response.read(end - start + 2)
            validate_range(
                data, response.status, response.headers.get("Content-Range"), start, end, self.total
            )
        path, receipt = self.cache_paths(start, end)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)
        r = dict(
            url=self.url,
            total=self.total,
            start=start,
            end=end,
            sha256=hashlib.sha256(data).hexdigest(),
        )
        receipt.write_text(json.dumps(r, indent=2) + "\n")
        return data

    def entry(self, metadata):
        start = metadata["header_offset"]
        if type(start) is not int or not 0 <= start < self.total:
            raise ValueError("invalid archive local offset")
        length = 30 + len(metadata["name"].encode("utf-8")) + 4096 + metadata["compressed_size"]
        blob = self.fetch(start, min(self.total - 1, start + length - 1))
        return extract_entry(blob, metadata)


def central_location(tail, total):
    """Validate a single-disk classic/ZIP64 footer inside a bounded archive tail."""
    if not 22 <= len(tail) <= 65557 or len(tail) > total:
        raise ValueError("bounded ZIP footer required")
    pos = tail.rfind(b"PK\x05\x06")
    if pos < 0 or pos + 22 > len(tail):
        raise ValueError("ZIP footer missing")
    _, disk, cd_disk, on_disk, count, size, offset, comment = struct.unpack_from(
        "<4s4H2LH", tail, pos
    )
    if disk or cd_disk or on_disk != count or pos + 22 + comment != len(tail):
        raise ValueError("unsupported/truncated ZIP footer")
    footer_start = total - len(tail) + pos
    if count == 65535 or size == 0xFFFFFFFF or offset == 0xFFFFFFFF:
        loc = pos - 20
        if loc < 0:
            raise ValueError("ZIP64 locator missing")
        sig, disk64, off64, disks = struct.unpack_from("<4sLQL", tail, loc)
        j = off64 - (total - len(tail))
        if sig != b"PK\x06\x07" or disk64 or disks != 1 or not 0 <= j <= loc - 56:
            raise ValueError("ZIP64 locator invalid")
        fields = struct.unpack_from("<4sQ2H2L4Q", tail, j)
        if (
            fields[0] != b"PK\x06\x06"
            or fields[1] < 44
            or fields[1] > 4096
            or j + 12 + fields[1] != loc
            or fields[4]
            or fields[5]
            or fields[6] != fields[7]
        ):
            raise ValueError("ZIP64 footer invalid")
        count, size, offset = fields[7:10]
        footer_start = off64
    if not 0 < count < 65535 or not 0 < size <= MAX_RANGE or not 0 <= offset <= footer_start - size:
        raise ValueError("central directory outside bounds")
    return offset, size, count


def parse_central(data, count):
    if not 0 < len(data) <= MAX_RANGE or not 0 < count < 65535:
        raise ValueError("bounded central directory required")
    footer = struct.pack("<4s4H2LH", b"PK\x05\x06", 0, 0, count, count, len(data), 0, 0)
    try:
        with zipfile.ZipFile(io.BytesIO(data + footer)) as archive:
            infos = archive.infolist()
            if len(infos) != count or len({i.filename for i in infos}) != count:
                raise ValueError("central entry count/duplicates mismatch")
            return [
                dict(
                    name=i.filename,
                    header_offset=i.header_offset,
                    compressed_size=i.compress_size,
                    size=i.file_size,
                    method=i.compress_type,
                    crc32=i.CRC,
                )
                for i in infos
            ]
    except (zipfile.BadZipFile, struct.error) as ex:
        raise ValueError("invalid central directory") from ex


def metadata_selection(records):
    routes = {}
    for record in records:
        parts = record["name"].split("/")
        if len(parts) >= 4 and parts[0] == "Chunk_3" and parts[1].startswith("99c94dc769b5d96e|"):
            try:
                segment = int(parts[2])
            except ValueError:
                continue
            routes.setdefault(parts[1], set()).add(segment)
    out, days = [], set()
    for route, segments in sorted(routes.items()):
        pairs = [s for s in sorted(segments) if s + 1 in segments]
        day = route.split("|", 1)[1][:10]
        if not pairs or day in days:
            continue
        days.add(day)
        i = len(out) + 1
        out.append(
            dict(
                id=f"civic_{i:02d}",
                route_id=route,
                day=day,
                segments=[pairs[0], pairs[0] + 1],
                phase="validation" if i <= 2 else "test",
            )
        )
        if i == 7:
            break
    if len(out) != 7:
        raise ValueError("seven eligible Civic dates required")
    return out


def verify_metadata(root, plan, index):
    base = Path(root) / "data/reference/vehicle_native_stage9"
    receipt = json.loads((base / "archive_metadata_receipt.json").read_bytes())
    if receipt["url"] != index["archive_url"] or receipt["total"] != index["archive_size"]:
        raise ValueError("archive metadata identity changed")
    buffers = {}
    for name, record in receipt["ranges"].items():
        data = (base / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != record["sha256"]:
            raise ValueError("archive metadata bytes changed")
        validate_range(
            data, 206, record["content_range"], record["start"], record["end"], receipt["total"]
        )
        buffers[name] = data
    offset, size, count = central_location(buffers["chunk3_tail.bin"], receipt["total"])
    central = buffers["chunk3_central.bin"]
    record = receipt["ranges"]["chunk3_central.bin"]
    if (
        record["start"] != offset
        or len(central) != size
        or hashlib.sha256(central).hexdigest() != index["central_directory_sha256"]
    ):
        raise ValueError("central location/hash mismatch")
    records = parse_central(central, count)
    lookup = {r["name"]: r for r in records}
    for entry in index["entries"]:
        if {k: entry[k] for k in lookup[entry["name"]]} != lookup[entry["name"]]:
            raise ValueError("selected entry differs from central metadata")
    if metadata_selection(records) != plan["trips"]:
        raise ValueError("predeclared metadata selection not reproducible")
    return dict(entries=count, selected_entries=len(index["entries"]), routes=7)
