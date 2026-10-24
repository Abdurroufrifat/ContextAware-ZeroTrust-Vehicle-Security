import io
import struct
import unittest
import zipfile

import numpy as np


class NativeArchiveTests(unittest.TestCase):
    def entry(self):
        b = io.BytesIO()
        np.save(b, np.arange(12, dtype=np.float64).reshape(4, 3), allow_pickle=False)
        z = io.BytesIO()
        with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("native/value", b.getvalue())
        with zipfile.ZipFile(z) as archive:
            i = archive.getinfo("native/value")
            info = dict(
                name=i.filename,
                compressed_size=i.compress_size,
                size=i.file_size,
                method=i.compress_type,
                crc32=i.CRC,
                header_offset=i.header_offset,
            )
        return z.getvalue(), info, b.getvalue()

    def test_range_requires_exact_status_offsets_and_length(self):
        from vehicle_native_archive import validate_range

        self.assertEqual(validate_range(b"123", 206, "bytes 10-12/100", 10, 12, 100), b"123")
        for status, header, data in [
            (200, "bytes 10-12/100", b"123"),
            (206, "bytes 11-13/100", b"123"),
            (206, "bytes 10-12/101", b"123"),
            (206, "bytes 10-12/100", b"12"),
        ]:
            with self.assertRaises(ValueError):
                validate_range(data, status, header, 10, 12, 100)

    def test_original_npy_bytes_preserved_and_crc_checked(self):
        from vehicle_native_archive import extract_entry

        blob, entry, expected = self.entry()
        self.assertEqual(extract_entry(blob, entry), expected)
        entry["crc32"] ^= 1
        with self.assertRaises(ValueError):
            extract_entry(blob, entry)

    def test_filename_mismatch_traversal_and_encryption_rejected(self):
        from vehicle_native_archive import extract_entry

        blob, entry, _ = self.entry()
        for name in ["wrong/value", "../native/value", "/native/value"]:
            e = dict(entry, name=name)
            with self.assertRaises(ValueError):
                extract_entry(blob, e)
        data = bytearray(blob)
        struct.pack_into("<H", data, 6, 1)
        with self.assertRaises(ValueError):
            extract_entry(bytes(data), entry)

    def test_invalid_sizes_header_and_unsupported_compression_rejected(self):
        from vehicle_native_archive import extract_entry

        blob, entry, _ = self.entry()
        for e in [
            dict(entry, size=128 * 1024 * 1024),
            dict(entry, method=99),
            dict(entry, compressed_size=1),
        ]:
            with self.assertRaises(ValueError):
                extract_entry(blob, e)
        with self.assertRaises(ValueError):
            extract_entry(b"broken" + blob[6:], entry)

    def test_object_wrong_shape_and_nonfinite_npy_rejected(self):
        from vehicle_native_archive import validate_npy

        for arr in [np.array([object()], dtype=object), np.ones((4, 2)), np.full((4, 3), np.nan)]:
            b = io.BytesIO()
            np.save(b, arr, allow_pickle=True)
            with self.assertRaises(ValueError):
                validate_npy(b.getvalue(), 3)
        b = io.BytesIO()
        np.save(b, np.ones((4, 3)), allow_pickle=False)
        self.assertEqual(validate_npy(b.getvalue(), 3).shape, (4, 3))

    def test_materialization_preserves_bytes_in_windows_safe_folder(self):
        import tempfile
        from pathlib import Path

        from acquire_vehicle_native_stage9 import materialize

        _, _, data = self.entry()
        entry = dict(
            name="Chunk_3/99c94dc769b5d96e|2018-01-01--00-00-00/5/processed_log/IMU/accelerometer/value",
            trip_id="civic_01",
            segment=5,
            crc32=0,
            size=len(data),
            compressed_size=100,
            header_offset=200,
            method=8,
        )
        with tempfile.TemporaryDirectory() as tmp:
            record = materialize(Path(tmp), entry, data)
            p = Path(tmp) / "data/raw/vehicle_native_stage9/civic_01/segment_5" / record["path"]
            self.assertEqual(p.read_bytes(), data)
            self.assertEqual(record["shape"], [4, 3])
            self.assertNotIn("|", str(p))

    def test_index_route_binding_rejects_other_vehicle(self):
        from acquire_vehicle_native_stage9 import check_index

        plan = dict(trips=[dict(id="civic_01", route_id="99c94dc769b5d96e|route", segments=[5])])
        channels = ["GNSS/live_gnss_ublox", "IMU/accelerometer", "CAN/wheel_speed"]
        entries = [
            dict(
                name=f"Chunk_3/99c94dc769b5d96e|route/5/processed_log/{c}/{s}",
                trip_id="civic_01",
                segment=5,
            )
            for c in channels
            for s in ["t", "value"]
        ]
        check_index(plan, dict(entries=entries))
        entries[0]["name"] = entries[0]["name"].replace("99c94dc769b5d96e", "b0c9d2329ad1606b")
        with self.assertRaises(ValueError):
            check_index(plan, dict(entries=entries))

    def test_cache_accepts_only_matching_pinned_receipt(self):
        import hashlib
        import json
        import tempfile
        from pathlib import Path

        from vehicle_native_archive import RangeArchive

        with tempfile.TemporaryDirectory() as tmp:
            reader = RangeArchive(
                "https://huggingface.co/datasets/a/b/resolve/pin/file.zip", 100, Path(tmp)
            )
            path, receipt = reader.cache_paths(10, 12)
            path.write_bytes(b"123")
            receipt.write_text(
                json.dumps(
                    dict(
                        url=reader.url,
                        total=100,
                        start=10,
                        end=12,
                        sha256=hashlib.sha256(b"123").hexdigest(),
                    )
                )
            )
            self.assertEqual(reader.read_cached(10, 12), b"123")
            path.write_bytes(b"124")
            self.assertIsNone(reader.read_cached(10, 12))

    def test_oversized_header_rejected_before_numpy_load(self):
        from unittest.mock import patch

        from vehicle_native_archive import validate_npy

        b = io.BytesIO()
        np.lib.format.write_array_header_1_0(
            b, dict(descr="<f8", fortran_order=False, shape=(10**12, 3))
        )
        with patch(
            "vehicle_native_archive.np.load", side_effect=AssertionError("allocation")
        ) as load:
            with self.assertRaises(ValueError):
                validate_npy(b.getvalue(), 3)
            load.assert_not_called()

    def test_archive_footer_and_central_bounds_verified(self):
        from vehicle_native_archive import central_location, parse_central

        blob, entry, _ = self.entry()
        offset, size, count = central_location(blob, len(blob))
        records = parse_central(blob[offset : offset + size], count)
        self.assertEqual(records, [{k: entry[k] for k in entry}])
        with self.assertRaises(ValueError):
            central_location(blob[:-1], len(blob) - 1)
        with self.assertRaises(ValueError):
            parse_central(blob[offset : offset + size], count + 1)

    def test_zip64_footer_offsets_and_selection_reproduced(self):
        from vehicle_native_archive import central_location, metadata_selection

        total = 10_000_000_000
        tail_start = total - 98
        tail = struct.pack(
            "<4sQ2H2L4Q", b"PK\x06\x06", 44, 45, 45, 0, 0, 12, 12, 200, tail_start - 200
        )
        tail += struct.pack("<4sLQL", b"PK\x06\x07", 0, tail_start, 1)
        tail += struct.pack("<4s4H2LH", b"PK\x05\x06", 0, 0, 12, 12, 200, 0xFFFFFFFF, 0)
        self.assertEqual(central_location(tail, total), (tail_start - 200, 200, 12))
        bad = bytearray(tail)
        struct.pack_into("<Q", bad, 48, total + 1)
        with self.assertRaises(ValueError):
            central_location(bytes(bad), total)
        records = [
            dict(name=f"Chunk_3/99c94dc769b5d96e|2018-01-{d:02d}--00-00-00/{seg}/x")
            for d in range(1, 8)
            for seg in (8, 9)
        ]
        selected = metadata_selection(records)
        self.assertEqual(len(selected), 7)
        self.assertEqual(selected[0]["segments"], [8, 9])
        self.assertEqual(selected[-1]["phase"], "test")
