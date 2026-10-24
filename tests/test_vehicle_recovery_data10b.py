import io
import tempfile
import unittest
from pathlib import Path

import numpy as np


class RecoveryDataTests(unittest.TestCase):
    def records(self):
        return [
            dict(
                name=f"Chunk_7/99c94dc769b5d96e|2018-06-{i:02d}--00-00-00/{s}/processed_log/{c}/{n}",
                header_offset=0,
                size=100,
                compressed_size=50,
                crc32=1,
                method=8,
            )
            for i in range(1, 11)
            for s in [3, 4]
            for c in ["GNSS/live_gnss_ublox", "IMU/accelerometer", "CAN/wheel_speed"]
            for n in ["t", "value"]
        ]

    def test_selection_excludes_prior_dates_and_uses_fixed_three_one_five(self):
        from vehicle_recovery_data10b import select_routes

        rows = select_routes(self.records(), {"2018-06-01"}, set())
        self.assertEqual(len(rows), 9)
        self.assertEqual(
            [r["phase"] for r in rows], ["calibration"] * 3 + ["validation"] + ["test"] * 5
        )
        self.assertEqual(rows[0]["day"], "2018-06-02")
        self.assertEqual(rows[-1]["segments"], [3, 4])

    def test_insufficient_new_dates_stop_without_synthetic_segments(self):
        from vehicle_recovery_data10b import select_routes

        with self.assertRaises(ValueError):
            select_routes(self.records(), {"2018-06-01", "2018-06-02"}, set())

    def test_index_binding_rejects_route_swap(self):
        from vehicle_recovery_data10b import check_index, select_routes, selected_index

        plan = select_routes(self.records(), {"2018-06-01"}, set())
        entries = selected_index(self.records(), plan)
        self.assertEqual(len(entries), 108)
        check_index(plan, entries)
        entries[0]["trip_id"] = "civic10b_09"
        with self.assertRaises(ValueError):
            check_index(plan, entries)

    def test_materialize_preserves_original_numeric_payload(self):
        from acquire_vehicle_native_stage10b import materialize

        arr = np.arange(12, dtype=np.float32).reshape(4, 3)
        stream = io.BytesIO()
        np.save(stream, arr, allow_pickle=False)
        entry = dict(
            name="Chunk_7/route/3/processed_log/IMU/accelerometer/value",
            trip_id="civic10b_01",
            segment=3,
            crc32=10,
            compressed_size=80,
            header_offset=100,
        )
        with tempfile.TemporaryDirectory() as tmp:
            record = materialize(Path(tmp), entry, stream.getvalue())
            path = (
                Path(tmp)
                / "data/raw/vehicle_native_stage10b/civic10b_01/segment_3"
                / record["path"]
            )
            self.assertEqual(path.read_bytes(), stream.getvalue())
            self.assertEqual(record["shape"], [4, 3])

    def test_acquisition_manifest_uses_predeclared_rule_field(self):
        from acquire_vehicle_native_stage10b import build_manifest

        result = build_manifest(dict(rule="fixed metadata selection"), [])
        self.assertEqual(result["selection_rule"], "fixed metadata selection")
