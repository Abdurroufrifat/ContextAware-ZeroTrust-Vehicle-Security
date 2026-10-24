import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


class TransferDataTests(unittest.TestCase):
    def fixture(self, root, start, folder):
        base = root / folder
        files = []
        gt = np.arange(0, 6, 0.2) + start
        it = np.arange(0, 6, 0.005) + start
        wt = np.arange(0, 6, 0.01) + start
        gv = np.zeros((len(gt), 6))
        gv[:, 2] = 10
        gv[:, 3] = 1e12 + gt * 1000
        iv = np.zeros((len(it), 3))
        iv[:, 2] = -9.8
        wv = np.full((len(wt), 4), 10.0)
        # A bounded native wheel receipt tie must not be manufactured away.
        wt = np.insert(wt, 10, wt[10])
        wv = np.insert(wv, 10, wv[10], axis=0)
        for channel, t, v in [
            ("GNSS/live_gnss_ublox", gt, gv),
            ("IMU/accelerometer", it, iv),
            ("CAN/wheel_speed", wt, wv),
        ]:
            for suffix, arr in [("t", t), ("value", v)]:
                n = "processed_log/" + channel + "/" + suffix
                p = base / n
                p.parent.mkdir(parents=True, exist_ok=True)
                with p.open("wb") as stream:
                    np.save(stream, arr, allow_pickle=False)
                files.append(
                    dict(
                        path=n,
                        sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                        shape=list(arr.shape),
                    )
                )
        (base / "source_manifest.json").write_text(json.dumps(dict(route_id="route", files=files)))

    def test_two_real_contiguous_parts_share_causal_imu_without_bridge(self):
        from vehicle_transfer_data import load_transfer_trip

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root, 0, "p1")
            self.fixture(root, 6, "p2")
            frames = load_transfer_trip(root, dict(route_id="route", parts=["p1", "p2"]))
            self.assertGreater(len(frames), 10)
            self.assertTrue(any(f.interval_start_ms < 6000 <= f.time_ms for f in frames))
            self.assertTrue(all(f.imu_accel == 0 for f in frames))

    def test_noncontiguous_parts_and_other_route_cannot_be_bridged(self):
        from vehicle_transfer_data import load_transfer_trip

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root, 0, "p1")
            self.fixture(root, 7, "p2")
            with self.assertRaisesRegex(ValueError, "continuity"):
                load_transfer_trip(root, dict(route_id="route", parts=["p1", "p2"]))
            with self.assertRaisesRegex(ValueError, "route"):
                load_transfer_trip(root, dict(route_id="other", parts=["p1", "p2"]))
