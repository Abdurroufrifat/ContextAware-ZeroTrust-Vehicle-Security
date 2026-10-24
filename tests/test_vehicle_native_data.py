import importlib
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def module():
    try:
        return importlib.import_module("vehicle_native_data")
    except ModuleNotFoundError:
        raise AssertionError("native data module not implemented") from None


class NativeDataTests(unittest.TestCase):
    def test_raw_imu_gap_straddling_interval_start_is_preserved(self):
        m = module()
        _, _, gap = m.causal_hold(
            np.array([-0.04, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]), np.ones(8), 0.0, 0.1
        )
        self.assertAlmostEqual(gap, 0.08)

    def test_causal_integral_does_not_use_future_measurement(self):
        m = module()
        mean, last, gap = m.causal_hold(
            np.array([0.0, 0.4, 1.2]), np.array([2.0, 4.0, 100.0]), 0.0, 1.0
        )
        self.assertAlmostEqual(mean, 3.2)
        self.assertAlmostEqual(last, 0.4)
        self.assertAlmostEqual(gap, 0.6)

    def test_integral_requires_prior_coverage(self):
        m = module()
        with self.assertRaises(ValueError):
            m.causal_hold(np.array([0.1, 0.2]), np.array([1.0, 2.0]), 0.0, 0.2)

    def test_chunk_hash_and_shape_are_enforced(self):
        m = module()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            p = root / "t.npy"
            np.save(p, np.array([1.0, 2.0]))
            manifest = {"files": [{"path": "t.npy", "sha256": m.sha(p.read_bytes()), "shape": [2]}]}
            self.assertEqual(
                m.validate_chunk(root, manifest), {"t.npy": manifest["files"][0]["sha256"]}
            )
            p.write_bytes(p.read_bytes() + b"x")
            with self.assertRaises(ValueError):
                m.validate_chunk(root, manifest)

    def test_object_array_is_rejected_even_with_matching_hash(self):
        m = module()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            p = root / "value.npy"
            np.save(p, np.array([{"v": 1}], dtype=object))
            with self.assertRaises(ValueError):
                m.validate_chunk(
                    root,
                    {
                        "files": [
                            {"path": "value.npy", "sha256": m.sha(p.read_bytes()), "shape": [1]}
                        ]
                    },
                )

    def test_path_escape_is_rejected(self):
        m = module()
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                m.validate_chunk(
                    Path(d), {"files": [{"path": "../outside", "sha256": "0" * 64, "shape": [1]}]}
                )

    def test_native_merge_orders_distinct_cross_segment_events(self):
        m = module()
        t, v = m.merge_channel(
            [
                (np.array([0.0, 1.005]), np.array([[0.0], [10.0]])),
                (np.array([1.0, 2.0]), np.array([[20.0], [30.0]])),
            ]
        )
        self.assertEqual(t.tolist(), [0.0, 1.0, 1.005, 2.0])
        self.assertEqual(v[:, 0].tolist(), [0.0, 20.0, 10.0, 30.0])

    def test_conflicting_equal_time_events_are_rejected(self):
        m = module()
        with self.assertRaises(ValueError):
            m.merge_channel(
                [
                    (np.array([0.0, 1.0]), np.array([[0.0], [1.0]])),
                    (np.array([1.0, 2.0]), np.array([[2.0], [3.0]])),
                ]
            )

    def test_native_source_ages_and_intervals_are_preserved(self):
        m = module()
        g = (
            np.array([0.0, 1.0, 2.0, 3.0]),
            np.array([[0, 0, 10, 1530000000000, 0, 0]] * 4, dtype=float),
        )
        i = (
            np.arange(0.0, 3.01, 0.01),
            np.column_stack([np.ones(301), np.zeros(301), np.zeros(301)]),
        )
        w = (np.arange(0.0, 3.01, 0.01), np.ones((301, 4)) * 12)
        frames = m.align_channels("trip", g, i, w, stride=1, edge_s=0.1)
        self.assertEqual(len(frames), 2)
        self.assertAlmostEqual(frames[0].imu_accel, 1.0)
        self.assertAlmostEqual(frames[0].wheel_speed, 12.0)
        self.assertEqual(frames[0].time_ms, 1000)
        self.assertEqual(frames[0].interval_start_ms, 0)
        self.assertLessEqual(frames[0].imu_time_ms, frames[0].time_ms)
        self.assertLessEqual(frames[0].wheel_time_ms, frames[0].time_ms)


if __name__ == "__main__":
    unittest.main()
