import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


class IntegrityDataTests(unittest.TestCase):
    def test_wheel_receipt_ties_use_last_native_log_update(self):
        from vehicle_integrity_data import wheel_receipt_view

        t = np.array([1.0, 2.0, 2.0, 3.0])
        v = np.array([[10.0] * 4, [11.0] * 4, [11.1] * 4, [12.0] * 4])
        tt, vv, a = wheel_receipt_view(t, v)
        self.assertEqual(tt.tolist(), [1.0, 2.0, 3.0])
        self.assertEqual(vv[1].tolist(), [11.1] * 4)
        self.assertEqual(a["duplicate_updates"], 1)
        self.assertEqual(v[1].tolist(), [11.0] * 4)

    def test_rollback_is_not_sorted_or_hidden(self):
        from vehicle_integrity_data import wheel_receipt_view

        with self.assertRaises(ValueError):
            wheel_receipt_view(np.array([1.0, 3.0, 2.0]), np.ones((3, 4)))

    def test_large_conflicting_tie_and_unbounded_burst_are_rejected(self):
        from vehicle_integrity_data import wheel_receipt_view

        with self.assertRaises(ValueError):
            wheel_receipt_view(
                np.array([1.0, 2.0, 2.0]), np.array([[10.0] * 4, [10.0] * 4, [12.0] * 4])
            )
        with self.assertRaises(ValueError):
            wheel_receipt_view(np.array([1.0] * 5 + [2.0]), np.ones((6, 4)))


if __name__ == "__main__":
    unittest.main()
