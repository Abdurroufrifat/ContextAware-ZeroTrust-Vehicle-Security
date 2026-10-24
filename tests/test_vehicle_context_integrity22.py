import unittest

import numpy as np

from coact_cps.vehicle.context_integrity22 import paired_effect, verify_pair


class ContextIntegrityTests(unittest.TestCase):
    def test_boundary_swap_counts_lost_and_gained_with_fixed_nodes(self):
        # Reversing the arms or using >= would break these expected counts.
        nodes = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 2.0], [2.0, 0.0, 2.0], [0.0, 0.0, 1.0]])
        result, reported, clean = paired_effect(
            nodes,
            1.0,
            np.array([3.0, 1.0, 3.0, 1.0]),
            np.array([1.0, 3.0, 1.0, 1.0]),
            np.array([True, True, True, False]),
        )
        self.assertEqual(reported.tolist(), [False, True, True, False])
        self.assertEqual(clean.tolist(), [True, False, True, False])
        self.assertEqual(result["active_lost_count"], 1)
        self.assertEqual(result["active_gained_count"], 1)
        self.assertEqual(result["active_boundary_raised_count"], 2)
        self.assertEqual(result["active_net_tpr_loss"], 0.0)

    def test_empty_active_denominator_is_null(self):
        r, _, _ = paired_effect(
            np.zeros((2, 3)), 1.0, np.ones(2), np.ones(2), np.zeros(2, dtype=bool)
        )
        self.assertIsNone(r["active_net_tpr_loss"])
        self.assertEqual(r["active_count"], 0)

    def test_identical_boundaries_have_zero_effect(self):
        r, a, b = paired_effect(
            np.array([[0.0, 0.0, 2.0]]), 1.0, np.ones(1), np.ones(1), np.ones(1, dtype=bool)
        )
        np.testing.assert_array_equal(a, b)
        self.assertEqual(r["active_lost_count"], 0)

    def test_inactive_flip_is_not_attack_recall_loss(self):
        r, _, _ = paired_effect(
            np.array([[0.0, 0.0, 2.0]]), 1.0, np.array([3.0]), np.ones(1), np.zeros(1, dtype=bool)
        )
        self.assertEqual(r["inactive_lost_count"], 1)
        self.assertEqual(r["active_lost_count"], 0)

    def test_labels_must_be_boolean(self):
        with self.assertRaises(ValueError):
            paired_effect(np.zeros((1, 3)), 1.0, np.ones(1), np.ones(1), np.ones(1))

    def test_nonfinite_boundary_rejected(self):
        with self.assertRaises(ValueError):
            paired_effect(
                np.zeros((1, 3)), 1.0, np.array([np.nan]), np.ones(1), np.ones(1, dtype=bool)
            )

    @staticmethod
    def context():
        return dict(
            time_ms=np.array([1000, 2000]),
            start_ms=np.array([0, 1000]),
            positions=np.array([1]),
            base_grant=np.array([False, True]),
            labels=np.array([False, False]),
        )

    def test_pair_permits_attack_labels_and_changed_features(self):
        a, b = self.context(), self.context()
        a["labels"][1] = True
        verify_pair(a, b)

    def test_timestamp_misalignment_rejected(self):
        a, b = self.context(), self.context()
        b["time_ms"][1] += 1
        with self.assertRaisesRegex(ValueError, "time_ms"):
            verify_pair(a, b)

    def test_clean_policy_mask_is_not_part_of_context_intervention(self):
        a, b = self.context(), self.context()
        b["base_grant"][1] = False
        verify_pair(a, b)

    def test_oracle_must_be_clean(self):
        a, b = self.context(), self.context()
        b["labels"][1] = True
        with self.assertRaisesRegex(ValueError, "normal"):
            verify_pair(a, b)


if __name__ == "__main__":
    unittest.main()
