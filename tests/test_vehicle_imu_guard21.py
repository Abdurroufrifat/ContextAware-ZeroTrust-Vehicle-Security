import unittest

import numpy as np

from coact_cps.vehicle.imu_guard21 import (
    boundaries,
    causal_context,
    decide,
    fit_partition,
    fit_profile,
    fit_quantile,
)


class IMUGuardTests(unittest.TestCase):
    def profile(self):
        x = {r: np.linspace(0, 1, 200) for r in ("civic13_07", "civic13_08")}
        partition = fit_partition(x)
        return x, fit_profile(x, {r: 2 * v + 1 for r, v in x.items()}, partition)

    def test_all_original_speed_alarms_are_preserved(self):
        z = np.array([[4, 0, 0], [0, 4, 0], [0, 0, 0.0]])
        np.testing.assert_array_equal(decide(z, 3.0, np.full(3, 100.0)), [True, True, False])

    def test_capped_guard_retains_high_imu_alarms(self):
        _, p = self.profile()
        b = boundaries(np.array([0.1, 0.2, 0.3]), p, 100.0, 3.0, cap=1.0)
        self.assertTrue(np.all((b >= 3) & (b <= 4)))
        z = np.array([[0, 0, 4.001], [0, 0, 4.0], [0, 0, 2.0]])
        np.testing.assert_array_equal(decide(z, 3.0, np.full(3, 4.0)), [True, False, False])

    def test_candidate_cannot_add_an_alarm(self):
        rng = np.random.default_rng(3)
        z = rng.normal(size=(300, 3)) * 4
        alarm = decide(z, 3.0, np.linspace(3, 4, 300))
        self.assertTrue(np.all(~alarm | (z.max(axis=1) > 3)))

    def test_unseen_context_restores_original_imu_decision(self):
        _, p = self.profile()
        b = boundaries(np.array([np.nan, 10.0, -1.0]), p, 100.0, 3.0, cap=1.0)
        np.testing.assert_array_equal(b, [3.0, 3.0, 3.0])

    def test_context_uses_only_past_speeds_and_no_imu_values(self):
        t = np.arange(1, 21, dtype=np.int64) * 200
        f = np.column_stack([t / 1000, t / 1000, np.ones(20)])
        original = causal_context(f, t, t - 200)
        f[:, 2] += 100
        np.testing.assert_allclose(original, causal_context(f, t, t - 200), equal_nan=True)
        f[12:, :2] += 500
        changed = causal_context(f, t, t - 200)
        np.testing.assert_allclose(original[:13], changed[:13], equal_nan=True)

    def test_gap_makes_context_unavailable(self):
        t = np.arange(1, 21, dtype=np.int64) * 200
        s = t - 200
        s[10] += 1
        f = np.column_stack([t / 1000, t / 1000, np.ones(20)])
        c = causal_context(f, t, s)
        self.assertTrue(np.isnan(c[11:16]).all())
        self.assertTrue(np.isfinite(c[16]))

    def test_clean_fit_roles_are_exact(self):
        with self.assertRaises(ValueError):
            fit_partition({"civic13_10": np.ones(200)})

    def test_bin_counts_do_not_duplicate_seeds(self):
        _, p = self.profile()
        self.assertEqual(sum(p["counts"]), 400)
        self.assertTrue(all(p["supported"]))

    def test_calibration_requires_supported_normal_rows(self):
        x, p = self.profile()
        q = fit_quantile(x["civic13_07"], 2 * x["civic13_07"] + 1, p)
        self.assertTrue(np.isfinite(q))
        with self.assertRaises(ValueError):
            fit_quantile(np.full(40, 100.0), np.ones(40), p)

    def test_invalid_boundary_cannot_weaken_floor(self):
        with self.assertRaises(ValueError):
            decide(np.ones((2, 3)), 3.0, np.array([2.0, 4.0]))

    def test_invalid_timestamps_rejected(self):
        with self.assertRaises(ValueError):
            causal_context(np.zeros((8, 3)), np.ones(8), np.zeros(8))


if __name__ == "__main__":
    unittest.main()
