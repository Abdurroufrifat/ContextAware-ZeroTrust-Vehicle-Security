import unittest

import numpy as np

from coact_cps.vehicle.physics17 import (
    STATUS,
    fit_physical_scale,
    hybrid_score,
    physical_residual,
    threshold_from_clean,
)


class PhysicsResidualTests(unittest.TestCase):
    def trace(self, dt=None):
        dt = np.full(12, 200) if dt is None else np.asarray(dt)
        end = np.cumsum(dt).astype(np.int64)
        start = end - dt
        speed = end / 1000 * 2
        return np.column_stack([speed, speed, np.full(len(dt), 2.0)]), end, start

    def test_perfect_kinematics(self):
        x, t, s = self.trace()
        r = physical_residual(x, t, s)
        self.assertTrue(np.isinf(r[:5]).all())
        np.testing.assert_allclose(r[5:], 0, atol=1e-14)

    def test_imu_bias(self):
        x, t, s = self.trace()
        x[:, 2] += 0.75
        np.testing.assert_allclose(physical_residual(x, t, s)[5:], 0.75)

    def test_nonuniform_intervals(self):
        x, t, s = self.trace([100, 210, 190, 300, 200, 100, 250, 150, 200, 190, 210, 200])
        np.testing.assert_allclose(physical_residual(x, t, s)[5:], 0, atol=1e-14)

    def test_gap_invalidates_five_intervals(self):
        x, t, s = self.trace()
        t[6:] += 20
        s[6:] += 20
        r = physical_residual(x, t, s)
        self.assertTrue(np.isinf(r[6:11]).all())
        self.assertTrue(np.isfinite(r[11]))

    def test_constant_speed_bias_is_invisible(self):
        x, t, s = self.trace()
        x[:, :2] += 5
        np.testing.assert_allclose(physical_residual(x, t, s)[5:], 0, atol=1e-14)

    def test_coherent_speed_acceleration_spoof_is_invisible(self):
        x, t, s = self.trace()
        x[:, :2] += t[:, None] / 1000
        x[:, 2] += 1
        np.testing.assert_allclose(physical_residual(x, t, s)[5:], 0, atol=1e-14)

    def test_hybrid_retains_other_channels_and_fails_closed(self):
        nodes = np.array([[8.0, 0, 100], [0, 9, 100], [0, 0, 100]])
        score = hybrid_score(nodes, np.zeros(3), np.ones(3), np.array([0.0, 0.0, np.inf]), 0.0, 1.0)
        np.testing.assert_equal(score, [8, 9, np.inf])

    def test_normal_roles_and_scale_floor(self):
        m, scale = fit_physical_scale({"civic13_07": np.zeros(10), "civic13_08": np.zeros(12)})
        self.assertEqual((m, scale), (0.0, 0.01))
        with self.assertRaises(ValueError):
            fit_physical_scale({"civic13_10": np.zeros(10)})

    def test_unavailable_warmup_ignored_only_when_fitting(self):
        a = np.array([np.inf, 0.0, 1.0, 2.0])
        m, scale = fit_physical_scale({"civic13_07": a, "civic13_08": a})
        self.assertTrue(np.isfinite([m, scale]).all())
        with self.assertRaises(ValueError):
            fit_physical_scale({"civic13_07": np.array([np.inf]), "civic13_08": a})

    def test_strict_alarm_threshold(self):
        threshold = threshold_from_clean(np.array([0.0, 1.0, 2.0]), 3.0)
        self.assertEqual(threshold, 3.0)
        np.testing.assert_equal(np.array([3.0, 3.001]) > threshold, [False, True])

    def test_reject_invalid_measurements_and_times(self):
        x, t, s = self.trace()
        for args in [
            (x[:-1], t, s),
            (x, t.astype(float), s),
            (x, t, t),
            (x * np.nan, t, s),
            (x, t, s - 1000),
        ]:
            with self.subTest(args=str(args)[:20]), self.assertRaises(ValueError):
                physical_residual(*args)

    def test_development_status(self):
        self.assertEqual(STATUS, "DEVELOPMENT_ONLY_NOT_PROMOTED")


if __name__ == "__main__":
    unittest.main()
