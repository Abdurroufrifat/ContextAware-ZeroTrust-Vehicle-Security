import importlib
import unittest

import numpy as np


def module():
    try:
        return importlib.import_module("coact_cps.vehicle.native_detectors")
    except ModuleNotFoundError:
        raise AssertionError("native detector mechanisms not implemented") from None


class NativeDetectorTests(unittest.TestCase):
    def test_innovation_energy_uses_full_covariance(self):
        m = module()
        self.assertAlmostEqual(m.innovation_energy(np.array([1.0, 2.0]), np.diag([1.0, 4.0])), 2.0)

    def test_window_one_matches_static_strict_threshold(self):
        m = module()
        a = m.Statistic("chi", 2.0)
        b = m.Statistic("window", 2.0, window=1)
        for z in [1.0, 2.0, 2.001, 0.5]:
            self.assertEqual(a.step(z).alarm, b.step(z).alarm)

    def test_window_waits_for_complete_history(self):
        m = module()
        s = m.Statistic("window", 5.0, window=3)
        self.assertFalse(s.step(2.0).scored)
        self.assertFalse(s.step(2.0).scored)
        result = s.step(2.0)
        self.assertTrue(result.scored)
        self.assertTrue(result.alarm)
        self.assertEqual(result.statistic, 6.0)

    def test_cusum_delayed_reset_discards_next_statistic(self):
        m = module()
        s = m.Statistic("cusum", 4.0, bias=3.0)
        self.assertFalse(s.step(6.0).alarm)
        self.assertFalse(s.step(6.0).alarm)
        r = s.step(100.0)
        self.assertTrue(r.alarm)
        self.assertFalse(r.scored)
        self.assertEqual(r.statistic, 0.0)
        self.assertFalse(s.step(6.0).alarm)

    def test_invalid_statistics_and_model_numbers_are_rejected(self):
        m = module()
        for value in [float("nan"), -1.0, True]:
            with self.assertRaises(ValueError):
                m.Statistic("chi", 3.0).step(value)
        with self.assertRaises(ValueError):
            m.ModelConfig(r_g=-1.0)
        with self.assertRaises(ValueError):
            m.Statistic("cusum", 3.0, bias=2.0)

    def test_variable_time_velocity_and_offset_prediction(self):
        m = module()
        o = m.Observer(m.ModelConfig(imu_offset=0.5))
        o.initialize(10.0, 10.0, 1000)
        self.assertTrue(o.predict(2.0, 1000, 2000))
        self.assertAlmostEqual(o.x[0], 11.5)
        self.assertGreaterEqual(np.linalg.eigvalsh(o.p).min(), 0.0)

    def test_unobserved_interval_cannot_be_silently_integrated(self):
        m = module()
        o = m.Observer(m.ModelConfig())
        o.initialize(10.0, 10.0, 1000)
        self.assertFalse(o.predict(2.0, 1200, 2000))
        self.assertIsNone(o.x)

    def test_kalman_update_keeps_positive_covariance(self):
        m = module()
        o = m.Observer(m.ModelConfig())
        o.initialize(10.0, 10.0, 1000)
        for k in range(1, 100):
            o.predict(0.1, 1000 + (k - 1) * 100, 1000 + k * 100)
            o.update(10.0 + k * 0.01, 10.0 + k * 0.01)
            self.assertGreaterEqual(np.linalg.eigvalsh(o.p).min(), -1e-12)

    def test_coherent_translation_has_identical_innovation(self):
        m = module()
        a = m.Observer(m.ModelConfig())
        b = m.Observer(m.ModelConfig())
        a.initialize(10.0, 10.0, 1000)
        b.initialize(18.0, 18.0, 1000)
        a.predict(0.2, 1000, 1200)
        b.predict(0.2, 1000, 1200)
        ra, sa = a.innovation(10.1, 10.1)
        rb, sb = b.innovation(18.1, 18.1)
        np.testing.assert_allclose(ra, rb, atol=1e-14)
        np.testing.assert_allclose(sa, sb, atol=1e-14)


if __name__ == "__main__":
    unittest.main()
