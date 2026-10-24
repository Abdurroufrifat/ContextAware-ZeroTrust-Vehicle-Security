import unittest


def module():
    from coact_cps.vehicle import integrity_window

    return integrity_window


class IntegrityWindowTests(unittest.TestCase):
    def monitor(self):
        return module().HorizonMonitor(
            bias=0.0, horizons=(2000, 5000), scales=(1.0, 1.0), threshold=2.0
        )

    def test_native_elapsed_time_warmup_and_coherent_motion(self):
        m = self.monitor()
        for i in range(31):
            t = i * 200
            m.ingest(1.0, max(0, t - 200), t) if i else None
            result = m.sample(10 + t / 1000, 10 + t / 1000, t)
            self.assertEqual(result.ready, i >= 10)
            self.assertFalse(result.alarm)
            self.assertAlmostEqual(result.score, 0.0)

    def test_nonadaptive_reference_detects_speed_ramp_without_learning(self):
        m = self.monitor()
        for i in range(31):
            t = i * 200
            if i:
                m.ingest(0.0, t - 200, t)
            r = m.sample(10 + t / 1000, 10 + t / 1000, t)
        self.assertTrue(r.alarm)
        self.assertAlmostEqual(r.score, 5.0)

    def test_same_epoch_cannot_advance_history_or_change_speeds(self):
        m = self.monitor()
        m.sample(10.0, 10.0, 1000)
        r = m.sample(10.0, 10.0, 1000)
        self.assertEqual(len(m.history), 1)
        with self.assertRaises(ValueError):
            m.sample(11.0, 10.0, 1000)
        self.assertFalse(r.ready)

    def test_missing_imu_interval_discards_old_speed_reference(self):
        m = self.monitor()
        m.sample(10.0, 10.0, 1000)
        m.ingest(0.0, 1000, 1200)
        m.sample(10.0, 10.0, 1200)
        self.assertFalse(m.ingest(0.0, 1400, 1600))
        r = m.sample(10.0, 10.0, 1600)
        self.assertFalse(r.ready)
        self.assertEqual(len(m.history), 1)

    def test_old_anchor_more_than_two_seconds_off_horizon_is_unavailable(self):
        m = self.monitor()
        m.sample(10.0, 10.0, 0)
        m.ingest(0.0, 0, 6000)
        result = m.sample(10.0, 10.0, 6000)
        self.assertFalse(result.ready)
        self.assertEqual(len(m.history), 2)

    def test_coherent_common_translation_is_invariant(self):
        a = self.monitor()
        b = self.monitor()
        for i in range(31):
            t = i * 200
            if i:
                a.ingest(0.4, t - 200, t)
                b.ingest(0.4, t - 200, t)
            x = a.sample(10 + 0.4 * t / 1000, 10 + 0.4 * t / 1000, t)
            y = b.sample(18 + 0.4 * t / 1000, 18 + 0.4 * t / 1000, t)
            self.assertAlmostEqual(x.score, y.score)


if __name__ == "__main__":
    unittest.main()
