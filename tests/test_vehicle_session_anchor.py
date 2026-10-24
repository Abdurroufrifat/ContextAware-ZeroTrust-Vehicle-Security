import unittest


class SessionAnchorTests(unittest.TestCase):
    def monitor(self):
        from coact_cps.vehicle.session_anchor import SessionAnchor

        return SessionAnchor(0.0, 0.2, 2.0)

    def step(self, m, i, speed=10.0, a=0.0):
        end = 1000 + 200 * i
        m.ingest(a, end - 200, end)
        return m.sample(speed, speed, end)

    def enrolled(self, m):
        for i in range(26):
            r = self.step(m, i, a=0.4)
        self.assertTrue(r.ready)
        self.assertAlmostEqual(r.bias, 0.4)
        return r

    def test_fits_effective_bias_once_from_five_seconds(self):
        m = self.monitor()
        for i in range(25):
            self.assertFalse(self.step(m, i, a=0.4).ready)
        r = self.step(m, 25, a=0.4)
        self.assertTrue(r.ready)
        self.assertFalse(r.alarm)
        self.assertAlmostEqual(r.bias, 0.4)
        for i in range(26, 40):
            self.step(m, i, speed=12, a=0.4)
        self.assertAlmostEqual(m.bias, 0.4)

    def test_gradual_common_speed_drift_is_not_assimilated(self):
        m = self.monitor()
        self.enrolled(m)
        for i in range(26, 55):
            r = self.step(m, i, speed=10 + 0.4 * (i - 25) * 0.2, a=0.4)
        self.assertTrue(r.alarm)

    def test_coherent_all_source_motion_is_a_negative_control(self):
        m = self.monitor()
        self.enrolled(m)
        for i in range(26, 55):
            r = self.step(m, i, speed=10 + 0.4 * (i - 25) * 0.2, a=0.8)
        self.assertFalse(r.alarm)

    def test_enrollment_speed_ramp_poisoning_is_a_negative_control(self):
        m = self.monitor()
        for i in range(55):
            r = self.step(m, i, speed=10 + 0.12 * i * 0.2)
        self.assertTrue(r.ready)
        self.assertFalse(r.alarm)
        self.assertAlmostEqual(r.bias, -0.12)

    def test_imu_gap_after_enrollment_latches_reference_loss(self):
        m = self.monitor()
        self.enrolled(m)
        m.ingest(0.4, 6400, 6600)
        r = m.sample(10, 10, 6600)
        self.assertTrue(r.lost)
        self.assertTrue(r.alarm)
        for i in range(29, 70):
            r = self.step(m, i, a=0.4)
        self.assertTrue(r.lost)
        self.assertTrue(r.alarm)

    def test_same_physical_epoch_cached_conflicting_copy_rejected(self):
        m = self.monitor()
        r = self.enrolled(m)
        n = len(m.samples)
        m.ingest(0.4, 5800, 6000)
        self.assertIs(m.sample(10, 10, 6000), r)
        self.assertEqual(len(m.samples), n)
        with self.assertRaises(ValueError):
            m.sample(11, 10, 6000)
        with self.assertRaises(ValueError):
            m.ingest(0.5, 5800, 6000)

    def test_invalid_initial_context_restarts_consecutive_enrollment(self):
        m = self.monitor()
        for i in range(24):
            self.step(m, i)
        m.invalidate_context()
        self.assertFalse(self.step(m, 24).ready)
        self.assertFalse(self.step(m, 25).ready)

    def test_rollback_and_invalid_configuration_rejected(self):
        from coact_cps.vehicle.session_anchor import SessionAnchor

        with self.assertRaises(ValueError):
            SessionAnchor(0, -1, 2)
        m = self.monitor()
        self.step(m, 0)
        with self.assertRaises(ValueError):
            m.ingest(0, 600, 800)
