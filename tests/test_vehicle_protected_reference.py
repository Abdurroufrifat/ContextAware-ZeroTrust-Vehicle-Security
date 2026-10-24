import unittest


class ProtectedReferenceTests(unittest.TestCase):
    def reference(self, bound=2.0):
        from coact_cps.vehicle.protected_reference import ProtectedReference

        return ProtectedReference(bound, lag_ms=2000, maximum_age_ms=90000)

    def enroll(self, reference):
        for end in range(200, 4200, 200):
            reference.ingest(-0.6, end - 200, end)
            reference.observe(10.0, -0.6, end, quiet=True)
        reference.freeze(4000)

    def test_fixed_bias_prediction_ignores_forged_measurements(self):
        r = self.reference()
        self.enroll(r)
        for end in range(4200, 24200, 200):
            r.ingest(-0.6, end - 200, end)
            r.observe(18.0, 3.0, end, quiet=True)
        good = r.check(10.0, 10.0, 24000)
        self.assertTrue(good.eligible)
        self.assertAlmostEqual(good.prediction, 10.0)
        self.assertFalse(r.check(18.0, 18.0, 24000).eligible)

    def test_snapshot_must_predate_quarantine_by_two_seconds(self):
        r = self.reference()
        self.enroll(r)
        self.assertEqual(r.check(10, 10, 4000).age_ms, 2000)

    def test_no_old_quiet_snapshot_means_no_fallback(self):
        r = self.reference()
        r.ingest(0, 0, 200)
        r.observe(10, 0, 200, quiet=True)
        r.freeze(200)
        self.assertFalse(r.check(10, 10, 200).eligible)

    def test_inclusive_velocity_bound(self):
        r = self.reference()
        self.enroll(r)
        self.assertTrue(r.check(12, 12, 4000).eligible)
        self.assertFalse(r.check(12.001, 10, 4000).eligible)

    def test_expiry_is_measured_from_selected_snapshot(self):
        r = self.reference()
        self.enroll(r)
        for end in range(4200, 92200, 200):
            r.ingest(-0.6, end - 200, end)
        self.assertTrue(r.check(10, 10, 92000).eligible)
        r.ingest(-0.6, 92000, 92200)
        self.assertTrue(r.check(10, 10, 92200).expired)
        self.assertFalse(r.check(10, 10, 92200).eligible)

    def test_imu_gap_loses_reference_until_release(self):
        r = self.reference()
        self.enroll(r)
        r.ingest(-0.6, 4200, 4400)
        r.ingest(-0.6, 4400, 4600)
        self.assertTrue(r.check(10, 10, 4600).lost)
        self.assertFalse(r.check(10, 10, 4600).eligible)
        r.release()
        self.assertFalse(r.check(10, 10, 4600).eligible)

    def test_duplicate_interval_does_not_integrate_twice(self):
        r = self.reference()
        self.enroll(r)
        before = r.check(10, 10, 4000)
        r.ingest(-0.6, 3800, 4000)
        self.assertEqual(r.check(10, 10, 4000), before)
        with self.assertRaises(ValueError):
            r.ingest(-0.5, 3800, 4000)
        with self.assertRaises(ValueError):
            r.ingest(-0.6, 3600, 3800)

    def test_nonquiet_snapshot_is_not_an_enrollment_source(self):
        r = self.reference()
        for end in range(200, 4200, 200):
            r.ingest(0, end - 200, end)
            r.observe(18, 0, end, quiet=False)
        r.freeze(4000)
        self.assertFalse(r.check(18, 18, 4000).eligible)

    def test_coherent_imu_and_speed_forgery_can_pass(self):
        r = self.reference()
        self.enroll(r)
        for end in range(4200, 6200, 200):
            r.ingest(1.9, end - 200, end)
        self.assertTrue(r.check(15, 15, 6000).eligible)

    def test_invalid_bounds_and_samples_rejected(self):
        for bad in (0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                self.reference(bad)
        r = self.reference()
        r.ingest(0, 0, 200)
        with self.assertRaises(ValueError):
            r.observe(float("nan"), 0, 200, quiet=True)
        with self.assertRaises(ValueError):
            r.check(-11, 10, 200)

    def test_check_exposes_frozen_snapshot_for_independent_audit(self):
        r = self.reference()
        self.enroll(r)
        check = r.check(10, 10, 4000)
        self.assertEqual(check.anchor_time_ms, 2000)
        self.assertEqual(check.snapshot_speed, 10)
        self.assertEqual(check.snapshot_bias, -0.6)
        r.release()
        self.assertEqual(check.anchor_time_ms, 2000)
