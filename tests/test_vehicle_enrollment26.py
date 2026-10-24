import unittest
import numpy as np
from coact_cps.vehicle.enrollment26 import enroll_profile, calibrate_threshold, score, screen


class EnrollmentTests(unittest.TestCase):
    def test_attack_label_cannot_enter_enrollment(self):
        with self.assertRaises(ValueError):
            enroll_profile(np.zeros((3,3)),np.array([False,True,False]))

    def test_normalization_uses_declared_median_iqr_and_floor(self):
        p=enroll_profile(np.array([[0.,1.,2.],[2.,1.,4.],[4.,1.,6.]]),np.zeros(3,dtype=bool))
        self.assertEqual(p,dict(median=[2.,1.,4.],scale=[2.,0.01,2.]))

    def test_calibration_rejects_attack_rows(self):
        with self.assertRaises(ValueError):
            calibrate_threshold(np.zeros((3,3)),np.ones(3,dtype=bool),
                                dict(median=[0]*3,scale=[1]*3))

    def test_quantile_is_normal_only_and_strict_alarm_boundary_is_separate(self):
        x=np.array([[0.,0.,0.],[1.,0.,0.],[2.,0.,0.]])
        p=dict(median=[0]*3,scale=[1]*3)
        self.assertAlmostEqual(calibrate_threshold(x,np.zeros(3,dtype=bool),p),1.9)
        self.assertEqual((score(x,p)>1.).tolist(),[False,False,True])

    def test_nonfinite_residuals_rejected(self):
        with self.assertRaises(ValueError):
            enroll_profile(np.full((2,3),np.nan),np.zeros(2,dtype=bool))

    def test_nonpositive_profile_scale_rejected(self):
        with self.assertRaises(ValueError):
            score(np.zeros((2,3)),dict(median=[0]*3,scale=[1,0,1]))

    def test_low_alarm_rate_cannot_override_missed_event(self):
        r=screen(0.02,0.3,0.9,0.7,0.,1,False,False)
        self.assertFalse(r['passed'])
        self.assertFalse(r['eligible_for_test'])

    def test_longer_grant_run_blocks_candidate(self):
        self.assertFalse(screen(0.02,0.3,0.9,0.7,0.,0,True,False)['passed'])

    def test_clean_screen_still_does_not_promote(self):
        r=screen(0.02,0.3,0.9,0.7,0.,0,False,False)
        self.assertTrue(r['passed'])
        self.assertFalse(r['eligible_for_test'])


if __name__=='__main__':unittest.main()
