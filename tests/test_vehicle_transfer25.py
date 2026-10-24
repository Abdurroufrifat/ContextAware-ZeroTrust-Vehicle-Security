import unittest

import numpy as np

from coact_cps.vehicle.transfer25 import normal_metrics, scaled_windows


class TransferTests(unittest.TestCase):
    def test_windows_do_not_cross_timestamp_gap(self):
        f = np.zeros((7, 3));t = np.array([100,200,300,500,600,700,800]);s = t - 100
        x,p = scaled_windows(f,t,s,3,dict(mean=[0]*3,scale=[1]*3))
        self.assertEqual(p.tolist(),[2,5,6])
        self.assertEqual(x.shape,(3,3,3))

    def test_original_float32_then_scaling_order_is_retained(self):
        f = np.full((3,3), 1.00000007)
        x,_ = scaled_windows(f,np.array([100,200,300]),np.array([0,100,200]),3,
                             dict(mean=[1.]*3,scale=[0.1]*3))
        expected=((f.astype(np.float32).astype(float)-1.)/0.1).astype(np.float32)
        np.testing.assert_array_equal(x[0],expected)

    def test_positive_scaler_required(self):
        with self.assertRaises(ValueError):
            scaled_windows(np.zeros((3,3)),np.array([1,2,3]),np.array([0,1,2]),3,
                           dict(mean=[0]*3,scale=[1,0,1]))

    def test_unscored_startup_is_not_false_alarm(self):
        r,score,alarm = normal_metrics(np.array([[1.,0.,0.],[2.,0.,0.]]),
            dict(median=[0]*3,scale=[1]*3),1.,np.array([2,3]),
            np.array([100,200,300,400]),np.array([0,100,200,300]))
        self.assertEqual(alarm.tolist(),[False,True])
        self.assertEqual(r['presumed_normal_alarm_fraction'],0.5)
        self.assertEqual(r['unscored_epochs'],2)

    def test_alarm_runs_break_at_missing_epoch(self):
        r,_,_ = normal_metrics(np.full((2,3),2.),dict(median=[0]*3,scale=[1]*3),1.,
            np.array([0,2]),np.array([100,200,300]),np.array([0,100,200]))
        self.assertEqual(r['longest_alarm_run_epochs'],1)

    def test_nonfinite_nodes_rejected(self):
        with self.assertRaises(ValueError):
            normal_metrics(np.array([[np.nan,0.,0.]]),dict(median=[0]*3,scale=[1]*3),
                           1.,np.array([0]),np.array([100]),np.array([0]))

    def test_duplicate_positions_rejected(self):
        with self.assertRaises(ValueError):
            normal_metrics(np.zeros((2,3)),dict(median=[0]*3,scale=[1]*3),1.,
                           np.array([0,0]),np.array([100]),np.array([0]))


if __name__ == '__main__':
    unittest.main()
