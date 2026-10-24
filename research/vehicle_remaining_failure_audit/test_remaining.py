"""Regressions for clock precision, descriptive denominators and training support."""
import unittest,numpy as np
import remaining_core as k
class RemainingTests(unittest.TestCase):
 def call(self,n,*args):
  f=getattr(k,n,None);self.assertTrue(callable(f),'missing diagnostic behavior: '+n);return f(*args)
 def test_epoch_bucket_keeps_one_nanosecond_boundary(self):
  t=1_700_000_000_000_000_000;np.testing.assert_array_equal(self.call('bucket_ids',np.array([t,t+19_999_999_999,t+20_000_000_000],dtype=np.int64),t),[0,0,1])
 def test_float_clock_rejected(self):
  with self.assertRaises(ValueError):self.call('bucket_ids',np.array([1.,2.]),0)
 def test_backward_clock_rejected(self):
  with self.assertRaises(ValueError):self.call('bucket_ids',np.array([2,1],dtype=np.int64),0)
 def test_run_includes_exact_gap_boundary(self):
  r=self.call('runs',np.array([1,150_000_001],dtype=np.int64),np.array([.6,.7]));self.assertEqual(len(r),1);self.assertEqual(r[0]['rows'],2)
 def test_run_splits_after_gap_boundary(self):
  self.assertEqual(len(self.call('runs',np.array([1,150_000_002],dtype=np.int64),np.array([.6,.7]))),2)
 def test_good_sample_splits_run(self):
  self.assertEqual(len(self.call('runs',np.array([1,2,3],dtype=np.int64),np.array([.6,.5,.7]))),2)
 def test_training_range_includes_equal_boundaries(self):
  a=np.array([[1.,2.],[3.,4.]]);b=np.array([[1.,4.],[0.,5.]]);np.testing.assert_array_equal(self.call('support_flags',a,b),[[False,False],[True,True]])
 def test_training_support_does_not_refit_on_query(self):
  train=np.array([[1.,2.],[3.,4.]]);a=self.call('support_flags',train,np.array([[2.,3.]]));b=self.call('support_flags',train,np.array([[2.,3.],[999.,999.]]));np.testing.assert_array_equal(a,b[:1])
 def test_nonfinite_training_support_rejected(self):
  with self.assertRaises(ValueError):self.call('support_flags',np.array([[np.nan]]),np.ones((1,1)))
 def test_head_terms_reproduce_uncapped_delta(self):
  h={'mean':[0.]*6,'scale':[2.]*6,'coefficient':[1.]*6,'intercept':-.5};terms,delta=self.call('head_terms',h,np.ones((1,6)));np.testing.assert_array_equal(terms,np.ones((1,6))*.5);np.testing.assert_allclose(delta,[2.5])
 def test_empty_metric_is_explicit(self):
  r=self.call('metrics',np.array([]),np.array([]));self.assertEqual(r['count'],0);self.assertIsNone(r['p95_mps']);self.assertIsNone(r['tail_overprediction_fraction'])
 def test_exact_limit_is_not_exceedance(self):
  r=self.call('metrics',np.array([1.]),np.array([1.5]));self.assertEqual(r['above_half_mps'],0)
 def test_tail_directions_use_tail_denominator(self):
  r=self.call('metrics',np.array([1.,2.,3.]),np.array([1.8,1.2,3.1]));self.assertEqual(r['above_half_mps'],2);self.assertEqual(r['tail_overprediction_fraction'],.5)
 def test_nonfinite_prediction_rejected(self):
  with self.assertRaises(ValueError):self.call('metrics',np.array([1.]),np.array([np.nan]))
 def test_statistics_count_missing_without_changing_total(self):
  r=self.call('stats',np.array([1.,np.nan,3.]));self.assertEqual(r['total'],3);self.assertEqual(r['finite'],2);self.assertEqual(r['median'],2.)
 def test_imu_ages_use_actual_lookup_age_fields(self):
  t=1_700_000_000_000_000_000;s={'bag_ns':np.array([t-5],dtype=np.int64),'header_ns':np.array([t-7],dtype=np.int64),'usable':np.array([True]),'acceleration_norm':np.ones(1),'angular_velocity_norm':np.ones(1)};q=self.call('imu_ages',s,np.array([t],dtype=np.int64));self.assertEqual(q['imu_header_age_ns'].tolist(),[7]);self.assertEqual(q['imu_arrival_age_ns'].tolist(),[5])
 def test_missing_latest_imu_has_missing_age_sentinel(self):
  s={'bag_ns':np.array([10],dtype=np.int64),'header_ns':np.array([10],dtype=np.int64),'usable':np.array([True]),'acceleration_norm':np.ones(1),'angular_velocity_norm':np.ones(1)};q=self.call('imu_ages',s,np.array([9],dtype=np.int64));self.assertEqual(q['imu_header_age_ns'].tolist(),[-1]);self.assertEqual(q['imu_status'].tolist(),[1])
if __name__=='__main__':unittest.main(verbosity=2)
