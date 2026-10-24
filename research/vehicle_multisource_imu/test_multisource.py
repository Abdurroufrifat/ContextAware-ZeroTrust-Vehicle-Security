import unittest
import numpy as np
import multisource_core as k
class MultisourceTests(unittest.TestCase):
 def call(self,name,*args,**kw):
  fn=getattr(k,name,None);self.assertTrue(callable(fn),'comparison behavior missing: '+name);return fn(*args,**kw)
 def test_primary_has120_features(self):self.assertEqual(self.call('assemble',np.zeros((2,75)),np.ones((2,45)),np.array([3.,4.]),np.array([5.,6.]),'radar_imu').shape,(2,120))
 def test_primary_preserves_signed_context_prefix(self):
  a=np.arange(150,dtype=float).reshape(2,75);x=self.call('assemble',a,np.ones((2,45)),np.array([3.,4.]),np.array([5.,6.]),'radar_imu');np.testing.assert_array_equal(x[:,:1],a[:,:1]);np.testing.assert_array_equal(x[:,3:75],a[:,3:]);np.testing.assert_array_equal(x[:,1],[3,4]);np.testing.assert_array_equal(x[:,2],[5,6])
 def test_imu_control_does_not_read_radar_anchor(self):
  im=np.arange(90,dtype=float).reshape(2,45);x=self.call('assemble',np.full((2,75),123.),im,np.array([999.,999.]),np.array([999.,999.]),'imu_only');np.testing.assert_array_equal(x,im)
 def test_unknown_method_rejected(self):
  with self.assertRaises(ValueError):self.call('assemble',np.zeros((2,75)),np.zeros((2,45)),np.zeros(2),np.zeros(2),'unknown')
 def test_misaligned_imu_rows_rejected(self):
  with self.assertRaises(ValueError):self.call('assemble',np.zeros((2,75)),np.zeros((1,45)),np.zeros(2),np.zeros(2),'radar_imu')
 def test_nan_imu_features_rejected(self):
  with self.assertRaises(ValueError):self.call('assemble',np.zeros((2,75)),np.full((2,45),np.nan),np.zeros(2),np.zeros(2),'radar_imu')
 def test_primary_uses_residual_target(self):
  target,anchor=self.call('targets_and_anchor',np.array([7.,8.]),np.array([3.,4.]),'radar_imu');np.testing.assert_array_equal(target,[4,4]);np.testing.assert_array_equal(anchor,[3,4])
 def test_imu_only_uses_direct_speed_and_zero_anchor(self):
  target,anchor=self.call('targets_and_anchor',np.array([7.,8.]),np.array([3.,4.]),'imu_only');np.testing.assert_array_equal(target,[7,8]);np.testing.assert_array_equal(anchor,[0,0])
 def test_candidate_cannot_revive_old_source_rejections(self):
  keep=self.call('endpoint_quality',np.array([1.,1.]),np.array([1.,1.]),np.array([False,True]),np.array([0,0],dtype=np.uint8));np.testing.assert_array_equal(keep,[False,True])
 def test_missing_imu_source_rejects_even_agreeing_model(self):
  keep=self.call('endpoint_quality',np.array([1.,1.]),np.array([1.,1.]),np.array([True,True]),np.array([0,4],dtype=np.uint8));np.testing.assert_array_equal(keep,[True,False])
 def test_unchanged_one_mps_peer_guard(self):
  keep=self.call('endpoint_quality',np.array([1.,1.000001]),np.array([0.,0.]),np.array([True,True]),np.array([0,0],dtype=np.uint8));np.testing.assert_array_equal(keep,[True,False])
 def test_binding_rejection_masks_both_exported_predictions(self):
  replay={'status':np.array([0,4],dtype=np.uint8),'prediction_mps':np.array([1.,np.nan]),'timing_prediction_mps':np.array([1.,2.])};out,ledger=self.call('bind_requests',replay,np.array([False,True]));self.assertEqual(out['status'].tolist(),[4,4]);self.assertTrue(np.isnan(out['prediction_mps'][0]));self.assertTrue(np.isnan(out['timing_prediction_mps'][0]));self.assertEqual(ledger,1)
 def test_binding_does_not_revive_quality_rejections(self):
  replay={'status':np.array([4],dtype=np.uint8),'prediction_mps':np.array([np.nan]),'timing_prediction_mps':np.array([1.])};out,ledger=self.call('bind_requests',replay,np.array([True]));self.assertEqual(out['status'].tolist(),[4]);self.assertTrue(np.isnan(out['prediction_mps'][0]));self.assertEqual(ledger,0)
class VerifierTests(unittest.TestCase):
 def check(self):
  import verify_multisource as v
  self.assertTrue(callable(getattr(v,'compare_replay',None)),'missing exact replay comparison');return v.compare_replay
 def test_epoch_clock_difference_of_one_ns_is_rejected(self):
  compare=self.check()
  with self.assertRaises(AssertionError):compare(np.array([1700781855718492373],dtype=np.int64),np.array([1700781855718492372],dtype=np.int64))
 def test_boolean_replay_flags_are_exact(self):
  compare=self.check()
  with self.assertRaises(AssertionError):compare(np.array([False]),np.array([True]))
 def test_float_predictions_keep_numeric_tolerance(self):self.check()(np.array([1.+5e-10]),np.array([1.]))
if __name__=='__main__':unittest.main(verbosity=2)
