"""Boundary tests for offline policy-input compromise, not hardware tampering."""
import importlib.util,unittest
import numpy as np
class Tests(unittest.TestCase):
 def core(self):
  self.assertIsNotNone(importlib.util.find_spec('compromise_core'),'source-compromise replay is not implemented')
  import compromise_core
  return compromise_core
 def vectors(self,case):
  c=self.core();t=np.array([9,10,19,20],np.int64);p={'fusion':np.array([4.,4.,4.,4.])};r=np.array([4.,4.,4.,4.]);q=r.copy()
  out=c.attack_vectors(t,10,20,r,p,q,case)
  np.testing.assert_array_equal(r,[4]*4);np.testing.assert_array_equal(p['fusion'],[4]*4);np.testing.assert_array_equal(q,[4]*4)
  return out
 def test_radar_alone_does_not_move_peer_or_fusion(self):
  r,p,q,a=self.vectors('radar_plus3');np.testing.assert_array_equal(r,[4,7,7,4]);np.testing.assert_array_equal(q,[4]*4);np.testing.assert_array_equal(p['fusion'],[4]*4)
 def test_shared_bias_preserves_agreement_and_window_edges(self):
  r,p,q,a=self.vectors('all_peer_plus3');np.testing.assert_array_equal(r,q);np.testing.assert_array_equal(r,p['fusion']);np.testing.assert_array_equal(a,[False,True,True,False]);np.testing.assert_array_equal(r,[4,7,7,4])
 def test_peer_alone_does_not_change_predictions(self):
  r,p,q,a=self.vectors('peer_plus3');np.testing.assert_array_equal(r,[4]*4);np.testing.assert_array_equal(q,[4,7,7,4]);np.testing.assert_array_equal(p['fusion'],[4]*4)
 def test_missing_prediction_remains_missing(self):
  c=self.core();r,p,q,a=c.attack_vectors(np.array([10]),10,20,np.array([np.nan]),{'fusion':np.array([np.nan])},np.array([np.nan]),'all_peer_plus3');self.assertTrue(np.isnan(r[0]) and np.isnan(p['fusion'][0]) and np.isnan(q[0]))
 def test_unknown_case_rejected(self):
  with self.assertRaises(ValueError):self.vectors('unknown')
 def test_nonfinite_peer_rejected_and_inclusive_threshold(self):
  c=self.core();keep=c.endpoint_quality(np.array([4.,4.,4.,4.]),np.array([5.,np.nextafter(5.,6.),np.nan,4.]),np.array([True,True,True,False]),np.array([0,0,0,0]));np.testing.assert_array_equal(keep,[True,False,False,False])
 def test_imu_denial_not_revived_by_agreement(self):
  c=self.core();keep=c.endpoint_quality(np.array([7.]),np.array([7.]),np.array([True]),np.array([1]));self.assertFalse(keep[0])
 def test_attacked_endpoint_survives_until_next_endpoint(self):
  c=self.core();events=np.array([0,100_000_000,200_000_000]);times=np.array([99_999_999,100_000_000,199_999_999,200_000_000]);p=np.array([4.,7.,4.]);q=c.make_branch(times,events,np.ones(3,bool),p,p,np.ones(3,bool),np.zeros(3,int),np.ones(4,bool),np.ones(4,bool));np.testing.assert_array_equal(q['prediction_mps'],[4,7,7,4])
 def test_binding_and_content_fail_closed(self):
  c=self.core();q=c.make_branch(np.array([0,1,2]),np.array([0]),np.ones(1,bool),np.array([4.]),np.array([4.]),np.ones(1,bool),np.zeros(1,int),np.array([True,False,True]),np.array([True,True,False]));np.testing.assert_array_equal(q['status'],[0,4,4]);self.assertTrue(np.isnan(q['prediction_mps'][1:]).all())
 def test_nominal_control_matches_saved_served_output_schema(self):
  c=self.core();c.branches('nominal',c.inputs())
 def test_negative_values_not_admitted(self):
  c=self.core();self.assertFalse(c.endpoint_quality(np.array([-1.]),np.array([-1.]),np.array([True]),np.array([0]))[0])
if __name__=='__main__':unittest.main()
