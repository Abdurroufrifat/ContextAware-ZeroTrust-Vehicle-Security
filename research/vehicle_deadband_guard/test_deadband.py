import unittest
import numpy as np
import deadband_core as c
def source(values):
 v=np.array(values,float);t=1_000_000_000+np.arange(len(v),dtype=np.int64)*10_000_000;z=np.zeros((len(v),3));a=z.copy();a[:,0]=v
 return {'bag_ns':t,'header_ns':t.copy(),'usable':np.ones(len(v),bool),'acceleration':a,'angular_velocity':z,'acceleration_norm':abs(v),'angular_velocity_norm':np.zeros(len(v))}
class Checks(unittest.TestCase):
 def test_subthreshold_jitter_does_not_renew(self):
  s=source([0.,.0001,0.,.0001]);np.testing.assert_array_equal(c.stamps(s,np.ones(6)),[s['bag_ns'][0]]*4)
 def test_strict_deadband_boundary(self):
  s=source([0.,.001,.00101]);np.testing.assert_array_equal(c.stamps(s,np.ones(6)),s['bag_ns'][[0,0,2]])
 def test_small_changes_accumulate_against_anchor(self):
  s=source([0.,.0004,.0008,.0012]);np.testing.assert_array_equal(c.stamps(s,np.ones(6)),s['bag_ns'][[0,0,0,3]])
 def test_scaled_axis_changes(self):
  s=source([0.,.0015]);scale=np.array([2.,1,1,1,1,1]);np.testing.assert_array_equal(c.stamps(s,scale),[s['bag_ns'][0]]*2)
 def test_invalid_values_cannot_renew(self):
  s=source([0.,99.,0.]);s['usable'][1]=False;np.testing.assert_array_equal(c.stamps(s,np.ones(6)),[s['bag_ns'][0],-1,s['bag_ns'][0]])
 def test_zero_epsilon_matches_exact_guard(self):
  s=source([0.,1e-12,0.]);np.testing.assert_array_equal(c.stamps(s,np.ones(6),epsilon=0.),s['bag_ns'])
 def test_threshold_aware_pulses_remain_evasion(self):
  s=source([0.,.002,0.,.002]);np.testing.assert_array_equal(c.stamps(s,np.ones(6)),s['bag_ns'])
 def test_invalid_scale_rejected(self):
  with self.assertRaises(ValueError):c.stamps(source([0.,1.]),np.zeros(6))
 def test_noninteger_clock_rejected(self):
  s=source([0.,1.]);s['bag_ns']=s['bag_ns'].astype(float)
  with self.assertRaises(ValueError):c.stamps(s,np.ones(6))
 def test_iqr_zero_axis_has_explicit_fallback(self):
  out=c.scale_from_vectors(np.array([[0.,1,0,0,0,0],[2.,1,0,0,0,0],[4.,1,0,0,0,0]]));self.assertEqual(out['scale'][0],2.);self.assertEqual(out['fallback_axes'],[1,2,3,4,5]);self.assertEqual(out['scale'][1],1.)
 def test_current_data_cannot_renew_old_prediction(self):
  np.testing.assert_array_equal(c.g.is_fresh(np.array([100],np.int64),np.array([201],np.int64),budget=100),[False])
if __name__=='__main__':unittest.main()
