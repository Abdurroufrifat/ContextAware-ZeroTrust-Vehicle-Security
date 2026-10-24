import importlib.util,unittest
import numpy as np
class Tests(unittest.TestCase):
 def core(self):
  self.assertIsNotNone(importlib.util.find_spec('calibrated_core'),'source-specific monitor not implemented')
  import calibrated_core
  return calibrated_core
 def bank(self,p,kind='return3',ok=None,ends=None,times=None,limit=1.):
  c=self.core();p=np.array(p,float);n=len(p);t=np.arange(n,dtype=np.int64)*50_000_000 if times is None else np.array(times,np.int64);e=t.copy() if ends is None else np.array(ends,np.int64)*50_000_000;return c.bank(t,p,np.ones(n,bool) if ok is None else np.array(ok,bool),e,limit,kind)
 def test_source_switch_does_not_compare_different_sources(self):
  a=self.bank([10,10,10]);b=self.bank([12,12,12]);self.assertTrue(a['allowed'].all() and b['allowed'].all())
 def test_latch_blocks_persistent_step(self):
  np.testing.assert_array_equal(self.bank([10,13,13,10,10,10],kind='latch')['allowed'],[1,0,0,0,0,0])
 def test_return_requires_three_distinct_fresh_endpoints(self):
  z=self.bank([10,13,10,10,10]);np.testing.assert_array_equal(z['allowed'],[1,0,0,0,1]);self.assertEqual(np.flatnonzero(z['recovery']).tolist(),[4])
 def test_repeated_endpoint_does_not_earn_recovery(self):
  z=self.bank([10,13,10,10,10,10],ends=[0,1,2,2,3,4]);np.testing.assert_array_equal(z['allowed'],[1,0,0,0,0,1])
 def test_unavailable_sample_breaks_recovery_run(self):
  z=self.bank([10,13,10,np.nan,10,10,10],ok=[1,1,1,0,1,1,1]);np.testing.assert_array_equal(z['allowed'],[1,0,0,0,0,0,1])
 def test_disagreement_breaks_recovery_run(self):
  z=self.bank([10,13,10,13,10,10,10]);np.testing.assert_array_equal(z['allowed'],[1,0,0,0,0,0,1])
 def test_gap_breaks_recovery_run(self):
  z=self.bank([10,13,10,10,10,10],times=[0,50_000_000,100_000_000,250_000_001,300_000_001,350_000_001]);np.testing.assert_array_equal(z['allowed'],[1,0,0,0,0,1])
 def test_no_rearm_to_persistent_false_plateau(self):
  z=self.bank([10,13,13,13,13]);self.assertEqual(int(z['recovery'].sum()),0)
 def test_startup_offset_not_detected(self):self.assertTrue(self.bank([13,13,13])['allowed'].all())
 def test_slow_drift_not_detected(self):self.assertTrue(self.bank(10+np.arange(20)*.2)['allowed'].all())
 def test_gap_can_hide_offset_before_quarantine(self):self.assertTrue(self.bank([10,13],times=[0,100_000_001])['allowed'].all())
 def test_anchor_collision_can_rearm_false_content(self):
  z=self.bank([10,13,10,10,10]);self.assertTrue(z['allowed'][-1]);self.assertTrue(z['recovery'][-1])
 def test_inclusive_limit(self):self.assertTrue(self.bank([10,11])['allowed'].all())
 def test_just_over_limit(self):self.assertFalse(self.bank([10,np.nextafter(11.,12.)])['allowed'][-1])
 def test_nominal_rejection_cannot_be_revived(self):self.assertFalse(self.bank([np.nan],ok=[0])['allowed'][0])
 def test_invalid_clock_rejected(self):
  with self.assertRaises(ValueError):self.bank([10,10],times=[0,0])
 def test_invalid_limit_rejected(self):
  with self.assertRaises(ValueError):self.bank([10],limit=-1)
 def test_causal_prefix(self):
  a=self.bank([10,13,10]);b=self.bank([10,13,10,10,10]);np.testing.assert_array_equal(a['allowed'],b['allowed'][:3])
 def test_pairs_include_every_within_100ms_lag(self):
  c=self.core();m=[{'route':'A','time_ns':t,'group':'A'} for t in (0,40_000_000,80_000_000,200_000_000)];a,b=c.pairs(m);self.assertEqual(list(zip(a,b)),[(0,1),(0,2),(1,2)])
 def test_pairs_do_not_cross_recordings(self):
  c=self.core();a,b=c.pairs([{'route':'A','time_ns':0,'group':'A'},{'route':'B','time_ns':1,'group':'B'}]);self.assertEqual(len(a),0)
 def test_duplicate_recording_time_rejected(self):
  c=self.core()
  with self.assertRaises(ValueError):c.pairs([{'route':'A','time_ns':0,'group':'A'},{'route':'A','time_ns':0,'group':'A'}])
 def test_maximum_uses_all_lags_without_labels(self):
  c=self.core();v=c.limit(np.array([0.,1.,2.]),np.array([0,0,1]),np.array([1,2,2]));self.assertEqual(v,2.)
if __name__=='__main__':unittest.main()
