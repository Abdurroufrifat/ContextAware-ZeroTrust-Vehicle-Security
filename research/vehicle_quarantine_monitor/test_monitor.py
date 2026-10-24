"""Behavior and explicit counterexamples for the quarantine heuristic."""
import importlib.util,unittest
import numpy as np
class Tests(unittest.TestCase):
 def core(self):
  self.assertIsNotNone(importlib.util.find_spec('monitor_core'),'quarantine monitor not implemented')
  import monitor_core
  return monitor_core
 def run_values(self,values,times=None,base=None):
  c=self.core();v=np.array(values,float);t=np.arange(len(v),dtype=np.int64)*50_000_000 if times is None else np.array(times,np.int64);b=np.ones(len(v),bool) if base is None else np.array(base,bool)
  return c.monitor(t,v,b)
 def test_abrupt_bias_is_withheld_at_first_bad_request(self):
  out=self.run_values([10,10,13,13]);np.testing.assert_array_equal(out['allowed'],[1,1,0,0]);self.assertEqual(out['first_alarm'],2)
 def test_quarantine_survives_long_clock_gap(self):
  out=self.run_values([10,13,10],[0,50_000_000,5_000_000_000]);np.testing.assert_array_equal(out['allowed'],[1,0,0])
 def test_base_rejection_cannot_be_revived(self):
  out=self.run_values([10,np.nan,10],base=[1,0,1]);np.testing.assert_array_equal(out['allowed'],[1,0,1])
 def test_inclusive_one_mps_step_is_allowed(self):
  out=self.run_values([10,11]);self.assertTrue(out['allowed'].all())
 def test_just_above_one_mps_step_latches(self):
  out=self.run_values([10,np.nextafter(11.,12.)]);self.assertFalse(out['allowed'][1])
 def test_exact_100ms_still_compares(self):
  out=self.run_values([10,13],[0,100_000_000]);self.assertFalse(out['allowed'][1])
 def test_gap_reanchor_is_explicit_counterexample(self):
  out=self.run_values([10,13],[0,100_000_001]);self.assertTrue(out['allowed'].all());self.assertEqual(out['reason'][1],4)
 def test_slow_drift_is_explicit_counterexample(self):
  out=self.run_values(10+np.minimum(np.arange(41)*.1,3));self.assertTrue(out['allowed'].all());self.assertIsNone(out['first_alarm'])
 def test_startup_bias_is_explicit_counterexample(self):
  self.assertTrue(self.run_values([13,13,13])['allowed'].all())
 def test_legitimate_fast_change_is_withheld(self):
  np.testing.assert_array_equal(self.run_values([10,11.01,11.01])['allowed'],[1,0,0])
 def test_duplicate_clock_is_rejected(self):
  with self.assertRaises(ValueError):self.run_values([10,10],[0,0])
 def test_accepted_nan_is_rejected(self):
  with self.assertRaises(ValueError):self.run_values([10,np.nan])
 def test_integer_flags_are_rejected(self):
  c=self.core()
  with self.assertRaises(ValueError):c.monitor(np.array([0],np.int64),np.array([10.]),np.array([1],np.uint8))
 def test_no_accepted_values_has_no_anchor_or_alarm(self):
  out=self.run_values([np.nan,np.nan],base=[0,0]);self.assertFalse(out['allowed'].any());self.assertIsNone(out['first_alarm'])
 def test_prefix_is_unchanged_by_future_values(self):
  prefix=self.run_values([10,10.2]);long=self.run_values([10,10.2,30,10]);np.testing.assert_array_equal(prefix['allowed'],long['allowed'][:2]);np.testing.assert_array_equal(prefix['reason'],long['reason'][:2])
if __name__=='__main__':unittest.main()
