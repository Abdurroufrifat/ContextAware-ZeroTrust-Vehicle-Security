import unittest
import numpy as np
import reference_core as c
class Checks(unittest.TestCase):
 def test_join_requires_exact_window_and_integer_nanoseconds(self):
  t=1700000000000000000;records=[{'window_id':'a','time_ns':t}]
  self.assertEqual(c.exact_ids(records,[('a',t),('a',t+1),('b',t)]),[0,-1,-1])
 def test_duplicate_reference_keys_rejected(self):
  with self.assertRaises(ValueError):c.exact_ids([{'window_id':'a','time_ns':1}]*2,[])
 def test_float_epoch_rejected(self):
  with self.assertRaises(ValueError):c.exact_ids([{'window_id':'a','time_ns':1.7e18}],[])
 def test_empty_comparison_is_explicit(self):
  self.assertEqual(c.metrics([],[]),{'count':0,'mae_mps':None,'rmse_mps':None,'p95_mps':None,'maximum_mps':None,'bias_mps':None,'above_half_mps':0,'above_2_mps':0})
 def test_tail_retained(self):
  self.assertEqual(c.metrics([0.,0.,0.],[0.,1.,3.]),{'count':3,'mae_mps':4/3,'rmse_mps':np.sqrt(10/3),'p95_mps':2.8,'maximum_mps':3.,'bias_mps':4/3,'above_half_mps':2,'above_2_mps':1})
 def test_missing_comparisons_not_filled(self):
  np.testing.assert_array_equal(c.select([True,False,True],[True,True,True],[True,True,False]),[True,False,False])
 def test_row_reordering_rejected(self):
  with self.assertRaises(ValueError):c.check_ids(np.array([1,2]),np.array([2,1]))
 def test_nonfinite_matched_speed_rejected(self):
  with self.assertRaises(ValueError):c.metrics([1.],[np.nan])
if __name__=='__main__':unittest.main()
