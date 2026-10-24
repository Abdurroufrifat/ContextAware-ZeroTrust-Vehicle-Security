import importlib,importlib.util,unittest
import numpy as np

class QualityTests(unittest.TestCase):
 def setUp(self):
  self.assertIsNotNone(importlib.util.find_spec('quality_core'),'quality rule missing')
  self.m=importlib.import_module('quality_core')
 def test_disagreement_above_one_withholds_estimate(self):self.assertFalse(self.m.usable(0.,10.,True))
 def test_boundary_is_inclusive(self):self.assertTrue(self.m.usable(3.,4.,True))
 def test_missing_history_withholds_even_when_models_agree(self):self.assertFalse(self.m.usable(3.,3.,False))
 def test_nonfinite_predictions_withhold(self):
  for x in (float('nan'),float('inf'),-float('inf')):self.assertFalse(self.m.usable(3.,x,True))
 def test_negative_speed_is_invalid(self):self.assertFalse(self.m.usable(-1.,0.,True))
 def test_boolean_is_not_a_speed(self):self.assertFalse(self.m.usable(True,0.,True))
 def test_jointly_wrong_models_can_pass_quality_rule(self):
  self.assertTrue(self.m.usable(0.,0.,True));self.assertEqual(abs(0.-10.),10.)
 def test_random_control_matches_withheld_count(self):
  keep=np.array([True,False,True,False,True]);random=self.m.matched_random(keep,141);self.assertEqual(int(random.sum()),int(keep.sum()))
 def test_random_control_is_reproducible(self):
  keep=np.array([True,False,True,False,True]);np.testing.assert_array_equal(self.m.matched_random(keep,141),self.m.matched_random(keep,141))
 def test_all_withheld_does_not_report_zero_error(self):
  result=self.m.metrics(np.array([1.,2.]),np.array([1.,2.]),np.array([False,False]));self.assertIsNone(result['accepted']['p95_mps']);self.assertEqual(result['withheld']['count'],2)
 def test_error_accounting_retains_withheld_errors(self):
  result=self.m.metrics(np.array([0.,10.]),np.array([0.,0.]),np.array([True,False]));self.assertEqual(result['all_supported']['above_2_mps'],1);self.assertEqual(result['withheld']['above_2_mps'],1);self.assertEqual(result['accepted']['above_2_mps'],0)
 def test_bad_mask_shape_is_rejected(self):
  with self.assertRaises(ValueError):self.m.metrics(np.array([1.,2.]),np.array([1.,2.]),np.array([True]))
if __name__=='__main__':unittest.main()
