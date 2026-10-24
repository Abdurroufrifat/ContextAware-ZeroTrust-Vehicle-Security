"""Stored cohort identity stays exact; replay permits bounded arithmetic roundoff."""
import unittest
import numpy as np
import verify_multidrive as v

class NumericalReplayTests(unittest.TestCase):
 def setUp(self):
  self.assertTrue(callable(getattr(v,'check_calibration_cohort',None)),'the portable calibration-cohort check is missing')
 def check(self,*args):return v.check_calibration_cohort(*args)
 def values(self):return {'sg_feature':np.array([6.719282126639535,0.178524]),'y':np.array([3.,0.])}
 def test_machine_roundoff_in_rebuilt_feature_is_accepted(self):
  saved=self.values();old=self.values();rebuilt=self.values();rebuilt['sg_feature']+=1.2e-14
  self.check(rebuilt,saved,old)
 def test_stored_cohort_change_is_rejected_even_if_tiny(self):
  saved=self.values();saved['sg_feature'][0]+=1.2e-14
  with self.assertRaises(AssertionError):self.check(saved,saved,self.values())
 def test_material_replay_change_is_rejected(self):
  rebuilt=self.values();rebuilt['sg_feature'][0]+=.000001
  with self.assertRaises(AssertionError):self.check(rebuilt,self.values(),self.values())
 def test_missing_array_is_rejected(self):
  saved=self.values();del saved['y']
  with self.assertRaises((AssertionError,ValueError)):self.check(self.values(),saved,self.values())
if __name__=='__main__':unittest.main()
