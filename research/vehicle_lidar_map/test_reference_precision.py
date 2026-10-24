import unittest
import numpy as np
from run_benchmark import interp
class ReferencePrecision(unittest.TestCase):
 def test_large_epoch_integer_difference(self):
  ref=np.array([[1000000000000000,0],[1000000000001000,1]],dtype=float)
  self.assertAlmostEqual(interp(ref,1000000000000500001),.500001,places=12)
 def test_no_extrapolation(self):
  ref=np.array([[1000000000000000,0],[1000000000001000,1]],dtype=float)
  self.assertIsNone(interp(ref,999999999999999999))
  self.assertIsNone(interp(ref,1000000000001000001))
 def test_bracket_limit(self):
  self.assertIsNone(interp(np.array([[0.,0],[50001.,1]]),25000000))
  self.assertEqual(interp(np.array([[0.,0],[50000.,1]]),25000000),.5)
if __name__=="__main__":unittest.main(verbosity=2)
