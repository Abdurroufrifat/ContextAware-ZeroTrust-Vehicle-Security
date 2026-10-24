import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from map_core import interval_motion, eligible, select_available, firing_mask
from kiss_icp.preprocess import Preprocessor
class Checks(unittest.TestCase):
 def test_translation(self):
  a=np.eye(4); b=a.copy(); b[0,3]=.2
  r=interval_motion(a,b,1000000000,1100000000,np.eye(4))
  self.assertAlmostEqual(r["speed_mps"],2)
  self.assertEqual(r["midpoint_ns"],1050000000)
 def test_rear_axis_mount(self):
  m=np.eye(4);m[0,3]=2
  rear=np.eye(4);rear[:3,:3]=Rotation.from_euler("z",.1).as_matrix()
  lidar=np.linalg.inv(m)@rear@m
  r=interval_motion(np.eye(4),lidar,0,100000000,m)
  self.assertLess(r["speed_mps"],1e-10)
 def test_bad_interval(self):
  for b in (0,-1,1000000,300000000):
   with self.assertRaises(ValueError): interval_motion(np.eye(4),np.eye(4),0,b,np.eye(4))
 def test_age_boundary(self):
  self.assertTrue(eligible(100,100,100000100))
  self.assertFalse(eligible(100,100,100000101))
  self.assertFalse(eligible(101,100,100))
  self.assertFalse(eligible(99,101,100))
 def test_past_only(self):
  rows=[dict(completed_ns=90,receipt_ns=80),dict(completed_ns=101,receipt_ns=95)]
  self.assertIs(select_available(rows,100),rows[0])
  self.assertIsNone(select_available(rows,79))
 def test_future_firings(self):
  self.assertEqual(firing_mask(np.array([-0.000007,0,.1,np.nan,.3]),0,50000000).tolist(),[True,True,False,False,False])
 def test_native_end_deskew(self):
  p=Preprocessor(100,0,True,4);delta=np.eye(4);delta[0,3]=1
  out=p.preprocess(np.array([[10.,0,0],[10.,0,0],[10.,0,0]]),np.array([5.,5.05,5.1]),delta)
  np.testing.assert_allclose(out[:,0],[9,9.5,10],atol=1e-10)
if __name__=="__main__": unittest.main(verbosity=2)
