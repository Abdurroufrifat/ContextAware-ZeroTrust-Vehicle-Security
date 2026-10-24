import unittest
import numpy as np
try:
 from surface_diagnosis import surface_metrics
except ImportError:surface_metrics=None
class SurfaceTests(unittest.TestCase):
 def grid(self):
  x,y=np.meshgrid(np.arange(-10.,11),np.arange(-10.,11))
  return np.column_stack((x.ravel(),y.ravel(),np.zeros(x.size)))
 def test_tangential_sampling_error_is_distinct_from_surface_error(self):
  self.assertIsNotNone(surface_metrics)
  p=self.grid();s=p[(np.abs(p[:,0])<8)&(np.abs(p[:,1])<8)]+[.4,.4,0]
  q=surface_metrics(s,p,np.eye(4))
  self.assertGreater(q["euclidean_p95_m"],.3)
  self.assertLess(q["plane_p95_m"],1e-10)
 def test_single_plane_cannot_certify_full_pose(self):
  self.assertIsNotNone(surface_metrics)
  p=self.grid();q=surface_metrics(p,p,np.eye(4))
  self.assertLess(q["jacobian_rank"],6)
  self.assertFalse(q["physical_error_certified"])
 def test_normal_displacement_is_detected(self):
  self.assertIsNotNone(surface_metrics)
  p=self.grid();q=surface_metrics(p+[0,0,.4],p,np.eye(4))
  self.assertAlmostEqual(q["plane_p95_m"],.4)
if __name__=="__main__":unittest.main(verbosity=2)
