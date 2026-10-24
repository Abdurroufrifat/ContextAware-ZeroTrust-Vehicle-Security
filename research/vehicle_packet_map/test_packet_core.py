import unittest
import numpy as np
from scipy.spatial.transform import Rotation
try:
 from packet_core import scheduled_start, window_mask, scale_motion, interval, geometry, PacketMap
except ImportError:
 scheduled_start=window_mask=scale_motion=interval=geometry=PacketMap=None
class PacketTests(unittest.TestCase):
 def test_scheduler_waits_for_busy_worker(self):
  self.assertIsNotNone(scheduled_start,"scheduler is missing")
  self.assertEqual(scheduled_start(100,120),120)
 def test_window_excludes_future_and_expired(self):
  self.assertIsNotNone(window_mask)
  self.assertEqual(window_mask(np.array([79,80,90,100,101]),100,20).tolist(),[False,True,True,True,False])
 def test_scaling_translation(self):
  self.assertIsNotNone(scale_motion)
  d=np.eye(4);d[0,3]=.4
  self.assertAlmostEqual(scale_motion(d,.25)[0,3],.1)
 def test_screw_scaling_composes(self):
  self.assertIsNotNone(scale_motion)
  d=np.eye(4);d[:3,:3]=Rotation.from_euler("z",.5).as_matrix();d[:3,3]=[.3,.2,0]
  half=scale_motion(d,.5)
  np.testing.assert_allclose(half@half,d,atol=1e-10)
 def test_variable_interval_and_mount(self):
  self.assertIsNotNone(interval)
  m=np.eye(4);m[0,3]=2
  rear=np.eye(4);rear[:3,:3]=Rotation.from_euler("z",.01).as_matrix()
  lidar=np.linalg.inv(m)@rear@m
  r=interval(np.eye(4),lidar,1000000000,1040000000,m)
  self.assertLess(r["speed_mps"],1e-10);self.assertEqual(r["midpoint_ns"],1020000000)
 def test_interval_rejects_expired_state(self):
  self.assertIsNotNone(interval)
  with self.assertRaises(ValueError):interval(np.eye(4),np.eye(4),0,200000001,np.eye(4))
 def test_geometry_rejects_unmatched_cloud(self):
  self.assertIsNotNone(geometry)
  a=np.random.default_rng(10).uniform(-1,1,(500,3))
  self.assertFalse(geometry(a,a+10,np.eye(4))["passed"])
 def test_geometry_rejects_line(self):
  self.assertIsNotNone(geometry)
  a=np.zeros((500,3));a[:,0]=np.linspace(-10,10,500)
  self.assertFalse(geometry(a,a,np.eye(4))["passed"])
 def test_geometry_accepts_known_full_scene(self):
  self.assertIsNotNone(geometry)
  a=np.random.default_rng(10).uniform(-10,10,(500,3));p=np.eye(4);p[0,3]=.1
  self.assertTrue(geometry(a,a+[.1,0,0],p)["passed"])
 def test_known_translation_and_past_map(self):
  self.assertIsNotNone(PacketMap)
  model=PacketMap();points=np.random.default_rng(123).uniform(-15,15,(18000,3))
  first=model.step(points,np.linspace(0,.1,len(points)),1000000000,np.eye(4))
  second=model.step(points-[.04,0,0],np.linspace(0,.02,len(points)),1020000000,np.eye(4))
  self.assertTrue(first["bootstrap"]);self.assertFalse(first["quality_passed"])
  self.assertEqual(second["prior_updates"],1)
  self.assertTrue(second["quality_passed"])
  self.assertAlmostEqual(second["motion"]["speed_mps"],2.,delta=.03)
if __name__=="__main__":unittest.main(verbosity=2)
