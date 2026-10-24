import unittest
import numpy as np
try:
 from controls import warp_to_sensor,discrepancy,select_anchors
except ImportError:
 warp_to_sensor=discrepancy=select_anchors=None
class Controls(unittest.TestCase):
 def test_known_pose_recovers_world_points(self):
  self.assertIsNotNone(warp_to_sensor)
  p=np.array([[1.,2,3],[4,5,6]]);t=np.eye(4);t[0,3]=.16
  q=warp_to_sensor(p,t)
  np.testing.assert_allclose(q@t[:3,:3].T+t[:3,3],p)
 def test_identical_points_have_zero_residual(self):
  self.assertIsNotNone(discrepancy)
  p=np.random.default_rng(1).normal(size=(400,3))
  self.assertLess(discrepancy(p,p,np.eye(4))["p95_m"],1e-12)
 def test_sampling_can_reject_a_correct_pose(self):
  self.assertIsNotNone(discrepancy)
  target=np.column_stack((np.arange(400)*2.,np.zeros(400),np.zeros(400)))
  source=target+[.4,0,0]
  r=discrepancy(source,target,np.eye(4))
  self.assertGreater(r["p95_m"],.3)
  self.assertFalse(r["residual_screen_passed"])
 def test_only_training_calibration_anchors(self):
  self.assertIsNotNone(select_anchors)
  rows=[dict(recording="r",role=role,index=i,now_us=i) for role in ["train","calibration","evaluation","purged"] for i in range(12)]
  anchors=select_anchors(rows)
  self.assertEqual(len(anchors),6)
  self.assertEqual({a["role"] for a in anchors},{"train","calibration"})
  self.assertEqual([a["index"] for a in anchors[:3]],[2,5,8])
if __name__=="__main__":unittest.main(verbosity=2)
