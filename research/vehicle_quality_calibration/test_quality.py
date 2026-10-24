import unittest
import numpy as np
try:
 from quality_core import upper_quantile,eligible,split_roles,block_scores,extract_features
except ImportError:upper_quantile=eligible=split_roles=block_scores=extract_features=None
class QualityTests(unittest.TestCase):
 def test_insufficient_calibration_abstains(self):
  self.assertIsNotNone(upper_quantile)
  self.assertIsNone(upper_quantile(list(range(18)),.05))
 def test_correct_finite_sample_order_statistic(self):
  self.assertIsNotNone(upper_quantile)
  self.assertEqual(upper_quantile(list(range(19)),.05),18)
  self.assertEqual(upper_quantile([1.]*99,.05),1)
 def test_unbounded_estimate_cannot_authorize(self):
  self.assertIsNotNone(eligible)
  self.assertFalse(eligible(.1,None))
  self.assertFalse(eligible(.1,float("nan")))
  self.assertFalse(eligible(.1,6.))
  self.assertTrue(eligible(.1,5.))
 def test_chronological_roles_do_not_include_evaluation(self):
  self.assertIsNotNone(split_roles)
  r=[dict(recording="r",role=role,index=i,now_us=i*100000) for role in ["train","calibration","evaluation","purged"] for i in range(10)]
  q=split_roles(r)
  self.assertEqual(len(q),20)
  self.assertEqual(sum(x["quality_role"]=="fit" for x in q),10)
  self.assertEqual(sum(x["quality_role"]=="calibrate" for x in q),5)
  self.assertEqual(sum(x["quality_role"]=="check" for x in q),5)
  self.assertTrue(all(x["role"] in ("train","calibration") for x in q))
 def test_blocks_use_worst_score(self):
  self.assertIsNotNone(block_scores)
  r=[dict(recording="r",now_us=t,ratio=v) for t,v in [(0,1),(500000,4),(1000000,2)]]
  self.assertEqual(sorted(block_scores(r)),[2,4])
 def test_geometry_features_are_finite_and_do_not_need_reference(self):
  self.assertIsNotNone(extract_features)
  points=np.random.default_rng(5).normal(size=(1000,3))
  features=extract_features(points,points,np.eye(4),.1,0.)
  self.assertTrue(np.isfinite(features).all())
  self.assertGreater(len(features),10)
if __name__=="__main__":unittest.main(verbosity=2)
