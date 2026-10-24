import unittest
import numpy as np
import nonlinear_core as c

class Checks(unittest.TestCase):
 def test_tree_boundary_and_missing(self):
  model={'baseline':1.,'trees':[[{'value':0.,'feature_idx':0,'num_threshold':2.,'left':1,'right':2,'is_leaf':False,'missing_go_to_left':False},{'value':3.,'is_leaf':True},{'value':5.,'is_leaf':True}]]}
  np.testing.assert_array_equal(c.tree_predict(model,np.array([[2.],[3.],[np.nan]])),[4.,6.,6.])
 def test_ridge_unpenalized_intercept(self):
  x=np.arange(21.).reshape(3,7);m=c.fit_ridge(x,np.full(3,4.),np.ones(3));np.testing.assert_allclose(c.ridge_predict(m,x),4.)
 def test_training_scope_excludes_held(self):
  with self.assertRaises(ValueError):c.validate_split([0,1],[1,2],3)
  c.validate_split([0,1],[2],3)
 def test_requests_never_use_future(self):
  q=c.serve(np.array([9,10,111]),np.array([10]),np.array([True]),np.array([2.]),np.array([True]),np.array([True]),ttl=100)
  np.testing.assert_array_equal(q['status'],[1,0,2]);self.assertTrue(np.isnan(q['prediction_mps'][0]))
 def test_guard_cannot_revive_source(self):
  np.testing.assert_array_equal(c.quality(np.array([1.,1.,-1.,3.]),np.ones(4),np.array([False,True,True,True]),np.zeros(4)),[False,True,False,False])
 def test_binding_fail_closed(self):
  q=c.serve(np.array([10]),np.array([10]),np.array([True]),np.array([2.]),np.array([True]),np.array([False]));self.assertEqual(q['status'][0],4);self.assertTrue(np.isnan(q['prediction_mps'][0]))
 def test_metrics_include_tail(self):
  m=c.metrics(np.zeros(3),np.array([0.,1.,3.]));self.assertEqual(m['above_2_mps'],1);self.assertAlmostEqual(m['p95_mps'],2.8)
 def test_nonfinite_features_rejected(self):
  with self.assertRaises(ValueError):c.head_features(np.ones(2),np.full((2,6),np.nan))

if __name__=='__main__':unittest.main()
