"""Portable checks for matched-data comparison, independent of sklearn."""
import unittest
import numpy as np
import matched_core as b

class MatchedTests(unittest.TestCase):
 def setUp(self):
  self.x=np.array([[0.,7.],[2.,7.],[8.,7.]])
  self.w=np.array([1.,2.,1.])
 def test_weighted_training_scaler(self):
  s=b.scaler(self.x,self.w)
  np.testing.assert_allclose(s['mean'],[3.,7.])
  np.testing.assert_allclose(s['scale'],[3.,1.])
 def test_held_feature_changes_cannot_enter_scaler(self):
  x=np.vstack([self.x,[[999.,999.]]]);train=np.array([0,1,2]);s=b.scaler(x[train],self.w)
  x[3]*=1000
  self.assertEqual(s,b.scaler(x[train],self.w))
 def test_scaler_rejects_zero_negative_or_nonfinite_weights(self):
  for w in [[0.,0.,0.],[1.,-1.,1.],[1.,np.nan,1.]]:
   with self.assertRaises(ValueError):b.scaler(self.x,w)
 def test_scaler_rejects_bad_feature_shape(self):
  for x in [np.array([1.,2.,3.]),np.array([[np.nan],[1.],[2.]])]:
   with self.assertRaises(ValueError):b.scaler(x,self.w)
 def test_weighted_closed_form_exact_affine(self):
  s=b.scaler(self.x[:,:1],self.w);z=b.transform(self.x[:,:1],s)
  beta,intercept=b.least_squares(z,2*self.x[:,0]+3,self.w,0.)
  np.testing.assert_allclose(z@beta+intercept,2*self.x[:,0]+3,atol=1e-12)
 def test_ridge_penalty_shrinks_coefficient_not_intercept(self):
  x=np.array([[-1.],[1.]]);y=np.array([1.,5.]);w=np.ones(2)
  beta,intercept=b.least_squares(x,y,w,2.)
  np.testing.assert_allclose(beta,[1.]);self.assertAlmostEqual(intercept,3.)
 def test_least_squares_rejects_invalid_target(self):
  with self.assertRaises(ValueError):b.least_squares(self.x,[1.,np.inf,3.],self.w,0.)
 def test_model_export_clip_and_anchor(self):
  m={'indices':[0],'scaler':{'mean':[1.],'scale':[2.]},'coef':[4.],'intercept':-1.}
  np.testing.assert_allclose(b.predict(m,np.array([[0.,99.],[3.,99.]]),np.array([1.,2.])),[0.,5.])
 def test_prediction_rejects_nonfinite_anchor_and_bad_indices(self):
  m={'indices':[2],'scaler':{'mean':[1.],'scale':[2.]},'coef':[4.],'intercept':-1.}
  with self.assertRaises(ValueError):b.predict(m,self.x,np.zeros(3))
  m['indices']=[0]
  with self.assertRaises(ValueError):b.predict(m,self.x,np.array([0.,np.nan,0.]))
 def test_prediction_rejects_zero_scale(self):
  m={'indices':[0],'scaler':{'mean':[0.],'scale':[0.]},'coef':[1.],'intercept':0.}
  with self.assertRaises(ValueError):b.predict(m,self.x,np.zeros(3))
 def toy(self):
  members=[{'group':g,'route':g+'/Day','window_id':g+'_w'} for g in b.r.GROUPS]
  y=np.ones(3);p={name:np.ones(3) for name in b.MODELS}
  return members,y,p,np.arange(3)
 def test_complete_matched_accounting(self):
  m,y,p,a=self.toy();s=b.summarize(m,y,p,a,{})
  self.assertEqual(len(s['by_group']),3);self.assertEqual(len(s['by_window']),3)
  self.assertFalse(s['promotion_allowed']);self.assertFalse(s['witness_enabled'])
 def test_wrong_held_assignments_rejected(self):
  m,y,p,a=self.toy()
  with self.assertRaises(ValueError):b.summarize(m,y,p,np.array([0,0,2]),{})
 def test_missing_model_or_nan_rejected(self):
  m,y,p,a=self.toy();p.pop(b.BASELINES[0])
  with self.assertRaises(ValueError):b.summarize(m,y,p,a,{})
  m,y,p,a=self.toy();p[b.BASELINES[0]][1]=np.nan
  with self.assertRaises(ValueError):b.summarize(m,y,p,a,{})
 def test_paired_differences_use_same_rows(self):
  m,y,p,a=self.toy();p[b.BASELINES[0]]+=1
  s=b.summarize(m,y,p,a,{})
  self.assertAlmostEqual(s['pooled']['paired_hgb_minus_baseline'][b.BASELINES[0]]['mae_mps'],-1.)
 def test_failure_retained_without_promotion(self):
  m,y,p,a=self.toy();p['context_boosting'][1]+=1
  s=b.summarize(m,y,p,a,{})
  self.assertEqual(s['diagnostic_failures']['context_boosting']['windows'],['Urban03_w'])
  self.assertFalse(s['promotion_allowed']);self.assertTrue(s['parent_development_gate_still_failed'])

if __name__=='__main__':unittest.main(verbosity=2)
