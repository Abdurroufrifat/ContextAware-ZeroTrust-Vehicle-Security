import unittest,numpy as np
import axis_core as k
class AxisTests(unittest.TestCase):
 def call(self,n,*a,**kw):
  f=getattr(k,n,None);self.assertTrue(callable(f),'missing signed-axis behavior: '+n);return f(*a,**kw)
 def source(self):
  t=10_000_000_000+np.arange(11,dtype=np.int64)*50_000_000
  return {'bag_ns':t,'header_ns':t.copy(),'usable':np.ones(11,dtype=bool),'acceleration_norm':np.ones(11),'angular_velocity_norm':np.ones(11),'acceleration':np.column_stack([np.arange(11),np.ones(11),np.ones(11)*2]),'angular_velocity':np.column_stack([np.ones(11)*3,np.ones(11)*4,np.ones(11)*5])}
 def test_oracle_is_zero_inside_prediction_interval(self):np.testing.assert_array_equal(self.call('oracle_errors',np.array([2.]),np.array([1.]),np.array([3.])),[0.])
 def test_oracle_respects_both_sides(self):np.testing.assert_array_equal(self.call('oracle_errors',np.array([0.,4.]),np.array([1.,1.]),np.array([3.,3.])),[1.,1.])
 def test_oracle_is_lower_bound_for_every_convex_prediction(self):
  y=np.array([0.,2.,5.]);a=np.array([1.,1.,1.]);b=np.array([3.,3.,3.]);floor=self.call('oracle_errors',y,a,b)
  for alpha in [0.,.2,.5,.8,1.]:self.assertTrue(np.all(abs(y-((1-alpha)*a+alpha*b))>=floor))
 def test_signed_features_retain_axis_direction(self):
  a=self.source();x,coverage=self.call('axis_features',a,a['bag_ns'][-1:]);self.assertEqual(x.shape,(1,6));np.testing.assert_allclose(x,[[5,1,2,3,4,5]]);self.assertEqual(coverage.tolist(),[1.]);a['acceleration']=-a['acceleration'];y,_=self.call('axis_features',a,a['bag_ns'][-1:]);np.testing.assert_allclose(y[0,:3],-x[0,:3])
 def test_future_arrival_does_not_affect_axis_features(self):
  a=self.source();t=a['bag_ns'][-2:-1];x,_=self.call('axis_features',a,t);a['acceleration'][-1]=999;a['angular_velocity'][-1]=999;y,_=self.call('axis_features',a,t);np.testing.assert_array_equal(x,y)
 def test_invalid_newest_is_not_replaced_by_older_sample(self):
  a=self.source();a['usable'][-1]=False;x,c=self.call('axis_features',a,a['bag_ns'][-1:]);self.assertAlmostEqual(c[0],10/11);self.assertEqual(x[0,0],4.5)
 def test_all_missing_features_are_zero_and_counted(self):
  a=self.source();a['usable'][:]=False;x,c=self.call('axis_features',a,a['bag_ns'][-1:]);np.testing.assert_array_equal(x,np.zeros((1,6)));self.assertEqual(c.tolist(),[0.])
 def test_float_endpoint_clock_rejected(self):
  with self.assertRaises(ValueError):self.call('axis_features',self.source(),np.array([1.]))
 def test_head_fit_uses_weighted_intercept(self):
  x=np.zeros((3,6));h=self.call('fit_head',x,np.array([1.,3.,9.]),np.array([1.,2.,1.]));self.assertEqual(h['intercept'],4.);np.testing.assert_allclose(self.call('head_predict',h,x),[4,4,4])
 def test_zero_variance_scale_is_safe(self):
  h=self.call('fit_head',np.zeros((3,6)),np.array([1.,2.,3.]),np.ones(3));self.assertEqual(h['scale'],[1.]*6);self.assertEqual(h['coefficient'],[0.]*6)
 def test_head_fit_matches_augmented_normal_equation(self):
  x=np.arange(30,dtype=float).reshape(5,6);y=np.array([0.,1.,.3,2.,1.]);w=np.array([1.,2.,1.,3.,2.]);h=self.call('fit_head',x,y,w);z=(x-np.array(h['mean']))/np.array(h['scale']);q=np.column_stack([np.ones(len(x)),z]);penalty=np.eye(7);penalty[0,0]=0;oracle=np.linalg.solve(q.T@(q*w[:,None])+penalty,q.T@(w*y));np.testing.assert_allclose([h['intercept']]+h['coefficient'],oracle,rtol=0,atol=1e-10)
 def test_input_row_permutation_does_not_change_head(self):
  x=np.arange(30,dtype=float).reshape(5,6);y=np.arange(5,dtype=float);w=np.ones(5);a=self.call('fit_head',x,y,w);b=self.call('fit_head',x[::-1],y[::-1],w[::-1]);np.testing.assert_allclose(self.call('head_predict',a,x),self.call('head_predict',b,x),rtol=0,atol=1e-10)
 def test_nonpositive_weights_rejected(self):
  with self.assertRaises(ValueError):self.call('fit_head',np.ones((2,6)),np.zeros(2),np.array([0.,1.]))
 def test_nan_fitting_feature_rejected(self):
  with self.assertRaises(ValueError):self.call('fit_head',np.full((2,6),np.nan),np.zeros(2),np.ones(2))
 def test_raw_and_masked_copies_share_axis_inputs(self):
  x=np.arange(18).reshape(3,6);y=np.array([1.,2.,3.]);ids=np.array([0,2]);native=np.array([.8,2.5,.5,2.]);xx,target=self.call('correction_inputs',x,y,ids,native);np.testing.assert_array_equal(xx[:2],xx[2:]);np.testing.assert_allclose(target,[.2,.5,.5,1.])
 def test_held_targets_do_not_enter_correction_inputs(self):
  x=np.arange(18).reshape(3,6);y=np.array([1.,2.,3.]);ids=np.array([0,2]);native=np.array([.8,2.5,.5,2.]);a=self.call('correction_inputs',x,y,ids,native);y[1]=999999;b=self.call('correction_inputs',x,y,ids,native)
  for p,q in zip(a,b):np.testing.assert_array_equal(p,q)
 def test_duplicate_fitting_ids_rejected(self):
  with self.assertRaises(ValueError):self.call('correction_inputs',np.zeros((3,6)),np.ones(3),np.array([0,0]),np.zeros(4))
if __name__=='__main__':unittest.main(verbosity=2)
