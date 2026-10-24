import unittest,numpy as np
import context_core as c
class ContextTests(unittest.TestCase):
 def grid(self,v=2.):
  a=np.zeros((1,4,128));a[:,0]=v;a[:,1]=abs(v);a[:,3]=1;return a
 def test_constant_features(self):
  z=c.features(self.grid(),np.array([3.]),np.array([2.]));self.assertEqual(z.shape,(1,len(c.FEATURE_NAMES)));self.assertEqual(z[0,0],2.)
  for n in [5,11,21,41,81,128]:
   self.assertAlmostEqual(z[0,c.FEATURE_NAMES.index(f'w{n}_median')],20.)
   self.assertAlmostEqual(z[0,c.FEATURE_NAMES.index(f'w{n}_ols_slope')],0.)
 def test_masked_values_not_signal(self):
  x=self.grid();x[:,:2,:64]=999;x[:,3,:64]=0;x[:,2,:64]=1
  z=c.features(x,np.array([3.]),np.array([2.]));self.assertEqual(z[0,c.FEATURE_NAMES.index('w128_median')],20.);self.assertEqual(z[0,c.FEATURE_NAMES.index('w128_valid_fraction')],.5)
 def test_empty_window(self):
  x=self.grid();x[:,3,-5:]=0
  z=c.features(x,np.array([3.]),np.array([2.]));self.assertEqual(z[0,c.FEATURE_NAMES.index('w5_valid_fraction')],0);self.assertEqual(z[0,c.FEATURE_NAMES.index('w5_mean_age')],1.)
 def test_linear_trend(self):
  x=self.grid();t=np.arange(128)*.05;x[:,0]=(.2*t+2)/10;x[:,1]=abs(x[:,0]);z=c.features(x,np.array([3.]),np.array([2.]))
  self.assertAlmostEqual(z[0,c.FEATURE_NAMES.index('w128_ols_slope')],.2)
 def test_features_ignore_other_samples(self):
  x=np.concatenate([self.grid(2),self.grid(6)]);a=c.features(x,np.array([3.,4.]),np.array([2.,6.]));b=c.features(x[:1],np.array([3.]),np.array([2.]));np.testing.assert_array_equal(a[:1],b)
 def test_bad_feature_shape(self):
  with self.assertRaises(ValueError):c.features(np.zeros((1,4,127)),np.array([3.]),np.array([2.]))
 def test_bad_feature_values(self):
  x=self.grid();x[0,0,-1]=np.nan
  with self.assertRaises(ValueError):c.features(x,np.array([3.]),np.array([2.]))
 def test_nonbinary_mask_rejected(self):
  x=self.grid();x[0,3,0]=.5
  with self.assertRaises(ValueError):c.features(x,np.array([3.]),np.array([2.]))
 def test_fit_routes_exact(self):
  c.validate_fit_routes(c.FIT_ROUTES)
  for routes in [[c.RESERVED],['Urban01/2023-11-23-18-24-14-Night'],c.FIT_ROUTES[:-1]]:
   with self.assertRaises(ValueError):c.validate_fit_routes(routes)
 def test_tree_threshold_and_negative_clip(self):
  model={'feature_names':['x'],'baseline':0.,'trees':[[{'leaf':False,'feature':0,'threshold':1.,'left':1,'right':2},{'leaf':True,'value':-2.},{'leaf':True,'value':3.}]]}
  x=np.array([[1.],[np.nextafter(1.,np.inf)]]);np.testing.assert_array_equal(c.predict(model,x,np.array([1.,1.])),[0.,4.])
 def test_tree_invalid_input(self):
  with self.assertRaises(ValueError):c.predict({'feature_names':['x'],'baseline':0.,'trees':[]},np.array([[np.nan]]),np.array([1.]))
 def test_quality_never_revives(self):
  keep=c.quality(np.array([0.,1.,2.]),np.array([0.,1.,2.]),np.array([False,True,False]));np.testing.assert_array_equal(keep,[False,True,False])
 def test_quality_boundary(self):
  keep=c.quality(np.array([1.,1.000001,np.nan]),np.zeros(3),np.ones(3,dtype=bool));np.testing.assert_array_equal(keep,[True,False,False])
if __name__=='__main__':unittest.main()
