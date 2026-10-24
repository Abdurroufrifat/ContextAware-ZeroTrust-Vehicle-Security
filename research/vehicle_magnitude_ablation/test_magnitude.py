import unittest
import numpy as np
import magnitude_core as g

class MagnitudeTests(unittest.TestCase):
 def grid(self):
  x=np.zeros((2,4,128),dtype=np.float32);x[:,0]=np.linspace(-1,1,128);x[:,1]=abs(x[:,0]);x[:,2]=.25;x[:,3]=1;return x
 def test_conversion_preserves_age_mask_and_other_channel(self):
  x=self.grid();out=g.magnitude_grid(x);np.testing.assert_array_equal(out[:,0],abs(x[:,0]));np.testing.assert_array_equal(out[:,1:],x[:,1:])
 def test_conversion_does_not_mutate_source(self):
  x=self.grid();before=x.copy();g.magnitude_grid(x);np.testing.assert_array_equal(x,before)
 def test_invalid_grid_rejected(self):
  for x in [np.zeros((2,3,128)),np.full((1,4,128),np.nan)]:
   with self.assertRaises(ValueError):g.magnitude_grid(x)
 def test_nonbinary_mask_rejected(self):
  x=self.grid();x[0,3,0]=.5
  with self.assertRaises(ValueError):g.magnitude_grid(x)
 def test_assembly_changes_only_anchor_columns(self):
  base=np.arange(150,dtype=float).reshape(2,75);old=base.copy();a=np.array([1.,2.]);m=np.array([3.,4.]);out=g.assemble(base,a,m);np.testing.assert_array_equal(base,old);np.testing.assert_array_equal(out[:,1],a);np.testing.assert_array_equal(out[:,2],m);np.testing.assert_array_equal(out[:,3:],base[:,3:]);np.testing.assert_array_equal(out[:,0],base[:,0])
 def test_assembly_rejects_bad_alignment(self):
  with self.assertRaises(ValueError):g.assemble(np.zeros((2,75)),np.zeros(1),np.zeros(2))
 def test_assembly_rejects_negative_median(self):
  with self.assertRaises(ValueError):g.assemble(np.zeros((2,75)),np.zeros(2),np.array([-1.,1.]))
 def test_cached_assembly_matches_full_feature_call(self):
  x=g.magnitude_grid(self.grid());a=np.array([1.,2.]);m=np.array([3.,4.]);base=g.c.features(x,np.zeros(2),np.zeros(2));np.testing.assert_array_equal(g.assemble(base,a,m),g.c.features(x,a,m))
 def test_row_local_features_do_not_use_other_rows(self):
  x=g.magnitude_grid(self.grid());a=g.c.features(x[:1],np.zeros(1),np.zeros(1));x[1,:2]+=100.;b=g.c.features(x,np.zeros(2),np.zeros(2));np.testing.assert_array_equal(a[0],b[0])
 def test_feature_names_are_distinct_and_semantic(self):
  for variant in g.VARIANTS:
   names=g.feature_names(variant);self.assertEqual(len(names),75);self.assertEqual(len(set(names)),75);self.assertEqual(names[0],'current_magnitude_div10')
 def test_unknown_variant_rejected(self):
  with self.assertRaises(ValueError):g.feature_names('pick_best')
 def toy(self):
  groups=np.repeat(np.array(g.r.GROUPS),4);y=np.tile([.5,2.,7.,12.],3);data={'y':y,'median_feature':y/2.,'X':np.zeros((12,4,128),dtype=np.float32)};return data,groups,np.ones(12,dtype=bool),y/2.+.25
 def test_magnitude_anchor_excludes_held_route(self):
  data,groups,day,mag=self.toy();cal=g.new_anchor_calibration(data,groups,day,'Urban01',mag);self.assertTrue(all(groups[i]!='Urban01' for i in cal['train_ids']));self.assertTrue(all(groups[i]!='Urban01' and day[i] for i in cal['calibration_ids']))
 def test_held_labels_and_medians_cannot_change_anchor(self):
  data,groups,day,mag=self.toy();cal=g.new_anchor_calibration(data,groups,day,'Urban01',mag);data['y'][:4]+=1000;mag[:4]+=1000;self.assertEqual(cal,g.new_anchor_calibration(data,groups,day,'Urban01',mag))
 def test_weight_counts_match_parent(self):
  data,groups,day,mag=self.toy();a=g.r.calibrate(data,groups,day,'Urban01');b=g.new_anchor_calibration(data,groups,day,'Urban01',mag);self.assertEqual(a['bin_counts'],b['bin_counts']);np.testing.assert_array_equal(g.r.weights(data['y'][a['train_ids']],a),g.r.weights(data['y'][b['train_ids']],b))
 def test_saved_parent_calibration_survives_one_ulp_refit(self):
  rebuilt={'anchor':{'slope':1.0000000000000002,'intercept':.1}};saved={'anchor':{'slope':1.,'intercept':.1}};self.assertIs(g.validated_calibration(rebuilt,saved),saved)
 def test_changed_parent_calibration_rejected(self):
  with self.assertRaises(ValueError):g.validated_calibration({'anchor':{'slope':2.}},{'anchor':{'slope':1.}})
 def test_magnitude_median_population_must_align(self):
  data,groups,day,mag=self.toy()
  with self.assertRaises(ValueError):g.new_anchor_calibration(data,groups,day,'Urban01',mag[:-1])

if __name__=='__main__':unittest.main(verbosity=2)
