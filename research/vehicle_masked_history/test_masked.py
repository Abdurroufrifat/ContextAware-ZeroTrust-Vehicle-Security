import importlib,importlib.util,unittest
import numpy as np
class MaskedTests(unittest.TestCase):
 def setUp(self):
  self.assertIsNotNone(importlib.util.find_spec('masked_core'),'masked implementation missing')
  self.m=importlib.import_module('masked_core')
 def rows(self,n=401):return [{'bag_time_ns':i*50000000,'header_time_ns':i*50000000,'speed_raw':3.} for i in range(n)]
 def test_partial_history_is_explicitly_masked(self):
  x,anchor=self.m.features(self.rows(),1000000000,0)
  self.assertEqual(x.shape,(4,128));self.assertEqual(int(x[3].sum()),21);self.assertTrue(np.all(x[:2,x[3]==0]==0));self.assertTrue(np.all(x[2,x[3]==0]==1));self.assertEqual(anchor,3.)
 def test_less_than_sixteen_ticks_is_unready(self):self.assertIsNone(self.m.features(self.rows(),700000000,0))
 def test_current_endpoint_must_be_valid(self):
  rows=self.rows();rows[200]['header_time_ns']+=1
  self.assertIsNone(self.m.features(rows,10000000000,0))
 def test_gap_in_old_history_does_not_block_fresh_endpoint(self):
  rows=[r for i,r in enumerate(self.rows()) if not 160<=i<=170]
  x,anchor=self.m.features(rows,10000000000,0);self.assertGreater(int((x[3]==0).sum()),0);self.assertEqual(x[3,-1],1)
 def test_complete_history_matches_frozen_baseline(self):
  rows=self.rows();x,_=self.m.features(rows,15000000000,0)
  np.testing.assert_array_equal(x,self.m.e.RadarGrid(rows).at(15000000000,0))
 def test_future_sample_changes_cannot_change_past_features(self):
  rows=self.rows();a=self.m.features(rows,10000000000,0);rows[-1]['speed_raw']=999.;b=self.m.features(rows,10000000000,0)
  np.testing.assert_array_equal(a[0],b[0]);self.assertEqual(a[1],b[1])
 def test_training_masking_is_repeatable_and_keeps_current_tick(self):
  x=np.ones((8,4,128),dtype=np.float32);x[:,0:2]=.3;x[:,2]=0
  a=self.m.augment(x,np.random.default_rng(141));b=self.m.augment(x,np.random.default_rng(141));np.testing.assert_array_equal(a,b);self.assertTrue(np.all(a[:,3,-1]==1));self.assertTrue(np.any(a[:,3]==0));np.testing.assert_array_equal(x[:,3],np.ones((8,128)))
 def test_masked_values_and_ages_are_unambiguous(self):
  x=np.ones((16,4,128),dtype=np.float32);a=self.m.augment(x,np.random.default_rng(142));mask=a[:,3]==0
  self.assertTrue(np.all(a[:,0][mask]==0));self.assertTrue(np.all(a[:,1][mask]==0));self.assertTrue(np.all(a[:,2][mask]==1))
 def test_synthetic_prefix_keeps_minimum_support(self):
  x=np.ones((128,4,128),dtype=np.float32);a=self.m.augment(x,np.random.default_rng(143));self.assertTrue(np.all(a[:,3].sum(axis=1)>=16))
 def test_no_prehistory_anchor_leaks_across_block_start(self):
  rows=self.rows();self.assertIsNone(self.m.features(rows,10000000000,9900000000))
 def test_unordered_raw_time_is_rejected(self):
  rows=self.rows();rows[1]['bag_time_ns']=0
  with self.assertRaises(ValueError):self.m.features(rows,10000000000,0)
if __name__=='__main__':unittest.main()
