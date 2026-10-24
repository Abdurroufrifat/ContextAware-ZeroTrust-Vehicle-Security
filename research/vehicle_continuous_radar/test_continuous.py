import importlib,importlib.util,unittest,json
import numpy as np

class ContinuousTests(unittest.TestCase):
 def setUp(self):
  self.assertIsNotNone(importlib.util.find_spec('continuous_core'),'continuous implementation missing')
  self.m=importlib.import_module('continuous_core')
 def test_download_ranges_cover_source_without_overlap(self):self.assertEqual(self.m.ranges(18,8),[(0,7),(8,15),(16,17)])
 def test_reserved_route_cannot_be_acquired(self):
  with self.assertRaises(ValueError):self.m.check_route('Urban02/2023-11-23-17-53-52-Night','https://example.org/radars.bag')
 def test_only_fixed_development_url_is_allowed(self):
  with self.assertRaises(ValueError):self.m.check_route(self.m.ROUTE,'https://example.org/radars.bag')
 def test_segments_use_one_global_clock(self):
  self.assertEqual(self.m.segment(300000000000,0),1);self.assertEqual(self.m.segment(299999999999,0),0)
 def test_missing_truth_does_not_become_a_zero_speed(self):
  out=self.m.account(np.array([10.,np.nan,3.]),np.array([10.,0.,3.]),np.array([True,True,False]),5)
  self.assertEqual(out['usable_total'],2);self.assertEqual(out['usable_reference_missing'],1);self.assertEqual(out['accepted']['count'],1);self.assertEqual(out['accepted']['rmse_mps'],0.)
 def test_all_endpoint_costs_stay_in_denominator(self):
  out=self.m.account(np.array([10.,3.]),np.array([10.,3.]),np.array([True,False]),5)
  self.assertEqual(out['unready_raw'],3);self.assertEqual(out['quality_withheld'],1);self.assertEqual(out['usable_total'],1);self.assertAlmostEqual(out['whole_acceptance_fraction'],.2)
 def test_no_accepted_truth_has_undefined_accuracy(self):
  out=self.m.account(np.array([np.nan]),np.array([0.]),np.array([True]),1);self.assertIsNone(out['accepted']['p95_mps'])
 def test_invalid_total_is_rejected(self):
  with self.assertRaises(ValueError):self.m.account(np.ones(2),np.ones(2),np.ones(2,dtype=bool),1)
 def test_segment_boundary_does_not_restart_history(self):
  rows=[{'bag_time_ns':i*50000000,'header_time_ns':i*50000000,'speed_raw':3.} for i in range(6002)]
  data,index=self.m.prepare(rows);point=index[6000]
  self.assertEqual(point['segment'],1);self.assertTrue(point['ready']);self.assertFalse(point['warmup'])
 def test_future_radar_cannot_change_prepared_past_features(self):
  rows=[{'bag_time_ns':i*50000000,'header_time_ns':i*50000000,'speed_raw':3.} for i in range(141)]
  a,ix=self.m.prepare(rows);rows[-1]['speed_raw']=999.;b,iy=self.m.prepare(rows)
  j=ix[130]['array_index'];np.testing.assert_array_equal(a['X'][j],b['X'][iy[130]['array_index']])
 def test_frame_readiness_is_portable_json(self):
  rows=[{'bag_time_ns':i*50000000,'header_time_ns':i*50000000,'speed_raw':3.} for i in range(141)]
  _,index=self.m.prepare(rows);self.assertTrue(json.dumps(index))
if __name__=='__main__':unittest.main()
