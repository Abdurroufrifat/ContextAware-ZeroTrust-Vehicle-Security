import importlib, importlib.util, unittest
import numpy as np

class TailTests(unittest.TestCase):
 def setUp(self):
  self.assertIsNotNone(importlib.util.find_spec('tail_analysis'),'tail audit implementation missing')
  self.m=importlib.import_module('tail_analysis')
 def example(self):
  labels=[{'window_id':'a','time_ns':1,'reference_horizontal_mps':3.},{'window_id':'a','time_ns':2,'reference_horizontal_mps':4.}]
  index=[{'window_id':'a','time_ns':1,'ready':True,'array_index':0},{'window_id':'a','time_ns':2,'ready':True,'array_index':1}]
  return index,labels,np.array([3.,4.]),np.array([2.,4.])
 def test_exact_endpoint_join_retains_every_supported_row(self):self.assertEqual(len(self.m.joined_rows(*self.example())),2)
 def test_duplicate_labels_are_rejected(self):
  i,l,y,p=self.example();l.append(dict(l[0]))
  with self.assertRaises(ValueError):self.m.joined_rows(i,l,y,p)
 def test_array_index_misalignment_is_rejected(self):
  i,l,y,p=self.example();i[0]['array_index']=1
  with self.assertRaises(ValueError):self.m.joined_rows(i,l,y,p)
 def test_reference_target_mismatch_is_rejected(self):
  i,l,y,p=self.example();y[0]=99.
  with self.assertRaises(ValueError):self.m.joined_rows(i,l,y,p)
 def test_tail_squared_error_share_uses_full_denominator(self):
  s=self.m.summary(np.array([0.,0.,0.]),np.array([0.,1.,10.]));self.assertEqual(s['tails']['above_2_mps']['count'],1);self.assertAlmostEqual(s['tails']['above_2_mps']['squared_error_share'],100/101)
 def test_perfect_prediction_has_zero_tail_share(self):
  s=self.m.summary(np.zeros(3),np.zeros(3));self.assertEqual(s['tails']['above_2_mps']['squared_error_share'],0.)
 def test_event_does_not_cross_a_block_boundary(self):
  rows=[{'window_id':'a','time_ns':10},{'window_id':'b','time_ns':20}]
  self.assertEqual(len(self.m.clusters(rows,np.array([3.,4.]),2,100)),2)
 def test_event_does_not_bridge_a_long_gap(self):
  rows=[{'window_id':'a','time_ns':10},{'window_id':'a','time_ns':200}]
  self.assertEqual(len(self.m.clusters(rows,np.array([3.,4.]),2,100)),2)
 def test_causal_raw_window_excludes_future_and_old_samples(self):
  rows=[{'bag_time_ns':0,'header_time_ns':0,'speed_raw':99.},{'bag_time_ns':1500000000,'header_time_ns':1500000000,'speed_raw':-0.4},{'bag_time_ns':2000000000,'header_time_ns':2000000000,'speed_raw':6.},{'bag_time_ns':2100000000,'header_time_ns':2100000000,'speed_raw':999.}]
  s=self.m.radar_window(rows,2000000000);self.assertEqual(s['samples'],2);self.assertAlmostEqual(s['signed_median_native'],2.8);self.assertEqual(s['low_absolute_return_fraction'],.5)
 def test_radar_window_rejects_late_delivery(self):
  rows=[{'bag_time_ns':1000000000,'header_time_ns':0,'speed_raw':999.},{'bag_time_ns':2000000000,'header_time_ns':2000000000,'speed_raw':3.}]
  self.assertEqual(self.m.radar_window(rows,2000000000)['samples'],1)
if __name__=='__main__':unittest.main()
