"""NumPy-only safeguards for descriptive error diagnosis."""
import unittest
import numpy as np
import diagnosis_core as d

class DiagnosisTests(unittest.TestCase):
 def test_reference_speed_bin_boundaries(self):
  np.testing.assert_array_equal(d.speed_bins(np.array([0.,.99,1.,4.99,5.,9.99,10.,100.])),[0,0,1,1,2,2,3,3])
 def test_invalid_speed_rejected(self):
  for x in [np.array([-1.]),np.array([np.nan]),np.array([[1.]])]:
   with self.assertRaises(ValueError):d.speed_bins(x)
 def test_empty_metric_cell_retained(self):
  m=d.metrics(np.array([]),np.array([]));self.assertEqual(m['count'],0);self.assertIsNone(m['p95_mps']);self.assertEqual(m['above_half_mps'],0)
 def test_known_metric_and_signed_bias(self):
  m=d.metrics(np.array([1.,1.,1.]),np.array([0.,1.,3.]))
  self.assertAlmostEqual(m['mae_mps'],1.);self.assertAlmostEqual(m['rmse_mps'],np.sqrt(5/3));self.assertAlmostEqual(m['bias_prediction_minus_reference_mps'],1/3)
  self.assertEqual(m['above_half_mps'],2);self.assertEqual(m['above_2_mps'],0)
 def test_metric_threshold_is_strict(self):
  m=d.metrics(np.zeros(3),np.array([.5,2.,2.01]));self.assertEqual(m['above_half_mps'],2);self.assertEqual(m['above_2_mps'],1)
 def test_metric_rejects_misaligned_arrays(self):
  with self.assertRaises(ValueError):d.metrics(np.array([1.]),np.array([1.,2.]))
 def test_metric_rejects_invalid_prediction(self):
  for p in [np.array([np.inf]),np.array([-1.])]:
   with self.assertRaises(ValueError):d.metrics(np.array([0.]),p)
 def test_time_thirds_include_final_timestamp(self):
  np.testing.assert_array_equal(d.time_thirds(np.array([0,30,60,90],dtype=np.int64)),[0,1,2,2])
 def test_time_subtraction_preserves_epoch_precision(self):
  t=np.array([1700000000000000000+i for i in [0,30,60,90]],dtype=np.int64)
  np.testing.assert_array_equal(d.time_thirds(t),[0,1,2,2])
 def test_singleton_time_is_explicit_first_third(self):
  np.testing.assert_array_equal(d.time_thirds(np.array([5],dtype=np.int64)),[0])
 def test_time_rejects_duplicate_unsorted_or_float(self):
  for t in [np.array([1,1]),np.array([2,1]),np.array([1.,2.])]:
   with self.assertRaises(ValueError):d.time_thirds(t)
 def test_run_breaks_at_nonexceedance(self):
  runs=d.error_runs(np.array([0,50,100,150],dtype=np.int64),np.array([.6,.7,.2,.8]),75)
  self.assertEqual([v['rows'] for v in runs],[2,1]);self.assertEqual(runs[0]['span_ns'],50)
 def test_run_breaks_at_large_supported_gap(self):
  runs=d.error_runs(np.array([0,50,200],dtype=np.int64),np.array([.6,.7,.8]),75)
  self.assertEqual([v['rows'] for v in runs],[2,1]);self.assertEqual(runs[1]['span_ns'],0)
 def test_run_accepts_exact_gap_boundary(self):
  self.assertEqual(d.error_runs(np.array([0,75],dtype=np.int64),np.array([.6,.7]),75)[0]['rows'],2)
 def test_run_retains_singletons_and_ignores_exact_error_limit(self):
  runs=d.error_runs(np.array([0,50],dtype=np.int64),np.array([.5,.6]),75)
  self.assertEqual(len(runs),1);self.assertEqual(runs[0]['span_ns'],0);self.assertEqual(runs[0]['first_local_index'],1)
 def test_no_exceedance_has_no_runs(self):
  self.assertEqual(d.error_runs(np.array([0,50],dtype=np.int64),np.array([.1,.5]),75),[])
 def test_runs_reject_invalid_error_or_gap(self):
  for e,gap in [(np.array([-.6,.7]),75),(np.array([.6,np.nan]),75),(np.array([.6,.7]),0)]:
   with self.assertRaises(ValueError):d.error_runs(np.array([0,50],dtype=np.int64),e,gap)
 def test_overlap_uses_identical_rows(self):
  q=d.overlap(np.array([True,True,False]),np.array([False,True,True]));self.assertEqual(q,{'both':1,'hgb_only':1,'baseline_only':1,'neither':0})
 def test_overlap_rejects_nonboolean_or_bad_shape(self):
  for b in [np.array([1,0]),np.array([True])]:
   with self.assertRaises(ValueError):d.overlap(np.array([True,False]),b)
 def toy(self):
  members=[{'id':i,'route':'Urban01/Example-Day','group':'Urban01','day':True,'window_id':'test_window','time_ns':100+i*50} for i in range(4)]
  y=np.array([0.,1.,5.,10.]);pred={name:y.copy() for name in d.MODELS};return members,y,pred
 def test_all_speed_bins_and_thirds_account_for_every_row(self):
  members,y,pred=self.toy();r,arrays,runs=d.build(members,y,pred)
  self.assertEqual(sum(v['count'] for v in r['by_speed'].values()),4)
  self.assertEqual(sum(v['count'] for v in r['by_window_third']['test_window'].values()),4)
  self.assertTrue(r['parent_development_gate_still_failed']);self.assertFalse(r['promotion_allowed']);self.assertEqual(len(runs),0)
 def test_empty_speed_cells_not_dropped(self):
  members,y,pred=self.toy();y[:]=0.;pred={n:y.copy() for n in d.MODELS};r,_,_=d.build(members,y,pred)
  self.assertEqual(len(r['by_speed']),4);self.assertEqual(r['by_speed']['ge10']['count'],0)
 def test_missing_model_rejected(self):
  members,y,pred=self.toy();pred.pop(d.MODELS[1])
  with self.assertRaises(ValueError):d.build(members,y,pred)
 def test_mixed_recordings_in_window_rejected(self):
  members,y,pred=self.toy();members[0]['route']='Urban03/Example-Day'
  with self.assertRaises(ValueError):d.build(members,y,pred)
 def test_row_identity_and_duplicate_time_rejected(self):
  members,y,pred=self.toy();members[1]['id']=0
  with self.assertRaises(ValueError):d.build(members,y,pred)
  members,y,pred=self.toy();members[1]['time_ns']=members[0]['time_ns']
  with self.assertRaises(ValueError):d.build(members,y,pred)

if __name__=='__main__':unittest.main(verbosity=2)
