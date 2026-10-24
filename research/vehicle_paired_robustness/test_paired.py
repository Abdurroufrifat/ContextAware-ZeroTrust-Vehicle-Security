import unittest
import numpy as np
import paired_core as p
class PairedTests(unittest.TestCase):
 def members(self):
  return [{'id':i,'group':'Urban01' if i<3 else 'Urban03','route':'Urban01/a-Day' if i<3 else 'Urban03/b-Night','day':i<3,'window_id':'a' if i<3 else 'b','time_ns':100+i} for i in range(4)]
 def test_functions_exist(self):
  self.assertTrue(callable(getattr(p,'metrics',None)),'paired metrics are missing')
 def test_metric_matches_hand_values(self):
  m=p.metrics([0,0,0,0],[0,1,2,3]);self.assertEqual(m['mae_mps'],1.5);self.assertAlmostEqual(m['rmse_mps'],np.sqrt(3.5));self.assertAlmostEqual(m['p95_mps'],2.85)
 def test_metrics_strict_threshold(self):
  m=p.metrics([0,0,0],[.5,2.,2.01]);self.assertEqual(m['above_half_mps'],2);self.assertEqual(m['above_2_mps'],1)
 def test_empty_metrics_rejected(self):
  with self.assertRaises(ValueError):p.metrics([],[])
 def test_mismatched_metrics_rejected(self):
  with self.assertRaises(ValueError):p.metrics([1],[1,2])
 def test_nonfinite_or_negative_rejected(self):
  for a in [[np.nan],[np.inf],[-1]]:
   with self.assertRaises(ValueError):p.metrics([0],a)
 def test_equal_unit_differs_from_sample_weighting(self):
  a=p.balanced([0,0,0,0],[0,0,0,4],['a','a','a','b']);self.assertEqual(a['equal_unit_mae_mps'],2);self.assertAlmostEqual(a['equal_unit_rmse_mps'],np.sqrt(8));self.assertEqual(a['mean_unit_p95_mps'],2);self.assertEqual(a['units'],2)
 def test_balancing_does_not_average_rmse(self):
  a=p.balanced([0,0],[0,4],['a','b']);self.assertNotEqual(a['equal_unit_rmse_mps'],2)
 def test_mean_unit_p95_is_not_pooled_quantile(self):
  a=p.balanced([0,0,0,0],[0,0,0,4],['a','a','a','b']);self.assertNotEqual(a['mean_unit_p95_mps'],p.metrics([0]*4,[0,0,0,4])['p95_mps'])
 def test_balancing_rejects_wrong_labels(self):
  with self.assertRaises(ValueError):p.balanced([0],[1],[])
 def test_pair_counts_orientation_and_tolerance(self):
  a=p.pair_counts([0]*4,[0,2,.1,1],[1,1,.1+5e-13,1]);self.assertEqual(a['control_lower_abs_error'],1);self.assertEqual(a['control_higher_abs_error'],1);self.assertEqual(a['tied_abs_error'],2)
 def test_exceedance_pairs_sum_and_boundary(self):
  a=p.pair_counts([0]*4,[.5,.6,.6,.1],[.6,.5,.7,.1]);self.assertEqual(a['above_half_pairs'],{'both':1,'control_only':1,'comparator_only':1,'neither':1})
 def test_pair_alignment_rejected(self):
  with self.assertRaises(ValueError):p.pair_counts([0,0],[1],[1,2])
 def test_wins_orientation(self):
  a=p.win_counts([-1,0,1,5e-13]);self.assertEqual(a,{'control_lower':1,'tied':2,'control_higher':1})
 def test_empty_wins_rejected(self):
  with self.assertRaises(ValueError):p.win_counts([])
 def test_members_valid(self):
  self.assertIsNone(p.validate_members(self.members(),4))
 def test_duplicate_timestamps_rejected(self):
  m=self.members();m[1]['time_ns']=m[0]['time_ns']
  with self.assertRaises(ValueError):p.validate_members(m,4)
 def test_duplicate_row_id_rejected(self):
  m=self.members();m[1]['id']=0
  with self.assertRaises(ValueError):p.validate_members(m,4)
 def test_window_cross_recording_rejected(self):
  m=self.members();m[-1]['window_id']='a'
  with self.assertRaises(ValueError):p.validate_members(m,4)
 def test_route_group_mismatch_rejected(self):
  m=self.members();m[0]['group']='Urban04'
  with self.assertRaises(ValueError):p.validate_members(m,4)
 def test_day_mismatch_rejected(self):
  m=self.members();m[0]['day']=False
  with self.assertRaises(ValueError):p.validate_members(m,4)
 def test_inexact_timestamp_rejected(self):
  m=self.members();m[0]['time_ns']=100.0
  with self.assertRaises(ValueError):p.validate_members(m,4)
 def test_subset_alignment(self):
  a=p.omission_masks(['a','a','b','c']);self.assertEqual(list(a),['a','b','c']);np.testing.assert_array_equal(a['a'],[False,False,True,True])
 def test_cannot_omit_only_group(self):
  with self.assertRaises(ValueError):p.omission_masks(['a','a'])
 def test_nonfinite_deltas_rejected(self):
  with self.assertRaises(ValueError):p.win_counts([float('nan')])
 def test_balanced_prediction_alignment_rejected(self):
  with self.assertRaises(ValueError):p.balanced([0,0],[1],['a','b'])
 def test_fold_ids_match_each_member(self):
  self.assertIsNone(p.validate_fold_ids(self.members(),np.array([0,0,0,1])))
 def test_wrong_fold_rejected(self):
  with self.assertRaises(ValueError):p.validate_fold_ids(self.members(),np.array([0,0,0,2]))
 def test_float_fold_rejected(self):
  with self.assertRaises(ValueError):p.validate_fold_ids(self.members(),np.array([0.,0.,0.,1.]))
 def test_fold_shape_rejected(self):
  with self.assertRaises(ValueError):p.validate_fold_ids(self.members(),np.array([0,1]))
 def test_build_retains_main_cohort_with_omission_sensitivity(self):
  y=np.zeros(4);pred={name:np.ones(4)*(i+1) for i,name in enumerate(p.MODELS)};r=p.build(self.members(),y,pred,{})
  self.assertEqual(r['supported_rows'],4);self.assertEqual(r['pooled']['count'],4);self.assertFalse(r['omission_changes_main_cohort']);self.assertFalse(r['promotion_allowed'])
  self.assertEqual(len(r['diagnostic_failures'][p.CONTROL]['windows']),2)
  for family,omissions in r['diagnostic_omissions'].items():
   for row in omissions.values():self.assertEqual(row['omitted_rows']+row['retained']['count'],4)
 def test_build_missing_model_rejected(self):
  with self.assertRaises(ValueError):p.build(self.members(),np.zeros(4),{p.CONTROL:np.zeros(4)},{})
if __name__=='__main__':unittest.main(verbosity=2)
