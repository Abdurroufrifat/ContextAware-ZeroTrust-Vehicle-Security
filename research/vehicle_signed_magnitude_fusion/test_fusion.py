import unittest
import numpy as np
import fusion_core as f
class FusionTests(unittest.TestCase):
 def grid(self):
  x=np.zeros((2,4,128),dtype=np.float32);x[:,0]=np.linspace(-1,1,128);x[:,1]=abs(x[:,0]);x[:,2]=.2;x[:,3]=1;return x
 def test_feature_api_exists(self):self.assertTrue(callable(getattr(f,'features',None)),'fusion features missing')
 def test_exact_original_prefix(self):
  x=self.grid();a=np.array([1.,2.]);m=np.array([3.,4.]);out=f.features(x,a,m);np.testing.assert_array_equal(out[:,:75],f.c.features(x,a,m));self.assertEqual(out.shape,(2,148))
 def test_magnitude_suffix_definition(self):
  x=self.grid();a=np.array([1.,2.]);m=np.array([3.,4.]);out=f.features(x,a,m);np.testing.assert_array_equal(out[:,75:],f.c.features(f.g.magnitude_grid(x),a,m)[:,f.EXTRA])
 def test_source_not_mutated(self):
  x=self.grid();before=x.copy();f.features(x,np.zeros(2),np.zeros(2));np.testing.assert_array_equal(x,before)
 def test_feature_names_unique(self):self.assertEqual(len(f.FEATURE_NAMES),148);self.assertEqual(len(set(f.FEATURE_NAMES)),148);self.assertEqual(f.FEATURE_NAMES[:75],f.c.FEATURE_NAMES)
 def test_magnitude_does_not_replace_sign(self):
  x=self.grid();x[:,0]*=-1;out=f.features(x,np.zeros(2),np.zeros(2));self.assertLess(out[0,0],0);self.assertGreater(out[0,75],0)
 def test_masked_empty_features_finite(self):
  x=self.grid();x[:,3]=0;out=f.features(x,np.zeros(2),np.zeros(2));self.assertTrue(np.isfinite(out).all())
 def test_bad_grid_rejected(self):
  with self.assertRaises(ValueError):f.features(np.zeros((1,3,128)),np.zeros(1),np.zeros(1))
 def test_bad_anchor_alignment_rejected(self):
  with self.assertRaises(ValueError):f.features(self.grid(),np.zeros(1),np.zeros(2))
 def test_cached_assembly_matches_full(self):
  x=self.grid();a=np.array([1.,2.]);m=np.array([3.,4.]);base=f.features(x,np.zeros(2),np.zeros(2));np.testing.assert_array_equal(f.assemble(base,a,m),f.features(x,a,m))
 def test_cached_base_not_mutated(self):
  base=f.features(self.grid(),np.zeros(2),np.zeros(2));before=base.copy();f.assemble(base,np.ones(2),np.ones(2));np.testing.assert_array_equal(base,before)
 def test_cached_assembly_rejects_wrong_features(self):
  with self.assertRaises(ValueError):f.assemble(np.zeros((2,147)),np.zeros(2),np.zeros(2))
 def test_row_locality(self):
  x=self.grid();a=f.features(x[:1],np.zeros(1),np.zeros(1));x[1,0]+=50;b=f.features(x,np.zeros(2),np.zeros(2));np.testing.assert_array_equal(a[0],b[0])
 def test_quality_never_revives_rejection(self):
  np.testing.assert_array_equal(f.c.quality(np.array([1.,1.]),np.array([1.,1.]),np.array([False,True])),[False,True])
 def test_quality_threshold_boundary(self):
  np.testing.assert_array_equal(f.c.quality(np.array([2.,2.001]),np.ones(2),np.ones(2,dtype=bool)),[True,False])
 def test_request_newest_invalid_does_not_fall_back(self):
  out=f.av.serve(np.array([150],dtype=np.int64),np.array([0,100],dtype=np.int64),np.array([True,True]),np.array([1.,2.]),np.array([True,False]),100);self.assertEqual(out['status'][0],4);self.assertTrue(np.isnan(out['prediction_mps'][0]))
 def test_request_ttl_exact_boundary(self):
  out=f.av.serve(np.array([100,101],dtype=np.int64),np.array([0],dtype=np.int64),np.array([True]),np.array([1.]),np.array([True]),100);np.testing.assert_array_equal(out['status'],[0,2])
 def test_saved_calibration_survives_ulp(self):
  old={'slope':1.};self.assertIs(f.g.validated_calibration({'slope':np.nextafter(1.,2.)},old),old)
 def gate_fixture(self):
  import copy
  q={'usable_fraction':.95,'accepted':{'count':95,'missing_truth':5,'p95_mps':.5,'above_2_mps':0}}
  return {'requests':{'overall':q,'segments_300s':{'0':copy.deepcopy(q)}},'previous_supported_blocks':{'b':{'whole_acceptance_fraction':.95,'accepted':{'count':95,'p95_mps':.5}}}}
 def test_full_gates_accept_exact_boundaries(self):self.assertTrue(all(f.full_gates(self.gate_fixture()).values()))
 def test_segment_failure_cannot_hide_in_overall(self):
  p=self.gate_fixture();p['requests']['segments_300s']['0']['accepted']['p95_mps']=.500001;self.assertFalse(f.full_gates(p)['request_accepted_p95_le_half_mps_every_segment_and_overall'])
 def test_old_block_failure_is_retained(self):
  p=self.gate_fixture();p['previous_supported_blocks']['b']['accepted']['p95_mps']=.500001;self.assertFalse(f.full_gates(p)['old_supported_block_accepted_p95_le_half_mps'])
 def test_missing_reference_coverage_fails(self):
  p=self.gate_fixture();p['requests']['overall']['accepted']['missing_truth']=6;self.assertFalse(f.full_gates(p)['request_accepted_reference_coverage_ge95pct_every_segment_and_overall'])
 def test_extra_withholding_fails_retention(self):
  p=self.gate_fixture();p['previous_supported_blocks']['b']['whole_acceptance_fraction']=.949;self.assertFalse(f.full_gates(p)['old_supported_block_quality_retention_ge95pct'])
 def test_accepted_large_error_fails(self):
  p=self.gate_fixture();p['requests']['overall']['accepted']['above_2_mps']=1;self.assertFalse(f.full_gates(p)['zero_accepted_recorded_request_errors_above2mps'])
 def claims(self):
  return {'decision':'FUSION_ABLATION_COMPLETE_NO_PROMOTION','reserved_final_opened':False,'production_policy_changed':False,'thresholds_changed':False,'witness_enabled':False,'promotion_allowed':False,'journal_readiness_established':False,'full_urban01_night_development_reused':True,'full_development':{'no_formerly_rejected_endpoint_revived':True}}
 def test_claim_validator_exists(self):
  import verify_fusion as v
  self.assertTrue(callable(getattr(v,'validate_claims',None)),'explicit decision metadata validation missing')
 def test_conservative_claims_accepted(self):
  from verify_fusion import validate_claims
  self.assertIsNone(validate_claims(self.claims()))
 def test_unsafe_or_missing_claim_rejected(self):
  from verify_fusion import validate_claims
  for key in ['reserved_final_opened','production_policy_changed','thresholds_changed','witness_enabled','promotion_allowed','journal_readiness_established']:
   value=self.claims();value[key]=True
   with self.assertRaises(ValueError):validate_claims(value)
   del value[key]
   with self.assertRaises(ValueError):validate_claims(value)
 def test_development_reuse_cannot_be_hidden(self):
  from verify_fusion import validate_claims
  value=self.claims();value['full_urban01_night_development_reused']=False
  with self.assertRaises(ValueError):validate_claims(value)
 def test_promotion_decision_rejected(self):
  from verify_fusion import validate_claims
  value=self.claims();value['decision']='PROMOTED'
  with self.assertRaises(ValueError):validate_claims(value)
if __name__=='__main__':unittest.main(verbosity=2)
