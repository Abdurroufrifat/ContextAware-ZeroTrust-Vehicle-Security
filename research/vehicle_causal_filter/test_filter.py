import importlib,importlib.util,unittest,tempfile,json
from pathlib import Path
from unittest.mock import patch
import numpy as np
class FilterTests(unittest.TestCase):
 def setUp(self):
  self.assertIsNotNone(importlib.util.find_spec('filter_core'),'causal filter implementation missing');self.f=importlib.import_module('filter_core')
 def run_filter(self,t,z,keep=None,r=.025):
  z=np.asarray(z,dtype=float);keep=np.ones(len(z),dtype=bool) if keep is None else np.asarray(keep,dtype=bool)
  return self.f.filter_events(np.asarray(t,dtype=np.int64),z,keep,r,1.,100000000)
 def test_first_valid_packet_has_no_hidden_warmup(self):
  out=self.run_filter([0],[3.]);self.assertEqual(out['estimate_mps'][0],3.);self.assertTrue(out['reset'][0])
 def test_constant_speed_stays_constant(self):
  out=self.run_filter(np.arange(100)*50000000,np.full(100,3.));np.testing.assert_allclose(out['estimate_mps'],3.,atol=1e-12)
 def test_no_future_packet_can_modify_past_output(self):
  times=np.arange(10)*50000000;a=self.run_filter(times,np.arange(10));b=self.run_filter(times,np.r_[np.arange(9),999.]);np.testing.assert_array_equal(a['estimate_mps'][:9],b['estimate_mps'][:9])
 def test_rejection_clears_state_and_emits_no_estimate(self):
  out=self.run_filter([0,50000000,100000000],[3.,100.,7.],[True,False,True]);self.assertTrue(np.isnan(out['estimate_mps'][1]));self.assertEqual(out['estimate_mps'][2],7.);self.assertTrue(out['reset'][2])
 def test_long_gap_resets_instead_of_extrapolating(self):
  out=self.run_filter([0,100000001],[3.,7.]);np.testing.assert_array_equal(out['estimate_mps'],[3.,7.]);self.assertTrue(out['reset'][1])
 def test_exact_age_limit_does_not_reset(self):
  out=self.run_filter([0,100000000],[3.,7.]);self.assertFalse(out['reset'][1]);self.assertLess(out['estimate_mps'][1],7.)
 def test_duplicate_clock_is_rejected(self):
  with self.assertRaises(ValueError):self.run_filter([1,1],[1.,2.])
 def test_invalid_covariance_is_rejected(self):
  for r in [0.,-1.,float('nan')]:
   with self.assertRaises(ValueError):self.run_filter([0],[1.],r=r)
 def test_nonfinite_accepted_measurement_is_rejected(self):
  with self.assertRaises(ValueError):self.run_filter([0],[float('nan')])
 def test_zero_output_is_not_missing(self):
  out=self.run_filter([0,50000000],[0.,0.]);np.testing.assert_array_equal(out['estimate_mps'],[0.,0.])
 def test_covariance_remains_positive_semidefinite(self):
  t=np.arange(100)*50000000;out=self.run_filter(t,2.+np.sin(np.arange(100)))
  self.assertGreaterEqual(float(np.linalg.eigvalsh(out['covariance']).min()),-1e-12)
 def test_fit_noise_cannot_use_development_labels(self):
  with self.assertRaises(ValueError):self.f.fit_noise(np.array([1.,2.]),np.array([1.,2.]),'calibration')
 def test_fit_noise_floor_handles_perfect_fit(self):
  self.assertEqual(self.f.fit_noise(np.array([1.,2.,3.]),np.array([1.,2.,3.]),'fit')['variance_mps2'],.0001)
 def test_postfilter_peer_rejection_resets_next_update(self):
  out=self.f.filter_events(np.array([0,50000000,100000000]),np.array([1.,10.,4.]),np.ones(3,dtype=bool),.025,1.,100000000,peer=np.array([1.,10.,4.]))
  self.assertFalse(out['quality'][1]);self.assertTrue(out['reset'][2]);self.assertEqual(out['estimate_mps'][2],4.)
 def test_source_archive_vectors_are_read_once(self):
  import run_filter
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory);index=[{'time_ns':0,'ready':True,'array_index':0},{'time_ns':50,'ready':True,'array_index':1},{'time_ns':100,'ready':False,'array_index':None}]
   for name in ['vehicle_continuous_radar','vehicle_masked_history']:
    folder=root/name;folder.mkdir();(folder/'frame_index.json').write_text(json.dumps(index),encoding='utf-8')
    if name=='vehicle_continuous_radar':
     np.savez_compressed(folder/'predictions.npz',causal_residual_tcn=[1.,2.],paper_inspired_cnn=[1.,2.]);np.savez_compressed(folder/'quality_masks.npz',guard=np.ones(2,dtype=bool))
    else:
     np.savez_compressed(folder/'predictions.npz',**{c+'__'+k:np.array([1.,2.]) for c in ['clean_control','masked_training'] for k in ['causal_residual_tcn','masked_cnn_4ch']});np.savez_compressed(folder/'decisions.npz',**{c+'__quality':np.array([True,True,False]) for c in ['clean_control','masked_training']})
   cls=type(np.load(root/'vehicle_continuous_radar/predictions.npz'));original=cls.__getitem__;calls=[]
   def tracked(instance,key):calls.append(key);return original(instance,key)
   with patch.object(run_filter,'ROOT',root/'new'),patch.object(cls,'__getitem__',tracked):out=run_filter.sources()
   np.testing.assert_array_equal(out['original']['pred'][:2],[1.,2.]);self.assertTrue(np.isnan(out['original']['pred'][2]));self.assertLessEqual(len(calls),9,'Prediction vectors must not be repeatedly decompressed per endpoint')
if __name__=='__main__':unittest.main()
