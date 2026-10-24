import math,struct,unittest
import numpy as np
import trace_core as t

def string(s):
 b=s.encode();return struct.pack('<I',len(b))+b
def header(ns):return struct.pack('<III',7,ns//10**9,ns%10**9)+string('sensor')
def reference_message(ns=1_000_000_000):return header(ns)+string('INSPVA')+struct.pack('<HBIBHI',507,0,1,160,2200,1000)+struct.pack('<9d',1.,2.,3.,3.,4.,0.,0.,0.,0.)+struct.pack('<I',3)
def radar_message(ns=1_000_000_000):return struct.pack('<f',-.25)+string('outbound')+struct.pack('<f',1.)+string('radar')+struct.pack('<f',1.)+header(ns)+struct.pack('<IIII',0,0,0,0)+struct.pack('<f',0.)
def ref(ns,v,status=3):return {'header_time_ns':ns,'bag_time_ns':ns,'horizontal_mps':v,'speed_3d_mps':v,'status':status}
class TraceTests(unittest.TestCase):
 def test_causal_reference_not_future(self):
  a=t.match_reference([ref(100,1),ref(200,2)],np.array([150],dtype=np.int64));self.assertEqual(a['reference_header_ns'][0],100);self.assertEqual(a['reference_mps'][0],1)
 def test_reference_max_age_inclusive(self):
  a=t.match_reference([ref(0,1)],np.array([100_000_000,100_000_001],dtype=np.int64));np.testing.assert_array_equal(a['present'],[True,False])
 def test_newest_invalid_reference_has_no_old_valid_fallback(self):
  a=t.match_reference([ref(100,1),ref(200,2,0)],np.array([210],dtype=np.int64));self.assertFalse(a['present'][0])
 def test_reference_before_first_missing(self):
  self.assertFalse(t.match_reference([ref(100,1)],np.array([99],dtype=np.int64))['present'][0])
 def test_epoch_integer_precision(self):
  base=1_700_000_000_000_000_000;a=t.match_reference([ref(base,1)],np.array([base+7],dtype=np.int64));self.assertEqual(a['reference_age_ns'][0],7)
 def test_duplicate_reference_headers_rejected(self):
  with self.assertRaises(ValueError):t.match_reference([ref(1,1),ref(1,2)],np.array([2],dtype=np.int64))
 def test_proxy_uses_two_past_valid_samples(self):
  a=t.match_reference([ref(0,1),ref(10_000_000,1.1)],np.array([15_000_000],dtype=np.int64));self.assertAlmostEqual(a['local_age_proxy_mps'][0],.05)
 def test_proxy_missing_for_invalid_previous_sample(self):
  a=t.match_reference([ref(0,1,0),ref(10_000_000,1.1)],np.array([15_000_000],dtype=np.int64));self.assertTrue(math.isnan(a['local_age_proxy_mps'][0]));self.assertTrue(a['present'][0])
 def test_proxy_missing_across_large_reference_gap(self):
  a=t.match_reference([ref(0,1),ref(200_000_000,1.1)],np.array([205_000_000],dtype=np.int64));self.assertTrue(math.isnan(a['local_age_proxy_mps'][0]))
 def test_nonfinite_reference_rejected_as_match(self):
  a=t.match_reference([ref(0,float('nan'))],np.array([1],dtype=np.int64));self.assertFalse(a['present'][0])
 def test_signed_median_cancellation_explicit_raw_units(self):
  h=t.history_summary(np.array([-1.,1.]));self.assertTrue(h['mixed_sign']);self.assertEqual(h['abs_signed_median_raw'],0);self.assertEqual(h['median_abs_raw'],1)
 def test_exact_zeros_are_not_both_signs(self):
  self.assertFalse(t.history_summary(np.array([0.,1.]))['mixed_sign'])
 def radar(self):return [{'bag_time_ns':i*50_000_000,'header_time_ns':i*50_000_000,'speed_raw':float(i%5-2)} for i in range(128)]
 def test_complete_grid_reconstruction(self):
  r=self.radar();x,h=t.replay_grid(r,np.array([r[-1]['bag_time_ns']],dtype=np.int64),0);self.assertEqual(x.shape,(1,4,128));self.assertEqual(x.dtype,np.float32);np.testing.assert_array_equal(x[0,3],np.ones(128));np.testing.assert_array_equal(x[0,2],np.zeros(128));self.assertEqual(h['raw_history_count'][0],21)
 def test_future_radar_not_used(self):
  r=self.radar();end=r[-1]['bag_time_ns'];x,_=t.replay_grid(r+[{'bag_time_ns':end+1,'header_time_ns':end+1,'speed_raw':999.}],np.array([end],dtype=np.int64),0);self.assertEqual(x[0,0,-1],np.float32(r[-1]['speed_raw']/10))
 def test_invalid_current_delivery_cannot_be_supported(self):
  r=self.radar();r[-1]['header_time_ns']=r[-1]['bag_time_ns']-250_000_001
  with self.assertRaises(ValueError):t.replay_grid(r,np.array([r[-1]['bag_time_ns']],dtype=np.int64),0)
 def test_radar_grid_cannot_cross_window_start(self):
  r=self.radar()
  with self.assertRaises(ValueError):t.replay_grid(r,np.array([r[-1]['bag_time_ns']],dtype=np.int64),1)
 def test_radar_duplicate_endpoint_rejected(self):
  r=self.radar();r.append(dict(r[-1]))
  with self.assertRaises(ValueError):t.replay_grid(r,np.array([r[-1]['bag_time_ns']],dtype=np.int64),0)
 def test_independent_radar_payload_probe(self):
  a=t.independent_message('radar',radar_message());self.assertEqual(a['speed_raw'],-.25);self.assertEqual(a['header_time_ns'],1_000_000_000)
 def test_independent_reference_payload_probe(self):
  a=t.independent_message('INSPVA',reference_message());self.assertEqual(a['horizontal_mps'],5.);self.assertEqual(a['status'],3)
 def test_decoder_and_oracle_agree_on_fixtures(self):
  for kind,blob in [('radar',radar_message()),('INSPVA',reference_message())]:
   native=t.io.decode_message('radar',blob,0) if kind=='radar' else t.refcore.decode_reference('INSPVA',blob,0)
   for k,v in t.independent_message(kind,blob).items():self.assertEqual(native[k],v)
 def test_truncated_payload_rejected(self):
  with self.assertRaises((ValueError,struct.error)):t.independent_message('INSPVA',reference_message()[:25])
 def test_bad_bag_magic_rejected(self):
  with self.assertRaises(ValueError):t.decode_bag_blob(b'not a ROS bag','reference')
 def test_reference_age_bin_boundaries(self):
  np.testing.assert_array_equal(t.age_bins(np.array([0,19_999_999,20_000_000,49_999_999,50_000_000,100_000_000],dtype=np.int64)),[0,0,1,1,2,2])
 def test_reference_age_outside_support_rejected(self):
  for a in [-1,100_000_001]:
   with self.assertRaises(ValueError):t.age_bins(np.array([a],dtype=np.int64))

class ProbePortabilityTests(unittest.TestCase):
 def probe(self,source):return [{'source':source,'message_kind':'radar','payload_base64':'AAAA','expected_fields':{'header_time_ns':123,'speed_raw':1.}}]
 def test_windows_and_posix_source_separators_match(self):
  import verify_trace as v
  v.compare_probes(self.probe('vehicle_reference_check\\reference_bags\\fit.bag'),self.probe('vehicle_reference_check/reference_bags/fit.bag'))
 def test_different_source_file_still_rejected(self):
  import verify_trace as v
  with self.assertRaises(ValueError):v.compare_probes(self.probe('vehicle_reference_check/reference_bags/other.bag'),self.probe('vehicle_reference_check/reference_bags/fit.bag'))
 def test_different_payload_still_rejected(self):
  import verify_trace as v
  actual=self.probe('a\\b');saved=self.probe('a/b');actual[0]['payload_base64']='AAAB'
  with self.assertRaises(ValueError):v.compare_probes(actual,saved)
 def test_different_decoded_timestamp_still_rejected(self):
  import verify_trace as v
  actual=self.probe('a\\b');saved=self.probe('a/b');actual[0]['expected_fields']['header_time_ns']+=1
  with self.assertRaises(ValueError):v.compare_probes(actual,saved)
 def test_comparison_does_not_mutate_either_input(self):
  import copy,verify_trace as v
  actual=self.probe('a\\b');saved=self.probe('a/b');a=copy.deepcopy(actual);b=copy.deepcopy(saved);v.compare_probes(actual,saved);self.assertEqual(actual,a);self.assertEqual(saved,b)

if __name__=='__main__':unittest.main(verbosity=2)
