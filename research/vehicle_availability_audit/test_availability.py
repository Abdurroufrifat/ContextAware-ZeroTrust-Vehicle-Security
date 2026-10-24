import importlib,importlib.util,unittest
import numpy as np

class AvailabilityTests(unittest.TestCase):
 def setUp(self):
  self.assertIsNotNone(importlib.util.find_spec('availability_core'),'availability implementation is missing')
  self.a=importlib.import_module('availability_core')
 def serve(self,times,events,ready=None,quality=None,ttl=100):
  n=len(events)
  ready=np.ones(n,dtype=bool) if ready is None else np.array(ready,dtype=bool)
  return self.a.serve(np.asarray(times,dtype=np.int64),np.asarray(events,dtype=np.int64),ready,np.where(ready,np.arange(n,dtype=float),np.nan),ready.copy() if quality is None else np.array(quality,dtype=bool),ttl)
 def test_fixed_schedule_counts_requests_without_packets(self):
  np.testing.assert_array_equal(self.a.request_times(7,208,50),[7,57,107,157,207])
 def test_schedule_uses_integer_nanoseconds_at_large_epoch(self):
  t=1700781855718492372
  np.testing.assert_array_equal(self.a.request_times(t,t+101,50),[t,t+50,t+100])
 def test_invalid_schedule_rejected(self):
  for args in [(2,1,50),(0,5,0),(True,5,1)]:
   with self.assertRaises(ValueError):self.a.request_times(*args)
 def test_future_packet_cannot_serve_earlier_request(self):
  x=self.serve([5,10,15],[10,20]);np.testing.assert_array_equal(x['latest_endpoint'],[-1,0,0]);self.assertEqual(x['status'][0],1)
 def test_ttl_boundary_and_expiry_count_in_denominator(self):
  x=self.serve([0,100,101],[0]);np.testing.assert_array_equal(x['status'],[0,0,2])
 def test_invalid_latest_packet_revokes_older_accepted_value(self):
  x=self.serve([0,50,60],[0,50],ready=[True,False]);np.testing.assert_array_equal(x['status'],[0,3,3]);self.assertTrue(np.isnan(x['prediction_mps'][1]))
 def test_latest_quality_rejection_does_not_search_backwards(self):
  x=self.serve([0,50,60],[0,50],quality=[True,False]);np.testing.assert_array_equal(x['status'],[0,4,4])
 def test_future_changes_leave_past_requests_identical(self):
  a=self.serve([0,25,49],[0,50]);b=self.serve([0,25,49],[0,999])
  for key in a:np.testing.assert_array_equal(a[key],b[key])
 def test_ambiguous_packet_clock_rejected(self):
  with self.assertRaises(ValueError):self.serve([10],[0,0])
 def test_zero_accepted_requests_have_undefined_accuracy(self):
  out=self.a.scores(np.array([1.,np.nan]),np.array([np.nan,np.nan]),np.array([False,False]));self.assertEqual(out['count'],0);self.assertIsNone(out['p95_mps']);self.assertEqual(out['missing_truth'],0)
 def test_missing_reference_is_counted_without_zero_imputation(self):
  out=self.a.scores(np.array([2.,np.nan]),np.array([2.,7.]),np.array([True,True]));self.assertEqual(out['count'],1);self.assertEqual(out['missing_truth'],1);self.assertEqual(out['rmse_mps'],0.)
 def test_diagnostic_distinguishes_current_invalid_from_stale_history(self):
  rows=[{'bag_time_ns':i*50000000,'header_time_ns':i*50000000,'speed_raw':3.} for i in range(401)]
  rows[200]['header_time_ns']=rows[200]['bag_time_ns']+1
  reasons=self.a.endpoint_reasons(rows)
  self.assertEqual(reasons[0]['reason'],'startup');self.assertEqual(reasons[200]['reason'],'current_sample_invalid');self.assertEqual(reasons[201]['reason'],'ready')
 def test_history_gap_cost_persists_after_packets_resume(self):
  rows=[{'bag_time_ns':i*50000000,'header_time_ns':i*50000000,'speed_raw':3.} for i in range(401) if not 160<=i<=170]
  reasons=self.a.endpoint_reasons(rows)
  self.assertEqual(next(x for x in reasons if x['time_ns']==8550000000)['reason'],'history_stale')
  self.assertEqual(reasons[-1]['reason'],'ready')
 def test_unready_prediction_and_quality_cannot_be_forged(self):
  with self.assertRaises(ValueError):self.a.serve(np.array([0]),np.array([0]),np.array([False]),np.array([3.]),np.array([True]),100)
if __name__=='__main__':unittest.main()
