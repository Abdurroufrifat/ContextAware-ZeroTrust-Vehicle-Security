import unittest
import previous_temporal_core as p
from data_stream import build_stream

PARAM={'centers':[0,0,0,0],'allowance_mps':.1,'cusum_limit':.3,'hard_limit':2.5}
class TemporalTests(unittest.TestCase):
 def step(self,policy,t,residual=0,valid=True):return policy.step(int(t*1e9),5+residual,5+residual,5,valid)
 def test_startup_common_bias_has_no_admission(self):
  x=p.TemporalPolicy(PARAM)
  for i in range(100):self.assertFalse(self.step(x,i*.05,7))
  self.assertTrue(x.latched)
 def test_subthreshold_persistent_residual_accumulates_to_alarm(self):
  x=p.TemporalPolicy(PARAM);alarms=[]
  for i in range(100):self.step(x,i*.05,.6);alarms.append(x.confirmed_now)
  self.assertTrue(any(alarms))
 def test_alarm_requires_fresh_physical_recovery_span_and_new_endpoint(self):
  x=p.TemporalPolicy(PARAM);self.step(x,0,7)
  for i in range(1,22):self.assertFalse(self.step(x,i*.05,0))
  self.assertTrue(self.step(x,1.1,0));self.assertFalse(x.latched)
 def test_microburst_of_clean_points_does_not_satisfy_time_span(self):
  x=p.TemporalPolicy(PARAM);self.step(x,0,7)
  for i in range(1,100):self.assertFalse(self.step(x,i*.0001))
 def test_isolated_soft_warning_denies_without_erasing_a_confirmed_alarm(self):
  x=p.TemporalPolicy({**PARAM,'cusum_limit':10})
  for i in range(30):self.step(x,i*.05)
  self.assertTrue(x.admitted);self.assertFalse(self.step(x,1.5,1.2));self.assertTrue(self.step(x,1.55))
  self.step(x,1.6,7);self.assertFalse(self.step(x,1.65));self.assertTrue(x.latched)
 def test_clock_gap_and_missing_evidence_require_readmission(self):
  x=p.TemporalPolicy(PARAM)
  for i in range(30):self.step(x,i*.05)
  self.assertFalse(self.step(x,3));self.assertFalse(self.step(x,3.05,valid=False));self.assertFalse(self.step(x,3.1))
 def test_backward_clock_fails_closed(self):
  x=p.TemporalPolicy(PARAM)
  for i in range(30):self.step(x,i*.05)
  self.assertFalse(self.step(x,1));self.assertFalse(self.step(x,1.5))
 def test_constructor_copies_parameters(self):
  d={**PARAM,'centers':list(PARAM['centers'])};x=p.TemporalPolicy(d);d['hard_limit']=100;d['centers'][2]=99
  self.assertFalse(self.step(x,0,7));self.assertTrue(x.confirmed_now)
 def test_calibration_refuses_evaluation_data(self):
  with self.assertRaises(ValueError):p.calibrate([{'role':'evaluation_urban03'}],[])
 def test_invalid_numbers_cannot_grant(self):
  x=p.TemporalPolicy(PARAM)
  for i in range(30):self.step(x,i*.05)
  self.assertFalse(x.step(1500000000,float('nan'),5,5,True))
class TimelineTests(unittest.TestCase):
 def fixture(self):
  radar=[{'bag_time_ns':i*50000000,'header_time_ns':i*50000000,'speed_raw':5.} for i in range(61)]
  obd=[{'bag_time_ns':i*50000000,'header_time_ns':i*50000000,'speed_raw':18.,'unit':'kph'} for i in range(61)]
  gnss=[{'bag_time_ns':i*50000000,'header_time_ns':i*50000000,'speed_raw':5.,'status':0,'motion_source':1} for i in range(61)]
  return radar,obd,gnss
 def test_missing_gnss_keeps_every_scored_endpoint_and_counts_prefix(self):
  radar,obd,_=self.fixture();rows,stats=build_stream(radar,obd,[],0,3000000000,{'slope':1,'intercept':0})
  self.assertEqual(len(rows),41);self.assertEqual(stats['warmup_denied_endpoints'],20);self.assertEqual(stats['missing_or_invalid_endpoints'],41);self.assertFalse(any(r['valid'] for r in rows))
 def test_future_radar_payload_cannot_change_earlier_prediction(self):
  radar,obd,gnss=self.fixture();model={'slope':1,'intercept':0};before,_=build_stream(radar,obd,gnss,0,3000000000,model)
  radar[-1]['speed_raw']=500;after,_=build_stream(radar,obd,gnss,0,3000000000,model)
  self.assertEqual(before[:-1],after[:-1]);self.assertTrue(all(r['valid'] for r in before))
 def test_duplicate_radar_time_rejected(self):
  radar,obd,gnss=self.fixture();radar[-1]['bag_time_ns']=radar[-2]['bag_time_ns']
  with self.assertRaises(ValueError):build_stream(radar,obd,gnss,0,3000000000,{'slope':1,'intercept':0})
if __name__=='__main__':unittest.main()
