import unittest
from cadence_core import CadencePolicy,calibrate_cadence
from data_stream import build_stream

PARAM={'centers':[0,0,0,0],'allowance_mps':.1,'cusum_limit':10.,'hard_limit':2.5,'continuity_gap_ns':300000000}
class CadenceTests(unittest.TestCase):
 def admit(self,p):
  for t in range(0,1500000000,50000000):p.step(t,5,5,5)
 def test_fresh_point_after_150ms_gap_preserves_admission(self):
  p=CadencePolicy(PARAM);self.admit(p)
  self.assertTrue(p.step(1600000000,5,5,5))
 def test_gap_over_limit_quarantines(self):
  p=CadencePolicy(PARAM);self.admit(p)
  self.assertFalse(p.step(1750000001,5,5,5));self.assertFalse(p.step(1800000000,5,5,5))
 def test_invalid_endpoint_is_denied_even_inside_cadence_limit(self):
  p=CadencePolicy(PARAM);self.admit(p)
  self.assertFalse(p.step(1500000000,5,5,5,False));self.assertFalse(p.step(1550000000,5,5,5))
 def test_gaps_within_limit_do_not_reset_cusum(self):
  p=CadencePolicy({**PARAM,'cusum_limit':.3});alarms=[]
  for t in range(0,3000000000,150000000):p.step(t,5.7,5.7,5);alarms.append(p.confirmed_now)
  self.assertTrue(any(alarms))
 def test_common_bias_at_start_never_admitted(self):
  p=CadencePolicy(PARAM)
  for t in range(0,3000000000,150000000):self.assertFalse(p.step(t,12,12,5))
 def test_recovery_requires_span_and_additional_fresh_point(self):
  p=CadencePolicy(PARAM);p.step(0,12,12,5)
  for t in range(150000000,1200000000,150000000):self.assertFalse(p.step(t,5,5,5))
  self.assertFalse(p.step(1200000000,5,5,5));self.assertTrue(p.step(1350000000,5,5,5))
 def test_duplicate_clock_does_not_count_as_fresh_evidence(self):
  p=CadencePolicy(PARAM);self.admit(p)
  self.assertFalse(p.step(1450000000,5,5,5));self.assertFalse(p.step(1500000000,5,5,5))
 def test_nonfinite_values_fail_closed(self):
  p=CadencePolicy(PARAM);self.admit(p)
  self.assertFalse(p.step(1500000000,float('nan'),5,5))
 def test_gap_parameter_cannot_exceed_safety_cap(self):
  for gap in (99999999,300000001,True,150000000.5):
   with self.assertRaises(ValueError):CadencePolicy({**PARAM,'continuity_gap_ns':gap})
 def test_calibration_rejects_evaluation_role(self):
  with self.assertRaises(ValueError):calibrate_cadence([{'role':'evaluation_urban02_night'}],[])
 def test_deadline_is_derived_from_timing_and_capped(self):
  def rows(role):return [{'role':role,'window_id':role,'time_ns':t,'valid':True,'gnss_mps':5.,'label_mps':5.,'prediction_mps':5.} for t in (0,50000000,200000000,480000000)]
  p=calibrate_cadence(rows('fit'),rows('calibration'))
  self.assertEqual(p['continuity_gap_ns'],300000000)
  self.assertGreater(p['cusum_limit'],0)
class FreshnessTests(unittest.TestCase):
 def fixture(self):
  r=[{'bag_time_ns':i*50000000,'header_time_ns':i*50000000,'speed_raw':5.} for i in range(61)]
  o=[{'bag_time_ns':i*50000000,'header_time_ns':i*50000000,'speed_raw':18.,'unit':'kph'} for i in range(61)]
  g=[{'bag_time_ns':i*50000000,'header_time_ns':i*50000000,'speed_raw':5.,'status':0,'motion_source':1} for i in range(61)]
  return r,o,g
 def test_stale_gnss_rejected_despite_300ms_continuity_budget(self):
  r,o,g=self.fixture()
  for q in g:q['header_time_ns']-=150000000
  rows,_=build_stream(r,o,g,0,3000000000,{'slope':1,'intercept':0})
  self.assertTrue(rows);self.assertFalse(any(q['valid'] for q in rows))
 def test_stale_obd_rejected_despite_300ms_continuity_budget(self):
  r,o,g=self.fixture()
  for q in o:q['header_time_ns']-=150000000
  rows,_=build_stream(r,o,g,0,3000000000,{'slope':1,'intercept':0})
  self.assertFalse(any(q['valid'] for q in rows))
 def test_stale_radar_rejected_despite_300ms_continuity_budget(self):
  r,o,g=self.fixture()
  for q in r:q['header_time_ns']-=260000000
  rows,_=build_stream(r,o,g,0,3000000000,{'slope':1,'intercept':0})
  self.assertFalse(any(q['valid'] for q in rows))
if __name__=='__main__':unittest.main()
