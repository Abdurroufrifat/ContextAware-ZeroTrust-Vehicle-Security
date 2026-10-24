import struct,unittest
import numpy as np
import imu_core as k
BASE=10_000_000_000
def payload(acc=(0.,0.,1.),gyro=(1.,2.,3.),cov=None,nanos=0):
 c=[0.]*9 if cov is None else cov
 return struct.pack('<IIII',1,10,nanos,5)+b'frame'+struct.pack('<37d',*([0.,0.,0.,1.]+[0.]*9+list(gyro)+list(c)+list(acc)+list(c)))
def raw(blob=None):
 b=payload() if blob is None else blob
 return {'bag_ns':np.array([BASE],dtype=np.int64),'payload_length':np.array([len(b)],dtype=np.uint16),'payload_bytes':np.frombuffer(b,dtype=np.uint8).reshape(1,-1).copy()}
def source():
 t=BASE+np.arange(128,dtype=np.int64)*50_000_000
 return {'bag_ns':t,'header_ns':t.copy(),'usable':np.ones(128,dtype=bool),'acceleration_norm':1.+np.arange(128)*.01,'angular_velocity_norm':2.+np.arange(128)*.02}
class ImuTests(unittest.TestCase):
 def call(self,name,*args,**kw):
  fn=getattr(k,name,None);self.assertTrue(callable(fn),'IMU behavior missing: '+name);return fn(*args,**kw)
 def test_raw_acceleration_scale(self):np.testing.assert_array_equal(self.call('decode_payloads',raw())['acceleration'],[[0,0,9.8]])
 def test_header_and_writer_integer_precision(self):
  a=self.call('decode_payloads',raw(payload(nanos=7)));self.assertEqual(int(a['header_ns'][0]),BASE+7);self.assertEqual(int(a['bag_ns'][0]),BASE)
 def test_unknown_covariance_not_a_bound(self):
  a=self.call('decode_payloads',raw());self.assertTrue(a['usable'][0]);self.assertTrue(a['accel_cov_unknown'][0])
 def test_unavailable_covariance_invalid(self):self.assertFalse(self.call('decode_payloads',raw(payload(cov=[-1]+[0]*8)))['usable'][0])
 def test_nonfinite_measurement_invalid(self):self.assertFalse(self.call('decode_payloads',raw(payload(acc=(np.nan,0,1))))['usable'][0])
 def test_scaled_overflow_invalid(self):self.assertFalse(self.call('decode_payloads',raw(payload(acc=(1e308,0,1))))['usable'][0])
 def test_large_representable_norm_stays_finite(self):self.assertTrue(np.isfinite(self.call('decode_payloads',raw(payload(acc=(1e200,0,1))))['acceleration_norm'][0]))
 def test_bad_nanoseconds_rejected(self):
  with self.assertRaises(ValueError):self.call('decode_payloads',raw(payload(nanos=1_000_000_000)))
 def test_truncated_record_rejected(self):
  with self.assertRaises(ValueError):self.call('decode_payloads',raw(payload()[:-1]))
 def test_duplicate_arrivals_rejected(self):
  a=raw();a={name:np.repeat(v,2,axis=0) for name,v in a.items()}
  with self.assertRaises(ValueError):self.call('decode_payloads',a)
 def test_float_clock_rejected(self):
  a=raw();a['bag_ns']=a['bag_ns'].astype(float)
  with self.assertRaises(ValueError):self.call('decode_payloads',a)
 def test_padding_cannot_hide_payload_bytes(self):
  a=raw();a['payload_bytes']=np.column_stack([a['payload_bytes'],np.array([1],dtype=np.uint8)])
  with self.assertRaises(ValueError):self.call('decode_payloads',a)
 def test_latest_future_header_has_no_fallback(self):
  a=source();a['header_ns'][-1]+=1;out=self.call('lookup',a,np.array([a['bag_ns'][-1]],dtype=np.int64));self.assertEqual(out['status'].tolist(),[3]);self.assertEqual(out['latest_message'].tolist(),[127])
 def test_latest_invalid_has_no_fallback(self):
  a=source();a['usable'][-1]=False;out=self.call('lookup',a,np.array([a['bag_ns'][-1]],dtype=np.int64));self.assertEqual(out['status'].tolist(),[4]);self.assertTrue(np.isnan(out['acceleration_norm'][0]))
 def test_both_clocks_must_be_fresh(self):
  a=source();t=a['bag_ns'][-1];a['header_ns'][-1]-=100_000_001;self.assertEqual(self.call('lookup',a,np.array([t],dtype=np.int64))['status'].tolist(),[2])
 def test_float_header_clock_rejected(self):
  a=source();a['header_ns']=a['header_ns'].astype(float)
  with self.assertRaises(ValueError):self.call('lookup',a,np.array([BASE],dtype=np.int64))
 def test_no_prior_message_status(self):self.assertEqual(self.call('lookup',source(),np.array([BASE-1],dtype=np.int64))['status'].tolist(),[1])
 def test_context_has45_features(self):
  a=source();x,_=self.call('features',a,a['bag_ns'][-1:]);self.assertEqual(x.shape,(1,45));self.assertEqual(x[0,2],1.)
 def test_fixed_grid_mean_and_slope(self):
  a=source();x,_=self.call('features',a,a['bag_ns'][-1:]);self.assertAlmostEqual(x[0,3],np.mean(a['acceleration_norm'][-5:]));self.assertAlmostEqual(x[0,5],.2);self.assertAlmostEqual(x[0,8],.4);self.assertEqual(x[0,9],1.)
 def test_missing_context_uses_zero_and_coverage(self):
  a=source();a['usable'][:]=False;x,current=self.call('features',a,a['bag_ns'][-1:]);np.testing.assert_array_equal(x,np.zeros((1,45)));self.assertEqual(current['status'].tolist(),[4])
 def test_one_valid_history_sample_has_zero_slope(self):
  a=source();a['usable'][:-1]=False;x,_=self.call('features',a,a['bag_ns'][-1:]);self.assertEqual(x[0,5],0.);self.assertEqual(x[0,4],0.);self.assertEqual(x[0,9],.2)
 def test_future_measurement_cannot_change_features(self):
  a=source();t=a['bag_ns'][-2:-1];x,_=self.call('features',a,t);a['acceleration_norm'][-1]=999.;a['angular_velocity_norm'][-1]=999.;y,_=self.call('features',a,t);np.testing.assert_array_equal(x,y)
 def test_feature_input_endpoints_must_be_sorted(self):
  with self.assertRaises(ValueError):self.call('features',source(),np.array([BASE+1,BASE],dtype=np.int64))
 def current(self):return {'endpoint_ns':np.array([BASE],dtype=np.int64),'bag_ns':np.array([BASE-10],dtype=np.int64),'header_ns':np.array([BASE-20],dtype=np.int64),'status':np.array([0],dtype=np.uint8)}
 def test_prediction_expires_at_oldest_used_clock(self):
  q=np.array([BASE+99_999_980,BASE+99_999_981],dtype=np.int64);out=self.call('prediction_binding',q,np.array([0,0]),self.current());self.assertEqual(out.tolist(),[True,False])
 def test_missing_endpoint_does_not_borrow_binding(self):self.assertEqual(self.call('prediction_binding',np.array([BASE]),np.array([-1]),self.current()).tolist(),[False])
 def test_prediction_not_available_before_its_endpoint(self):self.assertEqual(self.call('prediction_binding',np.array([BASE-1]),np.array([0]),self.current()).tolist(),[False])
 def test_invalid_source_cannot_bind_prediction(self):
  a=self.current();a['status'][0]=4;self.assertEqual(self.call('prediction_binding',np.array([BASE]),np.array([0]),a).tolist(),[False])
class SourceAuditTests(unittest.TestCase):
 def test_matching_invalid_nan_norms_are_equal(self):
  import source_audit as s
  self.assertTrue(callable(getattr(s,'norms_equal',None)),'missing invalid-norm comparison');self.assertTrue(s.norms_equal(float('nan'),float('nan')))
 def test_different_valid_norms_are_not_equal(self):
  import source_audit as s
  self.assertTrue(callable(getattr(s,'norms_equal',None)),'missing invalid-norm comparison');self.assertFalse(s.norms_equal(1.,2.))
if __name__=='__main__':unittest.main(verbosity=2)
