import struct,unittest
import numpy as np
import aux_core as a
class AuxiliaryTests(unittest.TestCase):
 def payload(self,ns=0,cov=None,acc=(0.,0.,1.),gyro=(1.,2.,3.)):
  c=[0.]*9 if cov is None else cov;values=[0.,0.,0.,1.]+[0.]*9+list(gyro)+list(c)+list(acc)+list(c);return struct.pack('<IIII',1,10,ns,5)+b'frame'+struct.pack('<37d',*values)
 def row(self,bag=10_000_000_000,**kw):return a.decode_imu(self.payload(**kw),bag,'/livox/imu')
 def test_scaled_acceleration_overflow_invalid(self):
  row=self.row(acc=(1e308,0,1));self.assertFalse(row['usable']);q=a.context([row],np.array([10_000_000_000],dtype=np.int64));self.assertEqual(q['status'][0],4);self.assertTrue(np.isnan(q['acceleration_norm'][0]))
 def test_large_representable_norm_is_finite(self):
  row=self.row(acc=(1e200,0,1));self.assertTrue(row['usable']);q=a.context([row],np.array([10_000_000_000],dtype=np.int64));self.assertTrue(np.isfinite(q['acceleration_norm'][0]));self.assertAlmostEqual(q['acceleration_norm'][0]/1e200,9.8)
 def test_true_norm_overflow_invalid(self):self.assertFalse(self.row(gyro=(1.7e308,1.7e308,1.7e308))['usable'])
 def test_scaled_covariance_overflow_invalid(self):self.assertFalse(self.row(cov=[1e308,0,0,0,1,0,0,0,1])['usable'])
 def test_context_rechecks_finite_norm(self):
  row=self.row();row['acceleration']=[float('inf'),0.,0.];q=a.context([row],np.array([10_000_000_000],dtype=np.int64));self.assertEqual(q['status'][0],4);self.assertTrue(np.isnan(q['acceleration_norm'][0]))
 def test_decoder_exists(self):self.assertTrue(callable(getattr(a,'decode_imu',None)))
 def test_livox_documented_scale(self):self.assertEqual(self.row()['acceleration'],[0.,0.,9.8])
 def test_xsens_not_rescaled(self):self.assertEqual(a.decode_imu(self.payload(),10_000_000_000,'/xsens/imu')['acceleration'],[0.,0.,1.])
 def test_gyro_nominal_rad_per_second(self):self.assertEqual(self.row()['angular_velocity'],[1.,2.,3.])
 def test_unknown_covariance_remains_unknown(self):
  r=self.row();self.assertTrue(r['acceleration_covariance_unknown']);self.assertTrue(r['angular_velocity_covariance_unknown']);self.assertTrue(r['usable'])
 def test_unavailable_covariance_marks_invalid(self):
  r=self.row(cov=[-1.]+[0.]*8);self.assertFalse(r['usable'])
 def test_nonfinite_vector_preserved_but_invalid(self):
  r=self.row(acc=(np.nan,0,1));self.assertFalse(r['usable']);self.assertTrue(np.isnan(r['acceleration'][0]))
 def test_reference_imu_is_forbidden(self):
  with self.assertRaises(ValueError):a.decode_imu(self.payload(),10_000_000_000,'/novatel/imu')
 def test_filtered_channel_is_forbidden(self):
  with self.assertRaises(ValueError):a.decode_imu(self.payload(),10_000_000_000,'/xsens/filter/twist')
 def test_truncated_payload_rejected(self):
  with self.assertRaises(ValueError):a.decode_imu(self.payload()[:-1],10_000_000_000,'/livox/imu')
 def test_extra_payload_rejected(self):
  with self.assertRaises(ValueError):a.decode_imu(self.payload()+b'x',10_000_000_000,'/livox/imu')
 def test_invalid_nanoseconds_rejected(self):
  with self.assertRaises(ValueError):self.row(ns=1_000_000_000)
 def test_header_integer_precision(self):self.assertEqual(self.row(ns=7)['header_ns'],10_000_000_007)
 def test_no_prior_source_returns_missing(self):self.assertEqual(a.context([],np.array([10],dtype=np.int64))['status'][0],1)
 def test_exact_ttl_and_one_ns_stale(self):
  q=a.context([self.row()],np.array([10_100_000_000,10_100_000_001],dtype=np.int64));np.testing.assert_array_equal(q['status'],[0,2])
 def test_future_header_newest_does_not_fall_back(self):
  rows=[self.row(),self.row(bag=10_020_000_000,ns=40_000_000)];q=a.context(rows,np.array([10_030_000_000],dtype=np.int64));self.assertEqual(q['status'][0],3);self.assertEqual(q['latest_row'][0],1)
 def test_newest_invalid_does_not_fall_back(self):
  rows=[self.row(),self.row(bag=10_020_000_000,ns=20_000_000,acc=(np.nan,0,1))];self.assertEqual(a.context(rows,np.array([10_030_000_000],dtype=np.int64))['status'][0],4)
 def test_future_arrival_not_used(self):
  q=a.context([self.row(bag=10_020_000_000)],np.array([10_010_000_000],dtype=np.int64));self.assertEqual(q['status'][0],1)
 def test_stale_arrival_cannot_be_hidden_by_header(self):
  q=a.context([self.row(ns=90_000_000)],np.array([10_100_000_001],dtype=np.int64));self.assertEqual(q['status'][0],2)
 def test_valid_magnitude_values(self):
  q=a.context([self.row()],np.array([10_000_000_000],dtype=np.int64));self.assertAlmostEqual(q['acceleration_norm'][0],9.8);self.assertAlmostEqual(q['angular_velocity_norm'][0],np.sqrt(14))
 def test_duplicate_arrival_rejected(self):
  with self.assertRaises(ValueError):a.context([self.row(),self.row()],np.array([10_000_000_000],dtype=np.int64))
 def test_mixed_sensor_frames_rejected(self):
  b=self.row(bag=10_010_000_000,ns=10_000_000);b['frame_id']='other'
  with self.assertRaises(ValueError):a.context([self.row(),b],np.array([10_020_000_000],dtype=np.int64))
 def test_float_endpoints_rejected(self):
  with self.assertRaises(ValueError):a.context([self.row()],np.array([10.]))
 def test_status_counts_keep_all_ticks(self):
  q=a.context([self.row()],np.array([9_999_999_999,10_000_000_000,10_100_000_001],dtype=np.int64));s=a.context_stats(q);self.assertEqual(sum(s['status_counts'].values()),3);self.assertEqual(s['available_ticks'],1)
 def test_timestamps_not_mutated(self):
  t=np.array([10_000_000_000],dtype=np.int64);old=t.copy();a.context([self.row()],t);np.testing.assert_array_equal(t,old)
 def test_acceleration_covariance_scales_as_variance(self):
  r=self.row(cov=[1.,0,0,0,1,0,0,0,1]);self.assertAlmostEqual(r['acceleration_covariance'][0],9.8**2)
 def test_pending_fit_source_allowed(self):
  self.assertIsNone(a.check_metadata_source(a.PENDING['Urban03'],'imus.bag'))
 def test_reserved_metadata_source_rejected(self):
  with self.assertRaises(ValueError):a.check_metadata_source('Urban02/2023-11-23-17-53-52-Night','imus.bag')
 def test_reference_payload_source_rejected(self):
  with self.assertRaises(ValueError):a.check_metadata_source(a.PENDING['Urban03'],'reference.bag')
 def test_metadata_range_boundaries(self):
  self.assertIsNone(a.allowed_metadata_range(0,4095,100000,'header'));self.assertIsNone(a.allowed_metadata_range(90000,99999,100000,'index'))
 def test_measurement_chunk_cannot_be_requested(self):
  with self.assertRaises(ValueError):a.allowed_metadata_range(4096,8191,100000,'index')
 def test_excessive_index_range_rejected(self):
  with self.assertRaises(ValueError):a.allowed_metadata_range(4096,3_000_000,3_000_001,'index')
if __name__=='__main__':unittest.main(verbosity=2)
