import unittest,numpy as np
import frame_reference_core as k
class FrameReferenceTests(unittest.TestCase):
 def call(self,name,*args):
  f=getattr(k,name,None);self.assertTrue(callable(f),'missing source-audit behavior: '+name);return f(*args)
 def test_reference_join_uses_window_and_exact_integer_time(self):
  t=1_700_000_000_000_000_000;r=self.call('reference_lookup',[{'window_id':'a','time_ns':t},{'window_id':'b','time_ns':t+1}]);self.assertIn(('a',t),r);self.assertNotIn(('a',t+1),r)
 def test_duplicate_reference_key_rejected(self):
  with self.assertRaises(ValueError):self.call('reference_lookup',[{'window_id':'a','time_ns':1},{'window_id':'a','time_ns':1}])
 def test_float_epoch_key_rejected(self):
  with self.assertRaises(ValueError):self.call('reference_lookup',[{'window_id':'a','time_ns':1.}])
 def test_sigma_exact_freshness_limit_included(self):
  self.assertTrue(self.call('sigma_available',{'uncertainty_header_time_ns':1,'north_velocity_stdev':.01,'east_velocity_stdev':.02},1_500_000_001))
 def test_sigma_one_nanosecond_stale_rejected(self):
  self.assertFalse(self.call('sigma_available',{'uncertainty_header_time_ns':1,'north_velocity_stdev':.01,'east_velocity_stdev':.02},1_500_000_002))
 def test_sigma_future_rejected(self):
  self.assertFalse(self.call('sigma_available',{'uncertainty_header_time_ns':2,'north_velocity_stdev':.01,'east_velocity_stdev':.02},1))
 def test_unknown_sigma_not_filled(self):
  self.assertFalse(self.call('sigma_available',{'uncertainty_header_time_ns':1,'north_velocity_stdev':None,'east_velocity_stdev':.02},1))
 def test_zero_quaternion_cannot_supply_orientation(self):self.assertFalse(self.call('orientation_available',np.zeros((1,4)),np.zeros((1,9)))[0])
 def test_unavailable_orientation_sentinel_respected(self):
  c=np.zeros((1,9));c[0,0]=-1;self.assertFalse(self.call('orientation_available',np.array([[0.,0.,0.,1.]]),c)[0])
 def test_unit_quaternion_with_unknown_covariance_is_orientation_not_bound(self):self.assertTrue(self.call('orientation_available',np.array([[0.,0.,0.,1.]]),np.zeros((1,9)))[0])
 def test_reference_horizontal_speed_ignores_vertical_component(self):self.assertEqual(self.call('horizontal_speed',3.,4.,12.),5.)
 def test_horizontal_speed_unchanged_by_north_east_swap(self):self.assertEqual(self.call('horizontal_speed',-3.,4.,0.),self.call('horizontal_speed',4.,-3.,0.))
 def test_missing_reference_not_silently_dropped(self):self.assertEqual(self.call('join_ids',[('a',1),('a',2)],{('a',1):0}),[0,-1])
 def test_alternative_reference_uses_same_rows(self):
  r=self.call('paired_metrics',np.array([1.,2.]),np.array([1.6,1.4]),np.array([1.1,1.9]));self.assertEqual(r['primary_reference']['count'],2);self.assertEqual(r['obd_comparison']['count'],2);self.assertEqual(r['primary_tail_also_above_half_against_obd'],0)
 def test_node_name_is_not_child_frame_identity(self):
  r=self.call('launch_geometry','<launch><node pkg="tf2_ros" type="static_transform_publisher" name="imu" args="0 0 0 0 0 0 base lidar" /></launch>','imu');self.assertFalse(r['captured_frame_in_transform_tree']);self.assertEqual(r['nodes_named_as_captured_frame'][0]['child'],'lidar')
 def test_actual_child_frame_is_connected(self):self.assertTrue(self.call('launch_geometry','<launch><node pkg="tf2_ros" type="static_transform_publisher" name="anything" args="0 0 0 0 0 0 base imu" /></launch>','imu')['captured_frame_in_transform_tree'])
 def test_duplicate_child_publishers_are_reported(self):self.assertEqual(self.call('launch_geometry','<launch><node pkg="tf2_ros" type="static_transform_publisher" name="a" args="0 0 0 0 0 0 base lidar" /><node pkg="tf2_ros" type="static_transform_publisher" name="b" args="0 0 0 0 0 0 base lidar" /></launch>','imu')['duplicate_child_frames'],['lidar'])
class PublisherEncodingTests(unittest.TestCase):
 def verify_with_ambient_encoding(self,encoding):
  import importlib
  from pathlib import Path
  from unittest.mock import patch
  v=importlib.import_module('verify_frame_reference');result=k.read(k.ROOT/'results.json');original=Path.open
  def ambient_open(path,mode='r',buffering=-1,encoding=None,errors=None,newline=None):
   selected=ambient_open.default if encoding in (None,'locale') else encoding
   return original(path,mode,buffering,None if 'b' in mode else selected,errors,newline)
  ambient_open.default=encoding
  with patch.object(Path,'open',ambient_open):
   try:actual=v.check_geometry(result)
   except (ValueError,UnicodeError) as exc:self.fail('Unchanged UTF-8 publisher data must verify under '+encoding+': '+str(exc))
  self.assertEqual(actual,5)
 def test_publisher_hashes_under_windows_western_encoding(self):self.verify_with_ambient_encoding('cp1252')
 def test_publisher_hashes_under_windows_chinese_encoding(self):self.verify_with_ambient_encoding('gbk')
 def test_publisher_hashes_under_ascii_locale(self):self.verify_with_ambient_encoding('ascii')
 def test_corrupted_calibration_file_still_rejected(self):
  import importlib,shutil,tempfile
  from pathlib import Path
  from unittest.mock import patch
  v=importlib.import_module('verify_frame_reference');result=k.read(k.ROOT/'results.json')
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);shutil.copytree(k.ROOT/'sources',root/'sources');file=root/'sources/setup_tf_1.launch';file.write_bytes(file.read_bytes()+b'\nCORRUPTED\n')
   with patch.object(v,'ROOT',root):
    with self.assertRaisesRegex(ValueError,'publisher calibration hash mismatch'):v.check_geometry(result)
if __name__=='__main__':unittest.main(verbosity=2)
