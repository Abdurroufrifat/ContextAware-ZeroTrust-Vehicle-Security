import math,struct,unittest
import reference_core as c
class ChecksumPortabilityTests(unittest.TestCase):
 def test_utf8_publisher_file_read_correctly_under_windows_default_decoding(self):
  from pathlib import Path
  from tempfile import TemporaryDirectory
  from unittest.mock import patch
  from verify_reference import read_published_checksums
  with TemporaryDirectory() as folder:
   path=Path(folder)/'checksums.txt';path.write_text('Filename/Fichier: sample/reference.bag\nSHA256 Checksum/Sommes de contrôle: '+'a'*64+'\n',encoding='utf-8')
   with patch('io.text_encoding',side_effect=lambda encoding=None, stacklevel=2: encoding if encoding is not None else 'cp1252'):
    self.assertEqual(read_published_checksums(path),{'sample/reference.bag':'a'*64})
 def test_empty_or_unrecognized_publisher_file_has_clear_error(self):
  from pathlib import Path
  from tempfile import TemporaryDirectory
  from verify_reference import read_published_checksums
  with TemporaryDirectory() as folder:
   path=Path(folder)/'checksums.txt';path.write_text('invalid publisher file',encoding='utf-8')
   with self.assertRaisesRegex(ValueError,'No publisher checksums'):read_published_checksums(path)
def string(x):
 b=x.encode();return struct.pack('<I',len(b))+b
def message(extended=False,stamp=1000000000):
 b=struct.pack('<III',7,stamp//1000000000,stamp%1000000000)+string('ENU')+string('INSPVAX' if extended else 'INSPVA')+struct.pack('<HBIBHI',1465 if extended else 507,0,7,160,2288,1000)
 if extended:b+=struct.pack('<II3df6d9fIH',3,56,45.,-75.,100.,0.,3.,4.,12.,0.,0.,0.,.01,.01,.01,.1,.2,.3,.01,.01,.01,0,0)
 else:b+=struct.pack('<9dI',45.,-75.,100.,3.,4.,12.,0.,0.,0.,3)
 return b
class DecoderTests(unittest.TestCase):
 def test_inspva_decodes_north_east_up_in_mps(self):
  r=c.decode_reference('INSPVA',message(),1000000010)
  self.assertEqual(r['horizontal_mps'],5.);self.assertEqual(r['speed_3d_mps'],13.);self.assertEqual(r['status'],3)
 def test_inspvax_decodes_uncertainty_without_promoting_it_to_bound(self):
  r=c.decode_reference('INSPVAX',message(True),1000000010)
  self.assertAlmostEqual(r['north_velocity_stdev'],.1,places=6);self.assertEqual(r['uniform_error_bound_mps'],None)
 def test_trailing_bytes_rejected(self):
  with self.assertRaises(ValueError):c.decode_reference('INSPVA',message()+b'x',1000000010)
 def test_truncated_message_rejected(self):
  with self.assertRaises(ValueError):c.decode_reference('INSPVAX',message(True)[:-1],1000000010)
 def test_nonfinite_uncertainty_does_not_destroy_finite_velocity(self):
  b=bytearray(message(True));struct.pack_into('<f',b,len(b)-42+3*4,float('nan'));r=c.decode_reference('INSPVAX',bytes(b),1000000010)
  self.assertIsNone(r['north_velocity_stdev']);self.assertEqual(r['horizontal_mps'],5.)
class MatchingTests(unittest.TestCase):
 def row(self,t,status=3):return {'header_time_ns':t,'status':status,'horizontal_mps':5.,'speed_3d_mps':5.}
 def test_future_reference_never_used(self):
  index=c.ReferenceIndex([self.row(200)])
  self.assertIsNone(index.latest(100,100000000))
 def test_age_limit_inclusive_but_stale_rejected(self):
  index=c.ReferenceIndex([self.row(100)])
  self.assertIsNotNone(index.latest(100000100,100000000));self.assertIsNone(index.latest(100000101,100000000))
 def test_bad_status_stays_in_quality_counts_but_is_not_matched(self):
  index=c.ReferenceIndex([self.row(100,0)])
  self.assertEqual(index.status_counts,{'0':1});self.assertIsNone(index.latest(200,100000000))
 def test_duplicate_timestamps_rejected(self):
  with self.assertRaises(ValueError):c.ReferenceIndex([self.row(100),self.row(100)])
 def test_recent_invalid_sample_does_not_fall_back_to_older_good_sample(self):
  index=c.ReferenceIndex([self.row(100),self.row(200,0)])
  self.assertIsNone(index.latest(300,100000000))
 def test_backward_reference_time_rejected(self):
  with self.assertRaises(ValueError):c.ReferenceIndex([self.row(200),self.row(100)])
 def test_nonfinite_velocity_does_not_match(self):
  row=self.row(100);row['horizontal_mps']=float('nan');self.assertIsNone(c.ReferenceIndex([row]).latest(200,100000000))
class ClaimTests(unittest.TestCase):
 def test_sigma_or_quantile_does_not_qualify_as_uniform_bound(self):
  r=c.qualify({'uniform_error_bound_mps':.1,'bound_kind':'P95','raw_units_verified':True,'geometry_verified':True,'threat_independent':True,'authenticated_live_channel':True,'trusted_live_time':True,'acceleration_bound_mps2':1.})
  self.assertFalse(r['qualified']);self.assertIn('uniform_physical_bound',r['failed_requirements'])
 def test_postprocessed_reference_never_counts_as_live_witness(self):
  r=c.qualify({'postprocessed':True,'bound_kind':'physical_uniform','uniform_error_bound_mps':.1,'raw_units_verified':True,'geometry_verified':True,'threat_independent':True,'authenticated_live_channel':True,'trusted_live_time':True,'acceleration_bound_mps2':1.})
  self.assertFalse(r['qualified'])
 def test_missing_requirements_fail_closed(self):self.assertFalse(c.qualify({})['qualified'])
 def test_age_expands_ambiguity_limit(self):self.assertAlmostEqual(c.conditional_limit(.1,3.,.25),1.7)
 def test_hypothetical_limit_refuses_unknown_or_invalid_bounds(self):
  for values in ((None,3.,.1),(.1,-1.,.1),(.1,3.,float('nan'))):
   with self.assertRaises(ValueError):c.conditional_limit(*values)
 def test_measured_max_is_not_a_uniform_bound(self):
  self.assertFalse(c.qualify({'bound_kind':'observed_maximum','uniform_error_bound_mps':.01})['qualified'])
 def test_complete_hypothetical_claims_pass_metadata_check_without_enabling_production(self):
  r=c.qualify({'postprocessed':False,'bound_kind':'physical_uniform','uniform_error_bound_mps':.1,'raw_units_verified':True,'geometry_verified':True,'threat_independent':True,'authenticated_live_channel':True,'trusted_live_time':True,'acceleration_bound_mps2':1.,'max_reference_age_s':.1})
  self.assertTrue(r['qualified']);self.assertFalse(r['failed_requirements']);self.assertIn('does not authenticate',r['interpretation'])
 def test_boolean_or_infinite_physical_bounds_rejected(self):
  for v in (True,float('inf')):
   with self.assertRaises(ValueError):c.conditional_limit(v,3.,.1)
if __name__=='__main__':unittest.main()
