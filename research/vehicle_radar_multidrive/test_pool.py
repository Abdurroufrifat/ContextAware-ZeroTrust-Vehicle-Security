"""Split integrity tests for the expanded development cohort."""
import copy, importlib, importlib.util, unittest

class PoolTests(unittest.TestCase):
 def module(self):
  self.assertIsNotNone(importlib.util.find_spec('pool_development'),'the development-only split guard is missing')
  return importlib.import_module('pool_development')
 def rows(self):
  return {'fit':[{'id':'a','source_role':'fit','source_route':'Urban01/day','start_ns':0,'end_ns':90}], 'calibration':[{'id':'b','source_role':'calibration','source_route':'Urban01/night','start_ns':100,'end_ns':190}]}
 def test_distinct_recordings_are_accepted(self):self.module().validate_split(self.rows())
 def test_calibration_cannot_enter_training(self):
  rows=self.rows();rows['fit'][0]['source_role']='calibration'
  with self.assertRaisesRegex(ValueError,'calibration'):self.module().validate_split(rows)
 def test_reserved_route_is_rejected(self):
  rows=self.rows();rows['fit'][0]['source_route']='Urban02/2023-11-23-17-53-52-Night'
  with self.assertRaisesRegex(ValueError,'reserved'):self.module().validate_split(rows)
 def test_same_drive_cannot_appear_in_both_roles(self):
  rows=self.rows();rows['calibration'][0]['source_route']='Urban01/day'
  with self.assertRaisesRegex(ValueError,'recording'):self.module().validate_split(rows)
 def test_duplicate_windows_are_rejected(self):
  rows=self.rows();rows['fit'].append(copy.deepcopy(rows['fit'][0]))
  with self.assertRaisesRegex(ValueError,'duplicate'):self.module().validate_split(rows)
 def test_overlap_within_a_recording_is_rejected(self):
  rows=self.rows();w=copy.deepcopy(rows['fit'][0]);w.update(id='c',start_ns=80,end_ns=180);rows['fit'].append(w)
  with self.assertRaisesRegex(ValueError,'overlap'):self.module().validate_split(rows)
 def test_unknown_source_role_is_rejected(self):
  rows=self.rows();rows['fit'][0]['source_role']='evaluation_unapproved'
  with self.assertRaisesRegex(ValueError,'source role'):self.module().validate_split(rows)
if __name__=='__main__':unittest.main()
