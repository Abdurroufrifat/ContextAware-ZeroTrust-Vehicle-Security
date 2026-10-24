import unittest
try:
 from completion_core import complete,metric_gate
except ImportError:complete=metric_gate=None
class CompletionTests(unittest.TestCase):
 def test_software_tests_alone_do_not_complete_research(self):
  self.assertIsNotNone(complete)
  self.assertFalse(complete({"software":"passed","research":"unknown"}))
 def test_all_explicit_requirements_must_pass(self):
  self.assertIsNotNone(complete)
  self.assertTrue(complete({"software":"passed","research":"passed"}))
  self.assertFalse(complete({}))
 def test_zero_accepts_cannot_count_as_success(self):
  self.assertIsNotNone(metric_gate)
  self.assertFalse(metric_gate(0,117,None,0))
 def test_original_thresholds_apply_together(self):
  self.assertIsNotNone(metric_gate)
  self.assertTrue(metric_gate(95,100,.5,0))
  self.assertFalse(metric_gate(94,100,.1,0))
  self.assertFalse(metric_gate(99,100,.500001,0))
  self.assertFalse(metric_gate(99,100,.1,1))
if __name__=="__main__":unittest.main(verbosity=2)
