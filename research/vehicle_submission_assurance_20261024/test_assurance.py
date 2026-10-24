import unittest
from assurance_core import facts,summary,dominates,quantile
def card():
 return {"certificate":{"issued_hold_steps":10,"max_hold_steps":100,"execution_error":2.5,"refresh_status":"range_conflict"},"point_budget_satisfied":False,"actuator_status":"authenticated","requested_command_mmps2":300,"accepted_command_mmps2":300}
class AssuranceTests(unittest.TestCase):
 def test_accepted_command_does_not_certify_speed(self):
  x=facts(card());self.assertTrue(x["command_gate_accepted"]);self.assertFalse(x["speed_budget_certified"]);self.assertIn("no certified",x["speed_text"])
 def test_backup_is_distinct(self):
  c=card();c["actuator_status"]="missing_frame";c["accepted_command_mmps2"]=-4000
  self.assertFalse(facts(c)["command_gate_accepted"]);self.assertEqual(facts(c)["accepted_mps2"],-4)
 def test_conditional_certification(self):
  c=card();c["point_budget_satisfied"]=True;x=facts(c);self.assertIn("Conditional",x["speed_text"]);self.assertIn("honest initial",x["assumptions_text"])
 def test_lifetimes_not_conflated(self):
  f=facts(card());self.assertEqual(f["hold_ms"],100);self.assertEqual(f["maximum_hold_ms"],1000)
 def test_no_truth_or_efficacy_claim(self):
  f=facts(card());self.assertIn("does not",f["measurement_truth_text"]);self.assertFalse(f["human_effectiveness_evaluated"])
 def test_pareto_requires_both_objectives(self):
  a={"contact_fraction":0,"mean_progress_m":20};b={"contact_fraction":.1,"mean_progress_m":100}
  self.assertFalse(dominates(a,b));self.assertFalse(dominates(b,a))
 def test_quantile(self):
  self.assertEqual(quantile([4,1,3,2],.5),2);self.assertIsNone(quantile([], .95))
 def test_missing_certification_fails_closed(self):
  c=card();c["point_budget_satisfied"]=None;self.assertFalse(facts(c)["speed_budget_certified"])
if __name__=="__main__":unittest.main()
