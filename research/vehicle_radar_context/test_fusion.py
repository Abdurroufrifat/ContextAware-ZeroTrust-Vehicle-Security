import unittest
from fusion_rules import fusion_reason
class FusionReasonTests(unittest.TestCase):
    def test_context_rejection_is_not_reported_as_acceptance_reason(self):
        self.assertEqual(fusion_reason("context_no_ood_plus_radar",True,"authenticated",True,"authenticated",True,False),"learned_context_rejection")
    def test_corrected_age_failure_cannot_be_called_agreement(self):
        self.assertEqual(fusion_reason("radar_agreement",True,"authenticated",False,"measurement_stale",False,True),"radar_measurement_stale")
    def test_candidate_authentication_failure_has_priority(self):
        self.assertEqual(fusion_reason("radar_agreement",False,"bad_tag",True,"authenticated",True,True),"candidate_bad_tag")
if __name__=="__main__": unittest.main()
