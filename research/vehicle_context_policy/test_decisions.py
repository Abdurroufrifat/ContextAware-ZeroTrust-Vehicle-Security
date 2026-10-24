import unittest
from run_experiment import decisions
from context_core import fit_model
from monitor_core import Cusum,Ewma
class DecisionTests(unittest.TestCase):
    def setUp(self):
        features=dict(peer=2.,yaw=.02,steering=.1,acceleration=.2)
        rows=[dict(candidate=2.,features=features) for _ in range(30)]
        self.model=fit_model(rows[:20],rows[20:])
        self.checked=dict(candidate_ok=True,context_ok=True,candidate=2.,features=features,
            candidate_status="authenticated",context_status="authenticated")
    def call(self,checked):
        return decisions(self.model,checked,2.,(Cusum(),Ewma()))
    def test_authentication_failure_cannot_be_overruled_by_ai(self):
        self.checked["candidate_ok"]=False
        out=self.call(self.checked)
        self.assertFalse(any(v["accept"] for p,v in out.items() if p!="unguarded"))
    def test_context_failure_blocks_context_policies(self):
        self.checked["context_ok"]=False
        out=self.call(self.checked)
        self.assertTrue(out["authenticated_only"]["accept"])
        self.assertFalse(any(out[p]["accept"] for p in ("fixed_agreement","global_learned","context_learned","cusum","ewma")))
    def test_ood_ablation_changes_only_support_check(self):
        self.checked["features"]=dict(peer=20.,yaw=.02,steering=.1,acceleration=.2)
        self.checked["candidate"]=20.
        out=self.call(self.checked)
        self.assertTrue(out["context_no_ood"]["accept"])
        self.assertFalse(out["context_learned"]["accept"])
    def test_coordinated_plausible_source_compromise_is_not_certified(self):
        self.checked["features"]["peer"]=2.3
        self.checked["candidate"]=2.3
        self.assertTrue(self.call(self.checked)["context_learned"]["accept"])
if __name__=="__main__": unittest.main()
