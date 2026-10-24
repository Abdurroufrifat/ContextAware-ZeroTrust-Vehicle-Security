import copy
import json
from pathlib import Path
import unittest
import run_evidence

class ReportTests(unittest.TestCase):
    def setUp(self):
        self.prior=json.loads((Path(__file__).resolve().parent/'input_evidence/run_report.json').read_text())

    def test_verified_full_attack_counts_preserved(self):
        r=run_evidence.build_report(self.prior,40)
        self.assertEqual(r['prior_attack_evidence']['scenes'],979)
        self.assertEqual(r['prior_attack_evidence']['verified_decisions'],19994163)
        self.assertEqual(r['prior_attack_evidence']['coordinated_accepted_disagreement_gt2'],486874)

    def test_bad_prior_verification_refused(self):
        for field,value in [('verification_passed',False),('verified_decisions',1),('input_lock_sha256','wrong')]:
            p=copy.deepcopy(self.prior);p['verification'][field]=value
            with self.assertRaises(ValueError):run_evidence.build_report(p,40)

    def test_authentication_never_fabricates_real_sensor_evidence(self):
        r=run_evidence.build_report(self.prior,40)
        self.assertFalse(r['project_promotion_allowed'])
        self.assertFalse(r['physical_witness_enabled'])
        self.assertTrue(all(value=='not_established' for value in r['actual_measurement_evidence'].values()))
        self.assertFalse(r['limits']['protects_against_compromised_signer'])
        self.assertFalse(r['limits']['protects_against_replay_database_snapshot_rollback'])

if __name__=='__main__':unittest.main()
