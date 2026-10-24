import unittest
import run_durable as r
class ReplayTests(unittest.TestCase):
    def test_complete_fixed_inventory(self):
        self.assertEqual(len(r.expected_keys()),5616)
        self.assertIn(('ingress','durable','p000','restart_replay'),r.expected_keys())
        self.assertIn(('capability','memory','p035','restart_token_reuse'),r.expected_keys())

    def test_verifier_rejects_missing_or_duplicate_rows(self):
        import verify_durable as v
        with self.assertRaises(ValueError): v.verify_rows([])

    def test_independent_count_includes_snapshot_grants(self):
        import verify_durable as v
        rows=[dict(phase='ingress',kind='attack',outcome='RESTRICT',contract_passed=True)]
        self.assertEqual(v.independent_counts(rows)['adverse_grants'],1)
if __name__=='__main__': unittest.main()
