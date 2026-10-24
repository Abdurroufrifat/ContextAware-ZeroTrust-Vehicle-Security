import copy,json,unittest
from pathlib import Path
import validate_audit

HERE=Path(__file__).resolve().parent

class AuditTests(unittest.TestCase):
    def report(self):return json.loads((HERE/'input_evidence/scene_full_report.json').read_text())
    def test_completed_full_audit_checks(self):
        r=validate_audit.validate(self.report());self.assertEqual(r['paired_tables_checked'],350)
        self.assertEqual(r['pooled_cells_checked'],490);self.assertEqual(r['nominal_zero_sd_scenes'],29)
        self.assertEqual(r['nominal_subnormal_sd_scenes'],2);self.assertFalse(r['raw_scene_flags_recomputed_here'])
    def test_false_completion_count_rejected(self):
        report=self.report();report['verification']['verified_paired_tables']=349
        with self.assertRaises(ValueError):validate_audit.validate(report)
    def test_false_physical_claim_rejected(self):
        report=self.report();report['analysis']['real_source_protection_established']=True
        with self.assertRaises(ValueError):validate_audit.validate(report)

if __name__=='__main__':unittest.main()
