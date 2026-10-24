"""Mutation tests for explanation fidelity; removing checks must break these tests."""
import copy
import gzip
import json
from pathlib import Path
import shutil
import tempfile
import unittest
try:
    import audit_explanations as a
except ModuleNotFoundError:
    a = None

HERE = Path(__file__).resolve().parent


class ExplanationAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with gzip.open(HERE/'lab_benchmark/decisions.jsonl.gz', 'rt') as f:
            cls.rows = [json.loads(line) for line in f]

    def analyze_change(self, field, value):
        rows = copy.deepcopy(self.rows)
        rows[0][field] = value
        return a.analyze(rows, 128)

    def test_all_saved_decisions_have_matching_explanations(self):
        self.assertIsNotNone(a, 'Explanation auditor is not implemented')
        report, details = a.analyze(self.rows, 128)
        self.assertEqual(report['decisions'], 6912)
        self.assertEqual(report['mismatched_decisions'], 0)
        self.assertEqual(len(details), 6912)
        self.assertTrue(report['explanation_validation_passed'])

    def test_known_reason_replacement_is_caught(self):
        report, details = self.analyze_change('reason', 'candidate_outside_budget_fallback')
        self.assertEqual(report['mismatched_decisions'], 1)
        self.assertFalse(details[0]['explanation_valid'])
        self.assertIn('reason', details[0]['mismatched_fields'])
        self.assertFalse(report['explanation_validation_passed'])

    def test_forged_authentication_status_is_caught(self):
        report, details = self.analyze_change('candidate_status', 'authenticated')
        self.assertEqual(report['mismatched_decisions'], 1)
        self.assertIn('candidate_status', details[0]['mismatched_fields'])

    def test_physical_authority_escalation_is_caught(self):
        report, details = self.analyze_change('physical_speed_authorized', True)
        self.assertEqual(report['mismatched_decisions'], 1)
        self.assertFalse(details[0]['physical_speed_authorized'])

    def test_wrong_selection_and_value_are_caught(self):
        rows = copy.deepcopy(self.rows)
        rows[0].update(selection='anchor', output_mmps=999999)
        report, details = a.analyze(rows, 128)
        self.assertEqual(report['mismatched_decisions'], 1)
        self.assertIn('output_mmps', details[0]['mismatched_fields'])
        self.assertIn('selection', details[0]['mismatched_fields'])

    def test_injected_free_text_is_not_rendered(self):
        report, details = self.analyze_change('reason', '<script>trusted physical truth</script>')
        self.assertEqual(report['mismatched_decisions'], 1)
        self.assertNotIn('<script>', details[0]['statement'])

    def test_signer_compromise_is_not_called_a_detected_attack(self):
        report, details = a.analyze(self.rows, 128)
        cell = report['preserved_counterexamples']['signed_coordinated_bias_gateway']
        self.assertEqual(cell['accepted_synthetic_error_gt2_count'], 64)
        self.assertEqual(cell['mismatched_explanations'], 0)
        row = next(x for x in details if x['case']=='signed_coordinated_bias' and x['policy']=='integrated_gateway' and x['index']==32)
        self.assertIn('does not establish measurement truth', row['evidence_limits'])
        self.assertFalse(row['measurement_truth_established'])

    def test_state_rollback_failure_remains_visible(self):
        report, _ = a.analyze(self.rows, 128)
        self.assertEqual(report['preserved_counterexamples']['database_snapshot_rollback_gateway']['replayed_evidence_admissions'], 1)

    def test_replay_reason_changes_at_inclusive_freshness_boundary(self):
        for policy in ('integrated_gateway','authenticated_candidate_control'):
            self.assertEqual(a.expected_trace('candidate_replay',policy,36,128)[1],'replay_or_reorder')
            self.assertEqual(a.expected_trace('candidate_replay',policy,37,128)[1],'measurement_stale')
        self.assertEqual(a.expected_trace('anchor_replay','integrated_gateway',36,128)[0],'anchor_replay_or_reorder')
        self.assertEqual(a.expected_trace('anchor_replay','integrated_gateway',37,128)[0],'anchor_measurement_stale')

    def test_rehashed_reason_attack_passes_old_scalar_check_but_fails_new_audit(self):
        with tempfile.TemporaryDirectory(dir=HERE.parent) as tmp:
            source=Path(tmp)/'source';shutil.copytree(HERE/'lab_benchmark',source)
            rows=copy.deepcopy(self.rows);rows[0]['reason']='research_candidate_within_budget'
            raw=''.join(json.dumps(row)+'\n' for row in rows).encode()
            (source/'decisions.jsonl.gz').write_bytes(gzip.compress(raw,mtime=0))
            hashes=json.loads((source/'result_hashes.json').read_text())
            hashes['decisions.jsonl.gz']=a.base.digest(source/'decisions.jsonl.gz')
            (source/'result_hashes.json').write_text(json.dumps(hashes))
            self.assertTrue(a.scalar.verify(source)['verification_passed'])
            report=a.audit(source,Path(tmp)/'result')
            self.assertEqual(report['summary']['mismatched_decisions'],1)
            self.assertFalse(report['summary']['explanation_validation_passed'])

    def test_paired_comparisons_keep_all_active_requests(self):
        report, _ = a.analyze(self.rows, 128)
        self.assertEqual(len(report['paired_scenario_responses']), 51)
        for cell in report['paired_scenario_responses']:
            self.assertEqual(cell['paired_active_requests'], 64)

    def test_complete_denominators_include_abstentions(self):
        report, details = a.analyze(self.rows, 128)
        self.assertEqual(sum(c['decisions'] for c in report['by_policy'].values()),6912)
        self.assertEqual(sum(c['decisions'] for c in report['by_reason'].values()),6912)
        self.assertTrue(any(d['selection']=='abstain' for d in details))

    def test_truncation_and_extra_rows_are_rejected(self):
        for rows in (self.rows[:-1], self.rows+[self.rows[-1]]):
            with self.assertRaises(ValueError): a.analyze(rows,128)

    def test_reordered_rows_are_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows[0],rows[1] = rows[1],rows[0]
        with self.assertRaises(ValueError): a.analyze(rows,128)

    def test_bool_index_and_output_are_rejected(self):
        for field in ('index','output_mmps'):
            rows=copy.deepcopy(self.rows);rows[0][field]=True
            with self.assertRaises(ValueError): a.analyze(rows,128)

    def test_missing_reason_is_rejected(self):
        rows=copy.deepcopy(self.rows);del rows[0]['reason']
        with self.assertRaises(ValueError): a.analyze(rows,128)

    def test_reserved_path_rejected_before_open(self):
        with self.assertRaises(ValueError): a.audit(Path('Urban02/absent'), Path('unused'))

    def test_output_cannot_overwrite_benchmark_or_code(self):
        for path in (HERE,HERE/'lab_benchmark',HERE.parent):
            with self.assertRaises(ValueError): a.audit(HERE/'lab_benchmark',path)

    def test_pipeline_verifies_saved_results_and_refuses_changed_output(self):
        with tempfile.TemporaryDirectory(dir=HERE.parent) as tmp:
            out=Path(tmp)/'audit'
            report=a.audit(HERE/'lab_benchmark',out)
            self.assertTrue(report['summary']['explanation_validation_passed'])
            before=(out/'explanations.jsonl.gz').read_bytes()
            a.audit(HERE/'lab_benchmark',out)
            self.assertEqual(before,(out/'explanations.jsonl.gz').read_bytes())
            summary=json.loads((out/'summary.json').read_text())
            summary['mismatched_decisions']=1
            (out/'summary.json').write_text(json.dumps(summary))
            with self.assertRaises(ValueError): a.audit(HERE/'lab_benchmark',out)


if a is None:
    for name in dir(ExplanationAuditTests):
        if name.startswith('test_') and name!='test_all_saved_decisions_have_matching_explanations':
            setattr(ExplanationAuditTests,name,unittest.skip('auditor not yet implemented')(getattr(ExplanationAuditTests,name)))

if __name__=='__main__': unittest.main()
