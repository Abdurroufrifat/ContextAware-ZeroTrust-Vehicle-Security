import gzip,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import benchmark_core as b
import run_benchmark as run
import verify_benchmark as independent

TMP=Path(__file__).resolve().parent/'_temporary';TMP.mkdir(exist_ok=True)

class BenchmarkTests(unittest.TestCase):
    def exercise(self,case,policy,n=16):
        with tempfile.TemporaryDirectory(dir=TMP) as folder:return list(b.trial(case,policy,Path(folder),n))
    def test_nominal_gateway_outputs_known_bounded_candidate(self):
        events=self.exercise('nominal','integrated_gateway')
        self.assertEqual(len(events),16)
        for e in events:self.assertEqual(e['output_mmps'],e['reference_mmps']+200)
    def test_candidate_bias_falls_back_to_anchor(self):
        e=self.exercise('candidate_bias','integrated_gateway')
        self.assertEqual(e[4]['selection'],'anchor');self.assertEqual(e[4]['output_mmps'],5080)
    def test_candidate_tamper_controls_differ(self):
        self.assertEqual(self.exercise('candidate_tamper','unsigned_candidate_control')[4]['output_mmps'],8280)
        self.assertIsNone(self.exercise('candidate_tamper','authenticated_candidate_control')[4]['output_mmps'])
        self.assertEqual(self.exercise('candidate_tamper','integrated_gateway')[4]['selection'],'anchor')
    def test_anchor_tamper_fails_closed(self):self.assertIsNone(self.exercise('anchor_tamper','integrated_gateway')[4]['output_mmps'])
    def test_candidate_replay_falls_back(self):self.assertEqual(self.exercise('candidate_replay','integrated_gateway')[4]['selection'],'anchor')
    def test_anchor_replay_abstains(self):self.assertIsNone(self.exercise('anchor_replay','integrated_gateway')[4]['output_mmps'])
    def test_context_mismatch_not_accepted(self):
        self.assertEqual(self.exercise('candidate_context','integrated_gateway')[4]['selection'],'anchor')
        self.assertIsNone(self.exercise('anchor_context','integrated_gateway')[4]['output_mmps'])
    def test_signed_compromise_failure_preserved(self):
        e=self.exercise('signed_coordinated_bias','integrated_gateway')[4]
        self.assertEqual(e['output_mmps']-e['reference_mmps'],3200)
    def test_missing_evidence_not_auto_approved(self):
        for case in ('anchor_assumption_missing','anchor_key_revoked'):
            self.assertTrue(all(e['output_mmps'] is None for e in self.exercise(case,'integrated_gateway')))
    def test_restricted_context_and_resume(self):
        for case in ('risk_lockdown','unsupported_operation'):
            e=self.exercise(case,'integrated_gateway');self.assertIsNone(e[4]['output_mmps']);self.assertIsNotNone(e[12]['output_mmps'])
    def test_elevated_context_anchor_only(self):self.assertEqual(self.exercise('risk_elevated','integrated_gateway')[4]['selection'],'anchor')
    def test_stale_anchor_abstains(self):self.assertIsNone(self.exercise('anchor_measurement_stale','integrated_gateway')[4]['output_mmps'])
    def test_clock_rollback_abstains_and_recovers(self):
        e=self.exercise('receiver_clock_rollback','integrated_gateway');self.assertIsNone(e[4]['output_mmps']);self.assertIsNotNone(e[12]['output_mmps'])
    def test_database_rollback_counterexample_preserved(self):
        e=self.exercise('database_snapshot_rollback','integrated_gateway')
        self.assertIsNotNone(e[4]['output_mmps']);self.assertTrue(e[4]['replayed_evidence_used'])
        self.assertIsNone(e[5]['output_mmps'])
    def test_latency_is_nonnegative_and_physical_flags_false(self):
        for e in self.exercise('nominal','integrated_gateway'):
            self.assertIs(type(e['latency_ns']),int);self.assertGreaterEqual(e['latency_ns'],0)
            self.assertFalse(e['physical_speed_authorized'])
    def test_metrics_keep_abstention_denominator(self):
        m=b.Metrics()
        for e in self.exercise('anchor_missing','integrated_gateway'):m.add(e)
        r=m.report()['all'];self.assertEqual(r['requests'],16);self.assertEqual(r['accepted'],8)
        self.assertFalse(r['diagnostic_gates_pass']);self.assertEqual(r['latency_samples'],16)
    def test_all_scenarios_match_independent_oracle(self):
        for case in b.CASES:
            for policy in b.POLICIES:
                for event in self.exercise(case,policy):
                    value,selection=independent.expected(case,policy,event['index'],16)
                    self.assertEqual(event['output_mmps'],value,(case,policy,event['index']))
                    self.assertEqual(event['selection'],selection)
    def test_separate_metric_reduction_matches(self):
        for case in ('nominal','candidate_bias','anchor_missing','signed_coordinated_bias','database_snapshot_rollback'):
            rows=self.exercise(case,'integrated_gateway');m=b.Metrics()
            for row in rows:m.add(row)
            expected={g:independent.reduce(rows if g=='all' else [e for e in rows if e['phase']==g]) for g in b.GROUPS}
            independent.compare(m.report(),expected,'independent_metrics')
    def test_unsupported_fixture_inventory_rejected(self):
        with tempfile.TemporaryDirectory(dir=TMP) as folder:
            with self.assertRaises(ValueError):list(b.trial('unknown','integrated_gateway',Path(folder)))
    def test_reserved_paths_refused_before_payload_access(self):
        with patch.object(Path,'open',side_effect=AssertionError('Payload accessed')):
            for name in ('Urban02_NIGHT','RAV4','vehicle_final_evaluation'):
                with self.assertRaises(ValueError):run.run(Path(name))


class PipelineTests(unittest.TestCase):
    def test_completed_benchmark_reverification_and_rehashed_tamper(self):
        # Keep the runner's own scratch-state directory protected. This isolated
        # output stays elsewhere inside the project's research directory.
        with tempfile.TemporaryDirectory(dir=run.HERE.parent) as folder:
            output=Path(folder)/'results';checked=run.run(output)
            self.assertEqual(checked['verified_decisions'],6912);self.assertEqual(checked['verified_policy_phase_cells'],216)
            self.assertEqual(run.run(output),checked)
            name=output/'decisions.jsonl.gz'
            rows=[json.loads(line) for line in gzip.decompress(name.read_bytes()).decode().splitlines()]
            rows[0]['output_mmps']+=1
            run.atomic(name,gzip.compress(('\n'.join(json.dumps(r) for r in rows)+'\n').encode(),mtime=0))
            hashes=run.read(output/'result_hashes.json');hashes[name.name]=run.digest(name);run.write(output/'result_hashes.json',hashes)
            with self.assertRaises(ValueError):independent.verify(output)
    def test_source_directory_output_refused(self):
        with self.assertRaisesRegex(ValueError,'overlaps'):run.run(run.HERE)

if __name__=='__main__':unittest.main()
