import copy,gzip,json,shutil,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import scene_core as c
import run_scene_comparison as run
import verify_scene_comparison as independent

HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'vehicle_monitor_comparison'
TMP=HERE/'_temporary';TMP.mkdir(exist_ok=True)

def cell():
    return dict(requests=10,accepted=10,coverage_percent=100.,scored_accepted=10,accepted_unscorable=0,
                mae_reference_disagreement_mps=.1,p95_reference_disagreement_mps=.2,max_reference_disagreement_mps=.3,
                accepted_disagreement_gt2_count=0,trace_bound_violation_count=0,max_deviation_from_original_wheel_mps=0.,
                selection_counts=dict(abstain=0,candidate=10,protected_copy=0),diagnostic_gates_pass=True,
                monitor=dict(requests=10,monitored=8,alarms=0,warmup=2,invalid_calibration=0,numerical_failure=0,missing=0,unavailable=0,alarm_percent_of_monitored=0.))

class SummaryTests(unittest.TestCase):
    def test_supplied_reports_metadata_and_arithmetic(self):
        self.assertEqual(c.validate_report(run.io.read(HERE/'input_evidence/monitor_full_report.json'),True)['pooled_metric_cells_checked'],280)
        self.assertEqual(c.validate_report(run.io.read(HERE/'input_evidence/conditional_full_report.json'),False)['pooled_metric_cells_checked'],210)
    def test_false_completed_decision_count_rejected(self):
        report=run.io.read(HERE/'input_evidence/monitor_full_report.json');report['verification']['verified_decisions']-=1
        with self.assertRaises(ValueError):c.validate_report(report,True)
    def test_independent_gate_matches_all_supplied_cells(self):
        for file in ('monitor_full_report.json','conditional_full_report.json'):
            r=run.io.read(HERE/'input_evidence'/file)
            for case in r['summary']['case_metrics'].values():
                for policy in case.values():
                    for x in policy.values():self.assertEqual(c.gate(x),independent.passes(x))
    def test_valid_cell(self):self.assertTrue(c.validate_cell(cell(),True))
    def test_gate_boundary_inclusive(self):
        x=cell();x.update(requests=100,accepted=95,coverage_percent=95.,scored_accepted=95,p95_reference_disagreement_mps=.5,max_reference_disagreement_mps=.5)
        x['selection_counts'].update(abstain=5,candidate=95);x.pop('monitor')
        self.assertTrue(c.validate_cell(x,False))
    def test_false_pass_rejected(self):
        x=cell();x['coverage_percent']=94
        with self.assertRaises(ValueError):c.validate_cell(x,True)
    def test_counter_partition_rejected(self):
        x=cell();x['monitor']['numerical_failure']=1
        with self.assertRaises(ValueError):c.validate_cell(x,True)
    def test_infinite_metric_rejected(self):
        x=cell();x['p95_reference_disagreement_mps']=float('inf')
        with self.assertRaises(ValueError):c.validate_cell(x,True)
    def test_no_alarm_on_unmonitored_requests(self):
        x=cell();x['monitor']['alarms']=9
        with self.assertRaises(ValueError):c.validate_cell(x,True)
    def test_zero_denominator_is_retained_failure(self):
        x=cell()
        for k in ('requests','accepted','scored_accepted','accepted_disagreement_gt2_count'):x[k]=0
        for k in ('coverage_percent','mae_reference_disagreement_mps','p95_reference_disagreement_mps','max_reference_disagreement_mps','max_deviation_from_original_wheel_mps'):x[k]=None
        x['selection_counts']=dict(abstain=0,candidate=0,protected_copy=0);x['diagnostic_gates_pass']=False;x.pop('monitor')
        self.assertFalse(c.validate_cell(x,False))
    def test_gate_requires_zero_large_disagreements(self):
        x=cell();x['accepted_disagreement_gt2_count']=1;x['max_reference_disagreement_mps']=3.;x['diagnostic_gates_pass']=False
        self.assertFalse(c.validate_cell(x,True))
    def test_pair_outcomes_keep_both_fail(self):
        self.assertEqual([c.outcome(a,b) for a,b in [(True,True),(True,False),(False,True),(False,False)]],['both_pass','candidate_only','comparator_only','both_fail'])
    def test_all_scenes_stay_in_paired_denominator(self):
        p=c.Pairs()
        for a,b in [(True,True),(True,False),(False,True),(False,False)]:p.add('nominal',0,'all',a,b,0 if not a and not b else 1)
        r=p.report()['nominal'][0]['all'];self.assertEqual(r['scenes'],4);self.assertEqual(r['both_fail'],1)
        self.assertEqual(r['empty_request_scenes'],1);self.assertEqual(r['candidate_minus_comparator_percentage_points'],0.)
    def test_subnormal_sd_is_reported_without_floor(self):
        a=c.Calibration()
        a.add('nominal','unprotected',dict(n=5,mean=1e-320,sd=5e-321,valid=True))
        x=a.report()['nominal']['unprotected'];self.assertEqual(x['valid_scenes'],1);self.assertEqual(x['subnormal_sd_scenes'],1);self.assertEqual(x['min_positive_sd'],5e-321)


class PipelineTests(unittest.TestCase):
    def inputs(self):return HERE.parent/'vehicle_conditional_source'/'pilot_results',BASE/'pilot_numeric_results'
    def test_saved_results_audit_replay_and_rehashed_tamper(self):
        old,new=self.inputs()
        with tempfile.TemporaryDirectory(dir=TMP) as folder:
            output=Path(folder)/'analysis';result=run.run(old,new,output)
            self.assertTrue(result['verification_passed']);self.assertEqual(result['verified_policy_phase_flags'],490)
            self.assertEqual(result['verified_paired_tables'],350);self.assertEqual(run.run(old,new,output),result)
            name=output/'paired_scene_gates.jsonl.gz'
            with gzip.open(name,'rt') as stream:record=json.loads(stream.readline())
            record['gates'][0][0][0]=not record['gates'][0][0][0]
            with run.monitor_io.gzip_writer(name) as stream:stream.write(json.dumps(record)+'\n')
            hashes=run.io.read(output/'result_hashes.json');hashes[name.name]=run.io.digest(name);run.io.write(output/'result_hashes.json',hashes)
            with self.assertRaises(ValueError):independent.verify(old,new,output)
    def test_changed_source_hash_rejected(self):
        old,new=self.inputs()
        with tempfile.TemporaryDirectory(dir=TMP) as folder:
            copydir=Path(folder)/'results';shutil.copytree(new,copydir)
            with (copydir/'summary.json').open('a') as stream:stream.write(' ')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):run.sources(old,copydir)
    def test_duplicate_scene_not_dropped(self):
        old,new=self.inputs()
        with tempfile.TemporaryDirectory(dir=TMP) as folder:
            target=Path(folder)
            for source,name in [(old,'old'),(new,'new')]:
                (target/name).mkdir()
                with gzip.open(source/'scene_metrics.jsonl.gz','rt') as stream:line=stream.readline()
                with run.monitor_io.gzip_writer(target/name/'scene_metrics.jsonl.gz') as stream:stream.write(line+line)
            with self.assertRaisesRegex(ValueError,'duplicated'):list(run.scene_records(target/'old',target/'new'))
    def test_shorter_scene_ledger_rejected(self):
        old,new=self.inputs()
        with tempfile.TemporaryDirectory(dir=TMP) as folder:
            target=Path(folder);(target/'old').mkdir();(target/'new').mkdir()
            with gzip.open(old/'scene_metrics.jsonl.gz','rt') as stream:line=stream.readline()
            with run.monitor_io.gzip_writer(target/'old'/'scene_metrics.jsonl.gz') as stream:stream.write(line)
            with run.monitor_io.gzip_writer(target/'new'/'scene_metrics.jsonl.gz') as stream:pass
            with self.assertRaisesRegex(ValueError,'lengths differ'):list(run.scene_records(target/'old',target/'new'))
    def test_reserved_paths_block_before_access(self):
        with patch.object(Path,'open',side_effect=AssertionError('Payload accessed')):
            for reserved in ('Urban02_NIGHT','RAV4','vehicle_final_evaluation'):
                with self.assertRaises(ValueError):run.run(Path(reserved),Path('monitor'),Path('analysis'))
    def test_output_overlap_rejected(self):
        old,new=self.inputs()
        with self.assertRaisesRegex(ValueError,'overlaps'):run.run(old,new,new/'analysis')

if __name__=='__main__':unittest.main()
