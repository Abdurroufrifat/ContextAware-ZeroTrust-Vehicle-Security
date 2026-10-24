import copy,gzip,json,math,tempfile,unittest
from pathlib import Path
import conditional_core as cc
import run_conditional as run
import verify_conditional as verify

TMP=Path(__file__).resolve().parent/'_temporary';TMP.mkdir(exist_ok=True)

def row(i,t,w=5,p=5):
    return dict(pose_index=i,pose_utime=t,wheel_utime=t,wheel_index=i,wheel_speed_mps=w,
                pose_speed_mps=p,receipt_age_us=0,status='valid',signed_disagreement_mps=w-p)

class SelectionTests(unittest.TestCase):
    def test_uses_candidate_at_inclusive_error_budget(self):
        self.assertEqual(cc.choose(5.25,0,5,.25,0,0,True),(5.25,1))
    def test_falls_back_outside_error_budget(self):
        self.assertEqual(cc.choose(5.250001,0,5,.25,0,0,True),(5,2))
    def test_age_boundary(self):
        self.assertEqual(cc.choose(5,0,5,.25,0,100000,True),(5,1))
        self.assertEqual(cc.choose(5,0,5,.25,0,100001,True),(None,0))
    def test_bad_candidate_falls_back(self):
        for c in [None,True,'5',float('nan')]:
            self.assertEqual(cc.choose(c,0,5,.25,0,0,True),(5,2))
    def test_stale_candidate_falls_back(self):
        self.assertEqual(cc.choose(5,0,5,.25,100001,100001,True),(5,2))
    def test_future_candidate_falls_back(self):
        self.assertEqual(cc.choose(5,1,5,.25,0,0,True),(5,2))
    def test_unknown_protection_abstains(self):
        self.assertEqual(cc.choose(5,0,5,.25,0,0,False),(None,0))
    def test_missing_or_invalid_protected_value_abstains(self):
        for v in [None,True,'5',float('inf')]:
            self.assertEqual(cc.choose(5,0,v,.25,0,0,True),(None,0))
    def test_uncertainty_boundary_and_invalid(self):
        self.assertEqual(cc.choose(5,0,5,.5,0,0,True),(5,1))
        for u in [-.1,.500001,True,None,float('nan')]:
            self.assertEqual(cc.choose(5,0,5,u,0,0,True),(None,0))
    def test_future_protected_time_abstains(self):
        self.assertEqual(cc.choose(5,0,5,.25,1,0,True),(None,0))
    def test_reference_not_a_selection_argument(self):
        a=[row(0,0),row(1,5000000),row(2,6000000),row(3,15000000)]
        b=copy.deepcopy(a)
        for r in b:r['pose_speed_mps']=99
        # Protected-copy values come only from the original wheel trace.
        ca=cc.events(a,'coordinated_bias_plus3');cb=cc.events(b,'coordinated_bias_plus3')
        self.assertEqual([e[1] for e in ca],[e[1] for e in cb])
        self.assertEqual([e[5][1:] for e in ca],[e[5][1:] for e in cb])
    def test_coordinated_candidate_bias_uses_protected_copy(self):
        e=cc.events([row(0,0),row(1,5000000)],'coordinated_bias_plus3')[1]
        self.assertEqual(e[5][0],8)
        self.assertAlmostEqual(e[5][1],5.25)
        self.assertAlmostEqual(e[5][2],5.25)
    def test_compromised_copy_is_not_oracle_rejected(self):
        e=cc.events([row(0,0),row(1,5000000)],'protected_copy_compromised')[1]
        self.assertTrue(e[7]);self.assertGreater(e[5][2],7)
    def test_missing_and_stale_copy_abstain_during_attack(self):
        rows=[row(0,0),row(1,5000000),row(2,15000000)]
        for case in ['protected_copy_missing','protected_copy_stale']:
            e=cc.events(rows,case)
            self.assertEqual(e[1][5][1:],[None,None]);self.assertEqual(e[2][5][1:],[5,5])
    def test_unestablished_real_evidence_abstains_everywhere(self):
        e=cc.events([row(0,0),row(1,5000000)],'unestablished_real_evidence')
        self.assertTrue(all(x[5][1:]==[None,None] for x in e))
    def test_bound_violation_is_retained(self):
        e=cc.events([row(0,0),row(1,5000000)],'protected_bound_violated')[1]
        self.assertTrue(e[7]);self.assertIsNotNone(e[5][2])
    def test_high_uncertainty_abstains(self):
        e=cc.events([row(0,0),row(1,5000000)],'coordinated_u075')
        self.assertTrue(all(x[5][1:]==[None,None] for x in e))
    def test_no_hidden_raw_modification(self):
        rows=[row(0,0),row(1,5000000)];original=copy.deepcopy(rows)
        for case in cc.CASES:cc.events(rows,case)
        self.assertEqual(rows,original)
    def test_independent_events_match(self):
        rows=[row(0,0),row(1,4999999),row(2,5000000),row(3,5100001),row(4,14999999),row(5,15000000)]
        for case in cc.CASES:
            run.baseverify.equal(cc.events(rows,case),verify.reconstruct(rows,case),case)
    def test_trace_bound_under_all_certified_inputs(self):
        # Exhaust finite adverse values; selection must stay within0.5 of latent clean trace.
        for clean in [0,5,20]:
            for u in [0,.1,.25,.5]:
                for shift in [0,u]:
                    center=clean+shift
                    for candidate in [0,clean,clean+.5,clean+3,100]:
                        value,code=cc.choose(candidate,0,center,u,0,0,True)
                        self.assertLessEqual(abs(value-clean),.5+1e-12)
    def test_metrics_keep_abstention_and_proxy_reference(self):
        m=cc.Metrics();m.add([8,None,5],[1,0,2],5,5,'during')
        r=m.report();self.assertEqual(r['legacy_agreement']['all']['accepted_disagreement_gt2_count'],1)
        self.assertEqual(r['conditional_anchor_only']['all']['requests'],1)
        self.assertFalse(r['conditional_anchor_only']['all']['diagnostic_gates_pass'])
        self.assertTrue(r['conditional_bounded_candidate']['all']['diagnostic_gates_pass'])
    def test_numeric_protection_flag_does_not_grant_eligibility(self):
        self.assertEqual(cc.choose(5,0,5,.25,0,0,1),(None,0))
    def test_historical_comparison_detects_changed_old_result(self):
        prior=run.baseio.read(run.HERE/'previous_run_report.json')['summary']
        metrics={case:{'legacy_agreement':copy.deepcopy(prior['scenario_metrics'][case]['age_and_agreement'])} for case in run.attacks.SCENARIOS}
        summary=dict(baseline_input_snapshot_sha256=prior['baseline_input_snapshot_sha256'],case_metrics=metrics)
        self.assertEqual(run.historical_check(summary)['matched_metric_cells'],350)
        metrics['coordinated_bias_plus3']['legacy_agreement']['all']['accepted_disagreement_gt2_count']=0
        with self.assertRaises(ValueError):run.historical_check(summary)
    def test_unknown_case_rejected(self):
        with self.assertRaises(ValueError):cc.events([row(0,0)],'tuned_case')

class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=TMP);self.root=Path(self.tmp.name)
        self.data=self.root/'data';self.data.mkdir();self.baseline=self.root/'baseline';self.output=self.root/'outputs'
        stamps=[0,4000000,5000000,5100000,14000000,15000000]
        poses=[{'utime':t,'vel':[5,0,0]} for t in stamps]
        wheels=[dict(utime=t,**{k:5/(2*math.pi*.305/60) for k in run.baseio.core.WHEELS}) for t in stamps]
        (self.data/'scene-0001_pose.json').write_text(json.dumps(poses))
        (self.data/'scene-0001_zoe_veh_info.json').write_text(json.dumps(wheels))
        run.baseio.run(self.data,self.baseline)
    def tearDown(self):self.tmp.cleanup()
    def test_full_run_independent_verify_and_repeat(self):
        run.run(self.data,self.baseline,self.output)
        v=verify.verify(self.data,self.baseline,self.output)
        self.assertEqual(v['verified_event_rows'],6*len(cc.CASES))
        self.assertEqual(v['verified_decisions'],6*len(cc.CASES)*3)
        before=(self.output/'run_report.json').read_bytes()
        run.run(self.data,self.baseline,self.output)
        self.assertEqual(before,(self.output/'run_report.json').read_bytes())
    def test_changed_raw_inputs_rejected(self):
        run.run(self.data,self.baseline,self.output)
        (self.data/'scene-0001_pose.json').write_text('[]')
        with self.assertRaises(ValueError):run.run(self.data,self.baseline,self.output)
    def test_forged_summary_with_updated_hash_rejected(self):
        run.run(self.data,self.baseline,self.output)
        p=self.output/'summary.json';s=run.baseio.read(p)
        s['case_metrics']['protected_copy_compromised']['conditional_bounded_candidate']['during']['accepted_disagreement_gt2_count']=0
        run.baseio.write(p,s)
        hashes=run.baseio.read(self.output/'result_hashes.json');hashes[p.name]=run.baseio.digest(p)
        run.baseio.write(self.output/'result_hashes.json',hashes)
        with self.assertRaises(ValueError):verify.verify(self.data,self.baseline,self.output)
    def test_missing_stream_denominator_retained(self):
        (self.data/'scene-0002_pose.json').write_text(json.dumps([{'utime':0,'vel':[5,0,0]}]))
        other=self.root/'baseline2';run.baseio.run(self.data,other)
        run.run(self.data,other,self.output)
        s=run.baseio.read(self.output/'summary.json')
        self.assertEqual(s['case_metrics']['nominal']['conditional_bounded_candidate']['all']['requests'],7)
        self.assertEqual(s['rows_without_query_time'],1)
    def test_forged_event_with_updated_hash_rejected(self):
        run.run(self.data,self.baseline,self.output)
        p=self.output/'scene-0001_events.jsonl.gz'
        with gzip.open(p,'rt',encoding='utf-8') as f:values=[json.loads(line) for line in f]
        values[0][6][2]=8
        with gzip.open(p,'wt',encoding='utf-8') as f:
            for value in values:f.write(json.dumps(value)+'\n')
        hashes=run.baseio.read(self.output/'result_hashes.json');hashes[p.name]=run.baseio.digest(p)
        run.baseio.write(self.output/'result_hashes.json',hashes)
        with self.assertRaises(ValueError):verify.verify(self.data,self.baseline,self.output)
    def test_output_overlap_rejected(self):
        with self.assertRaises(ValueError):run.run(self.data,self.baseline,self.baseline)

if __name__=='__main__':unittest.main()
