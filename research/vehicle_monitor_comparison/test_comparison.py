import copy, gzip, json, math, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
import monitor_core as m
import run_comparison as run
import verify_comparison as verify

TMP=Path(__file__).resolve().parent/'_temporary';TMP.mkdir(exist_ok=True)

def rows():
    return [dict(pose_index=i,pose_utime=t,wheel_utime=t,wheel_index=i,
                 wheel_speed_mps=5.+(i%3)*.1,pose_speed_mps=5.,
                 receipt_age_us=0,status='valid')
            for i,t in enumerate([0,1000000,2000000,3000000,4000000,5000000,6000000,15000000,16000000])]

class MonitorTests(unittest.TestCase):
    def test_tiny_variance_overflow_retained_without_infinite_json(self):
        a=rows()
        for i,r in enumerate(a):
            r['pose_speed_mps']=0.
            r['wheel_speed_mps']=(1+i%2)*1e-320 if i<5 else 5.
        e,cal=m.events(a,'nominal')
        self.assertTrue(cal['unprotected']['valid'])
        self.assertGreater(cal['unprotected']['sd'],0)
        self.assertEqual(e[5][2][0],'numerical_failure')
        self.assertEqual(e[5][5][:2],[None,None])
        self.assertEqual(e[5][4][:2],[None,None])
        json.dumps([e,cal],allow_nan=False)
        b,bc=verify.reconstruct(a,'nominal')
        run.equal(e,b,'tiny.events');run.equal(cal,bc,'tiny.calibration')
    def test_numeric_failure_stays_closed_until_scene_end(self):
        a=rows()
        for i,r in enumerate(a):
            r['pose_speed_mps']=0.;r['wheel_speed_mps']=(1+i%2)*1e-320 if i<5 or i>6 else 5.
        e,_=m.events(a,'nominal')
        self.assertEqual(e[-1][2][0],'numerical_failure')
        self.assertEqual(e[-1][5][:2],[None,None])
    def test_detector_rejects_nonfinite_input_without_state_mutation(self):
        for detector in (m.Cusum(),m.Ewma()):
            before=detector.state()
            for value in (float('inf'),float('-inf'),float('nan')):
                with self.assertRaises(ArithmeticError):detector.step(value)
                self.assertEqual(detector.state(),before)
    def test_cusum_finite_accumulation_overflow_preserves_state(self):
        d=m.Cusum();d.step(1e308);before=d.state()
        with self.assertRaises(ArithmeticError):d.step(1e308)
        self.assertEqual(d.state(),before)
    def test_independent_calibration_handles_subnormal_variance(self):
        a=[1e-320,2e-320,1e-320,2e-320]
        self.assertTrue(verify.calibration(a)['valid'])
        self.assertGreater(verify.calibration(a)['sd'],0)
    def test_tiny_finite_updates_match_separate_verifier(self):
        a=rows()
        for i,r in enumerate(a):r['pose_speed_mps']=0.;r['wheel_speed_mps']=(1+i%2)*1e-320
        for case in m.CASES:
            e,c=m.events(a,case);other,fit=verify.reconstruct(a,case)
            run.equal(e,other,'subnormal.events');run.equal(c,fit,'subnormal.fit')
    def test_protected_numerical_failure_uses_only_eligible_fallback(self):
        with patch.object(m,'fit',return_value=dict(n=5,mean=0.,sd=1e-320,valid=True)):
            e,_=m.events(rows(),'nominal')
        self.assertEqual(e[5][2][1],'numerical_failure')
        self.assertEqual(e[5][6][2:],[2,2])
        self.assertEqual(e[5][4][2:],[None,None])
    def test_numeric_failure_keeps_metric_denominators(self):
        a=rows()
        for i,r in enumerate(a):r['pose_speed_mps']=0.;r['wheel_speed_mps']=(1+i%2)*1e-320 if i<5 else 5.
        events,_=m.events(a,'nominal');main=m.Metrics();independent=verify.Metrics()
        for e,r in zip(events,a):
            for obj in (main,independent):obj.add(e,r['pose_speed_mps'],r['wheel_speed_mps'],run.previous.attacks.phase(r['pose_utime'],0))
        result=main.report()
        self.assertEqual(result['cusum_unprotected']['all']['requests'],len(a))
        self.assertEqual(result['cusum_unprotected']['all']['monitor']['numerical_failure'],4)
        run.equal(result,independent.report(),'numeric.metrics')
    def test_nist_cusum_first_alarm(self):
        values=[324.925,324.675,324.725,324.350,325.350,325.225,324.125,324.525,325.225,324.600,324.625,325.150,328.325,327.250,327.825,328.500,326.675,327.775,326.875,328.350]
        detector=m.Cusum(k=.3175,h=4.1959)
        alarms=[detector.step(x-325)[1] for x in values]
        self.assertEqual(alarms.index(True)+1,14)
    def test_cusum_strict_threshold(self):
        d=m.Cusum();self.assertFalse(d.step(5.5)[1]);self.assertTrue(d.step(.5001)[1])
    def test_cusum_negative_shift(self):
        self.assertTrue(m.Cusum().step(-6)[1])
    def test_cusum_continues_after_alarm(self):
        d=m.Cusum();d.step(6);self.assertEqual(d.step(0)[0],[5.,0.])
    def test_nist_ewma_sequence(self):
        d=m.Ewma();state,alarm=d.step((52-50)/2.0539)
        self.assertAlmostEqual(50+state*2.0539,50.6);self.assertFalse(alarm)
        state,_=d.step((47-50)/2.0539)
        self.assertAlmostEqual(50+state*2.0539,49.52)
    def test_ewma_limits_use_square_root(self):
        self.assertAlmostEqual(m.Ewma().limit,3*math.sqrt(.3/1.7))
        self.assertAlmostEqual(50+m.Ewma().limit*2.0539,52.5884,places=4)
    def test_fit_sample_variance(self):
        p=m.fit([1,2,3]);self.assertEqual(p,dict(n=3,mean=2.,sd=1.,valid=True))
    def test_zero_variance_no_invented_scale(self):
        self.assertFalse(m.fit([0,0])['valid'])
    def test_short_calibration_invalid(self):
        self.assertFalse(m.fit([])['valid']);self.assertFalse(m.fit([1])['valid'])
    def test_exact_transition(self):
        e,cal=m.events(rows(),'wheel_bias_plus3')
        self.assertEqual(e[4][2],['warmup','warmup'])
        self.assertEqual(e[5][2],['monitored','monitored'])
        self.assertEqual(cal['unprotected']['n'],5)
    def test_future_attack_does_not_change_earlier_decisions(self):
        a=rows();b=copy.deepcopy(a)
        for r in b[5:]:r['wheel_speed_mps']=999.
        self.assertEqual(m.events(a,'nominal')[0][:5],m.events(b,'nominal')[0][:5])
    def test_fit_excludes_attack_observations(self):
        self.assertEqual(m.events(rows(),'nominal')[1],m.events(rows(),'wheel_bias_plus3')[1])
    def test_p_track_does_not_use_pose(self):
        a=rows();b=copy.deepcopy(a)
        for r in b:r['pose_speed_mps']=900.
        x,_=m.events(a,'wheel_bias_plus3');y,_=m.events(b,'wheel_bias_plus3')
        self.assertEqual([e[5][2:] for e in x],[e[5][2:] for e in y])
    def test_coordinated_shift_invisible_to_u_residual(self):
        a,_=m.events(rows(),'nominal');b,_=m.events(rows(),'coordinated_bias_plus3')
        for x,y in zip(a,b):self.assertAlmostEqual(x[1][0],y[1][0],places=12)
    def test_missing_protected_source_abstains(self):
        e,_=m.events(rows(),'protected_copy_missing')
        self.assertEqual(e[5][5][2:],[None,None])
    def test_stale_protected_source_abstains(self):
        e,_=m.events(rows(),'protected_copy_stale')
        self.assertEqual(e[5][5][2:],[None,None])
    def test_high_uncertainty_abstains(self):
        e,_=m.events(rows(),'coordinated_u075')
        self.assertTrue(all(x[5][2:]==[None,None] for x in e))
    def test_real_evidence_not_established(self):
        e,_=m.events(rows(),'unestablished_real_evidence')
        self.assertTrue(all(x[5][2:]==[None,None] for x in e))
    def test_zero_variance_p_uses_anchor_with_explicit_status(self):
        e,c=m.events(rows(),'coordinated_u0')
        self.assertFalse(c['protected']['valid']);self.assertEqual(e[5][2][1],'invalid_calibration')
        self.assertEqual(e[5][6][2:],[2,2])
    def test_compromised_copy_not_oracle_rejected(self):
        e,_=m.events(rows(),'protected_copy_compromised')
        self.assertTrue(all(v>7 for v in e[5][5][2:]))
    def test_missing_query_retains_row(self):
        a=rows();a[-1]['pose_utime']=None
        e,_=m.events(a,'nominal');self.assertEqual(len(e),len(a));self.assertEqual(e[-1][5],[None]*4)
    def test_missing_candidate_pauses_state(self):
        a=rows();a[6]['wheel_speed_mps']=None
        e,_=m.events(a,'nominal');self.assertEqual(e[6][2],['missing','missing'])
        self.assertEqual(e[6][3],e[5][3])
    def test_no_raw_mutation(self):
        a=rows();b=copy.deepcopy(a)
        for case in m.CASES:m.events(a,case)
        self.assertEqual(a,b)
    def test_independent_reconstruction_all_cases(self):
        for case in m.CASES:
            a,ac=m.events(rows(),case);b,bc=verify.reconstruct(rows(),case)
            run.equal(a,b,case+'.events');run.equal(ac,bc,case+'.calibration')
    def test_independent_reduction_all_cases(self):
        for case in m.CASES:
            events,_=m.events(rows(),case);a=m.Metrics();b=verify.Metrics()
            for e,r in zip(events,rows()):
                phase=run.previous.attacks.phase(r['pose_utime'],0)
                for obj in (a,b):obj.add(e,r['pose_speed_mps'],r['wheel_speed_mps'],phase)
            run.equal(a.report(),b.report(),case)
    def test_all_phase_denominators_retained(self):
        a=m.Metrics()
        for e,r in zip(m.events(rows(),'nominal')[0],rows()):
            a.add(e,r['pose_speed_mps'],r['wheel_speed_mps'],run.previous.attacks.phase(r['pose_utime'],0))
        for result in a.report().values():
            self.assertEqual(result['all']['requests'],9)
            self.assertEqual(sum(result[g]['requests'] for g in m.GROUPS[1:]),9)
            self.assertEqual(sum(result['all']['selection_counts'].values()),9)
    def test_freshness_inclusive(self):
        for delta,expected in [(100000,'monitored'),(100001,'missing')]:
            a=rows();a[6]['wheel_utime']-=delta
            self.assertEqual(m.events(a,'nominal')[0][6][2][0],expected)
    def test_ineligible_future_receipt_pauses(self):
        a=rows();a[6]['wheel_utime']+=1
        e,_=m.events(a,'nominal');self.assertEqual(e[6][2],['missing','missing'])
    def test_unknown_time_no_monitor_update(self):
        a=rows();a[6]['pose_utime']=None
        e,_=m.events(a,'nominal');self.assertEqual(e[6][3],e[5][3])
    def test_reserved_paths_block_before_payload_access(self):
        with patch.object(Path,'open',side_effect=AssertionError('payload accessed')):
            for reserved in ('Urban02_NIGHT','RAV4','vehicle_final_evaluation'):
                with self.assertRaises(ValueError):run.run(Path(reserved),Path('baseline'),Path('output'))
    def test_input_output_overlap_refused(self):
        with self.assertRaisesRegex(ValueError,'overlaps'):run.run(Path('inputs'),Path('base'),Path('inputs/out'))
    def test_source_output_overlap_refused(self):
        with self.assertRaisesRegex(ValueError,'source module'):run.run(Path('inputs'),Path('base'),run.HERE)


class PipelineTests(unittest.TestCase):
    def test_native_tiny_variance_pipeline_verifies_without_dropping_rows(self):
        with tempfile.TemporaryDirectory(dir=TMP) as folder:
            root=Path(folder);data=root/'data';data.mkdir();baseline=root/'baseline';out=root/'comparison'
            poses=[];wheels=[]
            for i,r in enumerate(rows()):
                t=r['pose_utime'];speed=(1+i%2)*1e-320 if i<5 else 5.
                poses.append(dict(utime=t,vel=[0.,0.,0.]))
                wheels.append(dict(utime=t,**{name:speed*60/(2*math.pi*.305) for name in ('FL_wheel_speed','FR_wheel_speed','RL_wheel_speed','RR_wheel_speed')}))
            run.io.write(data/'scene-0001_pose.json',poses)
            run.io.write(data/'scene-0001_zoe_veh_info.json',wheels)
            run.io.run(data,baseline)
            result=run.run(data,baseline,out)
            self.assertEqual(result['verified_event_rows'],14*len(poses))
            summary=run.io.read(out/'summary.json')
            c=summary['case_metrics']['nominal']['cusum_unprotected']['all']
            self.assertEqual(c['requests'],len(poses));self.assertEqual(c['monitor']['numerical_failure'],4)
            self.assertFalse(c['diagnostic_gates_pass'])
    def test_compressed_results_published_only_after_durable_close(self):
        with tempfile.TemporaryDirectory(dir=TMP) as folder:
            target=Path(folder)/'ledger.jsonl.gz'
            with run.gzip_writer(target) as stream:
                stream.write('{"test":1}\n')
                self.assertFalse(target.exists())
            self.assertEqual(gzip.decompress(target.read_bytes()),b'{"test":1}\n')
            self.assertFalse(target.with_name(target.name+'.tmp').exists())
    def test_pilot_run_replay_and_tampering(self):
        baseline=run.HERE.parent/'vehicle_can_corroboration'/'pilot_results'
        data=run.HERE.parent/'vehicle_can_corroboration'/'pilot_inputs'
        with tempfile.TemporaryDirectory(dir=TMP) as folder:
            output=Path(folder)/'results'
            result=run.run(data,baseline,output)
            self.assertTrue(result['verification_passed']);self.assertEqual(result['verified_decisions'],53984)
            self.assertEqual(run.run(data,baseline,output),result)
            name=output/'scene-0001_events.jsonl.gz'
            with gzip.open(name,'rt') as f:lines=[json.loads(line) for line in f]
            lines[0][6][0]=99.
            with gzip.open(name,'wt') as f:
                for line in lines:f.write(json.dumps(line)+'\n')
            hashes=run.io.read(output/'result_hashes.json');hashes[name.name]=run.io.digest(name)
            run.io.write(output/'result_hashes.json',hashes)
            with self.assertRaises(ValueError):verify.verify(data,baseline,output,baseline_checked=True)
    def test_nonempty_unlocked_output_not_overwritten(self):
        baseline=run.HERE.parent/'vehicle_can_corroboration'/'pilot_results'
        data=run.HERE.parent/'vehicle_can_corroboration'/'pilot_inputs'
        with tempfile.TemporaryDirectory(dir=TMP) as folder:
            output=Path(folder)/'results';output.mkdir();(output/'keep').write_text('preserve')
            with self.assertRaisesRegex(ValueError,'Nonempty'):run.run(data,baseline,output)
            self.assertEqual((output/'keep').read_text(),'preserve')

if __name__=='__main__':unittest.main()
