"""Behavioral fixtures for the frozen offline decision-level attack audit."""
import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest
import attack_core as ac
import run_attacks as runner
import verify_attacks as verifier


def row(i, t, speed=5, age=0, comparator=5):
    return dict(pose_index=i, pose_utime=t, wheel_index=i, wheel_utime=t-age,
                receipt_age_us=age, wheel_speed_mps=speed, pose_speed_mps=comparator,
                signed_disagreement_mps=speed-comparator, status='valid')


class TransformTests(unittest.TestCase):
    def setUp(self):
        self.rows=[row(i,t) for i,t in enumerate([0,4_999_999,5_000_000,5_100_000,6_000_000,14_999_999,15_000_000,16_000_000])]

    def test_nominal_keeps_values(self):
        e=ac.transform(self.rows,'nominal')
        self.assertTrue(all(not r[1] and r[4:6]==[5,5] for r in e))

    def test_bias_window_boundaries(self):
        e=ac.transform(self.rows,'wheel_bias_plus3')
        self.assertEqual([r[4] for r in e],[5,5,8,8,8,8,5,5])
        self.assertEqual([r[1] for r in e],[False,False,True,True,True,True,False,False])

    def test_source_rows_unmodified(self):
        before=copy.deepcopy(self.rows); ac.transform(self.rows,'coordinated_bias_plus3')
        self.assertEqual(before,self.rows)

    def test_coordinated_attack_bypasses_agreement(self):
        e=ac.transform(self.rows,'coordinated_bias_plus3')[2]
        self.assertEqual(e[4:6],[8,8]); self.assertEqual(e[6],7)
        self.assertEqual(e[4]-self.rows[2]['pose_speed_mps'],3)

    def test_single_bias_blocked_by_agreement(self):
        self.assertEqual(ac.transform(self.rows,'wheel_bias_plus3')[2][6],3)

    def test_receipt_age_inclusive(self):
        self.assertEqual(ac.mask(5,5,100_000,0),7)
        self.assertEqual(ac.mask(5,5,100_001,0),1)
        self.assertEqual(ac.mask(5,5,0,1),0)

    def test_peer_difference_inclusive(self):
        self.assertEqual(ac.mask(6,5,0,0),7)
        self.assertEqual(ac.mask(6.000001,5,0,0),3)

    def test_invalid_values_abstain(self):
        for v in [None,True,'5',float('nan'),float('inf')]:
            self.assertEqual(ac.mask(v,5,0,0),0)
        self.assertEqual(ac.mask(5,None,0,0),3)

    def test_dropout_holds_old_receipt_then_abstains(self):
        e=ac.transform(self.rows,'wheel_dropout_hold')
        self.assertEqual(e[2][2:4],[1,4_999_999])
        self.assertEqual(e[3][6],1)
        self.assertEqual(e[6][2:4],[6,15_000_000])
        self.assertEqual(e[6][6],7)

    def test_replay_retains_causal_history(self):
        r=[row(0,0,2,comparator=2),row(1,4_000_000,3,comparator=3),row(2,5_000_000,7,comparator=7)]
        e=ac.transform(r,'wheel_replay_1s_retimed')[2]
        self.assertEqual(e[2],1); self.assertEqual(e[4],3)
        self.assertEqual(e[3],5_000_000); self.assertEqual(e[6],3)
        old=ac.transform(r,'wheel_replay_1s_original_receipt')[2]
        self.assertEqual(old[3],4_000_000); self.assertEqual(old[6],1)

    def test_replay_does_not_skip_invalid_latest_history(self):
        r=[row(0,0),row(1,4_000_000),row(2,5_000_000)]
        r[1]['wheel_speed_mps']=None
        self.assertIsNone(ac.transform(r,'wheel_replay_1s_retimed')[2][4])

    def test_ramp_fixed_without_clipping(self):
        e=ac.transform(self.rows,'wheel_drift_to3')
        self.assertEqual(e[2][4],5)
        self.assertAlmostEqual(e[4][4],5.3)
        self.assertEqual(e[6][4],5)

    def test_scalar_reconstruction_matches(self):
        for scenario in ac.SCENARIOS:
            self.assertEqual(ac.transform(self.rows,scenario),verifier.reconstruct(self.rows,scenario))

    def test_unknown_times_retained(self):
        r=row(0,0); r.update(pose_utime=None,wheel_utime=None,wheel_speed_mps=None,pose_speed_mps=None)
        for scenario in ac.SCENARIOS:
            self.assertEqual(len(ac.transform([r],scenario)),1)
            self.assertEqual(ac.transform([r],scenario)[0][6],0)

    def test_metrics_all_requests_and_unscorable(self):
        bank=ac.Bank()
        bank.add([0,True,0,0,8,8,7],5,'during')
        bank.add([1,True,None,None,None,None,0],5,'during')
        s=bank.report()['age_and_agreement']['all']
        self.assertEqual((s['requests'],s['accepted'],s['accepted_disagreement_gt2_count']),(2,1,1))
        self.assertEqual(s['coverage_percent'],50)
        self.assertFalse(s['diagnostic_gates_pass'])
        bank.add([2,False,2,1,5,None,3],None,'after')
        self.assertEqual(bank.report()['age_only']['all']['accepted_unscorable'],1)

    def test_reference_two_boundary_strict(self):
        bank=ac.Bank(); bank.add([0,True,0,0,7,7,7],5,'during')
        self.assertEqual(bank.report()['unrestricted']['all']['accepted_disagreement_gt2_count'],0)

    def test_zero_acceptance_cannot_pass(self):
        bank=ac.Bank();bank.add([0,True,0,0,8,5,3],5,'during')
        result=bank.report()['age_and_agreement']['all']
        self.assertEqual(result['accepted'],0)
        self.assertIsNone(result['p95_reference_disagreement_mps'])
        self.assertFalse(result['diagnostic_gates_pass'])

    def test_phase_denominators_partition_all(self):
        bank=ac.Bank()
        for i,group in enumerate(['before','during','after','unavailable']):
            bank.add([i,False,i,0,5,5,7],5,group)
        result=bank.report()['age_only']
        self.assertEqual(result['all']['requests'],sum(result[g]['requests'] for g in ac.PHASES if g!='all'))

    def test_unknown_scenario_rejected(self):
        with self.assertRaises(ValueError): ac.transform(self.rows,'tuned_new_attack')


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.root=Path(self.tmp.name)
        self.data=self.root/'data';self.data.mkdir()
        self.base=self.root/'base';self.out=self.root/'attack'
        p=[{'utime':t,'vel':[5,0,0]} for t in [0,4_000_000,5_000_000,5_100_000,10_000_000,15_000_000,16_000_000]]
        wheels=[dict(utime=r['utime'],**{k:5/(2*3.141592653589793*.305/60) for k in runner.baseio.core.WHEELS}) for r in p]
        (self.data/'scene-0001_pose.json').write_text(json.dumps(p))
        (self.data/'scene-0001_zoe_veh_info.json').write_text(json.dumps(wheels))
        runner.baseio.run(self.data,self.base)

    def tearDown(self): self.tmp.cleanup()

    def test_batch_full_verification_and_repeat(self):
        runner.run(self.data,self.base,self.out)
        v=verifier.verify(self.data,self.base,self.out)
        self.assertEqual(v['verified_event_rows'],7*len(ac.SCENARIOS))
        self.assertEqual(v['verified_decisions'],7*len(ac.SCENARIOS)*3)
        self.assertTrue(v['verification_passed'])
        report=runner.baseio.read(self.out/'run_report.json')
        self.assertFalse(report['summary']['promotion_allowed'])
        before=(self.out/'run_report.json').read_bytes()
        runner.run(self.data,self.base,self.out)
        self.assertEqual(before,(self.out/'run_report.json').read_bytes())

    def test_changed_native_input_refused(self):
        runner.run(self.data,self.base,self.out)
        (self.data/'scene-0001_pose.json').write_text('[]')
        with self.assertRaises(ValueError): runner.run(self.data,self.base,self.out)

    def test_forged_output_with_updated_hash_refused(self):
        runner.run(self.data,self.base,self.out)
        p=self.out/'attack_summary.json';s=runner.baseio.read(p)
        s['scenario_metrics']['coordinated_bias_plus3']['age_and_agreement']['during']['accepted_disagreement_gt2_count']=0
        runner.baseio.write(p,s)
        h=runner.baseio.read(self.out/'result_hashes.json');h[p.name]=runner.baseio.digest(p)
        runner.baseio.write(self.out/'result_hashes.json',h)
        with self.assertRaises(ValueError): verifier.verify(self.data,self.base,self.out)

    def test_missing_stream_denominator_retained(self):
        extra=[{'utime':0,'vel':[5,0,0]},{'utime':1,'vel':[5,0,0]}]
        (self.data/'scene-0002_pose.json').write_text(json.dumps(extra))
        other=self.root/'base2';runner.baseio.run(self.data,other)
        runner.run(self.data,other,self.out)
        s=runner.baseio.read(self.out/'attack_summary.json')
        self.assertEqual(s['scene_count'],2)
        self.assertEqual(s['scenario_metrics']['nominal']['age_only']['all']['requests'],9)
        self.assertEqual(s['rows_without_query_time'],2)

    def test_forged_event_with_updated_hash_refused(self):
        runner.run(self.data,self.base,self.out)
        p=self.out/'scene-0001_events.jsonl.gz'
        with gzip.open(p,'rt',encoding='utf-8') as f:
            values=[json.loads(line) for line in f]
        values[7+2][-1]=7
        with gzip.open(p,'wt',encoding='utf-8') as f:
            for value in values:f.write(json.dumps(value)+'\n')
        h=runner.baseio.read(self.out/'result_hashes.json');h[p.name]=runner.baseio.digest(p)
        runner.baseio.write(self.out/'result_hashes.json',h)
        with self.assertRaises(ValueError):verifier.verify(self.data,self.base,self.out)

    def test_baseline_verification_record_must_match(self):
        record=runner.baseio.read(self.base/'verification.json')
        record['verified_pair_rows']=1
        runner.baseio.write(self.base/'verification.json',record)
        with self.assertRaises(ValueError):runner.run(self.data,self.base,self.out)

    def test_truncated_ledger_with_updated_hash_refused(self):
        runner.run(self.data,self.base,self.out)
        p=self.out/'scene-0001_events.jsonl.gz'
        with gzip.open(p,'rt',encoding='utf-8') as f:lines=f.readlines()
        with gzip.open(p,'wt',encoding='utf-8') as f:f.writelines(lines[:-1])
        h=runner.baseio.read(self.out/'result_hashes.json');h[p.name]=runner.baseio.digest(p)
        runner.baseio.write(self.out/'result_hashes.json',h)
        with self.assertRaises(ValueError):verifier.verify(self.data,self.base,self.out)

    def test_output_cannot_overlap_baseline(self):
        with self.assertRaises(ValueError): runner.run(self.data,self.base,self.base)


if __name__=='__main__': unittest.main(verbosity=2)
