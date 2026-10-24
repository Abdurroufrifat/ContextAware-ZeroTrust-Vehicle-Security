"""Independent source construction, selector and statistical reductions."""
from array import array
import argparse,gzip,json,math
from pathlib import Path
import sys
import run_conditional as io
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'vehicle_can_attack_audit'))
import verify_attacks as previous

CASES=('nominal','wheel_bias_plus3','wheel_drift_to3','wheel_replay_1s_retimed','wheel_replay_1s_original_receipt','wheel_dropout_hold','coordinated_bias_plus3','coordinated_u0','coordinated_u075','protected_copy_missing','protected_copy_stale','protected_copy_compromised','protected_bound_violated','unestablished_real_evidence')
POLICIES=('legacy_agreement','conditional_anchor_only','conditional_bounded_candidate')
GROUPS=('all','before','during','after','unavailable')


def numeric(v):return type(v) in (int,float) and math.isfinite(v)


def receipt_ok(t,stamp):return type(t) is int and type(stamp) is int and stamp>=0 and 0<=t-stamp<=100000


def reconstruct(rows,case):
    if case not in CASES:raise ValueError('Unknown conditional case')
    scenario=case if case in previous.SCENARIOS else 'coordinated_bias_plus3'
    attacks=previous.reconstruct(rows,scenario);result=[]
    for i,r in enumerate(rows):
        event=attacks[i];t=r.get('pose_utime');clean=r.get('wheel_speed_mps')
        uncertainty={'coordinated_u0':0,'coordinated_u075':.75}.get(case,.25)
        center=float(clean)+(uncertainty if i%2 else 0) if numeric(clean) else None
        stamp=r.get('wheel_utime');assume=case!='unestablished_real_evidence'
        if event[1]:
            if case=='protected_copy_missing':center=None;stamp=None
            if case=='protected_copy_stale':stamp=t-100001
            if case=='protected_copy_compromised':center=center+3 if numeric(center) else None
            if case=='protected_bound_violated':center=center+1 if numeric(center) else None
        source_ok=(assume and numeric(center) and center>=0 and numeric(uncertainty) and 0<=uncertainty<=.5 and receipt_ok(t,stamp))
        outputs=[event[4] if (event[6]//4)%2 else None,None,None]
        codes=[1 if outputs[0] is not None else 0,0,0]
        if source_ok:
            outputs[1]=center;codes[1]=2
            candidate=event[4]
            use=(numeric(candidate) and candidate>=0 and receipt_ok(t,event[3]) and abs(candidate-center)+uncertainty<=.5)
            outputs[2]=candidate if use else center;codes[2]=1 if use else 2
        bad=bool(numeric(center) and numeric(clean) and abs(center-clean)>uncertainty+1e-12)
        result.append([event,center,uncertainty,stamp,assume,outputs,codes,bad])
    return result


class ScalarMetrics:
    def __init__(self):
        self.cells={(p,g):dict(n=0,k=0,unscored=0,errors=array('d'),trace_bad=0,trace_max=None,codes=[0,0,0]) for p in POLICIES for g in GROUPS}
    def add(self,values,codes,reference,clean,group):
        for j,p in enumerate(POLICIES):
            value=values[j]
            for g in ['all',group]:
                c=self.cells[p,g];c['n']+=1;c['codes'][codes[j]]+=1
                if value is not None:
                    c['k']+=1
                    if numeric(reference):c['errors'].append(abs(value-reference))
                    else:c['unscored']+=1
                    if numeric(clean):
                        delta=abs(value-clean)
                        if delta>.5+1e-12:c['trace_bad']+=1
                        if c['trace_max'] is None or delta>c['trace_max']:c['trace_max']=delta
    def report(self):
        out={p:{} for p in POLICIES}
        for (p,g),c in self.cells.items():
            e=sorted(c['errors']);n=len(e);coverage=c['k']/c['n']*100 if c['n'] else None
            p95=None
            if n:
                at=.95*(n-1);lo=math.floor(at);hi=math.ceil(at)
                p95=e[lo]*(1-(at-lo))+e[hi]*(at-lo)
            bad=len([x for x in e if x>2])
            out[p][g]=dict(requests=c['n'],accepted=c['k'],coverage_percent=coverage,scored_accepted=n,
                           accepted_unscorable=c['unscored'],mae_reference_disagreement_mps=sum(e)/n if n else None,
                           p95_reference_disagreement_mps=p95,max_reference_disagreement_mps=e[-1] if n else None,
                           accepted_disagreement_gt2_count=bad,trace_bound_violation_count=c['trace_bad'],
                           max_deviation_from_original_wheel_mps=c['trace_max'],
                           selection_counts=dict(zip(('abstain','candidate','protected_copy'),c['codes'])),
                           diagnostic_gates_pass=bool(c['n'] and coverage>=95 and p95 is not None and p95<=.5 and bad==0 and c['unscored']==0))
        return out


def verify(data,baseline,output,baseline_checked=False):
    data=io.baseio.safe(data);baseline=io.baseio.safe(baseline);output=io.baseio.safe(output)
    if not baseline_checked:
        checked=io.baseverify.verify(data,baseline)
        io.baseverify.equal(io.baseio.read(baseline/'verification.json'),checked,'baseline_verification')
    io.baseverify.equal(io.baseio.read(output/'input_lock.json'),io.freeze(baseline),'input_lock')
    base=io.baseio.read(baseline/'batch_summary.json')
    names={'input_lock.json','summary.json','scene_metrics.jsonl.gz'}|{s['scene']+'_events.jsonl.gz' for s in base['scenes']}
    hashes=io.baseio.read(output/'result_hashes.json')
    if set(hashes)!=names:raise ValueError('Conditional result inventory differs')
    for name in sorted(names):
        if io.baseio.digest(output/name)!=hashes[name]:raise ValueError('Conditional result hash mismatch: '+name)
    pooled={c:ScalarMetrics() for c in CASES};violations={c:0 for c in CASES};scenes=[];missing=0;count=0
    baseline_hashes=io.baseio.read(baseline/'result_hashes.json')
    with gzip.open(output/'scene_metrics.jsonl.gz','rt',encoding='utf-8') as stats:
        for number,s in enumerate(base['scenes'],1):
            name=s['scene']+'_pairs.jsonl.gz'
            if io.baseio.digest(baseline/name)!=baseline_hashes[name]:raise ValueError('Native baseline ledger changed')
            with gzip.open(baseline/name,'rt',encoding='utf-8') as f:rows=[json.loads(x) for x in f]
            times=[r['pose_utime'] for r in rows if type(r.get('pose_utime')) is int];first=times[0] if times else None
            absent=len(rows)-len(times);missing+=absent
            scenes.append(dict(scene=s['scene'],baseline_status=s['status'],pose_rows=s['pose_rows'],ledger_rows=len(rows),rows_without_query_time=absent))
            per={};per_violations={}
            with gzip.open(output/(s['scene']+'_events.jsonl.gz'),'rt',encoding='utf-8') as events:
                for case_id,case in enumerate(CASES):
                    local=ScalarMetrics();n_bad=0
                    for event in reconstruct(rows,case):
                        line=events.readline()
                        if not line:raise ValueError('Truncated conditional event ledger')
                        io.baseverify.equal(json.loads(line),[case_id]+event,s['scene']+'.'+case+'.event')
                        r=rows[event[0][0]];group=previous.partition(r.get('pose_utime'),first)
                        args=(event[5],event[6],r.get('pose_speed_mps'),r.get('wheel_speed_mps'),group)
                        local.add(*args);pooled[case].add(*args);n_bad+=event[7];count+=1
                    per[case]=local.report();per_violations[case]=n_bad;violations[case]+=n_bad
                if events.readline():raise ValueError('Unexpected extra conditional event row')
            line=stats.readline()
            if not line:raise ValueError('Truncated scene metrics')
            io.baseverify.equal(json.loads(line),dict(scene=s['scene'],metrics=per,protected_claim_bound_violations=per_violations),s['scene']+'.metrics')
            if number%50==0 or number==base['scene_count']:print('Verified conditional scenes: '+str(number)+'/'+str(base['scene_count']),flush=True)
        if stats.readline():raise ValueError('Unexpected extra scene metrics')
    expected=dict(protocol_id='conditional-protected-copy-v1',baseline_input_snapshot_sha256=base['input_snapshot_sha256'],
                  input_lock_sha256=io.baseio.digest(output/'input_lock.json'),scene_count=base['scene_count'],baseline_pose_rows=base['pooled']['pose_rows'],
                  unknown_pose_denominator_scenes=base['unknown_pose_denominator_scenes'],rows_without_query_time=missing,
                  cases=list(CASES),policy_names=list(POLICIES),case_metrics={c:pooled[c].report() for c in CASES},
                  protected_claim_bound_violations=violations,scenes=scenes,
                  interpretation='Conditional simulation with a protected copy of original wheel telemetry. Protection/uncertainty are assumptions; localization is an outcome proxy. No independent sensing, held-out confirmation or physical accuracy is established.',
                  conditional_simulation_only=True,real_source_protection_established=False,
                  physical_witness_enabled=False,production_policy_changed=False,promotion_allowed=False)
    expected['legacy_comparison']=io.historical_check(expected)
    io.baseverify.equal(io.baseio.read(output/'summary.json'),expected,'conditional_summary')
    return dict(verification_passed=True,verified_scenes=len(scenes),verified_event_rows=count,verified_decisions=count*3,
                verified_policy_windows=(len(scenes)+1)*14*3*5,input_lock_sha256=expected['input_lock_sha256'],
                baseline_input_snapshot_sha256=base['input_snapshot_sha256'],
                method='separate attack reconstruction, protected-copy model, eligibility/selection and scalar metric reduction',
                conditional_simulation_only=True,real_source_protection_established=False,promotion_allowed=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--baseline-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    try:print(json.dumps(verify(a.data_dir,a.baseline_dir,a.output_dir),indent=2))
    except (ValueError,OSError,KeyError) as e:
        print('CONDITIONAL_VERIFICATION_FAILED: '+str(e),file=sys.stderr);sys.exit(1)
