"""Replay frozen scores: stateless, conditional retained alert, strict missing witness.

The strict arm is analytically DENY when no witness is present. Positive signed
policy behavior is tested only with synthetic fixtures, not invented vehicle data.
"""
import argparse
from collections import defaultdict
import hashlib,json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from coact_cps.vehicle.session_evidence import RetainedAlert

PROTOCOL_PATH='research/session_evidence/REPLAY_PROTOCOL.json'
PROTOCOL_SHA='e5d6d76420d65a26ef12035a7eea347a25b12f6bccc190ce2d620d3ba4500b2d'
ARMS=('stateless','retained_alert_ablation','strict_missing_witness')
GRANTS={'stateless':'stateless_grant','retained_alert_ablation':'retained_grant','strict_missing_witness':'strict_grant'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def replay_segment(c,score,threshold,alert,prior_attack_seen=False):
    p=c['positions'];n=len(c['time_ms']);score=np.asarray(score)
    if score.shape!=p.shape or not np.isfinite(threshold):raise ValueError('score shape or threshold mismatch')
    initial=alert.retained;known=np.zeros(n,bool);known[p]=np.isfinite(score)
    alarm=np.zeros(n,bool);alarm[p]=np.isfinite(score)&(score>threshold)
    retained=np.array([alert.observe(bool(a)) for a in alarm],bool);base=c['base_grant']
    return dict(known=known,instant_alarm=alarm,retained_alert=retained,stateless_grant=base&known&~alarm,
        retained_grant=base&known&~retained,strict_grant=np.zeros(n,bool),retained_at_start=np.array(initial),prior_attack_seen=np.array(bool(prior_attack_seen)))

def policy_metrics(c,grant,known,prior_attack_seen=False):
    t,s,y,p=c['time_ms'],c['start_ms'],c['labels'],c['positions'];eligible=np.zeros(len(t),bool);eligible[p]=True;normal=~y
    if grant.shape!=t.shape or known.shape!=t.shape:raise ValueError('decision shape mismatch')
    def groups(mask):
        a=np.flatnonzero(mask)
        if not len(a):return []
        e=np.flatnonzero((np.diff(a)!=1)|(s[a[1:]]!=t[a[:-1]]))+1
        return np.split(a,e)
    attack=y&grant;den=int((y&eligible).sum());runs=groups(attack)
    # Ground-truth labels define accounting only; they never influence permits.
    after=normal&eligible&(bool(prior_attack_seen)|np.logical_or.accumulate(y))
    within=normal&eligible&(c['period']=='after')
    return dict(epochs=len(t),eligible_epochs=int(eligible.sum()),unknown_eligible_epochs=int((eligible&~known).sum()),
        normal_epochs=int(normal.sum()),eligible_normal_epochs=int((normal&eligible).sum()),
        eligible_normal_denied=int((normal&eligible&~grant).sum()),total_normal_denied=int((normal&~grant).sum()),
        scored_attack_epochs=den,attack_granted_epochs=int(attack.sum()),attack_granted_observed_span_ms=int((t-s)[attack].sum()),
        attack_longest_grant_run_epochs=max(map(len,runs),default=0),
        attack_policy_denial_fraction=1-int((y&eligible&grant).sum())/den if den else None,
        attack_events=len(groups(y)),attack_events_without_any_denial=sum(bool(grant[a].all()) for a in groups(y)),
        post_attack_eligible_normal_epochs=int(after.sum()),post_attack_eligible_normal_denied=int((after&~grant).sum()),
        within_segment_after_eligible_normal_epochs=int(within.sum()),within_segment_after_eligible_normal_denied=int((within&~grant).sum()))

def verify_inputs(root,protocol):
    source=root/protocol['input_artifact']
    for name,d in protocol['input_pins'].items():
        if sha(source/name)!=d:raise ValueError('source evidence pin mismatch: '+name)
    for name,d in json.loads((source/'file_hashes.json').read_text()).items():
        p=(source/name).resolve()
        if not p.is_relative_to(source.resolve()) or sha(p)!=d:raise ValueError('source payload mismatch')
    lock=json.loads((source/'model_lock.json').read_text())
    for name,d in lock['code_sha256'].items():
        if sha(root/name)!=d:raise ValueError('changed frozen producer: '+name)
    return source,json.loads((source/'summary.json').read_text()),lock

def run(root,output):
    root,output=Path(root).resolve(),Path(output).resolve()
    if output.exists():raise ValueError('output exists; keep completed results')
    if sha(root/PROTOCOL_PATH)!=PROTOCOL_SHA:raise ValueError('session replay protocol pin mismatch')
    protocol=json.loads((root/PROTOCOL_PATH).read_text());source,summary,lock=verify_inputs(root,protocol)
    selected=[r for r in summary['rows'] if r['method']=='joint_history_gru6'];assert len(selected)==180
    output.mkdir(parents=True);(output/'blocks').mkdir();write(output/'replay_protocol.json',protocol)
    names=['src/coact_cps/vehicle/session_evidence.py','src/coact_cps/vehicle/witness_certificate.py','src/coact_cps/vehicle/policy.py','scripts/run_vehicle_session_evidence.py','scripts/audit_vehicle_session_evidence.py']
    replay_lock=dict(protocol_sha256=PROTOCOL_SHA,input_pins=protocol['input_pins'],code_sha256={n:sha(root/n) for n in names},
        row_order='trip,case,seed,segment ascending; latch persists across segment boundaries',missing_independent_evidence=True,thresholds=lock['thresholds'],post_attack_cohort='eligible normal after any prior attack in manifest-ordered session; labels are accounting only')
    write(output/'replay_lock.json',replay_lock)
    groups=defaultdict(list)
    for r in selected:groups[r['trip'],r['case'],r['seed']].append(r)
    rows=[];count=0
    for (trip,case,seed),group in sorted(groups.items()):
        alert=RetainedAlert();attack_seen=False
        for r in sorted(group,key=lambda x:x['segment']):
            c=read(source/'contexts'/r['context']);b=read(source/'blocks'/r['block']);threshold=lock['thresholds'][f'joint_history_gru6__{seed}']
            assert float(b['threshold'])==threshold and np.array_equal(b['positions'],c['positions'])
            values=replay_segment(c,b['score'],threshold,alert,prior_attack_seen=attack_seen)
            np.testing.assert_array_equal(values['stateless_grant'],b['grant']);np.testing.assert_array_equal(values['instant_alarm'][c['positions']],b['alarm'])
            name=f"{trip}__{r['segment']}__{case}__{seed}.npz";np.savez_compressed(output/'blocks'/name,**values);count+=1
            for arm in ARMS:
                rows.append(dict(trip=trip,segment=r['segment'],case=case,seed=seed,arm=arm,source_context=r['context'],source_block=r['block'],block=name,metrics=policy_metrics(c,values[GRANTS[arm]],values['known'],prior_attack_seen=attack_seen)))
            attack_seen=attack_seen or bool(c['labels'].any())
    assert len(rows)==protocol['expected_policy_rows']==540 and count==180
    normal=[]
    for seed in protocol['seeds']:
        for trip in sorted({r['trip'] for r in selected}):
            for arm in ARMS:
                ms=[r['metrics'] for r in rows if r['case']=='normal' and r['seed']==seed and r['trip']==trip and r['arm']==arm]
                normal.append(dict(seed=seed,trip=trip,arm=arm,eligible_denial_fraction=sum(m['eligible_normal_denied'] for m in ms)/sum(m['eligible_normal_epochs'] for m in ms)))
    totals={}
    for arm in ARMS:
        ms=[r['metrics'] for r in rows if r['case']!='normal' and r['arm']==arm]
        totals[arm]={f:sum(m[f] for m in ms) for f in ('attack_granted_epochs','attack_granted_observed_span_ms','post_attack_eligible_normal_epochs','post_attack_eligible_normal_denied','within_segment_after_eligible_normal_epochs','within_segment_after_eligible_normal_denied')}
    verify_inputs(root,protocol)
    result=dict(status='POLICY_PROTOTYPE_EVIDENCE_GAP_NOT_PROMOTED',rows_expected=540,rows_observed=len(rows),saved_policy_blocks=count,source_streams=60,normal=normal,attacked_stream_totals=totals,rows=rows,
        post_attack_cost_scope='logical session including later-segment before windows',accounting_labels_used_for_policy=False,
        frozen_neural_failed_pairs=98,instantaneous_neural_scores_changed=False,verified_real_witness_records=0,
        positive_signed_evidence='synthetic protocol tests only',strict_missing_witness_normal_gate_pass=False,
        admission_and_recovery_on_real_witness_data_evaluated=False,pgd_reoptimized_for_policy=False,pgd_scope='saved transfer stress only',
        original_neural_gates_relaxed=False,new_training=False,threshold_changed=False,active_policy_changed=False,reserved_validation_accessed=False,
        stage16_final_accessed=False,eligible_for_validation=False,replay_lock_sha256=sha(output/'replay_lock.json'))
    assert all(r['eligible_denial_fraction']==1 for r in normal if r['arm']=='strict_missing_witness')
    write(output/'summary.json',result);write(output/'file_hashes.json',{p.relative_to(output).as_posix():sha(p) for p in sorted(output.rglob('*')) if p.is_file()})
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2));return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--output',type=Path)
    a=p.parse_args();run(a.root,a.output or a.root/'artifacts/vehicle_session_evidence')
