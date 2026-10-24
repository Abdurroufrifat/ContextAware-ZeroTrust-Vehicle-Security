"""Independent prefix-latch, grant and count audit; no models or physical witnesses invented."""
import argparse
from collections import defaultdict
import hashlib,json
from pathlib import Path
import numpy as np
PROTOCOL_SHA='e5d6d76420d65a26ef12035a7eea347a25b12f6bccc190ce2d620d3ba4500b2d'
GRANTS={'stateless':'stateless_grant','retained_alert_ablation':'retained_grant','strict_missing_witness':'strict_grant'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def verify_block(c,b,carry,source_block=None,prior_attack_seen=False):
    n=len(c['time_ms']);p=c['positions']
    if source_block is not None:
        known=np.zeros(n,bool);known[p]=np.isfinite(source_block['score'])
        alarm=np.zeros(n,bool);alarm[p]=np.isfinite(source_block['score'])&(source_block['score']>float(source_block['threshold']))
        assert np.array_equal(b['known'],known) and np.array_equal(b['instant_alarm'],alarm)
    else:known=b['known'];alarm=b['instant_alarm']
    assert bool(b['prior_attack_seen'])==bool(prior_attack_seen)
    assert bool(b['retained_at_start'])==carry
    retained=np.logical_or.accumulate(alarm)|carry
    expected=dict(retained_alert=retained,stateless_grant=c['base_grant']&known&~alarm,retained_grant=c['base_grant']&known&~retained,strict_grant=np.zeros(n,bool))
    for name,value in expected.items():assert b[name].dtype==np.bool_ and b[name].shape==(n,) and np.array_equal(b[name],value),name
    return bool(retained[-1])

def metrics(c,grant,known,prior_attack_seen=False):
    n=len(c['time_ms']);selected=set(map(int,c['positions']));normal=[i for i in range(n) if not c['labels'][i]]
    ns=[i for i in normal if i in selected];attack=[i for i in range(n) if c['labels'][i]];att=[i for i in attack if i in selected]
    ga=[i for i in attack if grant[i]];within=[i for i in ns if c['period'][i]=='after']
    seen=bool(prior_attack_seen);after=[]
    for i in range(n):
        seen=seen or bool(c['labels'][i])
        if seen and i in selected and not c['labels'][i]:after.append(i)
    def runs(indices):
        out=[]
        for i in indices:
            if out and i==out[-1][-1]+1 and c['start_ms'][i]==c['time_ms'][i-1]:out[-1].append(i)
            else:out.append([i])
        return out
    return dict(epochs=n,eligible_epochs=len(selected),unknown_eligible_epochs=sum(not known[i] for i in selected),
        normal_epochs=len(normal),eligible_normal_epochs=len(ns),eligible_normal_denied=sum(not grant[i] for i in ns),total_normal_denied=sum(not grant[i] for i in normal),
        scored_attack_epochs=len(att),attack_granted_epochs=len(ga),attack_granted_observed_span_ms=sum(int(c['time_ms'][i]-c['start_ms'][i]) for i in ga),
        attack_longest_grant_run_epochs=max((len(x) for x in runs(ga)),default=0),attack_policy_denial_fraction=1-sum(bool(grant[i]) for i in att)/len(att) if att else None,
        attack_events=len(runs(attack)),attack_events_without_any_denial=sum(all(grant[i] for i in event) for event in runs(attack)),
        post_attack_eligible_normal_epochs=len(after),post_attack_eligible_normal_denied=sum(not grant[i] for i in after),
        within_segment_after_eligible_normal_epochs=len(within),within_segment_after_eligible_normal_denied=sum(not grant[i] for i in within))

def audit(root,output):
    root,output=Path(root).resolve(),Path(output).resolve();index=json.loads((output/'file_hashes.json').read_text())
    assert set(index)=={p.relative_to(output).as_posix() for p in output.rglob('*') if p.is_file() and p.name!='file_hashes.json'}
    for n,d in index.items():
        p=(output/n).resolve();assert p.is_relative_to(output) and sha(p)==d
    lock=json.loads((output/'replay_lock.json').read_text());s=json.loads((output/'summary.json').read_text())
    assert sha(output/'replay_protocol.json')==lock['protocol_sha256']==PROTOCOL_SHA
    assert sha(root/'research/session_evidence/REPLAY_PROTOCOL.json')==PROTOCOL_SHA
    assert sha(output/'replay_lock.json')==s['replay_lock_sha256']
    for n,d in lock['code_sha256'].items():assert sha(root/n)==d,n
    protocol=json.loads((output/'replay_protocol.json').read_text());source=root/protocol['input_artifact'];assert lock['input_pins']==protocol['input_pins']
    for n,d in protocol['input_pins'].items():assert sha(source/n)==d
    source_index=json.loads((source/'file_hashes.json').read_text())
    for n,d in source_index.items():
        p=(source/n).resolve();assert p.is_relative_to(source.resolve()) and sha(p)==d
    old_lock=json.loads((source/'model_lock.json').read_text());assert lock['thresholds']==old_lock['thresholds']
    for n,d in old_lock['code_sha256'].items():assert sha(root/n)==d
    old=json.loads((source/'summary.json').read_text());grouped=defaultdict(list)
    for r in old['rows']:
        if r['method']=='joint_history_gru6':grouped[r['trip'],r['case'],r['seed']].append(r)
    actual={(r['trip'],r['segment'],r['case'],r['seed'],r['arm']):r for r in s['rows']};assert len(actual)==len(s['rows'])==540
    verified=[];blocks=edges=0
    for (trip,case,seed),group in sorted(grouped.items()):
        carry=False;attack_seen=False
        for number,r in enumerate(sorted(group,key=lambda x:x['segment'])):
            name=f"{trip}__{r['segment']}__{case}__{seed}.npz";c=read(source/'contexts'/r['context']);b=read(output/'blocks'/name);original=read(source/'blocks'/r['block'])
            assert float(original['threshold'])==lock['thresholds'][f'joint_history_gru6__{seed}']
            carry=verify_block(c,b,carry,original,prior_attack_seen=attack_seen);blocks+=1;edges+=int(number>0)
            np.testing.assert_array_equal(b['stateless_grant'],original['grant'])
            for arm,field in GRANTS.items():
                row=actual[trip,r['segment'],case,seed,arm]
                assert row['block']==name and row['source_context']==r['context'] and row['source_block']==r['block']
                value=metrics(c,b[field],b['known'],prior_attack_seen=attack_seen);assert value==row['metrics'],(name,arm)
                verified.append(dict(row,metrics=value))
            attack_seen=attack_seen or bool(c['labels'].any())
    assert blocks==180 and edges==90 and len(verified)==540 and len(s['normal'])==18
    for r in s['normal']:
        ms=[v['metrics'] for v in verified if v['case']=='normal' and all(v[k]==r[k] for k in ('trip','seed','arm'))]
        value=sum(m['eligible_normal_denied'] for m in ms)/sum(m['eligible_normal_epochs'] for m in ms)
        assert value==r['eligible_denial_fraction']
        if r['arm']=='strict_missing_witness':assert value==1.
    for arm,total in s['attacked_stream_totals'].items():
        for f,value in total.items():assert value==sum(r['metrics'][f] for r in verified if r['case']!='normal' and r['arm']==arm)
    assert s['frozen_neural_failed_pairs']==98 and s['verified_real_witness_records']==0
    for f in ('accounting_labels_used_for_policy','instantaneous_neural_scores_changed','strict_missing_witness_normal_gate_pass','admission_and_recovery_on_real_witness_data_evaluated','pgd_reoptimized_for_policy','original_neural_gates_relaxed','new_training','threshold_changed','active_policy_changed','reserved_validation_accessed','stage16_final_accessed','eligible_for_validation'):assert s[f] is False
    assert s['status']=='POLICY_PROTOTYPE_EVIDENCE_GAP_NOT_PROMOTED'
    return dict(status='PASS',output_payload_files_verified=len(index),policy_rows_independently_recomputed=len(verified),saved_policy_blocks_replayed=blocks,segment_edges_with_retained_state_checked=edges,source_output_files_verified=len(source_index),strict_missing_evidence_availability_cost_accounted=True,real_witness_records_verified=0,cryptographic_verifier_independently_reimplemented=False,reserved_validation_accessed=False)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--output',type=Path)
    a=p.parse_args();print(json.dumps(audit(a.root,a.output or a.root/'artifacts/vehicle_session_evidence'),indent=2))
