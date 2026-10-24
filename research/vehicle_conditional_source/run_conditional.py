"""Conditional protected-copy ablation: no real sensor independence is asserted."""
import argparse,gzip,json
from pathlib import Path
import sys
import conditional_core as cc
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'vehicle_can_corroboration'))
import run_can as baseio
import verify_can as baseverify
sys.path.insert(0,str(HERE.parent/'vehicle_can_attack_audit'))
import attack_core as attacks
CODE=('conditional_core.py','run_conditional.py','verify_conditional.py','test_conditional.py','protocol.json','previous_run_report.json')


def freeze(baseline):
    return dict(protocol='conditional-protected-copy-v1',
                baseline_sha256={n:baseio.digest(baseline/n) for n in ('input_snapshot.json','batch_summary.json','result_hashes.json','verification.json')},
                code_sha256={n:baseio.digest(HERE/n) for n in CODE},
                imported_attack_sha256={n:baseio.digest(HERE.parent/'vehicle_can_attack_audit'/n) for n in ('attack_core.py','verify_attacks.py')})


def ledger(baseline,scene):
    name=scene+'_pairs.jsonl.gz'
    expected=baseio.read(baseline/'result_hashes.json')[name]
    if baseio.digest(baseline/name)!=expected:raise ValueError('Baseline ledger changed: '+scene)
    with gzip.open(baseline/name,'rt',encoding='utf-8') as f:return [json.loads(x) for x in f]


def historical_check(summary):
    prior=baseio.read(HERE/'previous_run_report.json')['summary']
    if summary['baseline_input_snapshot_sha256']!=prior['baseline_input_snapshot_sha256']:
        return dict(applicable=False,reason='different baseline inventory; pilot or fixture only')
    fields=('requests','accepted','coverage_percent','scored_accepted','accepted_unscorable','mae_reference_disagreement_mps','p95_reference_disagreement_mps','max_reference_disagreement_mps','accepted_disagreement_gt2_count','diagnostic_gates_pass')
    checked=0
    for case in attacks.SCENARIOS:
        for group in cc.GROUPS:
            now=summary['case_metrics'][case]['legacy_agreement'][group]
            old=prior['scenario_metrics'][case]['age_and_agreement'][group]
            baseverify.equal({k:now[k] for k in fields},{k:old[k] for k in fields},'unchanged_legacy.'+case+'.'+group)
            checked+=len(fields)
    return dict(applicable=True,matched_metric_cells=checked,all_prior_legacy_failures_preserved=True)


def make_summary(base,metrics,scenes,violations,missing,lock):
    return dict(protocol_id='conditional-protected-copy-v1',baseline_input_snapshot_sha256=base['input_snapshot_sha256'],
                input_lock_sha256=lock,scene_count=base['scene_count'],baseline_pose_rows=base['pooled']['pose_rows'],
                unknown_pose_denominator_scenes=base['unknown_pose_denominator_scenes'],rows_without_query_time=missing,
                cases=list(cc.CASES),policy_names=list(cc.POLICIES),case_metrics=metrics,
                protected_claim_bound_violations=violations,scenes=scenes,
                interpretation='Conditional simulation with a protected copy of original wheel telemetry. Protection/uncertainty are assumptions; localization is an outcome proxy. No independent sensing, held-out confirmation or physical accuracy is established.',
                conditional_simulation_only=True,real_source_protection_established=False,
                physical_witness_enabled=False,production_policy_changed=False,promotion_allowed=False)


def run(data,baseline,output):
    data=baseio.safe(data);baseline=baseio.safe(baseline);output=baseio.safe(output)
    if any(output==p or output in p.parents or p in output.parents for p in (data,baseline)):
        raise ValueError('Output must not overlap native inputs/baseline')
    print('Research assumption: simulated protected wheel copy; no independent sensor claim.',flush=True)
    checked=baseverify.verify(data,baseline)
    baseverify.equal(baseio.read(baseline/'verification.json'),checked,'native_baseline_verification')
    frozen=freeze(baseline);output.mkdir(parents=True,exist_ok=True);lock=output/'input_lock.json'
    if lock.exists():
        if baseio.read(lock)!=frozen:raise ValueError('Frozen input/code changed; no automatic rescore')
    else:
        if any(output.iterdir()):raise ValueError('Nonempty output without input lock')
        baseio.write(lock,frozen)
    import verify_conditional
    if (output/'summary.json').exists() or (output/'result_hashes.json').exists():
        verified=verify_conditional.verify(data,baseline,output,baseline_checked=True)
    else:
        base=baseio.read(baseline/'batch_summary.json');pooled={c:cc.Metrics() for c in cc.CASES}
        violations={c:0 for c in cc.CASES};scenes=[];missing=0;names=[]
        with gzip.open(output/'scene_metrics.jsonl.gz','wt',encoding='utf-8') as stats:
            for number,s in enumerate(base['scenes'],1):
                rows=ledger(baseline,s['scene']);times=[r['pose_utime'] for r in rows if type(r.get('pose_utime')) is int]
                first=times[0] if times else None;absent=len(rows)-len(times);missing+=absent
                scenes.append(dict(scene=s['scene'],baseline_status=s['status'],pose_rows=s['pose_rows'],ledger_rows=len(rows),rows_without_query_time=absent))
                name=s['scene']+'_events.jsonl.gz';names.append(name);per={};per_violations={}
                with gzip.open(output/name,'wt',encoding='utf-8') as events:
                    for case_id,case in enumerate(cc.CASES):
                        local=cc.Metrics();n_bad=0
                        for event in cc.events(rows,case):
                            i=event[0][0];r=rows[i];group=attacks.phase(r.get('pose_utime'),first)
                            args=(event[5],event[6],r.get('pose_speed_mps'),r.get('wheel_speed_mps'),group)
                            local.add(*args);pooled[case].add(*args);n_bad+=event[7]
                            events.write(json.dumps([case_id]+event,separators=(',',':'),allow_nan=False)+'\n')
                        violations[case]+=n_bad;per_violations[case]=n_bad;per[case]=local.report()
                stats.write(json.dumps(dict(scene=s['scene'],metrics=per,protected_claim_bound_violations=per_violations),separators=(',',':'),allow_nan=False)+'\n')
                if number%50==0 or number==base['scene_count']:print('Scored conditional scenes: '+str(number)+'/'+str(base['scene_count']),flush=True)
        if frozen!=freeze(baseline):raise ValueError('Frozen baseline/code changed during scoring')
        summary=make_summary(base,{c:pooled[c].report() for c in cc.CASES},scenes,violations,missing,baseio.digest(lock))
        summary['legacy_comparison']=historical_check(summary)
        baseio.write(output/'summary.json',summary);names+=['input_lock.json','summary.json','scene_metrics.jsonl.gz']
        baseio.write(output/'result_hashes.json',{n:baseio.digest(output/n) for n in names});del pooled
        print('Independently reconstructing all conditional cases and decisions...',flush=True)
        verified=verify_conditional.verify(data,baseline,output,baseline_checked=True)
    baseio.write(output/'verification.json',verified)
    baseio.write(output/'run_report.json',dict(summary=baseio.read(output/'summary.json'),verification=verified))
    print(json.dumps(verified,sort_keys=True),flush=True)
    print('CONDITIONAL_SOURCE_AUDIT_COMPLETE_NO_PROMOTION',flush=True)
    return verified


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,default=HERE.parents[2]/'data'/'nuscenes_canbus'/'can_bus')
    p.add_argument('--baseline-dir',type=Path,default=HERE.parent/'vehicle_can_corroboration'/'local_batch')
    p.add_argument('--output-dir',type=Path,default=HERE/'local_batch')
    a=p.parse_args()
    try:run(a.data_dir,a.baseline_dir,a.output_dir)
    except (ValueError,OSError,KeyError) as e:
        print('CONDITIONAL_AUDIT_STOPPED: '+str(e),file=sys.stderr);sys.exit(1)
