"""Verify baseline, freeze protocol, run all seven scenarios, independently verify."""
import argparse
import gzip
import json
from pathlib import Path
import sys
import attack_core as ac

HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'vehicle_can_corroboration'
sys.path.insert(0,str(BASE))
import run_can as baseio
import verify_can as baseverify
CODE=('attack_core.py','run_attacks.py','verify_attacks.py','protocol.json','test_attacks.py')


def freeze(baseline):
    return dict(protocol_id='can-decision-attack-v1',
                baseline_sha256={n:baseio.digest(baseline/n) for n in ('input_snapshot.json','batch_summary.json','result_hashes.json','verification.json')},
                attack_code_sha256={n:baseio.digest(HERE/n) for n in CODE})


def ledger(baseline,scene):
    with gzip.open(baseline/(scene+'_pairs.jsonl.gz'),'rt',encoding='utf-8') as f:
        return [json.loads(s) for s in f]


def summary(base,metrics,scenes,missing_times,lock_sha):
    return dict(protocol_id='can-decision-attack-v1',input_lock_sha256=lock_sha,
                baseline_input_snapshot_sha256=base['input_snapshot_sha256'],
                baseline_pose_rows=base['pooled']['pose_rows'],scene_count=base['scene_count'],
                unknown_pose_denominator_scenes=base['unknown_pose_denominator_scenes'],
                rows_without_query_time=missing_times,scenarios=list(ac.SCENARIOS),
                policy_names=list(ac.POLICIES),scenario_metrics=metrics,scenes=scenes,
                interpretation='Synthetic decision-level attacks; original localization is an evaluation proxy, not independent truth. Descriptive cohort, not held-out confirmation.',
                independent_source_trust_established=False,physical_witness_enabled=False,
                production_policy_changed=False,promotion_allowed=False)


def run(data,baseline,output):
    data=baseio.safe(data);baseline=baseio.safe(baseline);output=baseio.safe(output)
    for other in (data,baseline):
        if output==other or other in output.parents or output in other.parents:
            raise ValueError('Attack output must not overlap raw input or baseline directory')
    print('Checking the complete native baseline and independent verification...',flush=True)
    checked=baseverify.verify(data,baseline)
    recorded=baseio.read(baseline/'verification.json')
    baseverify.equal(recorded,checked,'baseline_verification')
    frozen=freeze(baseline)
    output.mkdir(parents=True,exist_ok=True)
    lock=output/'input_lock.json'
    if lock.exists():
        if baseio.read(lock)!=frozen:
            raise ValueError('Frozen baseline/code changed; refusing automatic rescore')
    else:
        if any(output.iterdir()):
            raise ValueError('Nonempty attack output without a freeze lock')
        baseio.write(lock,frozen)
    import verify_attacks
    if (output/'attack_summary.json').exists() or (output/'result_hashes.json').exists():
        verification=verify_attacks.verify(data,baseline,output,baseline_checked=True)
    else:
        base=baseio.read(baseline/'batch_summary.json')
        pooled={s:ac.Bank() for s in ac.SCENARIOS}; scene_list=[]; missing_times=0
        names=[]
        with gzip.open(output/'scene_metrics.jsonl.gz','wt',encoding='utf-8') as stats:
            for number,s in enumerate(base['scenes'],1):
                scene=s['scene'];rows=ledger(baseline,scene)
                times=[r['pose_utime'] for r in rows if type(r.get('pose_utime')) is int]
                first=times[0] if times else None
                absent=len(rows)-len(times);missing_times+=absent
                info=dict(scene=scene,baseline_status=s['status'],pose_rows=s['pose_rows'],
                          ledger_rows=len(rows),rows_without_query_time=absent,
                          window_rows=sum(ac.phase(r.get('pose_utime'),first)=='during' for r in rows))
                scene_list.append(info);per={}
                name=scene+'_events.jsonl.gz';names.append(name)
                with gzip.open(output/name,'wt',encoding='utf-8') as events:
                    for scenario_id,scenario in enumerate(ac.SCENARIOS):
                        bank=ac.Bank()
                        for event in ac.transform(rows,scenario):
                            i=event[0];reference=rows[i].get('pose_speed_mps')
                            group=ac.phase(rows[i].get('pose_utime'),first)
                            bank.add(event,reference,group);pooled[scenario].add(event,reference,group)
                            events.write(json.dumps([scenario_id]+event,separators=(',',':'),allow_nan=False)+'\n')
                        per[scenario]=bank.report()
                stats.write(json.dumps(dict(scene=scene,metrics=per),separators=(',',':'),allow_nan=False)+'\n')
                if number%50==0 or number==len(base['scenes']):
                    print('Scored scenes: '+str(number)+'/'+str(len(base['scenes'])),flush=True)
        if freeze(baseline)!=frozen:
            raise ValueError('Baseline/code changed during audit; refusing finalization')
        metrics={s:pooled[s].report() for s in ac.SCENARIOS}
        baseio.write(output/'attack_summary.json',summary(base,metrics,scene_list,missing_times,baseio.digest(lock)))
        names+=['input_lock.json','attack_summary.json','scene_metrics.jsonl.gz']
        baseio.write(output/'result_hashes.json',{n:baseio.digest(output/n) for n in names})
        del pooled
        print('Reconstructing every attack event, decision and metric independently...',flush=True)
        verification=verify_attacks.verify(data,baseline,output,baseline_checked=True)
    baseio.write(output/'verification.json',verification)
    baseio.write(output/'run_report.json',dict(summary=baseio.read(output/'attack_summary.json'),verification=verification))
    print(json.dumps(verification,sort_keys=True),flush=True)
    print('CAN_ATTACK_AUDIT_COMPLETE_NO_PROMOTION',flush=True)
    return verification


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,default=HERE.parents[2]/'data'/'nuscenes_canbus'/'can_bus')
    p.add_argument('--baseline-dir',type=Path,default=BASE/'local_batch')
    p.add_argument('--output-dir',type=Path,default=HERE/'local_batch')
    a=p.parse_args()
    try:
        run(a.data_dir,a.baseline_dir,a.output_dir)
    except (ValueError,OSError,KeyError) as e:
        print('ATTACK_AUDIT_STOPPED: '+str(e),file=sys.stderr);return 1
    return 0


if __name__=='__main__':sys.exit(main())
