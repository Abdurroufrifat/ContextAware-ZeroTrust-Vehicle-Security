"""Separate injection reconstruction, policy decisions and metric reducers.
Shares baseline verification, filesystem and comparison helpers, not attack logic.
"""
from array import array
import argparse
import gzip
import json
import math
from pathlib import Path
import sys
import run_attacks as io

SCENARIOS=('nominal','wheel_bias_plus3','wheel_drift_to3','wheel_replay_1s_retimed','wheel_replay_1s_original_receipt','wheel_dropout_hold','coordinated_bias_plus3')
POLICIES=('unrestricted','age_only','age_and_agreement')
GROUPS=('all','before','during','after','unavailable')


def number(x):
    if isinstance(x,bool) or not isinstance(x,(int,float)):
        return False
    return math.isfinite(x)


def reconstruct(rows,scenario):
    if scenario not in SCENARIOS:
        raise ValueError('Unknown attack')
    clock=[(r['pose_utime'],j) for j,r in enumerate(rows) if type(r.get('pose_utime')) is int]
    start=clock[0][0]+5_000_000 if clock else None
    end=start+10_000_000 if start is not None else None
    held=None
    for stamp,j in clock:
        if stamp<start:
            held=j
    older=-1;events=[]
    for index,r in enumerate(rows):
        stamp=r.get('pose_utime');source=index
        wheel=r.get('wheel_speed_mps');comparison=r.get('pose_speed_mps');receipt=r.get('wheel_utime')
        active=bool(scenario!='nominal' and type(stamp) is int and start is not None and start<=stamp<end)
        if active:
            if scenario=='wheel_bias_plus3':
                wheel=wheel+3 if number(wheel) else None
            elif scenario=='coordinated_bias_plus3':
                wheel=wheel+3 if number(wheel) else None
                comparison=comparison+3 if number(comparison) else None
            elif scenario=='wheel_drift_to3':
                delta=(stamp-start)/10_000_000*3
                wheel=wheel+delta if number(wheel) else None
            else:
                if scenario=='wheel_dropout_hold':
                    source=held
                else:
                    while older+1<len(clock) and clock[older+1][0]<=stamp-1_000_000:
                        older+=1
                    source=clock[older][1] if older>=0 else None
                if source is None:
                    wheel=None;receipt=None
                else:
                    wheel=rows[source].get('wheel_speed_mps');receipt=rows[source].get('wheel_utime')
                if scenario=='wheel_replay_1s_retimed' and receipt is not None:
                    receipt=stamp
        accepted=[False,False,False]
        if number(wheel) and type(stamp) is int and type(receipt) is int and receipt<=stamp:
            accepted[0]=True
            accepted[1]=stamp-receipt<=100_000
            accepted[2]=accepted[1] and number(comparison) and abs(wheel-comparison)<=1
        packed=sum((2**j) for j,a in enumerate(accepted) if a)
        events.append([index,active,source,receipt,wheel,comparison,packed])
    return events


def partition(t,first):
    if type(t) is not int or first is None:
        return 'unavailable'
    elapsed=t-first
    if elapsed<5_000_000:return 'before'
    if elapsed<15_000_000:return 'during'
    return 'after'


class ScalarBank:
    def __init__(self):
        self.cells={(p,g):dict(requests=0,accepted=0,unscored=0,offered=0,errors=array('d')) for p in POLICIES for g in GROUPS}

    def add(self,event,reference,group):
        w=event[4]
        delta=abs(w-reference) if number(w) and number(reference) else None
        for i,p in enumerate(POLICIES):
            yes=(event[6]//(2**i))%2==1
            for g in ['all',group]:
                c=self.cells[p,g];c['requests']+=1
                if event[6]%2==1 and delta is not None and delta>2:c['offered']+=1
                if yes:
                    c['accepted']+=1
                    if delta is None:c['unscored']+=1
                    else:c['errors'].append(delta)

    def report(self):
        out={p:{} for p in POLICIES}
        for (p,g),c in self.cells.items():
            ordered=sorted(c['errors']);n=len(ordered);total=c['requests'];k=c['accepted']
            coverage=k/total*100 if total else None
            percentile=None
            if n:
                x=.95*(n-1);low=math.floor(x);high=math.ceil(x)
                percentile=(1-(x-low))*ordered[low]+(x-low)*ordered[high]
            bad=len([v for v in ordered if v>2]);offered=c['offered']
            out[p][g]=dict(requests=total,accepted=k,coverage_percent=coverage,
                           scored_accepted=n,accepted_unscorable=c['unscored'],
                           mae_reference_disagreement_mps=sum(ordered)/n if n else None,
                           p95_reference_disagreement_mps=percentile,max_reference_disagreement_mps=ordered[-1] if n else None,
                           accepted_disagreement_gt2_count=bad,offered_disagreement_gt2_count=offered,
                           blocked_disagreement_gt2_count=offered-bad,
                           block_rate_on_large_disagreement_percent=(offered-bad)/offered*100 if offered else None,
                           diagnostic_gates_pass=bool(total and coverage>=95 and percentile is not None and percentile<=.5 and bad==0 and c['unscored']==0))
        return out


def verify(data,baseline,output,baseline_checked=False):
    data=io.baseio.safe(data);baseline=io.baseio.safe(baseline);output=io.baseio.safe(output)
    if not baseline_checked:
        v=io.baseverify.verify(data,baseline)
        io.baseverify.equal(io.baseio.read(baseline/'verification.json'),v,'baseline_verification')
    io.baseverify.equal(io.baseio.read(output/'input_lock.json'),io.freeze(baseline),'frozen_attack_lock')
    base=io.baseio.read(baseline/'batch_summary.json')
    expected_files={'input_lock.json','attack_summary.json','scene_metrics.jsonl.gz'} | {s['scene']+'_events.jsonl.gz' for s in base['scenes']}
    hashes=io.baseio.read(output/'result_hashes.json')
    if set(hashes)!=expected_files:raise ValueError('Attack result inventory mismatch')
    for name in sorted(expected_files):
        if io.baseio.digest(output/name)!=hashes[name]:raise ValueError('Attack result hash mismatch: '+name)
    pooled={s:ScalarBank() for s in SCENARIOS};scene_list=[];missing=0;event_count=0
    with gzip.open(output/'scene_metrics.jsonl.gz','rt',encoding='utf-8') as stats:
        for scene_number,entry in enumerate(base['scenes'],1):
            name=entry['scene']
            with gzip.open(baseline/(name+'_pairs.jsonl.gz'),'rt',encoding='utf-8') as raw:
                rows=[json.loads(line) for line in raw]
            clock=[r['pose_utime'] for r in rows if type(r.get('pose_utime')) is int]
            first=clock[0] if clock else None
            absent=len(rows)-len(clock);missing+=absent
            scene_list.append(dict(scene=name,baseline_status=entry['status'],pose_rows=entry['pose_rows'],
                                   ledger_rows=len(rows),rows_without_query_time=absent,
                                   window_rows=sum(partition(r.get('pose_utime'),first)=='during' for r in rows)))
            per={}
            with gzip.open(output/(name+'_events.jsonl.gz'),'rt',encoding='utf-8') as f:
                for scenario_id,scenario in enumerate(SCENARIOS):
                    scalar=ScalarBank()
                    for event in reconstruct(rows,scenario):
                        line=f.readline()
                        if not line:raise ValueError('Truncated event ledger: '+name)
                        io.baseverify.equal(json.loads(line),[scenario_id]+event,name+'.event')
                        i=event[0];group=partition(rows[i].get('pose_utime'),first);ref=rows[i].get('pose_speed_mps')
                        scalar.add(event,ref,group);pooled[scenario].add(event,ref,group)
                        event_count+=1
                    per[scenario]=scalar.report()
                if f.readline():raise ValueError('Unexpected extra event rows: '+name)
            line=stats.readline()
            if not line:raise ValueError('Truncated per-scene metrics')
            io.baseverify.equal(json.loads(line),dict(scene=name,metrics=per),name+'.metrics')
            if scene_number%50==0 or scene_number==base['scene_count']:
                print('Verified attack scenes: '+str(scene_number)+'/'+str(base['scene_count']),flush=True)
        if stats.readline():raise ValueError('Unexpected extra per-scene metrics')
    expected=dict(protocol_id='can-decision-attack-v1',input_lock_sha256=io.baseio.digest(output/'input_lock.json'),
                  baseline_input_snapshot_sha256=base['input_snapshot_sha256'],baseline_pose_rows=base['pooled']['pose_rows'],
                  scene_count=base['scene_count'],unknown_pose_denominator_scenes=base['unknown_pose_denominator_scenes'],
                  rows_without_query_time=missing,scenarios=list(SCENARIOS),policy_names=list(POLICIES),
                  scenario_metrics={s:pooled[s].report() for s in SCENARIOS},scenes=scene_list,
                  interpretation='Synthetic decision-level attacks; original localization is an evaluation proxy, not independent truth. Descriptive cohort, not held-out confirmation.',
                  independent_source_trust_established=False,physical_witness_enabled=False,production_policy_changed=False,promotion_allowed=False)
    io.baseverify.equal(io.baseio.read(output/'attack_summary.json'),expected,'attack_summary')
    return dict(verification_passed=True,verified_scenes=len(scene_list),verified_event_rows=event_count,
                verified_decisions=event_count*3,verified_policy_windows=(len(scene_list)+1)*7*3*5,
                input_lock_sha256=expected['input_lock_sha256'],baseline_input_snapshot_sha256=base['input_snapshot_sha256'],
                method='separate scalar injection, chronological replay search, policy decisions and metric reducers',
                promotion_allowed=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--baseline-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    try:print(json.dumps(verify(a.data_dir,a.baseline_dir,a.output_dir),indent=2))
    except (ValueError,OSError,KeyError) as e:
        print('ATTACK_VERIFICATION_FAILED: '+str(e),file=sys.stderr);sys.exit(1)
