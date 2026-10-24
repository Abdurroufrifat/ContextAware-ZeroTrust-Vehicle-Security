"""Separate gate evaluation and Counter-based matched-scene reductions."""
import argparse,gzip,json,sys
from collections import Counter,defaultdict
from pathlib import Path
import run_scene_comparison as run
c=run.c


def passes(x):
    if x['requests']==0:return False
    coverage=100*x['accepted']/x['requests'];p95=x['p95_reference_disagreement_mps']
    return bool(coverage>=95 and p95 is not None and p95<=.5 and x['accepted_disagreement_gt2_count']==0 and x['accepted_unscorable']==0)


def verify(conditional,monitor,output):
    conditional=run.io.safe(conditional);monitor=run.io.safe(monitor);output=run.io.safe(output)
    a,b,pooled,evidence=run.sources(conditional,monitor)
    run.equal(run.io.read(output/'input_lock.json'),run.freeze(conditional,monitor),'input_lock')
    hashes=run.io.read(output/'result_hashes.json')
    c.require(set(hashes)=={'input_lock.json','analysis.json','paired_scene_gates.jsonl.gz'},'Analysis artifact inventory')
    for name,digest in hashes.items():c.require(run.io.digest(output/name)==digest,'Analysis artifact hash: '+name)
    counts=defaultdict(Counter);models=defaultdict(list);number=0;raw_totals=defaultdict(Counter)
    with gzip.open(output/'paired_scene_gates.jsonl.gz','rt',encoding='utf-8') as stream:
        for left,right in run.scene_records(conditional,monitor):
            gates=[];requests=[left['metrics']['nominal']['legacy_agreement'][g]['requests'] for g in c.GROUPS]
            for ci,case in enumerate(c.CASES):
                combined=dict(left['metrics'][case]);combined.update(right['metrics'][case])
                flags=[]
                for policy in c.POLICIES:
                    groups=combined[policy];c.validate_groups(groups,policy in c.NEW)
                    c.require([groups[g]['requests'] for g in c.GROUPS]==requests,'Request denominator changed')
                    flags.append([passes(groups[g]) for g in c.GROUPS])
                    for group in c.GROUPS:
                        cell=groups[group];key=(case,policy,group)
                        for field in c.COUNTS:raw_totals[key][field]+=cell[field]
                        for field,value in cell['selection_counts'].items():raw_totals[key]['selection.'+field]+=value
                        if policy in c.NEW:
                            for field in ('requests','alarms')+c.STATUSES:raw_totals[key]['monitor.'+field]+=cell['monitor'][field]
                gates.append(flags)
                for track in ('unprotected','protected'):models[case,track].append(right['calibration'][case][track])
                for pi,(candidate,comparator) in enumerate(c.PAIRS):
                    for gi,group in enumerate(c.GROUPS):
                        first=flags[c.POLICIES.index(candidate)][gi];second=flags[c.POLICIES.index(comparator)][gi]
                        counter=counts[case,pi,group];counter[first,second]+=1;counter['empty']+=requests[gi]==0
            line=stream.readline();c.require(bool(line),'Truncated scene flag ledger')
            run.equal(json.loads(line),dict(scene=left['scene'],requests=requests,gates=gates,calibration=right['calibration']),'independent_scene_flags')
            number+=1
        c.require(not stream.readline(),'Extra scene flag row')
    c.require(number==b['summary']['scene_count'],'Scene count differs')
    for (case,policy,group),total in raw_totals.items():run.equal(dict(total),run.counted(pooled[case][policy][group]),'independent_source_count_totals')
    tables={case:[] for case in c.CASES}
    for case in c.CASES:
        for pi in range(len(c.PAIRS)):
            groups={}
            for group in c.GROUPS:
                d=counts[case,pi,group];n=sum(d[x,y] for x in (False,True) for y in (False,True))
                groups[group]=dict(scenes=n,both_pass=d[True,True],candidate_only=d[True,False],comparator_only=d[False,True],both_fail=d[False,False],empty_request_scenes=d['empty'],candidate_pass_scenes=d[True,True]+d[True,False],comparator_pass_scenes=d[True,True]+d[False,True],candidate_minus_comparator_percentage_points=100*(d[True,False]-d[False,True])/n if n else None)
            tables[case].append(groups)
    calibration={case:{} for case in c.CASES}
    for (case,track),data in models.items():
        positives=sorted(x['sd'] for x in data if x['sd'] is not None and x['sd']>0);n=len(positives);median=None
        if n:median=positives[n//2] if n%2 else (positives[n//2-1]+positives[n//2])/2
        calibration[case][track]=dict(scenes=len(data),valid_scenes=sum(x['valid'] for x in data),short_prefix_scenes=sum(x['n']<2 for x in data),zero_sd_scenes=sum(x['sd']==0 for x in data),subnormal_sd_scenes=sum(x<sys.float_info.min for x in positives),min_positive_sd=positives[0] if n else None,median_positive_sd=median,max_positive_sd=positives[-1] if n else None)
    expected=run.make_analysis(a,b,pooled,evidence,tables,calibration,number,run.io.digest(output/'input_lock.json'))
    run.equal(run.io.read(output/'analysis.json'),expected,'independent_analysis')
    run.equal(run.freeze(conditional,monitor),run.io.read(output/'input_lock.json'),'unchanged_sources_after_verification')
    return dict(verification_passed=True,verified_scenes=number,verified_policy_phase_flags=number*14*7*5,verified_paired_tables=14*5*5,
                input_lock_sha256=expected['input_lock_sha256'],method='Separate per-scene gate calculation, Counter discordance tables, integer totals and sorted calibration diagnostics',
                source_pooled_metric_cells_checked=490,raw_event_reconstruction_repeated_here=False,promotion_allowed=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('conditional','monitor','output'):p.add_argument('--'+key+'-dir',type=Path,required=True)
    args=p.parse_args()
    try:print(json.dumps(verify(args.conditional_dir,args.monitor_dir,args.output_dir),indent=2))
    except (ValueError,OSError,KeyError,TypeError,EOFError,OverflowError) as e:
        print('SCENE_VERIFICATION_STOPPED: '+str(e),file=sys.stderr);sys.exit(1)
