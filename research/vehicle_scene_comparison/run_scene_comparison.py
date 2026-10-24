"""Audit existing scene results; never rerun native parsing, attacks or model fitting."""
import argparse,gzip,itertools,json,sys
from pathlib import Path
import scene_core as c
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'vehicle_monitor_comparison'))
import run_comparison as monitor_io
io=monitor_io.io
equal=monitor_io.equal
CODE=('scene_core.py','run_scene_comparison.py','verify_scene_comparison.py','test_scene_comparison.py','protocol.json','input_evidence/monitor_full_report.json','input_evidence/conditional_full_report.json')
META=('input_lock.json','summary.json','scene_metrics.jsonl.gz','result_hashes.json','verification.json','run_report.json')


def checked_source(directory,monitor):
    directory=io.safe(directory);hashes=io.read(directory/'result_hashes.json')
    for name in ('input_lock.json','summary.json','scene_metrics.jsonl.gz'):
        c.require(io.digest(directory/name)==hashes[name],'Source result hash mismatch: '+name)
    report=io.read(directory/'run_report.json')
    equal(report['summary'],io.read(directory/'summary.json'),'source.summary')
    equal(report['verification'],io.read(directory/'verification.json'),'source.verification')
    result=c.validate_report(report,monitor);lock=io.read(directory/'input_lock.json')
    c.require(io.digest(directory/'input_lock.json')==report['summary']['input_lock_sha256'],'Source input lock hash')
    c.require(lock['protocol_id' if monitor else 'protocol']==report['summary']['protocol_id'],'Locked protocol')
    module=HERE.parent/('vehicle_monitor_comparison' if monitor else 'vehicle_conditional_source')
    for name,digest in lock['code_sha256'].items():
        c.require(Path(name).name==name,'Unsafe locked code name')
        c.require(io.digest(module/name)==digest,'Locked source changed: '+name)
    imported=lock['imported_code_sha256'] if monitor else {'vehicle_can_attack_audit/'+k:v for k,v in lock['imported_attack_sha256'].items()}
    for name,digest in imported.items():
        relative=Path(name)
        c.require(len(relative.parts)==2 and relative.parts[0] in ('vehicle_can_corroboration','vehicle_can_attack_audit','vehicle_conditional_source') and relative.suffix=='.py','Unsafe imported code name')
        c.require(io.digest(HERE.parent/relative)==digest,'Imported source changed: '+name)
    return report,lock,result


def sources(conditional,monitor):
    a,al,av=checked_source(conditional,False);b,bl,bv=checked_source(monitor,True)
    equal(al['baseline_sha256'],bl['baseline_sha256'],'shared_baseline_metadata')
    for key in ('baseline_input_snapshot_sha256','scene_count','baseline_pose_rows','unknown_pose_denominator_scenes'):
        equal(a['summary'][key],b['summary'][key],'shared_inventory.'+key)
    equal(a['summary']['case_metrics'],b['summary']['historical_comparison']['case_metrics'],'preserved_historical_metrics')
    evidence={}
    for name,report in [('conditional',a),('monitor',b)]:
        frozen=io.read(HERE/'input_evidence'/(name+'_full_report.json'))
        applicable=frozen['summary']['baseline_input_snapshot_sha256']==report['summary']['baseline_input_snapshot_sha256']
        if applicable:equal(report,frozen,'supplied_full_'+name)
        evidence[name]=dict(matches_supplied_full_report=applicable,validation=av if name=='conditional' else bv)
    pooled={case:{**a['summary']['case_metrics'][case],**b['summary']['case_metrics'][case]} for case in c.CASES}
    return a,b,pooled,evidence


def freeze(conditional,monitor):
    return dict(protocol_id='matched-scene-comparison-v1',code_sha256={name:io.digest(HERE/name) for name in CODE},
                conditional_metadata_sha256={name:io.digest(conditional/name) for name in META},
                monitor_metadata_sha256={name:io.digest(monitor/name) for name in META})


def scene_records(conditional,monitor):
    seen=set()
    with gzip.open(io.safe(conditional)/'scene_metrics.jsonl.gz','rt',encoding='utf-8') as old,gzip.open(io.safe(monitor)/'scene_metrics.jsonl.gz','rt',encoding='utf-8') as new:
        for left,right in itertools.zip_longest(old,new):
            c.require(left is not None and right is not None,'Scene ledger lengths differ')
            a=json.loads(left);b=json.loads(right);name=a['scene']
            c.require(name==b['scene'] and name not in seen,'Mismatched, duplicated or reordered scenes');seen.add(name)
            yield a,b


def counted(x):
    result={key:x[key] for key in c.COUNTS}
    result.update({'selection.'+k:v for k,v in x['selection_counts'].items()})
    if 'monitor' in x:result.update({'monitor.'+k:x['monitor'][k] for k in ('requests','alarms')+c.STATUSES})
    return result


def accumulate(totals,key,cell):
    current=counted(cell);stored=totals.setdefault(key,{k:0 for k in current})
    for k,v in current.items():stored[k]+=v


def flags_record(a,b,totals,calibration):
    c.require(set(a['metrics'])==set(c.CASES) and set(b['metrics'])==set(c.CASES),'Scene case inventory')
    flags=[];requests=None
    for case in c.CASES:
        c.require(set(a['metrics'][case])==set(c.OLD) and set(b['metrics'][case])==set(c.NEW),'Scene policy inventory')
        merged={**a['metrics'][case],**b['metrics'][case]};scene_flags=[]
        for policy in c.POLICIES:
            groups=merged[policy];c.validate_groups(groups,policy in c.NEW)
            counts=[groups[g]['requests'] for g in c.GROUPS]
            if requests is None:requests=counts
            c.require(requests==counts,'Shared request denominators differ')
            scene_flags.append([c.gate(groups[g]) for g in c.GROUPS])
            for group in c.GROUPS:accumulate(totals,(case,policy,group),groups[group])
        flags.append(scene_flags)
        for track in ('unprotected','protected'):calibration.add(case,track,b['calibration'][case][track])
    return dict(scene=a['scene'],requests=requests,gates=flags,calibration=b['calibration'])


def make_analysis(a,b,pooled,evidence,pairs,calibration,count,lock):
    return dict(protocol_id='matched-scene-comparison-v1',scene_count=count,baseline_pose_rows=b['summary']['baseline_pose_rows'],
                baseline_input_snapshot_sha256=b['summary']['baseline_input_snapshot_sha256'],input_lock_sha256=lock,
                pairs=[dict(candidate=x,comparator=y,track='unprotected' if i<2 else 'simulated_protected') for i,(x,y) in enumerate(c.PAIRS)],
                cases=list(c.CASES),policies=list(c.POLICIES),phases=list(c.GROUPS),paired_scene_tables=pairs,
                calibration_diagnostics=calibration,pooled_metrics=pooled,source_evidence=evidence,
                source_monitor_verified_decisions=b['verification']['verified_decisions'],source_conditional_verified_decisions=a['verification']['verified_decisions'],
                interpretation='Post-result descriptive matched-scene gate counts. Every scene and zero-request phase retained. Source pooled metrics are preserved, not reconstructed by averaging scene quantiles. No p-values, independence assumption, held-out confirmation or recent full AV framework reproduction.',
                raw_event_reconstruction_repeated_here=False,real_source_protection_established=False,
                physical_witness_enabled=False,production_policy_changed=False,promotion_allowed=False)


def run(conditional,monitor,output):
    conditional=io.safe(conditional);monitor=io.safe(monitor);output=io.safe(output)
    for p in (conditional,monitor):c.require(output!=p and output not in p.parents and p not in output.parents,'Output overlaps inputs')
    c.require(output!=HERE and output not in HERE.parents,'Output overlaps source module')
    a,b,pooled,evidence=sources(conditional,monitor);locked=freeze(conditional,monitor)
    output.mkdir(parents=True,exist_ok=True);lock=output/'input_lock.json'
    if lock.exists():equal(io.read(lock),locked,'frozen_inputs_changed')
    else:
        c.require(not any(output.iterdir()),'Nonempty output without lock');io.write(lock,locked)
    import verify_scene_comparison
    if (output/'analysis.json').exists():verified=verify_scene_comparison.verify(conditional,monitor,output)
    else:
        pairs=c.Pairs();calibration=c.Calibration();totals={};count=0
        with monitor_io.gzip_writer(output/'paired_scene_gates.jsonl.gz') as stream:
            for left,right in scene_records(conditional,monitor):
                record=flags_record(left,right,totals,calibration)
                for ci,case in enumerate(c.CASES):
                    for index,(candidate,comparator) in enumerate(c.PAIRS):
                        for gi,group in enumerate(c.GROUPS):
                            pairs.add(case,index,group,record['gates'][ci][c.POLICIES.index(candidate)][gi],record['gates'][ci][c.POLICIES.index(comparator)][gi],record['requests'][gi])
                stream.write(json.dumps(record,separators=(',',':'),allow_nan=False)+'\n');count+=1
                if count%50==0:print('Audited saved scenes: '+str(count)+'/'+str(b['summary']['scene_count']),flush=True)
        c.require(count==b['summary']['scene_count'],'Missing source scenes')
        for (case,policy,group),values in totals.items():equal(values,counted(pooled[case][policy][group]),'scene_to_pooled_counts')
        c.require(freeze(conditional,monitor)==locked,'Sources/code changed during audit')
        analysis=make_analysis(a,b,pooled,evidence,pairs.report(),calibration.report(),count,io.digest(lock))
        io.write(output/'analysis.json',analysis)
        io.write(output/'result_hashes.json',{n:io.digest(output/n) for n in ('input_lock.json','analysis.json','paired_scene_gates.jsonl.gz')})
        verified=verify_scene_comparison.verify(conditional,monitor,output)
    io.write(output/'verification.json',verified)
    io.write(output/'run_report.json',dict(analysis=io.read(output/'analysis.json'),verification=verified))
    print(json.dumps(verified,sort_keys=True),flush=True);print('SCENE_COMPARISON_COMPLETE_NO_PROMOTION',flush=True)
    print('Upload: '+str(output/'run_report.json'),flush=True)
    return verified


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--conditional-dir',type=Path,default=HERE.parent/'vehicle_conditional_source'/'local_batch')
    p.add_argument('--monitor-dir',type=Path,default=HERE.parent/'vehicle_monitor_comparison'/'local_batch_numeric_fix')
    p.add_argument('--output-dir',type=Path,default=HERE/'local_analysis')
    args=p.parse_args()
    try:run(args.conditional_dir,args.monitor_dir,args.output_dir)
    except (ValueError,OSError,KeyError,TypeError,EOFError,OverflowError) as e:
        print('SCENE_COMPARISON_STOPPED: '+str(e),file=sys.stderr);sys.exit(1)
