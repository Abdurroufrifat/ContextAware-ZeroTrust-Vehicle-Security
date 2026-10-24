"""Fixed classical monitoring adaptations; run and independently verify in one command."""
import argparse,gzip,io as textio,json,os,sys
from contextlib import contextmanager
from pathlib import Path
import monitor_core as m
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'vehicle_conditional_source'))
import run_conditional as previous
io=previous.baseio
equal=previous.baseverify.equal
CODE=('monitor_core.py','run_comparison.py','verify_comparison.py','test_comparison.py','protocol.json','sources.json','historical_full_summary.json','historical_pilot_summary.json')


@contextmanager
def gzip_writer(path):
    """Finish gzip footer and sync bytes before exposing a completed result."""
    path=io.safe(path);temporary=path.with_name(path.name+'.tmp')
    with temporary.open('wb') as raw:
        with gzip.GzipFile(filename=path.name,mode='wb',fileobj=raw,mtime=0) as compressed:
            with textio.TextIOWrapper(compressed,encoding='utf-8',newline='\n') as text:
                yield text
        raw.flush();os.fsync(raw.fileno())
    os.replace(temporary,path)


def freeze(baseline):
    imported={}
    for folder in ('vehicle_can_corroboration','vehicle_can_attack_audit','vehicle_conditional_source'):
        for p in sorted((HERE.parent/folder).glob('*.py')):imported[folder+'/'+p.name]=io.digest(p)
    return dict(protocol_id='classical-monitor-comparison-v2-numerics',
                baseline_sha256={n:io.digest(baseline/n) for n in ('input_snapshot.json','batch_summary.json','result_hashes.json','verification.json')},
                code_sha256={n:io.digest(HERE/n) for n in CODE},imported_code_sha256=imported)


def historical(base):
    for name in ('historical_full_summary.json','historical_pilot_summary.json'):
        prior=io.read(HERE/name)
        if prior['summary']['baseline_input_snapshot_sha256']==base['input_snapshot_sha256']:
            if prior['summary']['scene_count']!=base['scene_count'] or prior['summary']['baseline_pose_rows']!=base['pooled']['pose_rows']:
                raise ValueError('Historical inventory count differs')
            return dict(applicable=True,source_file=name,source_sha256=io.digest(HERE/name),
                        evidence_scope=prior['provenance']['kind'],raw_historical_reverification_claimed=False,
                        comparison_by_track=dict(unprotected=['legacy_agreement'],protected=['conditional_anchor_only','conditional_bounded_candidate']),
                        case_metrics=prior['summary']['case_metrics'])
    return dict(applicable=False,reason='Baseline snapshot differs from both frozen historical inventories')


def summary(base,metrics,scene_count,lock):
    return dict(protocol_id='classical-monitor-comparison-v2-numerics',baseline_input_snapshot_sha256=base['input_snapshot_sha256'],
                input_lock_sha256=lock,scene_count=scene_count,baseline_pose_rows=base['pooled']['pose_rows'],
                unknown_pose_denominator_scenes=base['unknown_pose_denominator_scenes'],cases=list(m.CASES),policy_names=list(m.POLICIES),
                case_metrics=metrics,historical_comparison=historical(base),
                interpretation='Classical monitoring equations with project-specific causal calibration and speed-policy wrappers. Same-information tracks; simulated protected copy; localization outcome proxy. Seen development cohort, no recent full AV method reproduction or superiority claim.',
                conditional_simulation_only=True,real_source_protection_established=False,native_recent_av_method_reproduction_complete=False,
                physical_witness_enabled=False,production_policy_changed=False,promotion_allowed=False)


def run(data,baseline,output):
    data=io.safe(data);baseline=io.safe(baseline);output=io.safe(output)
    for p in (data,baseline):
        if output==p or output in p.parents or p in output.parents:raise ValueError('Output overlaps inputs')
    if output==HERE or output in HERE.parents:raise ValueError('Output overlaps source module')
    checked=previous.baseverify.verify(data,baseline)
    equal(io.read(baseline/'verification.json'),checked,'baseline_verification')
    frozen=freeze(baseline);output.mkdir(parents=True,exist_ok=True);lock=output/'input_lock.json'
    if lock.exists():equal(io.read(lock),frozen,'frozen_input_changed_no_automatic_rescore')
    else:
        if any(output.iterdir()):raise ValueError('Nonempty output without input lock')
        io.write(lock,frozen)
    import verify_comparison
    if (output/'summary.json').exists() or (output/'result_hashes.json').exists():
        verified=verify_comparison.verify(data,baseline,output,baseline_checked=True)
    else:
        base=io.read(baseline/'batch_summary.json');pooled={c:m.Metrics() for c in m.CASES};names=[]
        with gzip_writer(output/'scene_metrics.jsonl.gz') as stats:
            for number,s in enumerate(base['scenes'],1):
                rows=previous.ledger(baseline,s['scene']);times=[r['pose_utime'] for r in rows if type(r.get('pose_utime')) is int]
                first=times[0] if times else None;per={};cal={};delay={}
                name=s['scene']+'_events.jsonl.gz';names.append(name)
                with gzip_writer(output/name) as stream:
                    for ci,case in enumerate(m.CASES):
                        local=m.Metrics();events,cal[case]=m.events(rows,case);delay[case]={p:None for p in m.POLICIES}
                        for event in events:
                            r=rows[event[0]];group=previous.attacks.phase(r.get('pose_utime'),first)
                            args=(event,r.get('pose_speed_mps'),r.get('wheel_speed_mps'),group)
                            local.add(*args);pooled[case].add(*args)
                            for j,p in enumerate(m.POLICIES):
                                if group=='during' and event[4][j] is True and delay[case][p] is None:
                                    delay[case][p]=r['pose_utime']-first-5000000
                            stream.write(json.dumps([ci]+event,separators=(',',':'),allow_nan=False)+'\n')
                        per[case]=local.report()
                stats.write(json.dumps(dict(scene=s['scene'],calibration=cal,first_during_alarm_elapsed_us=delay,metrics=per),separators=(',',':'),allow_nan=False)+'\n')
                if number%10==0 or number==base['scene_count']:print('Scored comparison scenes: '+str(number)+'/'+str(base['scene_count']),flush=True)
        if freeze(baseline)!=frozen:raise ValueError('Frozen code/input changed during scoring')
        io.write(output/'summary.json',summary(base,{c:pooled[c].report() for c in m.CASES},len(base['scenes']),io.digest(lock)))
        names+=['input_lock.json','summary.json','scene_metrics.jsonl.gz']
        io.write(output/'result_hashes.json',{n:io.digest(output/n) for n in names});del pooled
        verified=verify_comparison.verify(data,baseline,output,baseline_checked=True)
    io.write(output/'verification.json',verified)
    io.write(output/'run_report.json',dict(summary=io.read(output/'summary.json'),verification=verified))
    print(json.dumps(verified,sort_keys=True),flush=True)
    print('MONITOR_COMPARISON_COMPLETE_NO_PROMOTION',flush=True)
    print('Upload: '+str(output/'run_report.json'),flush=True)
    return verified


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,default=HERE.parents[2]/'data'/'nuscenes_canbus'/'can_bus')
    p.add_argument('--baseline-dir',type=Path,default=HERE.parent/'vehicle_can_corroboration'/'local_batch')
    p.add_argument('--output-dir',type=Path,default=HERE/'local_batch_numeric_fix')
    a=p.parse_args()
    try:run(a.data_dir,a.baseline_dir,a.output_dir)
    except (ValueError,OSError,KeyError,TypeError,OverflowError) as e:
        print('MONITOR_COMPARISON_STOPPED: '+str(e),file=sys.stderr);sys.exit(1)
