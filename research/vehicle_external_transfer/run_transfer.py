"""Offline rerun, or selective download, of the frozen Civic proxy transfer test."""
import argparse, hashlib, json, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import transfer_core as io
HERE=Path(__file__).resolve().parent
for folder in ('vehicle_conditional_source','vehicle_monitor_comparison','vehicle_density_comparison'):
    sys.path.insert(0,str(HERE.parent/folder))
import conditional_core as conditional
import monitor_core as monitor
import density_core as density
import verify_conditional as check_source
import verify_comparison as check_monitor
import verify_density as check_density
equal=check_density.equal
FAMILIES=((conditional,check_source),(monitor,check_monitor),(density,check_density))

def freeze():
    paths=list(HERE.glob('*.py'))+[HERE/'selection.json',HERE/'protocol.json']
    # Include every imported project Python file; no sensor payloads are visited here.
    for folder in ('vehicle_can_corroboration','vehicle_can_attack_audit','vehicle_conditional_source',
                   'vehicle_monitor_comparison','vehicle_density_comparison'):
        paths+=list((HERE.parent/folder).glob('*.py'))
    return {str(p.relative_to(HERE.parent)):io.digest(p) for p in sorted(paths)}

def location(cache,record,entry):
    io.validate_member(entry['path'])
    return io.safe(cache/record['local_id']/('_'.join(entry['path'].split('/')[4:])+'.npy'))

def rows_from(cache,record):
    arrays=[io.load_array(location(cache,record,e).read_bytes()) for e in record['entries']]
    rows,audit=io.align(arrays[2],arrays[3],arrays[0],arrays[1])
    # Separate linear causal join and median implementation check the adapter.
    expected=[];j=-1;wt=[round(float(x)*1e6) for x in arrays[0]]
    for timestamp,values in zip(arrays[2],arrays[3]):
        t=round(float(timestamp)*1e6)
        while j+1<len(wt) and wt[j+1]<=t:j+=1
        stamp=wt[j] if j>=0 else None;wheel=None
        if j>=0 and t-stamp<=100000:
            v=arrays[1][j]
            if bool(io.np.all(io.np.isfinite(v))) and bool(io.np.all(v>=0)):wheel=float(io.np.median(v))
        ref=float(values[2]);ref=ref if io.math.isfinite(ref) and ref>=0 else None
        expected.append(dict(pose_utime=t,pose_speed_mps=ref,wheel_utime=stamp,wheel_speed_mps=wheel))
    equal(rows,expected,'independent_causal_adapter')
    if not rows:raise ValueError('Empty selected recording; no silent exclusion')
    return rows,audit

def bank(check=False):
    return {case:[(c.ScalarMetrics() if c is check_source else c.Metrics()) if check else m.Metrics()
                  for m,c in FAMILIES] for case in conditional.CASES}

def source_add(metric,e,r,group):
    metric.add(e[5],e[6],r['pose_speed_mps'],r['wheel_speed_mps'],group)

def report(b):
    return {case:{p:v for family in families for p,v in family.report().items()} for case,families in b.items()}

def evaluate(cache,selection):
    pooled=bank();oracle=bank(True);records=[];evaluation_rows=0;decisions=0
    for number,record in enumerate(selection['recordings'],1):
        rows,audit=rows_from(cache,record);local=bank();other=bank(True);h=hashlib.sha256();calibration={}
        first=rows[0]['pose_utime'];use=record['split']=='evaluation'
        for case in conditional.CASES:
            calibration[case]={}
            for k,(model,checker) in enumerate(FAMILIES):
                if k==0:events=model.events(rows,case);expected=checker.reconstruct(rows,case)
                else:
                    events,cal=model.events(rows,case);expected,calref=checker.reconstruct(rows,case)
                    equal(cal,calref,record['local_id']+'.'+case+'.calibration')
                    calibration[case][model.__name__]=cal
                equal(events,expected,record['local_id']+'.'+case+'.'+model.__name__)
                h.update(json.dumps([case,model.__name__,events],separators=(',',':'),allow_nan=False).encode())
                for e,ref in zip(events,expected):
                    i=e[0][0] if k==0 else e[0];r=rows[i];g=conditional.old.phase(r['pose_utime'],first)
                    targets=[(local[case][k],e),(other[case][k],ref)]
                    if use:targets += [(pooled[case][k],e),(oracle[case][k],ref)]
                    for metric,event in targets:
                        if k==0:source_add(metric,event,r,g)
                        else:metric.add(event,r['pose_speed_mps'],r['wheel_speed_mps'],g)
        metrics=report(local);equal(metrics,report(other),'independent_recording_metrics')
        records.append(dict(local_id=record['local_id'],route=record['route'],segment=record['segment'],day=record['day'],
                            split=record['split'],alignment=audit,event_digest_sha256=h.hexdigest(),calibration=calibration,case_metrics=metrics))
        if use:evaluation_rows+=len(rows);decisions+=len(rows)*14*11
        print(f'Checked external recordings: {number}/{len(selection["recordings"])}',flush=True)
    pooled_report=report(pooled);equal(pooled_report,report(oracle),'independent_pooled_metrics')
    return dict(protocol_id='external-civic-proxy-transfer-v1',recordings=records,
                evaluation_recordings=sum(r['split']=='evaluation' for r in records),
                evaluation_dates=sorted({r['day'] for r in records if r['split']=='evaluation'}),
                evaluation_reference_rows=evaluation_rows,evaluation_policy_decisions=decisions,
                evaluation_case_metrics=pooled_report,development_excluded_from_pooled=True,
                interpretation='Receiver GNSS-speed disagreement and synthetic protected-wheel-copy ablation. Public external transfer; no certified physical truth, acquisition authentication, native full AV-framework reproduction, or independence/significance claim.',
                production_policy_changed=False,physical_witness_enabled=False,promotion_allowed=False,
                real_source_protection_established=False,final_physical_validation_complete=False)

def run(cache,output,download=False):
    cache=io.safe(cache);output=io.safe(output)
    if cache==output or cache in output.parents or output in cache.parents:raise ValueError('Output/cache overlap')
    if output==HERE or output in HERE.parents:raise ValueError('Output overlaps module source')
    selection=json.loads((HERE/'selection.json').read_text());code=freeze()
    output.mkdir(parents=True,exist_ok=True);lock=output/'pre_payload_lock.json'
    if lock.exists():equal(json.loads(lock.read_text()),code,'frozen_code_or_selection_changed')
    else:
        if any(output.iterdir()):raise ValueError('Nonempty result directory without frozen lock')
        io.write(lock,code)
    jobs=[(r,e,location(cache,r,e)) for r in selection['recordings'] for e in r['entries']]
    if download:
        with ThreadPoolExecutor(max_workers=8) as pool:
            pending=[pool.submit(io.download,e,selection,p) for r,e,p in jobs]
            for i,f in enumerate(as_completed(pending),1):
                f.result()
                if i%8==0 or i==len(jobs):print(f'Public speed arrays ready: {i}/{len(jobs)}',flush=True)
    inputs={str(p.relative_to(cache)):io.digest(p) for _,_,p in jobs}
    for r,e,p in jobs:
        b=p.read_bytes()
        if len(b)!=e['size'] or io.zlib.crc32(b)!=e['crc']:raise ValueError('Cache size/CRC changed')
    input_lock=dict(code_sha256=code,payload_sha256=inputs)
    if (output/'input_lock.json').exists():equal(json.loads((output/'input_lock.json').read_text()),input_lock,'input_payload_changed')
    else:io.write(output/'input_lock.json',input_lock)
    result=evaluate(cache,selection)
    equal(freeze(),code,'code_changed_during_run')
    equal({str(p.relative_to(cache)):io.digest(p) for _,_,p in jobs},inputs,'payload_changed_during_run')
    if (output/'summary.json').exists():equal(json.loads((output/'summary.json').read_text()),result,'saved_report_reverification')
    else:io.write(output/'summary.json',result)
    verification=dict(verification_passed=True,method='Separate linear causal join; independent source/attack, CUSUM/EWMA and density reconstruction; every decision and calibration compared; separate per-recording/pooled reductions',
                      verified_recordings=len(selection['recordings']),evaluation_recordings=result['evaluation_recordings'],
                      evaluation_policy_decisions=result['evaluation_policy_decisions'],input_lock_sha256=io.digest(output/'input_lock.json'),
                      whole_archive_hash_verified=False,source_acquisition_authenticated=False,promotion_allowed=False)
    io.write(output/'verification.json',verification)
    io.write(output/'result_hashes.json',{n:io.digest(output/n) for n in ('pre_payload_lock.json','input_lock.json','summary.json','verification.json')})
    print(json.dumps(verification,sort_keys=True));print('EXTERNAL_PROXY_TRANSFER_COMPLETE_NO_PROMOTION')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache-dir',type=Path,default=HERE/'public_speed_arrays')
    p.add_argument('--output-dir',type=Path,default=HERE/'external_results')
    p.add_argument('--download',action='store_true',help='Download only the 84 frozen Civic NPY members using exact HTTP ranges')
    a=p.parse_args()
    try:run(a.cache_dir,a.output_dir,a.download)
    except (ValueError,OSError,KeyError,TypeError,ArithmeticError) as e:
        print('EXTERNAL_TRANSFER_STOPPED: '+str(e),file=sys.stderr);sys.exit(1)
