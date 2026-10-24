"""Retrospective diagnosis of a frozen external proxy test; never a policy fit."""
import argparse,bisect,csv,hashlib,io,json,math,statistics,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
EXTERNAL=HERE.parent/'vehicle_external_transfer'
sys.path.insert(0,str(EXTERNAL))
import run_transfer as prior
safe=prior.io.safe
CLAIMS=dict(independent_physical_truth_established=False,gnss_fault_proven=False,wheel_fault_proven=False,
            policy_changed=False,threshold_changed=False,promotion_allowed=False,physical_witness_enabled=False,
            untouched_confirmation_claimed=False)

def runs(flags):
    result=[];start=None
    for i,flag in enumerate(list(flags)+[False]):
        if flag and start is None:start=i
        elif not flag and start is not None:result.append((start,i-1));start=None
    return result

def longest_equal(values,times):
    if not values:return None
    groups=[];start=0
    for i in range(1,len(values)+1):
        if i==len(values) or values[i]!=values[start]:groups.append((start,i-1));start=i
    a,b=max(groups,key=lambda pair:(pair[1]-pair[0],-pair[0]))
    return dict(first_row=a,last_row=b,rows=b-a+1,elapsed_us=times[b]-times[a],speed_mps=values[a])

def surface_distance(lat1,lon1,lat2,lon2):
    a,b=math.radians(lat1),math.radians(lat2);dl=math.radians(lon2-lon1)
    q=math.sin((b-a)/2)**2+math.cos(a)*math.cos(b)*math.sin(dl/2)**2
    return 6371008.8*2*math.asin(math.sqrt(min(1.,max(0.,q))))

def describe(wheel,gnss,age,wheels):
    return dict(signed_wheel_minus_gnss_mps=wheel-gnss,absolute_gnss_disagreement_mps=abs(wheel-gnss),
                large_gnss_disagreement=abs(wheel-gnss)>2,recorded_age_gate_pass=0<=age<=100000,
                wheel_log_age_us=age,four_wheel_spread_mps=max(wheels)-min(wheels))

def stats(values):
    return dict(n=len(values),minimum=min(values),median=statistics.median(values),maximum=max(values)) if values else None

def json_bytes(value):return (json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()

def immutable_write(path,b):
    p=safe(path)
    if p.exists():
        if p.read_bytes()!=b:raise ValueError('Saved audit differs; old outputs preserved: '+p.name)
    else:p.write_bytes(b)

def check_inputs():
    manifest=json.loads(safe(EXTERNAL/'sha256.json').read_text())
    for name,h in manifest.items():
        p=safe(EXTERNAL/name)
        if EXTERNAL.resolve() not in p.parents:raise ValueError('External inventory path escapes module')
        if prior.io.digest(p)!=h:raise ValueError('Frozen external evidence changed: '+name)
    for name,h in json.loads(safe(EXTERNAL/'external_results/pre_payload_lock.json').read_text()).items():
        if prior.io.digest(safe(HERE.parent/name))!=h:raise ValueError('Original frozen implementation changed')
    return dict(external_inventory_sha256=prior.io.digest(EXTERNAL/'sha256.json'),
                original_input_lock_sha256=prior.io.digest(EXTERNAL/'external_results/input_lock.json'),
                audit_code_sha256={p.name:prior.io.digest(p) for p in sorted(HERE.glob('*.py'))},
                diagnosis_contract_sha256=prior.io.digest(HERE/'diagnostic_contract.json'))

def analyze():
    selection=json.loads((EXTERNAL/'selection.json').read_text());old=json.loads((EXTERNAL/'external_results/summary.json').read_text())
    records=[];failures=[];reproduced=0;checked_rows=0
    for record,previous in zip(selection['recordings'],old['recordings']):
        if record['local_id']!=previous['local_id']:raise ValueError('Recording order/identity changed')
        arrays=[prior.io.load_array(prior.location(EXTERNAL/'public_speed_arrays',record,e).read_bytes()) for e in record['entries']]
        rows,alignment=prior.rows_from(EXTERNAL/'public_speed_arrays',record)
        events=prior.conditional.events(rows,'nominal');oracle=prior.check_source.reconstruct(rows,'nominal')
        prior.equal(events,oracle,'separate_nominal_reconstruction');checked_rows+=len(rows)
        wheel_times=prior.io.timestamps(arrays[0]);query_times=[r['pose_utime'] for r in rows]
        flags=[];local=[];independent_bad=0;position_checks=[]
        for i,(row,event) in enumerate(zip(rows,events)):
            output=event[5][2];gnss=row['pose_speed_mps'];wheel=row['wheel_speed_mps']
            bad=output is not None and gnss is not None and abs(output-gnss)>2;flags.append(bad)
            # Independently calculate the nominal bounded selector, without choose()/events().
            j=bisect.bisect_right(wheel_times,row['pose_utime'])-1;actual=None;raw=[]
            if j>=0 and 0<=row['pose_utime']-wheel_times[j]<=100000:
                raw=[float(x) for x in arrays[1][j]]
                if all(math.isfinite(x) and x>=0 for x in raw):
                    ordered=sorted(raw);c=(ordered[1]+ordered[2])/2;a=c+(.25 if i%2 else 0)
                    actual=c if abs(c-a)+.25<=.5 else a
            prior.equal(output,actual,'independent_raw_array_selector')
            independent_bad+=actual is not None and gnss is not None and abs(actual-gnss)>2
            if i and all(math.isfinite(float(x)) for x in arrays[3][i-1:i+1,:2].flat):
                # Position and speed come from the same receiver; this is not independent truth.
                lat1,lon1=map(float,arrays[3][i-1,:2]);lat2,lon2=map(float,arrays[3][i,:2])
                if -90<=lat1<=90 and -90<=lat2<=90 and -180<=lon1<=180 and -180<=lon2<=180:
                    v=surface_distance(lat1,lon1,lat2,lon2)/((query_times[i]-query_times[i-1])/1e6)
                    if bad:position_checks.append(dict(row=i,position_step_speed_using_log_dt_mps=v,
                                                       reported_gnss_speed_mps=gnss,independent_truth=False))
            if bad:
                if not raw or wheel is None:raise ValueError('Unexpected missing selected wheel')
                detail=dict(local_id=record['local_id'],day=record['day'],split=record['split'],row=i,
                            elapsed_us=row['pose_utime']-query_times[0],query_us=row['pose_utime'],
                            wheel_timestamp_us=row['wheel_utime'],wheel_speed_mps=wheel,gnss_speed_mps=gnss,
                            selected_output_mps=output,selection_code=event[6][2],
                            assumed_anchor_mps=event[1],assumed_u_mps=event[2],
                            output_deviation_from_assumed_anchor_mps=abs(output-event[1]),
                            output_deviation_from_original_wheel_mps=abs(output-wheel),
                            **describe(wheel,gnss,row['pose_utime']-row['wheel_utime'],raw))
                detail.update({f'wheel_{k}_mps':v for k,v in enumerate(raw)})
                local.append(detail)
                if record['split']=='evaluation':failures.append(detail)
        prior.equal(independent_bad,sum(flags),'independent_failure_counter')
        prior.equal(sum(flags),previous['case_metrics']['nominal']['conditional_bounded_candidate']['all']['accepted_disagreement_gt2_count'],'original_nominal_failure_count')
        # Raw UTC field is retained; its observed scale is diagnostic, never a clock correction.
        utc=[float(v) for v in arrays[3][:,3]]
        ratios=[(b-a)/((query_times[i+1]-query_times[i])/1e6) for i,(a,b) in enumerate(zip(utc,utc[1:])) if math.isfinite(a) and math.isfinite(b) and b>a]
        clusters=[dict(first_row=a,last_row=b,rows=b-a+1,elapsed_span_us=query_times[b]-query_times[a]) for a,b in runs(flags)]
        records.append(dict(local_id=record['local_id'],split=record['split'],day=record['day'],alignment=alignment,
                            nominal_bounded_large_disagreements=sum(flags),failure_clusters=clusters,
                            failure_wheel_age_us=stats([x['wheel_log_age_us'] for x in local]),
                            failure_four_wheel_spread_mps=stats([x['four_wheel_spread_mps'] for x in local]),
                            longest_exact_gnss_speed_run=longest_equal([float(x) for x in arrays[3][:,2]],query_times),
                            raw_utc_units_per_boot_second=stats(ratios),
                            same_receiver_position_checks_on_failures=position_checks))
        if record['split']=='evaluation':reproduced+=sum(flags)
    expected=old['evaluation_case_metrics']['nominal']['conditional_bounded_candidate']['all']['accepted_disagreement_gt2_count']
    prior.equal(reproduced,expected,'original_pooled_failure_count')
    prior.equal(len(records),len(selection['recordings']),'recording_denominator')
    return dict(analysis_type='Retrospective source-conflict diagnosis of already exposed data; not new confirmation',
                checked_recordings=len(records),checked_nominal_rows=checked_rows,evaluation_failure_count=reproduced,
                affected_evaluation_recordings=sum(r['split']=='evaluation' and r['nominal_bounded_large_disagreements']>0 for r in records),
                all_failure_recorded_ages_pass=all(x['recorded_age_gate_pass'] for x in failures),
                all_failure_outputs_equal_original_wheel=all(x['output_deviation_from_original_wheel_mps']==0 for x in failures),
                failure_age_us=stats([x['wheel_log_age_us'] for x in failures]),
                failure_wheel_spread_mps=stats([x['four_wheel_spread_mps'] for x in failures]),
                records=records,claims=CLAIMS,
                interpretation='The fixed nominal selector and causal adapter reproduce the failures directly from source arrays. Four-wheel agreement, receipt/log freshness and a synthetic wheel-copy bound do not establish agreement to GNSS or physical motion. Repeated receiver speed is an observation, not proof of stale acquisition or GNSS fault.'),failures

def run(output):
    output=safe(output)
    if output==HERE or output in HERE.parents or output==EXTERNAL or EXTERNAL in output.parents or output in EXTERNAL.parents:
        raise ValueError('Audit output overlaps evidence/source root')
    frozen=check_inputs();output.mkdir(parents=True,exist_ok=True)
    immutable_write(output/'input_lock.json',json_bytes(frozen))
    result,failures=analyze();prior.equal(check_inputs(),frozen,'unchanged_evidence_after_diagnosis')
    text=io.StringIO(newline='');writer=csv.DictWriter(text,fieldnames=list(failures[0]) if failures else ['row']);writer.writeheader();writer.writerows(failures)
    immutable_write(output/'failures.csv',text.getvalue().encode())
    immutable_write(output/'summary.json',json_bytes(result))
    verification=dict(verification_passed=True,checked_recordings=result['checked_recordings'],checked_nominal_rows=result['checked_nominal_rows'],
                      reproduced_evaluation_failures=result['evaluation_failure_count'],external_evidence_unchanged=True,
                      independent_raw_selector_checked=True,independent_physical_truth_established=False,policy_changed=False,promotion_allowed=False)
    immutable_write(output/'verification.json',json_bytes(verification))
    immutable_write(output/'result_hashes.json',json_bytes({n:prior.io.digest(output/n) for n in ('input_lock.json','summary.json','failures.csv','verification.json')}))
    print(json.dumps(verification,sort_keys=True));print('SOURCE_CONFLICT_DIAGNOSIS_COMPLETE_NO_PROMOTION')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,default=HERE/'audit_results');a=p.parse_args()
    try:run(a.output_dir)
    except (ValueError,OSError,KeyError,TypeError,ArithmeticError) as e:print('SOURCE_AUDIT_STOPPED: '+str(e),file=sys.stderr);sys.exit(1)
