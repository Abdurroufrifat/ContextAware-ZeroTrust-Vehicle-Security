"""Independent scalar routing/metrics verifier over exact-hash prior replays.

Does not call the selector or summary reducer. Prior feature/model/guard
verification is reused by immutable input hashes, not repeated here.
"""
from pathlib import Path
import bisect,hashlib,io,json,math,sys,unittest,zipfile
import numpy as np

ROOT=Path(__file__).resolve().parent
RESEARCH=ROOT.parent
PROJECT=RESEARCH.parent
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:raise AssertionError('archive CRC '+str(path))
    with np.load(path,allow_pickle=False) as archive:return {key:archive[key] for key in archive.files}
def close(a,b,path=''):
    if isinstance(a,dict):
        if not isinstance(b,dict) or set(a)!=set(b):raise AssertionError('keys '+path)
        for key in a:close(a[key],b[key],path+'/'+str(key))
    elif isinstance(a,list):
        if len(a)!=len(b):raise AssertionError('length '+path)
        for i,(x,y) in enumerate(zip(a,b)):close(x,y,path+'/'+str(i))
    elif isinstance(a,float):
        if not isinstance(b,(float,int)) or not math.isclose(a,b,rel_tol=0,abs_tol=1e-9):raise AssertionError('number '+path+' '+str((a,b)))
    elif a!=b:raise AssertionError('value '+path+' '+str((a,b)))

def quantile(values,q):
    values=sorted(values);at=(len(values)-1)*q;i=int(at)
    return values[i]+(values[min(i+1,len(values)-1)]-values[i])*(at-i)
def scalar_stats(errors,missing):
    n=len(errors)
    return {'count':n,'missing_truth':missing,'mae_mps':math.fsum(errors)/n if n else None,
            'rmse_mps':math.sqrt(math.fsum(v*v for v in errors)/n) if n else None,
            'p95_mps':quantile(errors,.95) if n else None,'maximum_mps':max(errors) if n else None,
            'above_2_mps':sum(v>2. for v in errors)}

def new_cell():
    return {'requests':0,'routes':[0]*3,'primary':[0]*5,'radar':[0]*5,'errors':[],'missing':0,'steps':[],'switches':0}

def scalar_comparison(primary,radar,enable,times,truth,groups,names):
    n=len(times);fields=('status','prediction_mps','latest_endpoint','age_ns')
    out={'route':[],'status':[],'prediction_mps':[],'latest_endpoint':[],'age_ns':[],'primary_status':[],'radar_status':[]}
    cells={name:new_cell() for name in names};previous=None
    for i,t in enumerate(times):
        pstatus=int(primary['status'][i]);rstatus=int(radar['status'][i])
        if not 0<=pstatus<=4 or not 0<=rstatus<=4:raise AssertionError('invalid input status')
        for source,s in [(primary,pstatus),(radar,rstatus)]:
            if s==0 and (not math.isfinite(float(source['prediction_mps'][i])) or float(source['prediction_mps'][i])<0 or int(source['latest_endpoint'][i])<0 or not 0<=int(source['age_ns'][i])<=100_000_000):raise AssertionError('invalid accepted input')
        choice=primary if pstatus==0 else radar if enable and rstatus==0 else None
        route=1 if pstatus==0 else 2 if choice is not None else 0
        p=float(choice['prediction_mps'][i]) if choice is not None else math.nan
        out['route'].append(route);out['status'].append(0 if choice is not None else pstatus)
        out['prediction_mps'].append(p);out['latest_endpoint'].append(int(choice['latest_endpoint'][i]) if choice is not None else -1)
        out['age_ns'].append(int(choice['age_ns'][i]) if choice is not None else -1);out['primary_status'].append(pstatus);out['radar_status'].append(rstatus)
        current_groups=groups[i]
        for name in current_groups:
            cell=cells[name];cell['requests']+=1;cell['routes'][route]+=1;cell['primary'][pstatus]+=1;cell['radar'][rstatus]+=1
            if route:
                if math.isfinite(truth[i]):cell['errors'].append(abs(p-truth[i]))
                else:cell['missing']+=1
        if route and previous is not None:
            old_t,old_route,old_p,old_groups=previous
            if old_route and t-old_t<=50_000_000:
                for name in current_groups:
                    if name in old_groups:
                        cells[name]['steps'].append(abs(p-old_p));cells[name]['switches']+=route!=old_route
        previous=(t,route,p,current_groups)
    summaries={}
    for name,cell in cells.items():
        accepted=cell['routes'][1]+cell['routes'][2];availability=accepted/cell['requests'];errors=scalar_stats(cell['errors'],cell['missing']);p95=errors['p95_mps']
        gates={'availability_ge_95pct':availability>=.95,'p95_le_0_5_mps':p95 is not None and p95<=.5,
               'no_missing_accepted_truth':errors['missing_truth']==0,'no_accepted_error_above_2_mps':errors['above_2_mps']==0}
        summaries[name]={'requests':cell['requests'],'accepted_requests':accepted,'availability':availability,
                         'route_counts':{str(k):v for k,v in enumerate(cell['routes'])},
                         'primary_status_counts':{str(k):v for k,v in enumerate(cell['primary'])},
                         'radar_status_counts':{str(k):v for k,v in enumerate(cell['radar'])},
                         'accepted_errors':errors,'gates':gates,'joint_gate_pass':all(gates.values()),
                         'adjacent_accepted_pairs':len(cell['steps']),'source_transitions':cell['switches'],
                         'adjacent_prediction_step_p95_mps':quantile(cell['steps'],.95) if cell['steps'] else None,
                         'adjacent_prediction_step_maximum_mps':max(cell['steps']) if cell['steps'] else None}
    return out,summaries

def verify():
    protocol=read(ROOT/'protocol.json');lock=read(ROOT/'analysis_lock.json');result=read(ROOT/'results.json')
    if lock['protocol_sha256']!=digest(ROOT/'protocol.json'):raise AssertionError('protocol changed')
    for name,h in lock['code_sha256'].items():
        if digest(ROOT/name)!=h:raise AssertionError('frozen code changed '+name)
    for name,h in protocol['input_sha256'].items():
        if 'urban02' in name.lower() or 'rav4' in name.lower() or name.startswith(('vehicle_final_evaluation/','research/vehicle_final_evaluation/')):raise AssertionError('protected input')
        if digest(RESEARCH/name)!=h:raise AssertionError('input changed '+name)
    cases=('nominal','dropout','stale_header','future_header','accel_bias','gyro_bias','frozen_values','tiny_jitter','threshold_aware_pulses')
    methods=['radar_only']+[model+'__'+guard+'__'+strategy for model in ('radar_imu_parent','nonlinear7') for guard in ('none','exact','deadband') for strategy in ('withhold','fallback')]
    close(list(cases),protocol['cases']);close(methods,protocol['methods'])
    refs=read(RESEARCH/'vehicle_availability_audit/request_reference.json');times=[int(v['time_ns']) for v in refs]
    truth=[math.nan if v['reference_mps'] is None else float(v['reference_mps']) for v in refs]
    frames=read(RESEARCH/'vehicle_masked_history/frame_index.json');events=[int(v['time_ns']) for v in frames]
    if len(times)!=37340 or len(events)!=40690 or sum(bool(v['ready']) for v in frames)!=40678:raise AssertionError('schedule changed')
    original=load(RESEARCH/'vehicle_multisource_imu/development_replay.npz');np.testing.assert_array_equal(times,original['radar_imu__request__request_times_ns'])
    duration=events[-1]-events[0];start=events[0]+duration*2//5;end=events[0]+duration*3//5
    groups=[('all','before' if t<start else 'during' if t<end else 'after','segment_'+str((t-times[0])//300_000_000_000)) for t in times]
    names=['all','before','during','after']+['segment_'+str(k) for k in sorted({(t-times[0])//300_000_000_000 for t in times})]
    prior=read(RESEARCH/'vehicle_deadband_guard/results.json');trials=cells=0;nominal_radar=None
    if set(result['cases'])!=set(cases):raise AssertionError('case omitted')
    for ordinal,case in enumerate(cases):
        old=ordinal<7
        paths={'none':RESEARCH/('vehicle_sensor_faults/replays/'+case+'.npz') if old else RESEARCH/('vehicle_deadband_guard/replays/'+case+'__base.npz'),
               'exact':RESEARCH/('vehicle_content_guard/replays/'+case+'.npz') if old else RESEARCH/('vehicle_deadband_guard/replays/'+case+'__exact.npz'),
               'deadband':RESEARCH/('vehicle_deadband_guard/replays/'+case+'__deadband.npz')}
        archives={key:load(path) for key,path in paths.items()};actual=load(ROOT/'replays'/(case+'.npz'))
        expected_members={method+'__'+field for method in methods for field in ('route','status','prediction_mps','latest_endpoint','age_ns','primary_status','radar_status')}
        if set(actual)!=expected_members or set(result['cases'][case])!=set(methods):raise AssertionError('policy/field omitted')
        def source(archive,model):return {key:archive[model+'__request__'+key] for key in ('status','prediction_mps','latest_endpoint','age_ns')}
        radar=source(archives['none'],'radar_only')
        if nominal_radar is None:nominal_radar=radar
        for key in radar:np.testing.assert_array_equal(radar[key],nominal_radar[key])
        for archive in archives.values():
            for key in radar:np.testing.assert_array_equal(radar[key],source(archive,'radar_only')[key])
            for model in ('radar_only','radar_imu_parent','nonlinear7'):
                b=source(archive,model)
                for i,t in enumerate(times):
                    latest=bisect.bisect_right(events,t)-1
                    if latest!=int(b['latest_endpoint'][i]) or t-events[latest]!=int(b['age_ns'][i]):raise AssertionError('source request alignment')
        for method in methods:
            if method=='radar_only':primary=radar;enable=False
            else:
                model,guard,strategy=method.split('__');primary=source(archives[guard],model);enable=strategy=='fallback'
            expected,summary=scalar_comparison(primary,radar,enable,times,truth,groups,names)
            for field,values in expected.items():np.testing.assert_allclose(values,actual[method+'__'+field],rtol=0,atol=0,equal_nan=True)
            close(summary,result['cases'][case][method],case+'/'+method)
            if method=='radar_only' or method.endswith('__withhold'):
                model='radar_only' if method=='radar_only' else method.split('__')[0];guard='none' if method=='radar_only' else method.split('__')[1]
                old_summary=prior['cases'][case][guard][model]['requests']
                for name,cell in summary.items():
                    for key in ('requests','accepted_requests','availability','accepted_errors'):close(cell[key],old_summary[name][key],case+'/'+method+'/'+name+'/'+key)
            trials+=len(times);cells+=len(summary)
        print('verified',case,flush=True)
    if result['decision']!='OFFLINE_RADAR_FALLBACK_COMPARISON_COMPLETE_NO_PROMOTION':raise AssertionError('decision changed')
    for flag in ('models_retrained','labels_changed','threshold_search','production_policy_changed','reserved_final_opened','promotion_allowed'):
        if result[flag] is not False:raise AssertionError('unsupported claim '+flag)
    for name,h in read(ROOT/'sha256.json').items():
        if digest(ROOT/name)!=h:raise AssertionError('artifact changed '+name)
    snapshot=read(ROOT/'prior_snapshot_manifest.json');manifest=read(PROJECT/'PROJECT_MANIFEST.json');verified=missing=protected=0
    for name,h in snapshot.items():
        if name!='FINAL_PROJECT_README.md' and manifest.get(name)!=h:raise AssertionError('prior manifest changed '+name)
    for name,h in manifest.items():
        if 'urban02' in name.lower() or 'rav4' in name.lower() or name.startswith('research/vehicle_final_evaluation/'):protected+=1;continue
        p=PROJECT/name
        if not p.is_file():missing+=1;continue
        if digest(p)!=h:raise AssertionError('project hash '+name)
        verified+=1
    sys.path.insert(0,str(ROOT));import test_fallback
    stream=io.StringIO();tests=unittest.TextTestRunner(stream=stream).run(unittest.defaultTestLoader.loadTestsFromModule(test_fallback))
    if not tests.wasSuccessful() or tests.testsRun!=13:raise AssertionError(stream.getvalue())
    print('RADAR FALLBACK VERIFICATION PASS:',trials,'policy-request checks;',cells,'metric cells;',tests.testsRun,'tests')
    print('PROJECT:',verified,'verified;',missing,'missing;',protected,'protected paths not opened')
    print('OFFLINE_RADAR_FALLBACK_COMPARISON_COMPLETE_NO_PROMOTION')
    print('ADAPTIVE-PULSE FAILURE / RADAR-PEER ASSUMPTIONS / REUSED DEVELOPMENT LIMITS RETAINED')

if __name__=='__main__':verify()
