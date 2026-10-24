"""Offline, source-preserving degraded speed serving; no live policy grants."""
from pathlib import Path
import hashlib,io,json,os,tempfile
import numpy as np

ROOT=Path(__file__).resolve().parent
RESEARCH=ROOT.parent
PROJECT=RESEARCH.parent
CASES=('nominal','dropout','stale_header','future_header','accel_bias','gyro_bias','frozen_values','tiny_jitter','threshold_aware_pulses')
MODELS=('radar_imu_parent','nonlinear7')
GUARDS=('none','exact','deadband')
FIELDS=('status','prediction_mps','latest_endpoint','age_ns')
METHODS=('radar_only',)+tuple(model+'__'+guard+'__'+strategy for model in MODELS for guard in GUARDS for strategy in ('withhold','fallback'))

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def permitted(name):return not ('urban02' in name.lower() or 'rav4' in name.lower() or name.startswith(('vehicle_final_evaluation/','research/vehicle_final_evaluation/')))
def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def atomic_npz(path,arrays):
    buffer=io.BytesIO();np.savez_compressed(buffer,**arrays);temporary=None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.fallback-',delete=False) as out:
            temporary=Path(out.name);out.write(buffer.getvalue());out.flush();os.fsync(out.fileno())
        os.replace(temporary,path)
    finally:
        if temporary is not None:temporary.unlink(missing_ok=True)

def validate_branch(branch):
    if set(branch)!=set(FIELDS):raise ValueError('four source serving fields required')
    arrays={k:np.asarray(branch[k]) for k in FIELDS};shape=arrays['status'].shape
    if len(shape)!=1 or any(v.shape!=shape for v in arrays.values()):raise ValueError('aligned one-dimensional source arrays required')
    for key in ('status','latest_endpoint','age_ns'):
        if not np.issubdtype(arrays[key].dtype,np.integer):raise ValueError('integer source clocks/status required')
    status=arrays['status'];p=arrays['prediction_mps'];accepted=status==0
    if np.any((status<0)|(status>4)):raise ValueError('unknown source status')
    if np.any(accepted&(~np.isfinite(p)|(p<0))):raise ValueError('eligible prediction must be finite and nonnegative')
    if np.any(accepted&((arrays['latest_endpoint']<0)|(arrays['age_ns']<0)|(arrays['age_ns']>100_000_000))):raise ValueError('eligible source endpoint must be present and fresh')
    return arrays

def select(primary,radar,allow_fallback):
    """Prefer eligible fusion; otherwise independently eligible radar if enabled.

    No labels, case identity, error estimate or future records are inputs.
    Route 0 abstains, 1 uses primary, 2 uses radar. Keep both original statuses.
    """
    primary=validate_branch(primary);radar=validate_branch(radar)
    if primary['status'].shape!=radar['status'].shape:raise ValueError('source request counts differ')
    use_primary=primary['status']==0
    use_radar=(~use_primary)&(radar['status']==0)&bool(allow_fallback)
    route=np.where(use_primary,1,np.where(use_radar,2,0)).astype(np.uint8)
    return {'route':route,'status':np.where(route>0,0,primary['status']).astype(np.uint8),
            'prediction_mps':np.where(use_primary,primary['prediction_mps'],np.where(use_radar,radar['prediction_mps'],np.nan)),
            'latest_endpoint':np.where(use_primary,primary['latest_endpoint'],np.where(use_radar,radar['latest_endpoint'],-1)),
            'age_ns':np.where(use_primary,primary['age_ns'],np.where(use_radar,radar['age_ns'],-1)),
            'primary_status':primary['status'].copy(),'radar_status':radar['status'].copy()}

def paths(case):
    old=case in CASES[:7]
    return {'none':RESEARCH/('vehicle_sensor_faults/replays/'+case+'.npz') if old else RESEARCH/('vehicle_deadband_guard/replays/'+case+'__base.npz'),
            'exact':RESEARCH/('vehicle_content_guard/replays/'+case+'.npz') if old else RESEARCH/('vehicle_deadband_guard/replays/'+case+'__exact.npz'),
            'deadband':RESEARCH/('vehicle_deadband_guard/replays/'+case+'__deadband.npz')}

def branch(archive,model):return {key:archive[model+'__request__'+key] for key in FIELDS}

def cohort():
    refs=read(RESEARCH/'vehicle_availability_audit/request_reference.json')
    times=np.array([v['time_ns'] for v in refs],np.int64)
    y=np.array([np.nan if v['reference_mps'] is None else v['reference_mps'] for v in refs],float)
    frames=read(RESEARCH/'vehicle_masked_history/frame_index.json');events=np.array([v['time_ns'] for v in frames],np.int64)
    original=load(RESEARCH/'vehicle_multisource_imu/development_replay.npz')
    np.testing.assert_array_equal(times,original['radar_imu__request__request_times_ns'])
    if len(times)!=37340 or len(events)!=40690 or sum(v['ready'] for v in frames)!=40678 or np.any(np.diff(times)<=0):raise ValueError('fixed schedule changed')
    duration=int(events[-1])-int(events[0]);start=int(events[0])+duration*2//5;end=int(events[0])+duration*3//5
    segment=(times-times[0])//300_000_000_000
    masks={'all':np.ones(len(times),bool),'before':times<start,'during':(times>=start)&(times<end),'after':times>=end,
           **{'segment_'+str(int(k)):segment==k for k in np.unique(segment)}}
    return times,y,events,masks

def error_stats(y,p,mask):
    valid=mask&np.isfinite(y)&np.isfinite(p);e=np.abs(p[valid]-y[valid])
    return {'count':len(e),'missing_truth':int(np.sum(mask&~np.isfinite(y))),
            'mae_mps':float(e.mean()) if len(e) else None,'rmse_mps':float(np.sqrt(np.mean(e**2))) if len(e) else None,
            'p95_mps':float(np.quantile(e,.95)) if len(e) else None,'maximum_mps':float(e.max()) if len(e) else None,'above_2_mps':int(np.sum(e>2.))}

def summarize(out,times,y,masks):
    summaries={}
    for name,mask in masks.items():
        accepted=mask&(out['route']>0);errors=error_stats(y,out['prediction_mps'],accepted)
        availability=float(accepted.sum()/mask.sum());p95=errors['p95_mps']
        routes={str(k):int(np.sum(mask&(out['route']==k))) for k in range(3)}
        pairs=mask[1:]&mask[:-1]&(out['route'][1:]>0)&(out['route'][:-1]>0)&(np.diff(times)<=50_000_000)
        steps=np.abs(np.diff(out['prediction_mps']))[pairs]
        transitions=int(np.sum(pairs&(out['route'][1:]!=out['route'][:-1])))
        gates={'availability_ge_95pct':availability>=.95,'p95_le_0_5_mps':p95 is not None and p95<=.5,
               'no_missing_accepted_truth':errors['missing_truth']==0,'no_accepted_error_above_2_mps':errors['above_2_mps']==0}
        summaries[name]={'requests':int(mask.sum()),'accepted_requests':int(accepted.sum()),'availability':availability,'route_counts':routes,
                         'primary_status_counts':{str(k):int(np.sum(mask&(out['primary_status']==k))) for k in range(5)},
                         'radar_status_counts':{str(k):int(np.sum(mask&(out['radar_status']==k))) for k in range(5)},
                         'accepted_errors':errors,'gates':gates,'joint_gate_pass':bool(all(gates.values())),
                         'adjacent_accepted_pairs':len(steps),'source_transitions':transitions,
                         'adjacent_prediction_step_p95_mps':float(np.quantile(steps,.95)) if len(steps) else None,
                         'adjacent_prediction_step_maximum_mps':float(steps.max()) if len(steps) else None}
    return summaries

def check_lock(create=False):
    protocol=read(ROOT/'protocol.json')
    if protocol['cases']!=list(CASES) or protocol['methods']!=list(METHODS) or protocol['production_policy_changed'] or protocol['promotion_allowed']:raise ValueError('fixed comparison scope changed')
    for name,h in protocol['input_sha256'].items():
        if not permitted(name):raise ValueError('protected input')
        if digest(RESEARCH/name)!=h:raise ValueError('input changed '+name)
    lock={'protocol_sha256':digest(ROOT/'protocol.json'),'code_sha256':{name:digest(ROOT/name) for name in ('fallback_core.py','run_fallback.py')}}
    if (ROOT/'analysis_lock.json').exists():
        if read(ROOT/'analysis_lock.json')!=lock:raise ValueError('analysis lock changed')
    elif create:write(ROOT/'analysis_lock.json',lock)
    else:raise ValueError('lock analysis before scoring')

def generate():
    check_lock()
    if (ROOT/'results.json').exists() or (ROOT/'replays').exists():raise ValueError('one-shot comparison already exists')
    (ROOT/'replays').mkdir();times,y,events,masks=cohort();results={}
    for case in CASES:
        archives={policy:load(path) for policy,path in paths(case).items()};radar=branch(archives['none'],'radar_only')
        for policy in GUARDS:
            for key in FIELDS:np.testing.assert_array_equal(radar[key],branch(archives[policy],'radar_only')[key])
        expected=np.searchsorted(events,times,side='right')-1
        for archive in archives.values():
            for model in (*MODELS,'radar_only'):
                b=branch(archive,model);np.testing.assert_array_equal(b['latest_endpoint'],expected)
                np.testing.assert_array_equal(b['age_ns'],times-events[np.maximum(expected,0)])
        outputs={'radar_only':select(radar,radar,False)}
        for model in MODELS:
            for policy in GUARDS:
                primary=branch(archives[policy],model)
                for strategy in ('withhold','fallback'):
                    outputs[model+'__'+policy+'__'+strategy]=select(primary,radar,strategy=='fallback')
        results[case]={method:summarize(out,times,y,masks) for method,out in outputs.items()}
        atomic_npz(ROOT/'replays'/(case+'.npz'),{method+'__'+key:value for method,out in outputs.items() for key,value in out.items()})
        m=results[case]['nonlinear7__deadband__fallback']['during']
        print(case,'candidate deadband fallback:',m['accepted_requests'],'/',m['requests'],'P95',m['accepted_errors']['p95_mps'],'radar',m['route_counts']['2'],'joint',m['joint_gate_pass'],flush=True)
    flags={key:False for key in ('models_retrained','labels_changed','threshold_search','production_policy_changed','reserved_final_opened','promotion_allowed')}
    write(ROOT/'results.json',{'decision':'OFFLINE_RADAR_FALLBACK_COMPARISON_COMPLETE_NO_PROMOTION','cases':results,
                             'scope':'Post-failure engineering comparison on reused Urban01 NIGHT development; no independent final test or published-system reproduction',**flags})
    print('OFFLINE_RADAR_FALLBACK_COMPARISON_COMPLETE_NO_PROMOTION',flush=True)
