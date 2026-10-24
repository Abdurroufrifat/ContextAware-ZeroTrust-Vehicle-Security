"""Author-code component audit and explicit native-input availability contract."""
from pathlib import Path
from collections.abc import Mapping
import hashlib,importlib.util,itertools,json,math,numbers
import numpy as np

ROOT=Path(__file__).resolve().parent
RESEARCH=ROOT.parent
PROJECT=RESEARCH.parent
REQUIRED=('kappa_plan','kappa_exec','d_t','v_t','z_t')
CASES=('nominal','dropout','stale_header','future_header','accel_bias','gyro_bias','frozen_values','tiny_jitter','threshold_aware_pulses')
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def permitted(name):return not any(s in name.lower() for s in ('urban02','rav4','vehicle_final_evaluation'))

def validate_native_record(record):
    """Project prerequisite only, not a reproduction of RAIL's permissive API.

    No aliases, filling, interpolation, physical calibration or source-authentication
    inference. Passing confirms shape/types only, not physically valid provenance.
    """
    if not isinstance(record,Mapping):return {'eligible':False,'missing':list(REQUIRED),'invalid':['record']}
    missing=[key for key in REQUIRED if key not in record];invalid=[]
    for key in REQUIRED[:4]:
        if key not in record:continue
        value=record[key]
        if isinstance(value,(bool,np.bool_)) or not isinstance(value,numbers.Real) or not math.isfinite(float(value)) or (key in ('d_t','v_t') and float(value)<0):invalid.append(key)
    if 'z_t' in record:
        try:
            z=np.asarray(record['z_t'])
            good=z.shape==(72,) and z.dtype.kind in 'iuf' and np.isfinite(z).all() and (z>=0).all()
        except (ValueError,TypeError):good=False
        if not good:invalid.append('z_t')
    return {'eligible':not missing and not invalid,'missing':missing,'invalid':invalid}

def replay_paths():
    out={}
    for ordinal,case in enumerate(CASES):
        old=ordinal<7
        out[case]={
            'none':RESEARCH/('vehicle_sensor_faults/replays/'+case+'.npz') if old else RESEARCH/('vehicle_deadband_guard/replays/'+case+'__base.npz'),
            'exact':RESEARCH/('vehicle_content_guard/replays/'+case+'.npz') if old else RESEARCH/('vehicle_deadband_guard/replays/'+case+'__exact.npz'),
            'deadband':RESEARCH/('vehicle_deadband_guard/replays/'+case+'__deadband.npz')}
    return out

def check_lock(create=False):
    protocol=read(ROOT/'protocol.json')
    if protocol['cases']!=list(CASES) or protocol['same_cohort_comparison_executed'] or protocol['promotion_allowed']:raise ValueError('component scope changed')
    for name,h in protocol['input_sha256'].items():
        if not permitted(name):raise ValueError('protected input')
        if digest(RESEARCH/name)!=h:raise ValueError('input changed '+name)
    for name,h in protocol['author_sha256'].items():
        if digest(ROOT/name)!=h:raise ValueError('author source changed '+name)
    lock={'protocol_sha256':digest(ROOT/'protocol.json'),'code_sha256':{name:digest(ROOT/name) for name in ('component_core.py','run_component.py')}}
    if (ROOT/'analysis_lock.json').exists():
        if read(ROOT/'analysis_lock.json')!=lock:raise ValueError('component analysis lock changed')
    elif create:write(ROOT/'analysis_lock.json',lock)
    else:raise ValueError('lock before execution')

def author_class():
    provenance=read(ROOT/'source_provenance.json')
    for item in provenance['files']:
        data=(ROOT/item['local_name']).read_bytes()
        if hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()!=item['git_blob_sha1']:raise ValueError('author Git blob changed')
    spec=importlib.util.spec_from_file_location('_pinned_rail_iri',ROOT/'author_reference/iri_calculator.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.IRICalculator

def fixture_warmup():
    theta=np.arange(72,dtype=float)*2*math.pi/72
    return np.array([20.+.2*np.sin(theta+i*.3)+.02*i*np.cos(theta)+.004*i*(i%3) for i in range(30)])

def record_output(out):
    return {key:float(out[key]) for key in ('iri','r_curv','r_ttc','r_ood')}|{'i_star':out['i_star'],'shield':bool(out['shield']),'c_t':out['c_t'].tolist()}

def generate():
    check_lock()
    if (ROOT/'results.json').exists():raise ValueError('one-shot component audit already exists')
    cls=author_class();calculator=cls();cube=[]
    for cues in itertools.product([i/10 for i in range(11)],repeat=3):
        iri,dominant=calculator.noisy_or(*cues);cube.append({'cues':list(cues),'iri':iri,'dominant':dominant,'shield':iri>calculator.tau})
    curv=[{'plan':plan,'executed':executed,'value':calculator.r_curv(plan,executed)} for plan in (-.2,0.,.2) for executed in (-.8,-.4,0.,.4,.8)]
    ttc=[{'distance':d,'closing_speed':v,'value':calculator.r_ttc(d,v)} for d in (0.,1.,5.,20.,100.) for v in (0.,.1,1.,10.,30.)]
    warmup=fixture_warmup();scores=[calculator.r_ood(z) for z in warmup]
    probes=[warmup[-1],warmup.mean(0),warmup[-1]+.05,warmup[-1]+1.]
    ood=[{'ranges_m':z.tolist(),'value':calculator.r_ood(z)} for z in probes]
    integrated=[]
    for plan,executed,d,v in itertools.product((0.,),(-.8,0.,.8),(0.,10.,100.),(0.,5.,20.)):
        integrated.append({'plan':plan,'executed':executed,'distance':d,'closing_speed':v,'ranges_m':probes[0].tolist(),
                           'output':record_output(calculator.compute(plan,executed,d,v,probes[0]))})
    missing_calc=cls();missing=record_output(missing_calc.compute(0.,0.,100.,5.,None))
    missing_contract=validate_native_record({'kappa_plan':0.,'kappa_exec':0.,'d_t':100.,'v_t':5.,'z_t':None})
    schema=[]
    for case,policies in replay_paths().items():
        for policy,path in policies.items():
            with np.load(path,allow_pickle=False) as archive:
                keys=archive.files;count=len(archive['nonlinear7__request__status'])
                available=[key for key in REQUIRED if key in keys]
            # The archive has model outputs and serving metadata, not native RAIL cues.
            contract=validate_native_record({key:None for key in available})
            schema.append({'case':case,'policy':policy,'file':path.relative_to(RESEARCH).as_posix(),'requests':count,
                           'available_native_fields':available,'required_native_fields':list(REQUIRED),'input_contract':contract,
                           'native_RAIL_compute_executed_on_cohort':False})
    result={'decision':'RAIL_COMPONENT_AUDIT_COMPLETE_SAME_COHORT_COMPARISON_BLOCKED','author_commit':read(ROOT/'source_provenance.json')['commit'],
            'noisy_or_cube':cube,'curvature':curv,'ttc':ttc,'synthetic_ood':{'warmup_ranges_m':warmup.tolist(),'warmup_scores':scores,'model_ready_after_warmup':bool(calculator._ood_model_ready),'probes':ood},
            'integrated':integrated,'author_missing_lidar_behavior':missing,'project_missing_lidar_contract':missing_contract,
            'cohort_schema_checks':schema,'component_case_count':len(cube)+len(curv)+len(ttc)+len(ood)+len(integrated),
            'same_cohort_comparison_executed':False,'vehicle_models_retrained':False,'production_policy_changed':False,'reserved_final_opened':False,'promotion_allowed':False}
    write(ROOT/'results.json',result)
    print('AUTHOR COMPONENT EXECUTED:',result['component_case_count'],'numeric cases; 30 synthetic OOD warmup vectors; 27 replay schema checks')
    print('RAIL_COMPONENT_AUDIT_COMPLETE_SAME_COHORT_COMPARISON_BLOCKED')
