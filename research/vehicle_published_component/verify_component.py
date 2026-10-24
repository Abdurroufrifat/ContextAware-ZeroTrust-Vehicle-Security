"""Independent formulas and author replay; no native-cue substitution."""
from pathlib import Path
import hashlib,importlib.util,io,itertools,json,math,sys,unittest,zipfile
import numpy as np

ROOT=Path(__file__).resolve().parent;RESEARCH=ROOT.parent;PROJECT=RESEARCH.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def close(a,b):
    if isinstance(a,dict):
        if not isinstance(b,dict) or set(a)!=set(b):raise AssertionError('keys differ')
        for key in a:close(a[key],b[key])
    elif isinstance(a,(list,tuple,np.ndarray)):
        if len(a)!=len(b):raise AssertionError('length differs')
        for x,y in zip(a,b):close(x,y)
    elif isinstance(a,(float,np.floating)):
        if not math.isclose(a,b,abs_tol=1e-8,rel_tol=0):raise AssertionError('numeric difference '+str((a,b)))
    elif a!=b:raise AssertionError('value differs '+str((a,b)))
def sigmoid(x):return 1/(1+math.exp(-x))
def curv(p,e):return sigmoid(6*(abs(p-e)/(math.pi/4)-.5))
def ttc(d,v):return math.exp(-max(d,0)/(2*max(max(v,0),1e-3)))
def noisy(cues):
    terms=[w*c for w,c in zip((.3,.4,.3),cues)]
    iri=max(0.,min(1.,1.-math.prod(1-t for t in terms)))
    return iri,('curv','ttc','ood')[max(range(3),key=lambda i:terms[i])]
def out(cues):
    iri,dominant=noisy(cues)
    return {'iri':iri,'r_curv':cues[0],'r_ttc':cues[1],'r_ood':cues[2],'i_star':dominant,'shield':iri>.3,'c_t':list(cues)}

def verify():
    p=read(ROOT/'protocol.json');lock=read(ROOT/'analysis_lock.json');provenance=read(ROOT/'source_provenance.json');r=read(ROOT/'results.json')
    if sha(ROOT/'protocol.json')!=lock['protocol_sha256']:raise AssertionError('protocol changed')
    for name,h in lock['code_sha256'].items():
        if sha(ROOT/name)!=h:raise AssertionError('code changed '+name)
    for name,h in p['input_sha256'].items():
        if any(s in name.lower() for s in ('urban02','rav4','vehicle_final_evaluation')):raise AssertionError('protected input')
        if sha(RESEARCH/name)!=h:raise AssertionError('input changed '+name)
    if provenance['commit']!='58363ad49f1dd65c9135ff7d733f24a5f474cd4c' or r['author_commit']!=provenance['commit']:raise AssertionError('author revision changed')
    for item in provenance['files']:
        b=(ROOT/item['local_name']).read_bytes()
        if hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()!=item['git_blob_sha1']:raise AssertionError('author blob differs')
    for name,h in p['author_sha256'].items():
        if sha(ROOT/name)!=h:raise AssertionError('author source differs')
    spec=importlib.util.spec_from_file_location('_independent_author_iri',ROOT/'author_reference/iri_calculator.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);calc=m.IRICalculator()
    cube=list(itertools.product([i/10 for i in range(11)],repeat=3));close(len(r['noisy_or_cube']),1331)
    for cues,row in zip(cube,r['noisy_or_cube']):
        iri,dominant=noisy(cues);close(row,{'cues':cues,'iri':iri,'dominant':dominant,'shield':iri>.3});a,b=calc.noisy_or(*cues);close(a,iri);close(b,dominant)
    curv_grid=list(itertools.product((-.2,0.,.2),(-.8,-.4,0.,.4,.8)));close(len(r['curvature']),15)
    for (plan,executed),row in zip(curv_grid,r['curvature']):
        value=curv(plan,executed);close(row,{'plan':plan,'executed':executed,'value':value});close(calc.r_curv(plan,executed),value)
    ttc_grid=list(itertools.product((0.,1.,5.,20.,100.),(0.,.1,1.,10.,30.)));close(len(r['ttc']),25)
    for (d,v),row in zip(ttc_grid,r['ttc']):
        value=ttc(d,v);close(row,{'distance':d,'closing_speed':v,'value':value});close(calc.r_ttc(d,v),value)
    warmup=np.array([[20.+.2*math.sin(j*2*math.pi/72+i*.3)+.02*i*math.cos(j*2*math.pi/72)+.004*i*(i%3) for j in range(72)] for i in range(30)])
    close(warmup,r['synthetic_ood']['warmup_ranges_m'])
    # Independent sample covariance (ddof=1) and solve, not the author's inverse.
    mu=np.array([math.fsum(row[j] for row in warmup)/30 for j in range(72)])
    cov=np.array([[math.fsum((row[j]-mu[j])*(row[k]-mu[k]) for row in warmup)/29 for k in range(72)] for j in range(72)])+.001*np.eye(72)
    def mahal(z):
        delta=np.asarray(z)-mu;solution=np.linalg.solve(cov,delta)
        return math.sqrt(max(math.fsum(float(a)*float(b) for a,b in zip(delta,solution)),0.))
    distances=[mahal(z) for z in warmup];mean=math.fsum(distances)/30;std=max(math.sqrt(math.fsum((d-mean)**2 for d in distances)/30),1e-6)
    def ood(z):return sigmoid((mahal(z)-mean)/std)
    authored_warmup=[calc.r_ood(z) for z in warmup]
    close(authored_warmup,[0.]*30);close(r['synthetic_ood']['warmup_scores'],[0.]*30)
    if not calc._ood_model_ready or r['synthetic_ood']['model_ready_after_warmup'] is not True:raise AssertionError('warmup state')
    probes=[warmup[-1],mu,warmup[-1]+.05,warmup[-1]+1.];close(len(r['synthetic_ood']['probes']),4)
    for z,row in zip(probes,r['synthetic_ood']['probes']):close(row,{'ranges_m':z,'value':ood(z)});close(calc.r_ood(z),ood(z))
    grid=list(itertools.product((0.,),(-.8,0.,.8),(0.,10.,100.),(0.,5.,20.)));close(len(r['integrated']),27)
    for (plan,executed,d,v),row in zip(grid,r['integrated']):
        expected=out((curv(plan,executed),ttc(d,v),ood(probes[0])))
        close(row,{'plan':plan,'executed':executed,'distance':d,'closing_speed':v,'ranges_m':probes[0],'output':expected})
        actual=calc.compute(plan,executed,d,v,probes[0]);close(actual,expected)
    missing=out((curv(0.,0.),ttc(100.,5.),0.));close(r['author_missing_lidar_behavior'],missing)
    fresh=m.IRICalculator();close(fresh.compute(0.,0.,100.,5.,None),missing)
    if fresh._ood_step!=0:raise AssertionError('missing input advanced warmup')
    close(r['project_missing_lidar_contract'],{'eligible':False,'missing':[],'invalid':['z_t']})
    cases=('nominal','dropout','stale_header','future_header','accel_bias','gyro_bias','frozen_values','tiny_jitter','threshold_aware_pulses');required=['kappa_plan','kappa_exec','d_t','v_t','z_t']
    close(list(cases),p['cases']);close(len(r['cohort_schema_checks']),27);identities=set()
    for row in r['cohort_schema_checks']:
        case=row['case'];policy=row['policy']
        if case not in cases or policy not in ('none','exact','deadband') or (case,policy) in identities:raise AssertionError('schema identity')
        identities.add((case,policy));old=case in cases[:7]
        expected=('vehicle_sensor_faults/replays/'+case+'.npz' if old else 'vehicle_deadband_guard/replays/'+case+'__base.npz') if policy=='none' else ('vehicle_content_guard/replays/'+case+'.npz' if old else 'vehicle_deadband_guard/replays/'+case+'__exact.npz') if policy=='exact' else 'vehicle_deadband_guard/replays/'+case+'__deadband.npz'
        if row['file']!=expected:raise AssertionError('wrong replay schema source')
        with zipfile.ZipFile(RESEARCH/expected) as z:
            if z.testzip() is not None:raise AssertionError('bad replay CRC')
        with np.load(RESEARCH/expected,allow_pickle=False) as archive:
            present=[key for key in required if key in archive.files];count=len(archive['nonlinear7__request__status'])
        close(count,37340);close(present,[])
        close(row,{'case':case,'policy':policy,'file':expected,'requests':count,'available_native_fields':[],
                   'required_native_fields':required,'input_contract':{'eligible':False,'missing':required,'invalid':[]},'native_RAIL_compute_executed_on_cohort':False})
    close(r['component_case_count'],1402)
    for flag in ('same_cohort_comparison_executed','vehicle_models_retrained','production_policy_changed','reserved_final_opened','promotion_allowed'):
        if r[flag] is not False:raise AssertionError('unsupported claim '+flag)
    if r['decision']!='RAIL_COMPONENT_AUDIT_COMPLETE_SAME_COHORT_COMPARISON_BLOCKED':raise AssertionError('decision changed')
    for name,h in read(ROOT/'sha256.json').items():
        if sha(ROOT/name)!=h:raise AssertionError('artifact changed '+name)
    snapshot=read(ROOT/'prior_snapshot_manifest.json');manifest=read(PROJECT/'PROJECT_MANIFEST.json');verified=missing_count=protected=0
    for name,h in snapshot.items():
        if name!='FINAL_PROJECT_README.md' and manifest.get(name)!=h:raise AssertionError('historical manifest changed '+name)
    for name,h in manifest.items():
        if any(s in name.lower() for s in ('urban02','rav4','vehicle_final_evaluation')):protected+=1;continue
        path=PROJECT/name
        if not path.is_file():missing_count+=1;continue
        if sha(path)!=h:raise AssertionError('project hash differs '+name)
        verified+=1
    sys.path.insert(0,str(ROOT));import test_component
    stream=io.StringIO();tests=unittest.TextTestRunner(stream=stream).run(unittest.defaultTestLoader.loadTestsFromModule(test_component))
    if not tests.wasSuccessful() or tests.testsRun!=10:raise AssertionError(stream.getvalue())
    print('PUBLISHED COMPONENT VERIFICATION PASS: 1402 numeric cases; 30 synthetic warmup vectors; 27 replay schemas; 10 input-contract tests')
    print('PROJECT:',verified,'verified;',missing_count,'missing;',protected,'protected paths not opened')
    print('RAIL_COMPONENT_AUDIT_COMPLETE_SAME_COHORT_COMPARISON_BLOCKED')
    print('NO FULL RAIL / DRIVING PERFORMANCE / NOVELTY / PROMOTION CLAIM')

if __name__=='__main__':verify()
