"""Independent comparator inference, scalar serving/routing and complete metrics."""
import bisect,io,math,sys,unittest,zipfile
import numpy as np
from head_core import ROOT,RESEARCH,PROJECT,SCOPES,NEW,MODELS,METHODS,read,load,digest,ah,check_models,fitting_data,f,d
import verify_deadband as vd
sys.path.insert(0,str(RESEARCH/'vehicle_radar_fallback'));import verify_fallback as vr
vf=vd.vf;close=vf.close

def infer(model,x):
    # sklearn quantizes inputs to float32, then compares against double thresholds.
    # Keep those quantized values in float64 to prevent NumPy scalar promotion
    # from rounding a Python threshold down to float32 in the independent traversal.
    if model['method']=='extra_trees7':return vf.tree_infer(model,np.asarray(x,np.float32).astype(np.float64))
    return np.array([float(model['intercept'])+math.fsum((float(v)-a)/b*c for v,a,b,c in zip(row,model['mean'],model['scale'],model['coefficient'])) for row in x])

def verify():
    check_models();fit=fitting_data();result=read(ROOT/'results.json');trace=read(ROOT/'training_trace.json');y=fit['y'];members=fit['members'];expected_held={m:np.full(len(y),np.nan) for m in NEW};seen=np.zeros(len(y),int);training_rows=0
    if len(trace['runs'])!=8:raise AssertionError('fit omitted')
    for scope in SCOPES:
        cal=read(RESEARCH/'vehicle_multisource_imu/models'/scope/'calibration.json');ids=cal['train_ids'];held=cal['held_ids'];a=set(ids);b=set(held)
        if len(a)!=len(ids) or len(b)!=len(held) or a&b or a|b!=set(range(len(y))):raise AssertionError('scope partition')
        if scope!='all_fit' and (any(members[i]['group']==scope for i in ids) or any(members[i]['group']!=scope for i in held)):raise AssertionError('held-route leakage')
        parent=load(RESEARCH/'vehicle_multisource_imu/models'/scope/'radar_imu_export.npz')['native'];orig_axes=fit['axes'][ids];axes=np.concatenate([orig_axes,orig_axes],axis=0);x=np.column_stack([parent,axes]);target=np.concatenate([y[ids],y[ids]])-parent
        original_weights=[]
        for i in ids:
            bin_id=0 if y[i]<1 else 1 if y[i]<5 else 2 if y[i]<10 else 3
            original_weights.append((len(cal['calibration_ids'])/4)/cal['bin_counts'][bin_id]*.5)
        weights=np.array(original_weights+original_weights);hashes={'feature_sha256':ah(x),'residual_target_sha256':ah(target),'weight_sha256':ah(weights)}
        old=next(v for v in read(RESEARCH/'vehicle_nonlinear_correction/training_trace.json')['runs'] if v['scope']==scope and v['method']=='nonlinear7')
        for key,h in hashes.items():close(h,old[key])
        total=math.fsum(map(float,weights));mean=[math.fsum(float(w)*float(row[j]) for w,row in zip(weights,x))/total for j in range(7)];var=[math.fsum(float(w)*(float(row[j])-mean[j])**2 for w,row in zip(weights,x))/total for j in range(7)];scale=[math.sqrt(v) if v>1e-12 else 1. for v in var]
        for method in NEW:
            run=next(v for v in trace['runs'] if v['scope']==scope and v['method']==method)
            for key,h in hashes.items():close(h,run[key])
            close(run['augmented_rows'],len(x));close(run['held_rows'],len(held));model=read(ROOT/'models'/scope/(method+'.json'));artifact=load(ROOT/'models'/scope/(method+'_training.npz'));p=infer(model,x)
            np.testing.assert_allclose(p,artifact['native'],rtol=0,atol=1e-8);np.testing.assert_allclose(p,artifact['portable'],rtol=0,atol=1e-8);training_rows+=len(x)
            if method=='huber7':close(model['mean'],mean);close(model['scale'],scale)
            if scope!='all_fit':
                xx=np.column_stack([fit['parent'][held],fit['axes'][held]]);expected_held[method][held]=np.maximum(0.,fit['parent'][held]+infer(model,xx))
        if scope!='all_fit':seen[held]+=1
    if not np.all(seen==1):raise AssertionError('held coverage')
    expected_held.update(radar_imu_parent=fit['parent'],nonlinear7=fit['prior']['nonlinear7__prediction_mps'],linear7=fit['prior']['linear7__prediction_mps'])
    saved=load(ROOT/'held_predictions.npz');np.testing.assert_array_equal(y,saved['target_mps']);np.testing.assert_array_equal(fit['fold'],saved['fold_id']);held_cells=0
    if set(result['held'])!=set(expected_held):raise AssertionError('held model omitted')
    for method,p in expected_held.items():
        np.testing.assert_allclose(p,saved[method+'__prediction_mps'],rtol=0,atol=1e-8);masks={'all':np.ones(len(y),bool)}
        for field in ('group','route','window_id'):
            for value in sorted({v[field] for v in members}):masks[field+':'+value]=np.array([v[field]==value for v in members])
        close({key:vf.stats(y,p,mask) for key,mask in masks.items()},result['held'][method]);held_cells+=len(masks)
    data=f.inputs();times=list(map(int,data['times']));events=list(map(int,data['events']));truth=list(map(float,data['req_y']));scale=np.array(read(RESEARCH/'vehicle_deadband_guard/parameters.json')['scale']);start,end=data['start'],data['end']
    groups=[('all','before' if t<start else 'during' if t<end else 'after','segment_'+str((t-times[0])//300_000_000_000)) for t in times];names=['all','before','during','after']+['segment_'+str(k) for k in sorted({(t-times[0])//300_000_000_000 for t in times})];trials=cells=probes=0
    previous=read(RESEARCH/'vehicle_radar_fallback/results.json')
    if set(result['cases'])!=set(d.CASES):raise AssertionError('case omitted')
    for case in d.CASES:
        old=case in d.CASES[:7];base=load(RESEARCH/('vehicle_sensor_faults/replays/'+case+'.npz')) if old else load(RESEARCH/('vehicle_deadband_guard/replays/'+case+'__base.npz'));guard=load(RESEARCH/('vehicle_deadband_guard/replays/'+case+'__deadband.npz'));saved=load(ROOT/'replays'/(case+'.npz'))
        with zipfile.ZipFile(ROOT/'replays'/(case+'.npz')) as z:
            if z.testzip() is not None:raise AssertionError('bad replay CRC')
        source=vd.reference_source(data,case,scale);axes=vf.signed_features(source,data['events'])[data['ready']];parent=base['radar_imu_parent__ready_prediction_mps'];x=np.column_stack([parent,axes]);candidate=np.maximum(0.,parent+vf.tree_infer(data['head'],x));np.testing.assert_allclose(candidate,base['nonlinear7__ready_prediction_mps'],rtol=0,atol=1e-9)
        arrivals=list(map(int,source['bag_ns']))
        for i in np.unique(np.linspace(0,len(events)-1,25,dtype=int)):
            _,aa=vf.scalar_features(source,arrivals,events[i]);np.testing.assert_allclose(aa,vf.signed_features(source,np.array([events[i]],np.int64))[0],rtol=0,atol=1e-9);probes+=1
        linear=read(RESEARCH/'vehicle_nonlinear_correction/models/all_fit/linear7.json');pred={'linear7':np.maximum(0.,parent+np.array([linear['intercept']+math.fsum((float(v)-a)/b*c for v,a,b,c in zip(row,linear['mean'],linear['scale'],linear['coefficient'])) for row in x])),**{method:np.maximum(0.,parent+infer(read(ROOT/'models/all_fit'/(method+'.json')),x)) for method in NEW}}
        def branch(archive,model):return {key:archive[model+'__request__'+key] for key in ('status','prediction_mps','latest_endpoint','age_ns')}
        radar=branch(base,'radar_only');branches={'radar_only':radar,'radar_imu_parent':branch(guard,'radar_imu_parent'),'nonlinear7':branch(guard,'nonlinear7')}
        for method,values in pred.items():
            np.testing.assert_allclose(values,saved[method+'__ready_prediction_mps'],rtol=0,atol=1e-8);p=np.full(len(events),np.nan);p[data['ready']]=values
            quality=[bool(s) and int(imu)==0 and math.isfinite(float(v)) and math.isfinite(float(peer)) and v>=0 and peer>=0 and abs(v-peer)<=1. for v,peer,s,imu in zip(p,data['peer'],data['source'],base['current__status'])];statuses=[];served=[];latest=[];ages=[]
            for k,t in enumerate(times):
                i=bisect.bisect_right(events,t)-1;age=t-events[i] if i>=0 else -1;bound=i>=0 and int(base['current__status'][i])==0 and 0<=t-int(base['current__bag_ns'][i])<=100_000_000 and 0<=t-int(base['current__header_ns'][i])<=100_000_000
                if bound!=bool(base['binding'][k]):raise AssertionError('source binding differs')
                s=1 if i<0 else 2 if age>100_000_000 else 3 if not data['ready'][i] else 4
                if s==4 and quality[i] and bound and bool(guard['request_content_fresh'][k]):s=0
                statuses.append(s);served.append(float(p[i]) if s==0 else math.nan);latest.append(i);ages.append(age)
            branches[method]={'status':np.array(statuses,np.uint8),'prediction_mps':np.array(served),'latest_endpoint':np.array(latest,np.int64),'age_ns':np.array(ages,np.int64)}
        if set(result['cases'][case])!=set(METHODS):raise AssertionError('model/strategy omitted')
        for method in METHODS:
            if method=='radar_only':primary=radar;enabled=False;model='radar_only'
            else:model,_,strategy=method.split('__');primary=branches[model];enabled=strategy=='fallback'
            expected,summary=vr.scalar_comparison(primary,radar,enabled,times,truth,groups,names);close(summary,result['cases'][case][method]);trials+=len(times);cells+=len(summary)
            if model in (*NEW,'linear7'):
                for key,value in expected.items():np.testing.assert_allclose(value,saved[method+'__'+key],rtol=0,atol=1e-8,equal_nan=True)
            else:close(summary,previous['cases'][case][method])
        print('verified matched comparators:',case,flush=True)
    if result['decision']!='MATCHED_HEAD_COMPARISON_COMPLETE_NO_PROMOTION':raise AssertionError('decision differs')
    for flag in ('vehicle_original_models_changed','labels_changed','threshold_search','production_policy_changed','reserved_final_opened','promotion_allowed'):
        if result[flag] is not False:raise AssertionError('unsupported claim')
    for name,h in read(ROOT/'sha256.json').items():
        if digest(ROOT/name)!=h:raise AssertionError('artifact changed '+name)
    prior=read(ROOT/'prior_snapshot_manifest.json');manifest=read(PROJECT/'PROJECT_MANIFEST.json');verified=missing=protected=0
    for name,h in prior.items():
        if name!='FINAL_PROJECT_README.md' and manifest.get(name)!=h:raise AssertionError('prior manifest changed '+name)
    for name,h in manifest.items():
        if any(s in name.lower() for s in ('urban02','rav4','vehicle_final_evaluation')):protected+=1;continue
        p=PROJECT/name
        if not p.is_file():missing+=1;continue
        if digest(p)!=h:raise AssertionError('prior hash '+name)
        verified+=1
    import test_heads
    stream=io.StringIO();tests=unittest.TextTestRunner(stream=stream).run(unittest.defaultTestLoader.loadTestsFromModule(test_heads))
    if not tests.wasSuccessful():raise AssertionError(stream.getvalue())
    print('MATCHED HEAD VERIFICATION PASS:',training_rows,'training-export checks;',held_cells,'held metric cells;',trials,'policy-request checks;',cells,'replay metric cells;',probes,'feature probes;',tests.testsRun,'tests')
    print('PROJECT:',verified,'verified;',missing,'missing;',protected,'protected paths not opened')
    print('MATCHED_HEAD_COMPARISON_COMPLETE_NO_PROMOTION')
if __name__=='__main__':verify()
