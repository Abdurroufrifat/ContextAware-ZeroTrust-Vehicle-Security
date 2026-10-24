"""Fixed seven-input correction comparators; retrospective development only."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parent;RESEARCH=ROOT.parent;PROJECT=RESEARCH.parent
sys.path.insert(0,str(RESEARCH/'vehicle_deadband_guard'));import deadband_core as d
sys.path.insert(0,str(RESEARCH/'vehicle_radar_fallback'));import fallback_core as fb
f=d.f;n=f.n;read=n.read;write=n.write;load=n.load;digest=n.digest;head_features=n.head_features
SCOPES=('Urban01','Urban03','Urban04','all_fit');NEW=('huber7','extra_trees7')
MODELS=('radar_imu_parent','nonlinear7','linear7',*NEW)
METHODS=('radar_only',)+tuple(m+'__deadband__'+s for m in MODELS for s in ('withhold','fallback'))
def ah(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def scaler(x,w):
    x=np.asarray(x,float);w=np.asarray(w,float)
    if x.ndim!=2 or x.shape[1]!=7 or w.shape!=(len(x),) or not len(x) or not np.isfinite(x).all() or not np.isfinite(w).all() or np.any(w<0) or w.sum()<=0:raise ValueError('finite matched data and nonnegative weights required')
    mean=np.sum(x*w[:,None],axis=0)/w.sum();var=np.sum((x-mean)**2*w[:,None],axis=0)/w.sum()
    return mean,np.where(var>1e-12,np.sqrt(var),1.)
def predict(model,x):
    x=np.asarray(x,float)
    if x.ndim!=2 or x.shape[1]!=7 or not np.isfinite(x).all():raise ValueError('seven finite features required')
    if model['method']=='huber7':return float(model['intercept'])+((x-np.array(model['mean']))/np.array(model['scale']))@np.array(model['coefficient'])
    if model['method']!='extra_trees7' or not model['trees']:raise ValueError('unknown comparator or empty forest')
    return f.infer(model,x.astype(np.float32))
def export_extra(model):
    trees=[];count=len(model.estimators_)
    for estimator in model.estimators_:
        t=estimator.tree_;trees.append([{'leaf':int(t.children_left[i]<0),'value':float(t.value[i,0,0])/count,'feature':int(t.feature[i]),'threshold':float(t.threshold[i]),'left':int(t.children_left[i]),'right':int(t.children_right[i])} for i in range(t.node_count)])
    return {'method':'extra_trees7','baseline':0.,'trees':trees,'input_cast':'float32 like native sklearn trees','tree_values':'leaf means divided by ensemble size; sum all trees'}
def fitting_data():
    axes=load(RESEARCH/'vehicle_signed_axis_correction/feature_cache.npz')['fit_axes'];o=load(RESEARCH/'vehicle_multisource_imu/oof_predictions.npz');prior=load(RESEARCH/'vehicle_nonlinear_correction/oof_predictions.npz');members=read(RESEARCH/'vehicle_route_validation/membership.json');diag=load(RESEARCH/'vehicle_remaining_failure_audit/row_diagnostics.npz')
    if len(members)!=27178 or axes.shape!=(27178,6):raise ValueError('cohort changed')
    for key in ('target_mps','fold_id'):np.testing.assert_array_equal(o[key],prior[key])
    np.testing.assert_array_equal(o['target_mps'],diag['fit__target_mps']);np.testing.assert_array_equal(axes,diag['fit__signed_axes']);np.testing.assert_array_equal(o['radar_imu__prediction_mps'],prior['parent_radar_imu__prediction_mps'])
    return {'axes':axes,'y':o['target_mps'],'fold':o['fold_id'],'parent':o['radar_imu__prediction_mps'],'prior':prior,'members':members}
def training_batch(scope,data):
    cal=read(RESEARCH/'vehicle_multisource_imu/models'/scope/'calibration.json');ids=np.array(cal['train_ids'],int);held=np.array(cal['held_ids'],int);n.validate_split(ids,held,len(data['y']))
    groups=np.array([v['group'] for v in data['members']])
    if scope!='all_fit' and (np.any(groups[ids]==scope) or not np.all(groups[held]==scope)):raise ValueError('route leakage')
    parent=load(RESEARCH/'vehicle_multisource_imu/models'/scope/'radar_imu_export.npz')['native'];x=head_features(parent,np.tile(data['axes'][ids],(2,1)));target=np.tile(data['y'][ids],2)-parent;bins=np.digitize(data['y'][ids],[1,5,10]);w=np.tile((len(cal['calibration_ids'])/4)/np.array(cal['bin_counts'])[bins]*.5,2)
    old=next(v for v in read(RESEARCH/'vehicle_nonlinear_correction/training_trace.json')['runs'] if v['scope']==scope and v['method']=='nonlinear7')
    hashes={'feature_sha256':ah(x),'residual_target_sha256':ah(target),'weight_sha256':ah(w)}
    for key,value in hashes.items():
        if value!=old[key]:raise ValueError('unmatched original HGB training data '+key)
    return x,target,w,held,hashes
def check_lock(create=False):
    protocol=read(ROOT/'protocol.json')
    if protocol['scopes']!=list(SCOPES) or protocol['new_methods']!=list(NEW) or protocol['cases']!=list(d.CASES) or protocol['promotion_allowed']:raise ValueError('fixed matrix changed')
    for name,h in protocol['input_sha256'].items():
        if any(s in name.lower() for s in ('urban02','rav4','vehicle_final_evaluation')):raise ValueError('protected input')
        if digest(RESEARCH/name)!=h:raise ValueError('input changed '+name)
    lock={'protocol_sha256':digest(ROOT/'protocol.json'),'code_sha256':{name:digest(ROOT/name) for name in ('head_core.py','train_heads.py','score_heads.py')}}
    if (ROOT/'analysis_lock.json').exists():
        if read(ROOT/'analysis_lock.json')!=lock:raise ValueError('analysis code changed')
    elif create:write(ROOT/'analysis_lock.json',lock)
    else:raise ValueError('freeze before fitting')
    return protocol
def check_models():
    check_lock();lock=read(ROOT/'model_lock.json')
    if lock['analysis_lock_sha256']!=digest(ROOT/'analysis_lock.json') or lock['training_trace_sha256']!=digest(ROOT/'training_trace.json') or lock['all_eight_locked_before_scoring'] is not True:raise ValueError('model lock changed')
    for name,h in lock['files_sha256'].items():
        if digest(ROOT/name)!=h:raise ValueError('model artifact changed '+name)
def held_summary(data,p):
    members=data['members'];masks={'all':np.ones(len(p),bool)}
    for field in ('group','route','window_id'):
        values=np.array([v[field] for v in members]);masks.update({field+':'+value:values==value for value in sorted(set(values))})
    return {key:n.metrics(data['y'],p,mask) for key,mask in masks.items()}
