"""Frozen model/reference disagreement audit; no fitting, inference or relabeling."""
from pathlib import Path
import hashlib,io,json,os,sys,tempfile
import numpy as np
ROOT=Path(__file__).resolve().parent;RESEARCH=ROOT.parent;PROJECT=RESEARCH.parent
sys.path.insert(0,str(RESEARCH/'vehicle_nonlinear_reference'));import reference_core as ref
read=ref.read;write=ref.write;load=ref.load;digest=ref.digest
METHODS=(*ref.METHODS,'huber7','extra_trees7')
def eligibility(pred,peer,source,imu_status,history_fresh):
    p,v,s,i,h=map(np.asarray,(pred,peer,source,imu_status,history_fresh))
    if p.ndim!=1 or any(a.shape!=p.shape for a in (v,s,i,h)) or s.dtype.kind!='b' or h.dtype.kind!='b' or i.dtype.kind not in 'iu':raise ValueError('aligned numeric signals, integer status and boolean masks required')
    return s&(i==0)&h&np.isfinite(p)&np.isfinite(v)&(p>=0)&(v>=0)&(abs(p-v)<=1.)
def atomic_npz(path,arrays):
    buffer=io.BytesIO();np.savez_compressed(buffer,**arrays);temp=None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.reference-',delete=False) as stream:
            temp=Path(stream.name);stream.write(buffer.getvalue());stream.flush();os.fsync(stream.fileno())
        os.replace(temp,path)
    finally:
        if temp is not None:temp.unlink(missing_ok=True)
def inputs():
    data=ref.inputs();held=load(RESEARCH/'vehicle_matched_head_comparison/held_predictions.npz');nominal=load(RESEARCH/'vehicle_matched_head_comparison/replays/nominal.npz')
    np.testing.assert_array_equal(held['target_mps'],data['fit']['y'])
    for name,alias in [('parent_radar_imu','radar_imu_parent'),('nonlinear7','nonlinear7'),('linear7','linear7')]:np.testing.assert_array_equal(data['fit']['pred'][name],held[alias+'__prediction_mps'])
    for scope,archive,suffix in [('fit',held,'__prediction_mps'),('full',nominal,'__ready_prediction_mps')]:
        for method in ('huber7','extra_trees7'):
            p=archive[method+suffix]
            if p.shape!=data[scope]['y'].shape or not np.isfinite(p).all() or np.any(p<0):raise ValueError('invalid new frozen predictions')
            data[scope]['pred'][method]=p
    full=data['full'];ids=full['row_id'];frames=read(RESEARCH/'vehicle_masked_history/frame_index.json');np.testing.assert_array_equal(ids,np.flatnonzero([v['ready'] for v in frames]));np.testing.assert_array_equal(full['time_ns'],[frames[i]['time_ns'] for i in ids])
    parent=load(RESEARCH/'vehicle_multisource_imu/development_replay.npz');new=load(RESEARCH/'vehicle_nonlinear_correction/development_replay.npz');old=load(RESEARCH/'vehicle_signed_axis_correction/development_replay.npz');base=load(RESEARCH/'vehicle_sensor_faults/replays/nominal.npz');guard=load(RESEARCH/'vehicle_deadband_guard/replays/nominal__deadband.npz');decisions=load(RESEARCH/'vehicle_night_training/decisions.npz');night=load(RESEARCH/'vehicle_night_training/predictions.npz')
    fresh=guard['history_clean'][ids]&guard['endpoint_content_fresh'][ids];peer=night['day_plus_night__masked_cnn_4ch'];source=decisions['day_plus_night__quality'][ids];imu_status=base['current__status'][ids]
    if peer.shape!=full['y'].shape:raise ValueError('peer row alignment changed')
    own={'original_context':old['original_context__quality'][ids],
         'parent_radar_imu':parent['radar_imu__quality'][ids]&fresh,
         'signed6_previous':old['signed_axis_corrected__quality'][ids]&fresh,
         'linear7':new['linear7__quality'][ids]&fresh,'nonlinear7':new['nonlinear7__quality'][ids]&fresh}
    for method in ('linear7','nonlinear7'):
        np.testing.assert_array_equal(own[method],eligibility(full['pred'][method],peer,source,imu_status,fresh))
    for method in ('huber7','extra_trees7'):own[method]=eligibility(full['pred'][method],peer,source,imu_status,fresh)
    full['own_accepted']=own;data['fit']['own_accepted']={method:np.ones(len(data['fit']['y']),bool) for method in METHODS}
    return data
def categories(q,members,scope):
    n=len(q['y']);out={'pooled':{'all':np.ones(n,bool)},'by_window':{name:q['window']==name for name in sorted(set(q['window'].tolist()))}}
    if scope=='fit':
        for category,field in [('by_group','group'),('by_recording','route')]:
            labels=np.array([v[field] for v in members]);out[category]={name:labels==name for name in sorted(set(labels))}
    return out
def cell(q,mask,methods=METHODS):
    value=ref.cell({**q,'pred':{method:q['pred'][method] for method in methods}},mask)
    for model in value['models'].values():
        a=model['primary_reference']['p95_mps'];b=model['obd']['p95_mps'];model['both_reference_p95_le_0_5']=None if a is None or b is None else bool(a<=.5 and b<=.5)
    return value
def freeze(create=False):
    protocol=read(ROOT/'protocol.json')
    if protocol['methods']!=list(METHODS) or any(protocol[k] for k in ('models_refitted','labels_changed','thresholds_changed','reserved_final_opened','promotion_allowed')):raise ValueError('fixed reference scope changed')
    for name,h in protocol['input_sha256'].items():
        if any(s in name.lower() for s in ('urban02','rav4','vehicle_final_evaluation')):raise ValueError('protected input')
        if digest(RESEARCH/name)!=h:raise ValueError('input changed '+name)
    lock={'protocol_sha256':digest(ROOT/'protocol.json'),'code_sha256':{name:digest(ROOT/name) for name in ('head_reference_core.py','generate_head_reference.py')}}
    if (ROOT/'analysis_lock.json').exists():
        if read(ROOT/'analysis_lock.json')!=lock:raise ValueError('analysis lock changed')
    elif create:write(ROOT/'analysis_lock.json',lock)
    else:raise ValueError('freeze before scoring')
def compute():
    freeze();data=inputs();populations={};arrays={}
    for scope in ('fit','full'):
        q=data[scope];out={}
        for category,pops in categories(q,data['members'],scope).items():
            out[category]={}
            for name,mask in pops.items():
                matched=ref.select(q['present'],mask,np.ones(len(mask),bool));v={'parent_rows':int(mask.sum()),'matched_rows':int(matched.sum()),'missing_rows':int(np.sum(mask&~q['present'])),'all_matched':cell(q,matched)}
                if scope=='full':
                    v['fixed_hgb_accepted_matched']=cell(q,ref.select(q['present'],mask,q['accepted']));own={}
                    for method in METHODS:
                        admitted=mask&q['own_accepted'][method];selected=admitted&q['present']
                        own[method]={'admitted_parent_rows':int(admitted.sum()),'admitted_matched_rows':int(selected.sum()),'admitted_missing_rows':int(np.sum(admitted&~q['present'])),'comparison':cell(q,selected,(method,))}
                    v['own_admitted_matched']=own
                out[category][name]=v
        populations[scope]=out
        for key in ('row_id','time_ns','match_id','present','accepted'):arrays[scope+'__'+key]=q[key]
        for method,mask in q['own_accepted'].items():arrays[scope+'__own_accepted__'+method]=mask
    return {'decision':'FROZEN_HEAD_REFERENCE_COMPARISON_COMPLETE_NO_PROMOTION','populations':populations,'models_refitted':False,'labels_changed':False,'thresholds_changed':False,'reserved_final_opened':False,'production_policy_changed':False,'promotion_allowed':False,'reference_threat_independence_established':False},arrays
