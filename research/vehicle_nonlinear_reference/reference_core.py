"""Exact saved-reference comparisons; no model fitting, inference or relabeling."""
from pathlib import Path
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parent;RESEARCH=ROOT.parent;PROJECT=RESEARCH.parent
METHODS=('original_context','parent_radar_imu','signed6_previous','linear7','nonlinear7')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def exact_ids(records,keys):
 lookup={}
 for i,row in enumerate(records):
  key=(row['window_id'],row['time_ns'])
  if type(key[1]) is not int or key[1]<0 or key in lookup:raise ValueError('duplicate/noninteger exact reference key')
  lookup[key]=i
 result=[]
 for window,time in keys:
  if type(time) is not int or time<0:raise ValueError('integer endpoint clock required')
  result.append(lookup.get((window,time),-1))
 return result
def metrics(y,p):
 y=np.asarray(y,float);p=np.asarray(p,float)
 if y.ndim!=1 or p.shape!=y.shape or not np.isfinite(y).all() or not np.isfinite(p).all() or np.any(y<0) or np.any(p<0):raise ValueError('nonfinite/invalid matched speed')
 e=p-y;a=abs(e);n=len(e)
 return {'count':n,'mae_mps':float(a.mean()) if n else None,'rmse_mps':float(np.sqrt(np.mean(e**2))) if n else None,'p95_mps':float(np.quantile(a,.95)) if n else None,'maximum_mps':float(a.max()) if n else None,'bias_mps':float(e.mean()) if n else None,'above_half_mps':int((a>.5).sum()),'above_2_mps':int((a>2.).sum())}
def select(present,scope,accepted):
 a,b,c=map(np.asarray,[present,scope,accepted])
 if a.ndim!=1 or b.shape!=a.shape or c.shape!=a.shape or any(v.dtype.kind!='b' for v in [a,b,c]):raise ValueError('aligned boolean masks required')
 return a&b&c
def check_ids(expected,actual):
 a,b=map(np.asarray,[expected,actual])
 if a.dtype.kind not in 'iu' or b.dtype.kind not in 'iu' or a.shape!=b.shape or not np.array_equal(a,b):raise ValueError('exact integer row identity differs')
def inputs():
 diag=load(RESEARCH/'vehicle_remaining_failure_audit/row_diagnostics.npz');pairs=load(RESEARCH/'vehicle_frame_reference_audit/reference_pairs.npz');records=read(RESEARCH/'vehicle_reference_check/matched_endpoints.json');members=read(RESEARCH/'vehicle_route_validation/membership.json');oof=load(RESEARCH/'vehicle_nonlinear_correction/oof_predictions.npz');new=load(RESEARCH/'vehicle_nonlinear_correction/development_replay.npz');old=load(RESEARCH/'vehicle_signed_axis_correction/development_replay.npz');parent=load(RESEARCH/'vehicle_multisource_imu/development_replay.npz');out={}
 if len(members)!=27178:raise ValueError('route cohort differs')
 for scope in ['fit','full']:
  q={k[len(scope)+2:]:v for k,v in diag.items() if k.startswith(scope+'__')};n=len(q['row_id']);check_ids(q['row_id'],pairs[scope+'__row_id']);ids=np.array(exact_ids(records,zip(q['window_id'].tolist(),map(int,q['time_ns']))),dtype=np.int64);check_ids(ids,pairs[scope+'__reference_match_id']);present=ids>=0;np.testing.assert_array_equal(present,pairs[scope+'__comparison_present']);obd=np.full(n,np.nan)
  for i in np.flatnonzero(present):
   row=records[ids[i]]
   if row['reference_horizontal_mps']!=float(q['target_mps'][i]) or int(q['time_ns'][i])-row['reference_header_time_ns']!=int(q['reference_age_ns'][i]):raise ValueError('saved reference target or clock differs')
   obd[i]=row['label_mps']
  np.testing.assert_allclose(obd,pairs[scope+'__obd_mps'],rtol=0,atol=0,equal_nan=True)
  if np.any(present&~np.isfinite(obd)):raise ValueError('matched OBD missing')
  if scope=='fit':
   check_ids(q['row_id'],np.arange(n));np.testing.assert_array_equal(q['target_mps'],oof['target_mps']);check_ids(q['time_ns'],np.array([v['time_ns'] for v in members],np.int64));np.testing.assert_array_equal(q['window_id'],[v['window_id'] for v in members]);pred={m:oof[m+'__prediction_mps'] for m in METHODS};accepted=np.ones(n,bool)
  else:
   pred={m:new[m+'__ready_prediction_mps'] for m in ['linear7','nonlinear7']};pred.update(original_context=old['original_context__ready_prediction_mps'],parent_radar_imu=parent['radar_imu__ready_prediction_mps'],signed6_previous=old['signed_axis_corrected__ready_prediction_mps']);accepted=new['nonlinear7__quality'][q['row_id']]
  if n!=(27178 if scope=='fit' else 40678) or any(v.shape!=(n,) or not np.isfinite(v).all() for v in pred.values()):raise ValueError('prediction cohort differs')
  out[scope]={'row_id':q['row_id'],'time_ns':q['time_ns'],'window':q['window_id'],'y':q['target_mps'],'obd':obd,'present':present,'match_id':ids,'pred':pred,'accepted':accepted}
 out['members']=members;return out
def freeze(create=False):
 p=read(ROOT/'protocol.json')
 for flag in ['new_model_fitted','labels_changed','thresholds_changed','reserved_final_opened','promotion_allowed']:
  if p[flag] is not False:raise ValueError('scope changed')
 for n,h in p['input_sha256'].items():
  if 'urban02' in n.lower() or 'rav4' in n.lower() or n.startswith('vehicle_final_evaluation/'):raise ValueError('protected input')
  if digest(RESEARCH/n)!=h:raise ValueError('input changed: '+n)
 lock={'protocol_sha256':digest(ROOT/'protocol.json'),'code_sha256':{n:digest(ROOT/n) for n in ['reference_core.py','generate_reference.py']}}
 if (ROOT/'analysis_lock.json').exists():
  if read(ROOT/'analysis_lock.json')!=lock:raise ValueError('frozen implementation changed')
 elif create:write(ROOT/'analysis_lock.json',lock)
 else:raise ValueError('freeze before scoring')
def cell(q,mask):
 models={}
 for n,p in q['pred'].items():
  primary=metrics(q['y'][mask],p[mask]);obd=metrics(q['obd'][mask],p[mask]);a=abs(p[mask]-q['y'][mask])>.5;b=abs(p[mask]-q['obd'][mask])>.5
  models[n]={'primary_reference':primary,'obd':obd,'above_half_against_both':int((a&b).sum()),'primary_tail_but_obd_le_half':int((a&~b).sum()),'obd_tail_but_primary_le_half':int((~a&b).sum())}
 return {'rows':int(mask.sum()),'models':models,'obd_minus_primary':metrics(q['y'][mask],q['obd'][mask])}
def compute():
 freeze();data=inputs();pop={};arrays={}
 for scope in ['fit','full']:
  q=data[scope];n=len(q['y']);categories={'pooled':{'all':np.ones(n,bool)}}
  categories['by_window']={name:q['window']==name for name in sorted(set(q['window'].tolist()))}
  if scope=='fit':
   for category,key in [('by_group','group'),('by_recording','route')]:
    labels=np.array([v[key] for v in data['members']]);categories[category]={name:labels==name for name in sorted(set(labels.tolist()))}
  result={}
  for cat,masks in categories.items():
   result[cat]={}
   for name,mask in masks.items():
    matched=select(q['present'],mask,np.ones(n,bool));v={'parent_rows':int(mask.sum()),'matched_rows':int(matched.sum()),'missing_rows':int(np.sum(mask&~q['present'])),'all_matched':cell(q,matched)}
    if scope=='full':v['nonlinear_accepted_matched']=cell(q,select(q['present'],mask,q['accepted']))
    result[cat][name]=v
  pop[scope]=result
  for key in ['row_id','time_ns','match_id','present','accepted']:arrays[scope+'__'+key]=q[key]
 return {'decision':'FROZEN_NONLINEAR_ALTERNATE_REFERENCE_CHECK_COMPLETE_NO_PROMOTION','scope':'Retrospective diagnostic using exact saved OBD comparisons; OBD is not certified independent truth','primary_method':'nonlinear7','populations':pop,'primary_model_changed':False,'new_model_fitted':False,'labels_changed':False,'thresholds_changed':False,'reserved_final_opened':False,'promotion_allowed':False,'reference_threat_independence_established':False},arrays
