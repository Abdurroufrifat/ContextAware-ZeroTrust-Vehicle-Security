"""Fixed sensor-frame signed-axis correction; no promotion."""
from pathlib import Path
import hashlib,sys
import numpy as np
ROOT=Path(__file__).resolve().parent;RESEARCH=ROOT.parent
sys.path.insert(0,str(RESEARCH/'vehicle_multisource_imu'));import multisource_core as m
read=m.read;write=m.write;digest=m.digest;ALPHA=1.;WIDTH=11;SCOPES=m.SCOPES
SLOT_NAMES={'original_context':'original_context','radar_imu':'uncorrected_radar_imu','imu_only':'signed_axis_corrected'}
METHODS=tuple(SLOT_NAMES.values());FEATURE_NAMES=['mean11_acc_x','mean11_acc_y','mean11_acc_z','mean11_gyro_x','mean11_gyro_y','mean11_gyro_z']
def rename(value):
 if isinstance(value,dict):return {SLOT_NAMES.get(k,k):rename(v) for k,v in value.items()}
 if isinstance(value,list):return [rename(v) for v in value]
 return value
def oracle_errors(y,a,b):
 y=np.asarray(y,dtype=float);a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
 if y.ndim!=1 or a.shape!=y.shape or b.shape!=y.shape or not all(np.isfinite(v).all() for v in [y,a,b]):raise ValueError('unaligned/nonfinite oracle inputs')
 return np.maximum(0.,np.maximum(np.minimum(a,b)-y,y-np.maximum(a,b)))
def axis_features(source,endpoints):
 t=np.asarray(endpoints)
 if t.ndim!=1 or not len(t) or t.dtype.kind!='i' or np.any(t<0) or np.any(t[1:]<=t[:-1]):raise ValueError('strict integer axis endpoint clock required')
 if any(np.asarray(source[k]).shape!=(len(source['bag_ns']),3) for k in ['acceleration','angular_velocity']):raise ValueError('unaligned signed vectors')
 grid=t[:,None]+np.arange(-(WIDTH-1),1,dtype=np.int64)*50_000_000;q=m.im.lookup(source,grid);valid=q['status']==0;ids=np.maximum(q['latest_message'],0);count=valid.sum(1);safe=np.maximum(count,1);columns=[]
 for field in ['acceleration','angular_velocity']:
  values=np.where(valid[:,:,None],source[field][ids],0.);columns.append(values.sum(1)/safe[:,None])
 result=np.column_stack(columns)
 if not np.isfinite(result).all():raise ValueError('nonfinite signed-axis context')
 return result,count/WIDTH
def fit_head(x,y,w):
 x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float);w=np.asarray(w,dtype=float)
 if x.ndim!=2 or x.shape[1]!=6 or not len(x) or y.shape!=(len(x),) or w.shape!=y.shape or not all(np.isfinite(v).all() for v in [x,y,w]) or np.any(w<=0):raise ValueError('invalid ridge-head fitting arrays')
 total=w.sum();mean=(x*w[:,None]).sum(0)/total;variance=((x-mean)**2*w[:,None]).sum(0)/total;scale=np.sqrt(variance);scale[scale==0]=1.;z=(x-mean)/scale;intercept=float((y*w).sum()/total);coefficient=np.linalg.solve(z.T@(z*w[:,None])+ALPHA*np.eye(6),z.T@(w*(y-intercept)))
 return {'feature_names':FEATURE_NAMES,'alpha':ALPHA,'mean':mean.tolist(),'scale':scale.tolist(),'intercept':intercept,'coefficient':coefficient.tolist(),'meaning':'Empirical signed sensor-frame residual; no vehicle-axis or physical-bound claim'}
def head_predict(head,x):
 x=np.asarray(x,dtype=float);mean=np.asarray(head['mean']);scale=np.asarray(head['scale']);coefficient=np.asarray(head['coefficient'])
 if x.ndim!=2 or x.shape[1]!=6 or head['feature_names']!=FEATURE_NAMES or head['alpha']!=ALPHA or any(v.shape!=(6,) for v in [mean,scale,coefficient]) or np.any(scale<=0) or not all(np.isfinite(v).all() for v in [x,mean,scale,coefficient]) or not np.isfinite(head['intercept']):raise ValueError('invalid frozen ridge head')
 return head['intercept']+((x-mean)/scale)@coefficient
def correction_inputs(features,y,ids,parent_native):
 features=np.asarray(features,dtype=float);y=np.asarray(y,dtype=float);ids=np.asarray(ids);native=np.asarray(parent_native,dtype=float)
 if features.shape!=(len(y),6) or ids.ndim!=1 or ids.dtype.kind!='i' or not len(ids) or len(np.unique(ids))!=len(ids) or np.any(ids<0) or np.any(ids>=len(y)) or native.shape!=(2*len(ids),):raise ValueError('invalid scope-specific correction population')
 return np.tile(features[ids],(2,1)),np.tile(y[ids],2)-native
def diagnosis():
 members=read(m.r.ROOT/'membership.json');result=read(m.ROOT/'results.json')
 with np.load(m.ROOT/'oof_predictions.npz',allow_pickle=False) as z:y=z['target_mps'];a=z['original_context__prediction_mps'];b=z['radar_imu__prediction_mps']
 windows=np.array([v['window_id'] for v in members]);ea=a-y;eb=b-y;floor=oracle_errors(y,a,b);bywindow={}
 for name in sorted(set(windows.tolist())):
  mask=windows==name;bad=abs(eb[mask])>.5;bywindow[name]={'count':int(mask.sum()),'original_p95_mps':float(np.quantile(abs(ea[mask]),.95)),'uncorrected_p95_mps':float(np.quantile(abs(eb[mask]),.95)),'oracle_interval_p95_lower_bound_mps':float(np.quantile(floor[mask],.95)),'uncorrected_bias_mps':float(eb[mask].mean()),'uncorrected_tail_count':int(bad.sum()),'tail_same_sign_fraction':float(np.mean(ea[mask][bad]*eb[mask][bad]>0)) if bad.any() else None}
 return {'scope':'Retrospective paired-error diagnosis; oracle uses truth and is never a predictor','rows':len(y),'residual_correlation':float(np.corrcoef(ea,eb)[0,1]),'opposite_sign_residual_fraction':float(np.mean(ea*eb<0)),'by_window':bywindow,'unfiltered_windows_no_convex_blend_can_fix':[name for name,v in bywindow.items() if v['oracle_interval_p95_lower_bound_mps']>.5],'root_cause_established':False,'new_model_fitted':False,'reserved_final_opened':False}
def prepare():
 freeze()
 if (ROOT/'feature_cache.npz').exists() or (ROOT/'data_lock.json').exists():raise ValueError('signed-axis inputs already prepared')
 members=read(m.r.ROOT/'membership.json');index=read(m.c.MASK/'frame_index.json');fit=np.full((len(members),6),np.nan);fitcoverage=np.zeros(len(members));frames={}
 for key,route in m.sa.l.ROUTES.items():
  raw,source=m.sa.load(key);frames[key]=source['frame_id']
  if key=='Urban01_night':times=np.array([v['time_ns'] for v in index],dtype=np.int64);full,coverage=axis_features(source,times)
  else:
   ids=np.array([i for i,v in enumerate(members) if v['route']==route]);times=np.array([members[i]['time_ns'] for i in ids],dtype=np.int64);fit[ids],fitcoverage[ids]=axis_features(source,times)
 if not np.isfinite(fit).all() or set(frames.values())!={'livox_imu_link'}:raise ValueError('source assignment/frame differs')
 np.savez_compressed(ROOT/'feature_cache.npz',fit_axes=fit,fit_coverage=fitcoverage,full_axes=full,full_coverage=coverage);write(ROOT/'data_lock.json',{'analysis_lock_sha256':digest(ROOT/'analysis_lock.json'),'feature_cache_sha256':digest(ROOT/'feature_cache.npz'),'frame_ids':frames,'fitting_rows':27178,'full_endpoints':40690,'new_outcomes_scored':False})
def cache():
 lock=read(ROOT/'data_lock.json')
 if lock['analysis_lock_sha256']!=digest(ROOT/'analysis_lock.json') or lock['feature_cache_sha256']!=digest(ROOT/'feature_cache.npz') or lock['new_outcomes_scored'] is not False:raise ValueError('axis feature lock differs')
 with np.load(ROOT/'feature_cache.npz',allow_pickle=False) as z:values={k:z[k].copy() for k in z.files}
 if values['fit_axes'].shape!=(27178,6) or values['full_axes'].shape!=(40690,6):raise ValueError('axis fixed cohort differs')
 return values
def training_inputs(scope,data,axes):
 if scope not in SCOPES:raise ValueError('unknown/held-final scope')
 cal=read(m.ROOT/'models'/scope/'calibration.json');ids=np.array(cal['train_ids'],dtype=np.int64)
 with np.load(m.ROOT/'models'/scope/'radar_imu_export.npz',allow_pickle=False) as z:native=z['native'].copy()
 x,y=correction_inputs(axes,data['y'],ids,native);w=np.tile(m.r.weights(data['y'][ids],cal)*.5,2)
 return x,y,w,ids,native
def check_heads():
 freeze();cache();lock=read(ROOT/'head_lock.json');expected={f'heads/{scope}/{n}' for scope in SCOPES for n in ['head.json','training_check.npz']}
 if lock['analysis_lock_sha256']!=digest(ROOT/'analysis_lock.json') or lock['data_lock_sha256']!=digest(ROOT/'data_lock.json') or lock['training_trace_sha256']!=digest(ROOT/'training_trace.json') or lock['all_four_locked_before_new_scoring'] is not True or set(lock['files_sha256'])!=expected:raise ValueError('incomplete head lock')
 for n,h in lock['files_sha256'].items():
  if digest(ROOT/n)!=h:raise ValueError('frozen correction head changed')
def freeze(create=False):
 if not (ROOT/'analysis_lock.json').is_file() and not create:raise ValueError('signed-axis protocol must be frozen first')
 m.check_models();p=read(ROOT/'protocol.json')
 if p['width']!=WIDTH or p['alpha']!=ALPHA or p['scopes']!=list(SCOPES) or p['feature_names']!=FEATURE_NAMES or p['thresholds_changed'] or p['promotion_allowed']:raise ValueError('signed-axis prespecification changed')
 for name,h in p['input_sha256'].items():
  path=(ROOT/name).resolve()
  if not path.is_relative_to(RESEARCH) or digest(path)!=h:raise ValueError('signed-axis input changed: '+name)
 lock={'protocol_sha256':digest(ROOT/'protocol.json'),'code_sha256':{n:digest(ROOT/n) for n in ['axis_core.py','train_axis.py','score_axis.py']}};path=ROOT/'analysis_lock.json'
 if path.exists() and read(path)!=lock:raise ValueError('signed-axis implementation changed after freeze')
 if create and not path.exists():write(path,lock)
