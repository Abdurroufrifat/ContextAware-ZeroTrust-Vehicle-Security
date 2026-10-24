"""Fixed training-scaled deadband content guard; offline research only."""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parent;RESEARCH=ROOT.parent;PROJECT=RESEARCH.parent
sys.path.insert(0,str(RESEARCH/'vehicle_content_guard'));import guard_core as g
f=g.f;read=g.read;write=g.write;load=g.load;digest=g.digest
EPSILON=.001;BUDGET=100_000_000
FITS=('Urban01_day','Urban03_day','Urban04_day','Urban03_night','Urban04_night')
CASES=(*f.CASES,'tiny_jitter','threshold_aware_pulses')
POLICIES=('none','exact','deadband')
def scale_from_vectors(values):
 v=np.asarray(values,float)
 if v.ndim!=2 or v.shape[1]!=6 or not len(v) or not np.isfinite(v).all():raise ValueError('finite fitting vectors required')
 lo,hi=np.quantile(v,[.25,.75],axis=0);raw=hi-lo;fallback=np.flatnonzero(raw==0);scale=np.where(raw>0,raw,1.)
 return {'q25':lo.tolist(),'q75':hi.tolist(),'scale':scale.tolist(),'fallback_axes':fallback.tolist()}
def stamps(source,scale,epsilon=EPSILON):
 valid=g.content_stamps(source)>=0;scale=np.asarray(scale,float)
 if scale.shape!=(6,) or not np.isfinite(scale).all() or np.any(scale<=0) or not np.isfinite(epsilon) or epsilon<0:raise ValueError('positive finite scale and nonnegative epsilon required')
 values=np.column_stack([source['acceleration'],source['angular_velocity']]);limits=scale*epsilon;out=np.full(len(valid),-1,np.int64);anchor=None;stamp=-1
 for i in np.flatnonzero(valid):
  row=values[i]
  if anchor is None or np.any(abs(row-anchor)>limits):anchor=row.copy();stamp=int(source['bag_ns'][i])
  out[i]=stamp
 return out
def freeze(create=False):
 p=read(ROOT/'protocol.json')
 if p['epsilon']!=EPSILON or p['budget_ns']!=BUDGET or p['fitting_sources']!=list(FITS) or p['cases']!=list(CASES) or p['threshold_search'] or p['promotion_allowed']:raise ValueError('fixed deadband scope differs')
 for name,h in p['input_sha256'].items():
  if 'urban02' in name.lower() or 'rav4' in name.lower() or name.startswith('vehicle_final_evaluation/'):raise ValueError('protected input')
  if digest(RESEARCH/name)!=h:raise ValueError('input changed '+name)
 lock={'protocol_sha256':digest(ROOT/'protocol.json'),'code_sha256':{n:digest(ROOT/n) for n in ['deadband_core.py','run_deadband.py']}}
 if (ROOT/'analysis_lock.json').exists():
  if read(ROOT/'analysis_lock.json')!=lock:raise ValueError('frozen deadband code changed')
 elif create:write(ROOT/'analysis_lock.json',lock)
 else:raise ValueError('freeze before calibration')
def calibrate():
 freeze()
 if (ROOT/'parameters.json').exists() or (ROOT/'parameter_lock.json').exists():raise ValueError('already calibrated; no retry')
 rows=[];counts={}
 for key in FITS:
  source=f.im.decode_payloads(load(RESEARCH/'vehicle_livox_collection/runtime_export'/(key+'.npz')));v=np.column_stack([source['acceleration'],source['angular_velocity']]);v=v[source['usable']&np.isfinite(v).all(1)];rows.append(v);counts[key]=len(v)
 p=scale_from_vectors(np.concatenate(rows));p.update(epsilon=EPSILON,budget_ns=BUDGET,fitting_record_counts=counts,fitting_rows=sum(counts.values()),meaning='Empirical IQR scale; not sensor uncertainty, resolution or physical calibration');write(ROOT/'parameters.json',p);write(ROOT/'parameter_lock.json',{'analysis_lock_sha256':digest(ROOT/'analysis_lock.json'),'parameters_sha256':digest(ROOT/'parameters.json'),'locked_before_new_guard_outcomes':True});print('FITTING-ONLY SCALE LOCKED',p['fitting_rows'],'vectors; scales',p['scale'],flush=True)
def check_parameters():
 freeze();lock=read(ROOT/'parameter_lock.json')
 if lock['analysis_lock_sha256']!=digest(ROOT/'analysis_lock.json') or lock['parameters_sha256']!=digest(ROOT/'parameters.json') or lock['locked_before_new_guard_outcomes'] is not True:raise ValueError('parameter lock changed')
 return read(ROOT/'parameters.json')
def attack_source(data,case,scale):
 if case in f.CASES:return f.mutate(data['source_imu'],case,data['start'],data['end'])
 if case not in CASES:raise ValueError('unknown attack case')
 source=f.mutate(data['source_imu'],'frozen_values',data['start'],data['end']);ids=np.flatnonzero((source['bag_ns']>=data['start'])&(source['bag_ns']<data['end']));amplitude=1e-12 if case=='tiny_jitter' else 2*EPSILON*float(scale[0]);source['acceleration'][ids,0]+=(np.arange(len(ids))%2)*amplitude;source['acceleration_norm']=np.hypot.reduce(source['acceleration'],axis=1)
 return source
def predict_source(data,source):
 features,current=f.im.features(source,data['events']);a=f.axes(source,data['events']);ready=data['ready'];parent=np.maximum(0.,data['anchor']+f.infer(data['parent_model'],np.column_stack([data['radar'],features[ready]])));candidate=np.maximum(0.,parent+f.infer(data['head'],np.column_stack([parent,a[ready]])));pred={'radar_only':data['radar_only'],'radar_imu_parent':parent,'nonlinear7':candidate};latest=np.searchsorted(data['events'],data['times'],side='right')-1;binding=f.binding(data['times'],latest,current);out={}
 for method,values in pred.items():
  p=np.full(len(ready),np.nan);p[ready]=values;keep=data['source']&np.isfinite(p)&np.isfinite(data['peer'])&(p>=0)&(data['peer']>=0)&(abs(p-data['peer'])<=1.);bound=np.ones(len(latest),bool)
  if method!='radar_only':keep&=current['status']==0;bound=binding
  replay=f.n.serve(data['times'],data['events'],ready,p,keep,bound);out[method+'__ready_prediction_mps']=values;out[method+'__quality']=keep
  for k,v in replay.items():out[method+'__request__'+k]=v
 for k in ['status','bag_ns','header_ns']:out['current__'+k]=current[k]
 out['binding']=binding;return out
def overlay(data,base,source,policy,scale):
 stamp=stamps(source,scale) if policy=='deadband' else g.content_stamps(source);clean,current=g.history_clean(source,data['events'],stamp);endfresh=g.is_fresh(current,data['events']);latest=base['nonlinear7__request__latest_endpoint'];ids=np.maximum(latest,0);fresh=(latest>=0)&clean[ids]&g.is_fresh(current[ids],data['times']);arrays={'message_content_stamp_ns':stamp,'history_clean':clean,'endpoint_content_stamp_ns':current,'endpoint_content_fresh':endfresh,'request_content_fresh':fresh};results={}
 for method in f.METHODS:
  quality=base[method+'__quality'].copy();old={k:base[method+'__request__'+k] for k in ['latest_endpoint','age_ns','status','prediction_mps']};enabled=policy!='none' and method!='radar_only';replay=g.apply_guard(old,fresh if enabled else np.ones(len(latest),bool))
  if enabled:quality&=clean&endfresh
  if np.any((replay['status']==0)&(old['status']!=0)) or np.any(quality&~base[method+'__quality']):raise ValueError('prior rejection revived')
  p=np.full(len(data['events']),np.nan);p[data['ready']]=base[method+'__ready_prediction_mps'];value=f.summarize(data,p,quality,replay);value['additional_withheld_requests']=int(replay['content_guard_rejected'].sum());value['nominal_prediction_values_changed']=False;results[method]=value;arrays[method+'__quality']=quality
  for k,v in replay.items():arrays[method+'__request__'+k]=v
 return results,arrays
def source_checks(scale):
 out={}
 for key in g.SOURCES:
  source=f.im.decode_payloads(load(RESEARCH/'vehicle_livox_collection/runtime_export'/(key+'.npz')));stamp=stamps(source,scale);valid=stamp>=0;fresh=g.is_fresh(stamp,source['bag_ns']);out[key]={'captured_records':len(stamp),'valid_records':int(valid.sum()),'content_withheld_records':int(np.sum(valid&~fresh)),'calibration_source':key in FITS}
 return out
def controls(scale):
 t=1_000_000_000+np.arange(501,dtype=np.int64)*10_000_000;out={}
 for case,amplitude in [('constant_legitimate_possible',0.),('tiny_jitter',1e-12),('threshold_aware_pulses',2*EPSILON*float(scale[0]))]:
  a=np.zeros((501,3));a[:,2]=9.8;a[:,0]=np.arange(501)%2*amplitude;source={'bag_ns':t,'usable':np.ones(501,bool),'acceleration':a,'angular_velocity':np.zeros((501,3))};out[case]={'records':501,**{p:int(g.is_fresh(stamps(source,scale) if p=='deadband' else g.content_stamps(source),t).sum()) for p in ['exact','deadband']}}
 return out
