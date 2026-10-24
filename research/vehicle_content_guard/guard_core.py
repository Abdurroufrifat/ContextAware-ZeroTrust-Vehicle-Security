"""Offline exact-value freshness guard; heuristic withholding, not physical attestation."""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parent;RESEARCH=ROOT.parent;PROJECT=RESEARCH.parent
sys.path.insert(0,str(RESEARCH/'vehicle_sensor_faults'));import fault_core as f
read=f.read;write=f.write;load=f.load;digest=f.digest
BUDGET=100_000_000
SOURCES=('Urban01_day','Urban03_day','Urban04_day','Urban03_night','Urban04_night','Urban01_night')
def content_stamps(source):
 bag=np.asarray(source['bag_ns']);usable=np.asarray(source['usable']);a=np.asarray(source['acceleration']);g=np.asarray(source['angular_velocity']);count=len(bag)
 if bag.ndim!=1 or bag.dtype.kind!='i' or np.any(bag<0) or np.any(bag[1:]<=bag[:-1]) or usable.shape!=(count,) or usable.dtype.kind!='b' or a.shape!=(count,3) or g.shape!=(count,3):raise ValueError('strict aligned source clocks/vectors required')
 if not count:return np.empty(0,np.int64)
 values=np.column_stack([a,g]);valid=usable&np.isfinite(values).all(1);ids=np.flatnonzero(valid);changed=np.zeros(count,bool)
 if len(ids):changed[ids[np.r_[True,np.any(values[ids][1:]!=values[ids][:-1],axis=1)]]]=True
 stamps=np.maximum.accumulate(np.where(changed,bag,-1));stamps[~valid]=-1
 return stamps
def is_fresh(stamp,query,budget=BUDGET):
 s=np.asarray(stamp);q=np.asarray(query)
 if s.shape!=q.shape or s.dtype.kind!='i' or q.dtype.kind!='i' or type(budget) is not int or budget<0:raise ValueError('aligned integer content clocks required')
 return (s>=0)&(q>=s)&(q-s<=budget)
def history_clean(source,times,stamp):
 grid=times[:,None]+np.arange(-127,1,dtype=np.int64)*50_000_000;q=f.im.lookup(source,grid);ids=np.maximum(q['latest_message'],0);selected=np.where(q['latest_message']>=0,stamp[ids],-1);bad=(q['status']==0)&~is_fresh(selected,grid);current=q['latest_message'][:,-1];current_stamp=np.where(current>=0,stamp[np.maximum(current,0)],-1)
 return ~bad.any(1),current_stamp
def apply_guard(old,fresh):
 fresh=np.asarray(fresh)
 if fresh.dtype.kind!='b' or fresh.shape!=old['status'].shape:raise ValueError('aligned boolean guard required')
 out={k:v.copy() for k,v in old.items()};reject=(out['status']==0)&~fresh;out['status'][reject]=4;out['prediction_mps'][out['status']!=0]=np.nan;out['content_guard_rejected']=reject
 return out
def controls():
 t=1_000_000_000+np.arange(501,dtype=np.int64)*10_000_000;v=np.zeros((501,3));v[:,2]=9.8;out={}
 for case in ['constant_legitimate_possible','tiny_alternating_jitter']:
  a=v.copy()
  if case=='tiny_alternating_jitter':a[:,0]=np.arange(501)%2*1e-12
  source={'bag_ns':t,'usable':np.ones(501,bool),'acceleration':a,'angular_velocity':np.zeros((501,3))};stamps=content_stamps(source);fresh=is_fresh(stamps,t);out[case]={'records':501,'fresh':int(fresh.sum()),'withheld':int((~fresh).sum()),'meaning':'Possible benign constant content is withheld' if case=='constant_legitimate_possible' else 'Tiny value changes evade exact-equality guard'}
 return out
def source_checks():
 out={}
 for key in SOURCES:
  source=f.im.decode_payloads(load(RESEARCH/'vehicle_livox_collection/runtime_export'/(key+'.npz')));stamps=content_stamps(source);valid=source['usable']&np.isfinite(source['acceleration']).all(1)&np.isfinite(source['angular_velocity']).all(1);fresh=is_fresh(stamps,source['bag_ns']);age=source['bag_ns'][valid]-stamps[valid];out[key]={'captured_records':len(stamps),'valid_vector_records':int(valid.sum()),'unchanged_content_withheld':int(np.sum(valid&~fresh)),'maximum_content_age_ns':int(age.max()) if len(age) else None}
 return out
def summarize(data,old,case):
 source=f.mutate(data['source_imu'],case,data['start'],data['end']);stamp=content_stamps(source);clean,current_stamp=history_clean(source,data['events'],stamp);endpoint_fresh=is_fresh(current_stamp,data['events']);results={};arrays={'message_content_stamp_ns':stamp,'history_clean':clean,'endpoint_content_stamp_ns':current_stamp,'endpoint_content_fresh':endpoint_fresh};latest=old['nonlinear7__request__latest_endpoint'];ids=np.maximum(latest,0);request_fresh=(latest>=0)&clean[ids]&is_fresh(current_stamp[ids],data['times']);arrays['request_content_fresh']=request_fresh
 for method in f.METHODS:
  quality=old[method+'__quality'].copy();replay={k:old[method+'__request__'+k] for k in ['latest_endpoint','age_ns','status','prediction_mps']}
  if method!='radar_only':quality&=clean&endpoint_fresh;replay=apply_guard(replay,request_fresh)
  else:replay=apply_guard(replay,np.ones(len(latest),bool))
  if np.any((replay['status']==0)&(old[method+'__request__status']!=0)) or np.any(quality&~old[method+'__quality']):raise ValueError('guard revived rejected evidence')
  p=np.full(len(data['events']),np.nan);p[data['ready']]=old[method+'__ready_prediction_mps'];v=f.summarize(data,p,quality,replay);v['additional_withheld_requests']=int(replay['content_guard_rejected'].sum());v['nominal_prediction_values_changed']=False;results[method]=v;arrays[method+'__quality']=quality
  for k,a in replay.items():arrays[method+'__request__'+k]=a
 return results,arrays
def freeze(create=False):
 p=read(ROOT/'protocol.json')
 if p['budget_ns']!=BUDGET or p['cases']!=list(f.CASES) or p['thresholds_tuned'] or p['production_policy_changed'] or p['promotion_allowed']:raise ValueError('fixed guard scope differs')
 for name,h in p['input_sha256'].items():
  if 'urban02' in name.lower() or 'rav4' in name.lower() or name.startswith('vehicle_final_evaluation/'):raise ValueError('protected input')
  if digest(RESEARCH/name)!=h:raise ValueError('frozen guard input changed '+name)
 lock={'protocol_sha256':digest(ROOT/'protocol.json'),'code_sha256':{n:digest(ROOT/n) for n in ['guard_core.py','run_guard.py']}}
 if (ROOT/'analysis_lock.json').exists():
  if read(ROOT/'analysis_lock.json')!=lock:raise ValueError('frozen guard code changed')
 elif create:write(ROOT/'analysis_lock.json',lock)
 else:raise ValueError('freeze guard first')
