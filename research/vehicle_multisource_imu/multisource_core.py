"""Fixed multi-source model comparison; no policy promotion."""
from pathlib import Path
import hashlib,sys
import numpy as np
import imu_core as im
import source_audit as sa
ROOT=Path(__file__).resolve().parent;RESEARCH=ROOT.parent
sys.path.insert(0,str(RESEARCH/'vehicle_signed_magnitude_fusion'));import fusion_core as f
r=f.r;c=f.c;g=f.g;d=f.d;av=f.av;read=f.read;write=f.write;digest=f.digest
NEW_METHODS=('radar_imu','imu_only');METHODS=('original_context',*NEW_METHODS);SCOPES=tuple(r.GROUPS)+('all_fit',)
FEATURE_NAMES={'radar_imu':list(c.FEATURE_NAMES)+im.FEATURE_NAMES,'imu_only':im.FEATURE_NAMES}
def assemble(radar,imu,anchor,median,method):
 radar=np.asarray(radar);imu=np.asarray(imu,dtype=float);anchor=np.asarray(anchor);median=np.asarray(median)
 if method not in NEW_METHODS or imu.ndim!=2 or imu.shape[1]!=45 or radar.shape!=(len(imu),75) or anchor.shape!=(len(imu),) or median.shape!=anchor.shape or not np.isfinite(imu).all():raise ValueError('unaligned multi-source features/method')
 if method=='imu_only':return imu.copy()
 return np.column_stack([g.assemble(radar,anchor,median),imu])
def targets_and_anchor(y,anchor,method):
 y=np.asarray(y,dtype=float);anchor=np.asarray(anchor,dtype=float)
 if method not in NEW_METHODS or y.ndim!=1 or anchor.shape!=y.shape or not np.isfinite(y).all() or np.any(y<0):raise ValueError('invalid fixed target/anchor')
 if method=='imu_only':return y.copy(),np.zeros(len(y))
 if not np.isfinite(anchor).all():raise ValueError('invalid primary radar anchor')
 return y-anchor,anchor.copy()
def endpoint_quality(pred,peer,source,imu_status):
 status=np.asarray(imu_status)
 if status.shape!=np.asarray(pred).shape:raise ValueError('unaligned endpoint IMU eligibility')
 return c.quality(pred,peer,source)&(status==0)
def bind_requests(replay,binding):
 binding=np.asarray(binding)
 if binding.dtype.kind!='b' or binding.shape!=replay['status'].shape:raise ValueError('invalid source-binding flags')
 result={k:v.copy() for k,v in replay.items()};rejected=(result['status']==0)&~binding;result['status'][rejected]=4;result['prediction_mps'][~binding]=np.nan;result['timing_prediction_mps'][~binding]=np.nan;result['imu_binding_fresh']=binding.copy()
 return result,int(rejected.sum())
def prepare_features():
 freeze();files=['feature_cache.npz','data_lock.json']
 if any((ROOT/n).exists() for n in files):raise ValueError('feature preparation already performed; no overwrite')
 data,members,groups,day,support=r.load_data();fit=np.full((len(members),45),np.nan);currents={}
 for key,route in sa.l.ROUTES.items():
  raw,source=sa.load(key)
  if key=='Urban01_night':
   index=read(c.MASK/'frame_index.json');times=np.array([v['time_ns'] for v in index],dtype=np.int64);full,current=im.features(source,times);currents=current
  else:
   ids=np.array([i for i,v in enumerate(members) if v['route']==route],dtype=np.int64);times=np.array([members[i]['time_ns'] for i in ids],dtype=np.int64);fit[ids],_=im.features(source,times)
  print('IMU FEATURES PREPARED',key,len(times),'endpoints',flush=True)
 if not np.isfinite(fit).all():raise ValueError('unassigned fitting IMU context')
 np.savez_compressed(ROOT/'feature_cache.npz',fit_imu=fit,full_imu=full,**{'full_current__'+k:v for k,v in currents.items()});write(ROOT/'data_lock.json',{'analysis_lock_sha256':digest(ROOT/'analysis_lock.json'),'feature_cache_sha256':digest(ROOT/'feature_cache.npz'),'source_audit_sha256':digest(ROOT/'source_audit.json'),'fitting_rows':27178,'full_raw_endpoints':40690,'imu_features':45,'new_accuracy_scoring_started':False})
def cache():
 lock=read(ROOT/'data_lock.json')
 if lock['analysis_lock_sha256']!=digest(ROOT/'analysis_lock.json') or lock['feature_cache_sha256']!=digest(ROOT/'feature_cache.npz') or lock['source_audit_sha256']!=digest(ROOT/'source_audit.json') or lock['new_accuracy_scoring_started'] is not False:raise ValueError('multi-source preparation lock changed')
 with np.load(ROOT/'feature_cache.npz',allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
 if arrays['fit_imu'].shape!=(27178,45) or arrays['full_imu'].shape!=(40690,45) or not np.isfinite(arrays['fit_imu']).all() or not np.isfinite(arrays['full_imu']).all():raise ValueError('invalid fixed IMU feature cache')
 return arrays
def training_arrays(data,base,imu,cal,method):
 ids=np.array(cal['train_ids']);anchor=c.e.affine(cal['anchor'],data['median_feature'][ids]);aug=c.m.augment(data['X'][ids],np.random.default_rng(20261019));a=assemble(base[ids],imu[ids],anchor,data['median_feature'][ids],method)
 if method=='radar_imu':b=np.column_stack([c.features(aug,anchor,data['median_feature'][ids]),imu[ids]])
 else:b=imu[ids].copy()
 target,used_anchor=targets_and_anchor(data['y'][ids],anchor,method);x=np.concatenate([a,b]);y=np.tile(target,2);w=np.tile(r.weights(data['y'][ids],cal)*.5,2)
 return x,y,w,np.tile(used_anchor,2),aug
def check_models():
 freeze();cache();lock=read(ROOT/'model_lock.json')
 if lock['analysis_lock_sha256']!=digest(ROOT/'analysis_lock.json') or lock['data_lock_sha256']!=digest(ROOT/'data_lock.json') or lock['training_trace_sha256']!=digest(ROOT/'training_trace.json') or lock['all_eight_locked_before_new_scoring'] is not True or lock['new_scores_created'] is not False or len(lock['files_sha256'])!=20:raise ValueError('incomplete/changed multi-source model lock')
 expected={f'models/{scope}/{n}' for scope in SCOPES for n in ['calibration.json','radar_imu.json','radar_imu_export.npz','imu_only.json','imu_only_export.npz']}
 if set(lock['files_sha256'])!=expected:raise ValueError('multi-source model membership changed')
 for n,h in lock['files_sha256'].items():
  if digest(ROOT/n)!=h:raise ValueError('frozen multi-source model changed: '+n)
 return lock
def route_summary(members,y,pred,support):
 if set(pred)!=set(METHODS) or len(y)!=27178 or any(d.vector(v).shape!=y.shape for v in pred.values()):raise ValueError('route comparison population differs')
 def cell(mask):return {'count':int(mask.sum()),'models':{n:d.metrics(y[mask],v[mask]) for n,v in pred.items()}}
 result={'supported_rows':len(y),'pooled':cell(np.ones(len(y),dtype=bool))}
 for name,key in [('by_group','group'),('by_recording','route'),('by_window','window_id')]:
  labels=np.array([v[key] for v in members]);result[name]={n:cell(labels==n) for n in sorted(set(labels.tolist()))}
 bins=d.speed_bins(y);days=np.array([v['day'] for v in members]);result['by_reference_speed']={name:cell(bins==i) for i,name in enumerate(d.SPEED_NAMES)};result['by_day_night']={'day':cell(days),'night':cell(~days)};result['raw_support_costs']=support
 result['diagnostic_failures']={n:{'pooled':result['pooled']['models'][n]['p95_mps']>.5,'groups':[k for k,v in result['by_group'].items() if v['models'][n]['p95_mps']>.5],'recordings':[k for k,v in result['by_recording'].items() if v['models'][n]['p95_mps']>.5],'windows':[k for k,v in result['by_window'].items() if v['models'][n]['p95_mps']>.5]} for n in METHODS}
 return result
def full_summary(predictions,arrays):
 index,ready,x,median,anchor,peer,source,old=f.load_full();events=np.array([v['time_ns'] for v in index],dtype=np.int64);times=av.request_times(int(events[0]),int(events[-1]),50_000_000);current={k:arrays['full_current__'+k] for k in ['endpoint_ns','status','latest_message','bag_ns','header_ns']};np.testing.assert_array_equal(current['endpoint_ns'],events)
 rawtargets=read(RESEARCH/'vehicle_continuous_radar/reference_targets.json');requesttargets=read(RESEARCH/'vehicle_availability_audit/request_reference.json')
 if events.tolist()!=[v['time_ns'] for v in rawtargets] or times.tolist()!=[v['time_ns'] for v in requesttargets] or set(predictions)!=set(NEW_METHODS):raise ValueError('fixed full-development inputs changed')
 yr=np.array([np.nan if v['reference_mps'] is None else v['reference_mps'] for v in rawtargets]);yq=np.array([np.nan if v['reference_mps'] is None else v['reference_mps'] for v in requesttargets]);fullpeer=np.full(len(index),np.nan);fullpeer[ready]=peer;lookup={int(t):i for i,t in enumerate(events)};blocks={}
 for v in read(c.OLD/'prepared/calibration/index.json'):
  if v['ready']:blocks.setdefault(v['window_id'],[]).append(lookup[v['time_ns']])
 values={'original_context':old,**predictions};methods={};output={};replays={}
 for name in METHODS:
  if values[name].shape!=old.shape:raise ValueError('unaligned ready development predictions')
  p=np.full(len(index),np.nan);p[ready]=values[name];keep=c.quality(p,fullpeer,source) if name=='original_context' else endpoint_quality(p,fullpeer,source,current['status']);replay=av.serve(times,events,ready,p,keep,100_000_000);binding_rejected=0
  if name!='original_context':replay,binding_rejected=bind_requests(replay,im.prediction_binding(times,replay['latest_endpoint'],current))
  methods[name]={'endpoints':c.c.account(yr[ready],p[ready],keep[ready],len(index)),'requests':f.request_summary(replay,yq),'previous_supported_blocks':{k:c.c.account(yr[ids],p[ids],keep[ids],len(ids)) for k,ids in blocks.items()},'additional_rejected_endpoints':int(np.sum(source&~keep)),'imu_binding_additional_rejected_requests':binding_rejected,'ungated_ready_endpoints':av.scores(yr,p,ready)};replays[name]=replay;output[name+'__quality']=keep;output[name+'__ready_prediction_mps']=values[name]
  for k,v in replay.items():output[name+'__request__'+k]=v
 gates={n:f.full_gates(v) for n,v in methods.items()};segment=(times-times[0])//300_000_000_000;paired={}
 for name in NEW_METHODS:
  common=(replays['original_context']['status']==0)&(replays[name]['status']==0)
  def cell(mask):return {'shared_requests':int(mask.sum()),'models':{n:av.scores(yq,replays[n]['prediction_mps'],mask) for n in ['original_context',name]}}
  paired[name]={'overall':cell(common),'segments_300s':{str(int(k)):cell(common&(segment==k)) for k in np.unique(segment)}}
 return {'raw_endpoints':len(index),'ready_endpoints':int(ready.sum()),'fixed_requests':len(times),'methods':methods,'gates':gates,'all_gates_pass':{n:all(v.values()) for n,v in gates.items()},'common_accepted_requests':paired,'no_formerly_rejected_source_endpoint_revived':all(not np.any(output[n+'__quality']&~source) for n in METHODS)},output
def freeze(create=False):
 if not (ROOT/'analysis_lock.json').is_file() and not create:raise ValueError('multi-source implementation must be frozen before use')
 f.freeze();sa.l.freeze();p=read(ROOT/'protocol.json')
 if p['model']!=read(r.ROOT/'protocol.json')['model'] or p['scopes']!=list(SCOPES) or p['new_methods']!=list(NEW_METHODS) or p['imu_feature_names']!=im.FEATURE_NAMES or p['primary_feature_count']!=120 or p['thresholds_changed'] or p['promotion_allowed'] or p['all_models_locked_before_new_scoring'] is not True:raise ValueError('fixed multi-source protocol changed')
 for n,h in p['input_sha256'].items():
  path=(ROOT/n).resolve()
  if not path.is_relative_to(RESEARCH) or digest(path)!=h:raise ValueError('frozen multi-source input changed: '+n)
 lock={'protocol_sha256':digest(ROOT/'protocol.json'),'code_sha256':{n:digest(ROOT/n) for n in ['imu_core.py','source_audit.py','multisource_core.py','train_multisource.py','score_multisource.py']}};file=ROOT/'analysis_lock.json'
 if file.exists() and read(file)!=lock:raise ValueError('multi-source implementation changed after freeze')
 if create and not file.exists():write(file,lock)
 return p
