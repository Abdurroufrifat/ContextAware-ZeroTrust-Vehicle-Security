"""One fixed signed + magnitude feature fusion; no policy promotion."""
from pathlib import Path
import hashlib,sys
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'vehicle_magnitude_ablation'))
import magnitude_core as g
r=g.r;c=g.c;d=g.d;read=g.read;write=g.write;digest=g.digest;close=g.close
sys.path.insert(0,str(ROOT.parent/'vehicle_availability_audit'))
import availability_core as av
sys.path.insert(0,str(ROOT.parent/'vehicle_masked_history'))
from score_masked import request_summary
EXTRA=[0]+list(range(3,75));FEATURE_NAMES=list(c.FEATURE_NAMES)+['extra_magnitude_current_div10']+['extra_magnitude_'+c.FEATURE_NAMES[i] for i in EXTRA[1:]]
SCOPES=tuple(r.GROUPS)+('all_fit',);METHODS=('original_context','signed_magnitude_fusion')

def features(x,anchor,median):
 signed=c.features(x,anchor,median);mag=c.features(g.magnitude_grid(x),anchor,median);return np.column_stack([signed,mag[:,EXTRA]])

def assemble(base,anchor,median):
 base=np.asarray(base,dtype=float)
 if base.ndim!=2 or base.shape[1]!=148 or not np.isfinite(base).all():raise ValueError('invalid fusion feature cache')
 signed=g.assemble(base[:,:75],anchor,median);return np.column_stack([signed,base[:,75:]])

def original_base(data):return features(data['X'],np.zeros(len(data['y'])),np.zeros(len(data['y'])))

def calibration(data,groups,day,scope):
 if scope not in SCOPES:raise ValueError('unknown fixed fusion scope')
 if scope!='all_fit':return g.validated_calibration(r.calibrate(data,groups,day,scope),read(r.ROOT/'folds'/scope/'calibration.json'))
 return {'held_group':scope,'anchor':read(c.OLD/'classical_models.json')['reference_median_affine'],'train_ids':list(range(len(groups))),'held_ids':[],'calibration_ids':np.flatnonzero(day).tolist(),'bin_counts':np.bincount(np.digitize(data['y'][day],[1,5,10]),minlength=4).tolist()}

def fit_inputs(data,base,cal):
 ids=np.array(cal['train_ids']);anchor=c.e.affine(cal['anchor'],data['median_feature'][ids]);aug=c.m.augment(data['X'][ids],np.random.default_rng(20261019));a=assemble(base[ids],anchor,data['median_feature'][ids]);b=features(aug,anchor,data['median_feature'][ids]);x=np.concatenate([a,b]);y=np.tile(data['y'][ids]-anchor,2);w=np.tile(r.weights(data['y'][ids],cal)*.5,2);return x,y,w,np.tile(anchor,2),aug

def check_parent_inputs(scope,x,y,w):
 if scope=='all_fit':
  trace=read(c.ROOT/'training_trace.json');n=len(x)//2
  if g.array_digest(x[:n,:75])!=trace['original_feature_sha256'] or g.array_digest(x[n:,:75])!=trace['masked_feature_sha256']:raise ValueError('all-fit original signed feature prefix differs')
 else:
  trace=next(v for v in read(r.ROOT/'training_trace.json')['runs'] if v['held_group']==scope)
  if g.array_digest(x[:,:75])!=trace['feature_sha256'] or g.array_digest(y)!=trace['target_sha256'] or g.array_digest(w)!=trace['weight_sha256']:raise ValueError('fold original signed prefix/target/weight differs')

def load_full():
 index=read(c.MASK/'frame_index.json');ready=np.array([v['ready'] for v in index],dtype=bool)
 if [v['array_index'] for v in index if v['ready']]!=list(range(int(ready.sum()))):raise ValueError('full-development prepared row identity differs')
 with np.load(c.MASK/'prepared.npz',allow_pickle=False) as z:x=z['X'].copy();median=z['median_feature'].copy()
 with np.load(c.NIGHT/'predictions.npz',allow_pickle=False) as z:anchor=z['anchor'].copy();peer=z['day_plus_night__masked_cnn_4ch'].copy()
 with np.load(c.NIGHT/'decisions.npz',allow_pickle=False) as z:source=z['day_plus_night__quality'].copy()
 with np.load(c.ROOT/'replay.npz',allow_pickle=False) as z:old=z['context_prediction_mps'].copy()
 if len(x)!=int(ready.sum()) or any(v.shape!=(len(x),) for v in [median,anchor,peer,old]) or source.shape!=ready.shape or source.dtype.kind!='b' or np.any(source&~ready):raise ValueError('full-development alignment differs')
 return index,ready,x,median,anchor,peer,source,old

def route_summary(members,y,pred,support):
 if len(y)!=27178 or set(pred)!=set(METHODS) or any(d.vector(v).shape!=y.shape for v in pred.values()):raise ValueError('fusion route scoring cohort differs')
 def cell(mask):return {'count':int(mask.sum()),'models':{n:d.metrics(y[mask],v[mask]) for n,v in pred.items()}}
 result={'supported_rows':len(y),'pooled':cell(np.ones(len(y),bool))}
 for category,key in [('by_group','group'),('by_recording','route'),('by_window','window_id')]:
  labels=np.array([v[key] for v in members]);result[category]={n:cell(labels==n) for n in sorted(set(labels.tolist()))}
 bins=d.speed_bins(y);day=np.array([m['day'] for m in members]);result['by_reference_speed']={name:cell(bins==i) for i,name in enumerate(d.SPEED_NAMES)};result['by_day_night']={'day':cell(day),'night':cell(~day)};result['raw_support_costs']=support
 result['diagnostic_failures']={n:{'pooled':result['pooled']['models'][n]['p95_mps']>.5,'groups':[k for k,v in result['by_group'].items() if v['models'][n]['p95_mps']>.5],'recordings':[k for k,v in result['by_recording'].items() if v['models'][n]['p95_mps']>.5],'windows':[k for k,v in result['by_window'].items() if v['models'][n]['p95_mps']>.5]} for n in METHODS};return result

def full_gates(primary):
 groups=[primary['requests']['overall']]+list(primary['requests']['segments_300s'].values());b=list(primary['previous_supported_blocks'].values())
 return {'request_usable_fraction_ge95pct_every_segment_and_overall':all(v['usable_fraction']>=.95 for v in groups),'request_accepted_reference_coverage_ge95pct_every_segment_and_overall':all(v['accepted']['count']>0 and v['accepted']['count']/(v['accepted']['count']+v['accepted']['missing_truth'])>=.95 for v in groups),'request_accepted_p95_le_half_mps_every_segment_and_overall':all(v['accepted']['count']>0 and v['accepted']['p95_mps']<=.5 for v in groups),'zero_accepted_recorded_request_errors_above2mps':primary['requests']['overall']['accepted']['count']>0 and primary['requests']['overall']['accepted']['above_2_mps']==0,'old_supported_block_quality_retention_ge95pct':all(v['whole_acceptance_fraction']>=.95 for v in b),'old_supported_block_accepted_p95_le_half_mps':all(v['accepted']['count']>0 and v['accepted']['p95_mps']<=.5 for v in b)}

def full_summary(new):
 index,ready,x,median,anchor,peer,source,old=load_full();events=np.array([v['time_ns'] for v in index],dtype=np.int64);times=av.request_times(int(events[0]),int(events[-1]),50000000);rawtargets=read(ROOT.parent/'vehicle_continuous_radar/reference_targets.json');requesttargets=read(ROOT.parent/'vehicle_availability_audit/request_reference.json')
 if [int(v) for v in events]!=[v['time_ns'] for v in rawtargets] or [int(v) for v in times]!=[v['time_ns'] for v in requesttargets] or new.shape!=old.shape:raise ValueError('original development schedule differs')
 yr=np.array([np.nan if v['reference_mps'] is None else v['reference_mps'] for v in rawtargets]);yq=np.array([np.nan if v['reference_mps'] is None else v['reference_mps'] for v in requesttargets]);fullpeer=np.full(len(index),np.nan);fullpeer[ready]=peer;lookup={int(t):i for i,t in enumerate(events)};blocks={}
 for v in read(c.OLD/'prepared/calibration/index.json'):
  if v['ready']:blocks.setdefault(v['window_id'],[]).append(lookup[v['time_ns']])
 methods={};arrays={};replays={}
 for name,values in zip(METHODS,[old,new]):
  p=np.full(len(index),np.nan);p[ready]=values;keep=c.quality(p,fullpeer,source);replay=av.serve(times,events,ready,p,keep,100000000);methods[name]={'endpoints':c.c.account(yr[ready],p[ready],keep[ready],len(index)),'requests':request_summary(replay,yq),'previous_supported_blocks':{k:c.c.account(yr[ids],p[ids],keep[ids],len(ids)) for k,ids in blocks.items()},'additional_rejected_endpoints':int(np.sum(source&~keep)),'ungated_ready_endpoints':av.scores(yr,p,ready)};replays[name]=replay;arrays[name+'__quality']=keep;arrays[name+'__ready_prediction_mps']=values
  for k,v in replay.items():arrays[name+'__request__'+k]=v
 common=(replays[METHODS[0]]['status']==0)&(replays[METHODS[1]]['status']==0);segment=(times-times[0])//300000000000
 def paired(mask):return {'shared_requests':int(mask.sum()),'models':{n:av.scores(yq,v['prediction_mps'],mask) for n,v in replays.items()}}
 gates={n:full_gates(v) for n,v in methods.items()};result={'raw_endpoints':len(index),'ready_endpoints':int(ready.sum()),'fixed_requests':len(times),'methods':methods,'gates':gates,'all_gates_pass':{n:all(v.values()) for n,v in gates.items()},'common_accepted_requests':{'overall':paired(common),'segments_300s':{str(int(k)):paired(common&(segment==k)) for k in np.unique(segment)}},'no_formerly_rejected_endpoint_revived':all(not np.any(arrays[n+'__quality']&~source) for n in METHODS)}
 return result,arrays

def freeze(create=False):
 r.freeze();g.freeze();c.freeze();p=read(ROOT/'protocol.json')
 if p['feature_count']!=148 or p['model']!=read(r.ROOT/'protocol.json')['model'] or p['scopes']!=list(SCOPES) or not p['all_four_locked_before_new_scoring'] or p['thresholds_changed'] or p['promotion_allowed']:raise ValueError('fixed fusion design changed')
 for n,h in p['input_sha256'].items():
  f=(ROOT/n).resolve()
  if not f.is_relative_to(ROOT.parent) or digest(f)!=h:raise ValueError('fusion source changed: '+n)
 lock={'protocol_sha256':digest(ROOT/'protocol.json'),'code_sha256':{n:digest(ROOT/n) for n in ['fusion_core.py','train_fusion.py','score_fusion.py']}};path=ROOT/'analysis_lock.json'
 if path.exists() and read(path)!=lock:raise ValueError('fusion implementation changed after freeze')
 if create and not path.exists():write(path,lock)
 return p

def check_models():
 lock=read(ROOT/'model_lock.json')
 if not lock['all_four_locked_before_new_scoring'] or lock['new_scores_created'] or len(lock['files_sha256'])!=12 or lock['analysis_lock_sha256']!=digest(ROOT/'analysis_lock.json') or lock['training_trace_sha256']!=digest(ROOT/'training_trace.json'):raise ValueError('fusion model lock incomplete')
 for n,h in lock['files_sha256'].items():
  if digest(ROOT/n)!=h:raise ValueError('fusion model artifact changed: '+n)
 return lock
