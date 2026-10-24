"""One complete same-recording development evaluation after all models freeze."""
import sys
import numpy as np
from masked_core import ROOT,OLD,PREVIOUS,c,e,read,write,digest,freeze,MaskedGrid,ensemble
sys.path.insert(0,str(ROOT.parent/'vehicle_availability_audit'))
from availability_core import serve,scores,STATUS,request_times
def prepare():
 raw,_=c.decode_radar();grid=MaskedGrid(raw);start=raw[0]['bag_time_ns'];index=[];xs=[];med=[]
 for i,row in enumerate(raw):
  t=row['bag_time_ns'];pair=grid.at_masked(t,start);ready=pair is not None;index.append({'ordinal':i,'time_ns':t,'ready':ready,'array_index':len(xs) if ready else None,'valid_ticks':int(pair[0][3].sum()) if ready else None})
  if ready:xs.append(pair[0]);med.append(pair[1])
 return {'X':np.asarray(xs,dtype=np.float32),'median_feature':np.asarray(med)},index
def check_models():
 lock=read(ROOT/'model_lock.json')
 if lock['analysis_lock_sha256']!=digest(ROOT/'analysis_lock.json') or lock['training_trace_sha256']!=digest(ROOT/'training_trace.json'):raise ValueError('model training lock differs')
 if len(lock['models_sha256'])!=12:raise ValueError('incomplete matched model conditions')
 for name,h in lock['models_sha256'].items():
  if digest(ROOT/name)!=h:raise ValueError('frozen model changed: '+name)
 return lock
def predict(data):
 check_models();anchor=e.affine(read(OLD/'classical_models.json')['reference_median_affine'],data['median_feature']);out={'anchor':anchor}
 for condition in ['clean_control','masked_training']:
  for kind in ['causal_residual_tcn','masked_cnn_4ch']:
   models=[read(ROOT/f'models/{condition}/{kind}_{s}.json') for s in [141,142,143]];mean,seeds=ensemble(models,data['X'],anchor);key=condition+'__'+kind;out[key]=mean
   for seed,value in zip([141,142,143],seeds):out[key+'__'+str(seed)]=value
   print('MASK MODEL REPLAY',condition,kind,len(anchor),'endpoints',flush=True)
 return out
def request_summary(replay,y):
 times=replay['request_times_ns'];status=replay['status'];segment=(times-times[0])//300000000000
 def stats(ids):
  st=status[ids];keep=st==0;return {'requests':len(st),'status_counts':{label:int((st==code).sum()) for code,label in STATUS.items()},'usable_fraction':float(keep.mean()),'reference_missing_all_requests':int((~np.isfinite(y[ids])).sum()),'accepted':scores(y[ids],replay['prediction_mps'][ids],keep)}
 return {'overall':stats(np.ones(len(times),dtype=bool)),'segments_300s':{str(int(k)):stats(segment==k) for k in np.unique(segment)}}
def summarize(index,pred):
 raw_targets=read(PREVIOUS/'reference_targets.json');request_targets=read(ROOT.parent/'vehicle_availability_audit/request_reference.json');old_index=read(OLD/'prepared/calibration/index.json');events=np.array([x['time_ns'] for x in index],dtype=np.int64);ready=np.array([x['ready'] for x in index],dtype=bool);times=request_times(int(events[0]),int(events[-1]),50000000)
 if [int(t) for t in times]!=[p['time_ns'] for p in request_targets]:raise ValueError('request schedule differs from baseline')
 if [int(t) for t in events]!=[p['time_ns'] for p in raw_targets]:raise ValueError('endpoint population differs from baseline')
 yr=np.array([np.nan if x['reference_mps'] is None else x['reference_mps'] for x in raw_targets]);yq=np.array([np.nan if x['reference_mps'] is None else x['reference_mps'] for x in request_targets]);lookup={int(t):i for i,t in enumerate(events)};old_ids={}
 for point in old_index:
  if point['ready']:old_ids.setdefault(point['window_id'],[]).append(lookup[point['time_ns']])
 baseline_ready=np.array([x['ready'] for x in read(PREVIOUS/'frame_index.json')],dtype=bool);new=ready&~baseline_ready;conditions={};arrays={}
 for condition in ['clean_control','masked_training']:
  p=np.full(len(index),np.nan);cnn=np.full(len(index),np.nan);p[ready]=pred[condition+'__causal_residual_tcn'];cnn[ready]=pred[condition+'__masked_cnn_4ch'];keep=np.array([c.q.usable(float(a),float(b),bool(r)) for a,b,r in zip(p,cnn,ready)],dtype=bool)
  endpoint=c.account(yr[ready],p[ready],keep[ready],len(index));replay=serve(times,events,ready,p,keep,100000000);requests=request_summary(replay,yq);timing=request_summary(serve(times,events,ready,p,ready.copy(),100000000),yq);overlap={name:c.account(yr[ids],p[ids],keep[ids],len(ids)) for name,ids in old_ids.items()}
  conditions[condition]={'endpoints':endpoint,'requests':requests,'requests_timing_only':timing,'previous_supported_blocks':overlap,'newly_supported_endpoints':c.account(yr[new],p[new],keep[new],int(new.sum()))};arrays[condition+'__quality']=keep
  for key,value in replay.items():arrays[condition+'__request__'+key]=value
 primary=conditions['masked_training'];requests=primary['requests'];groups=[requests['overall']]+list(requests['segments_300s'].values());blocks=list(primary['previous_supported_blocks'].values());gates={'request_usable_fraction_ge_95pct_overall_and_each_segment':all(x['usable_fraction']>=.95 for x in groups),'request_accepted_reference_coverage_ge_95pct_overall_and_each_segment':all(x['accepted']['count']>0 and x['accepted']['count']/(x['accepted']['count']+x['accepted']['missing_truth'])>=.95 for x in groups),'request_accepted_p95_le_half_mps_overall_and_each_segment':all(x['accepted']['count']>0 and x['accepted']['p95_mps']<=.5 for x in groups),'zero_accepted_request_errors_above2mps':requests['overall']['accepted']['count']>0 and requests['overall']['accepted']['above_2_mps']==0,'previous_supported_block_quality_retention_ge95pct':all(x['whole_acceptance_fraction']>=.95 for x in blocks),'previous_supported_block_accepted_p95_le_half_mps':all(x['accepted']['count']>0 and x['accepted']['p95_mps']<=.5 for x in blocks)}
 result={'scope':'Same inspected development recording; matched clean-versus-masked fine-tuning; not held-out validation','raw_endpoints':len(index),'features_ready':int(ready.sum()),'unready':int((~ready).sum()),'newly_supported_endpoints':int(new.sum()),'prior_supported_endpoints_lost':int((baseline_ready&~ready).sum()),'fixed_requests':len(times),'conditions':conditions,'frozen_original_request_baseline':read(ROOT.parent/'vehicle_availability_audit/results.json')['overall'],'gates':gates,'all_gates_pass':all(gates.values()),'decision':'MASKED_DEVELOPMENT_GATES_PASS_NO_PROMOTION' if all(gates.values()) else 'STOP_AFTER_MASKED_DEVELOPMENT_FAILURE','reserved_final_opened':False,'witness_enabled':False,'production_policy_changed':False}
 return result,arrays
def score():
 freeze();check_models()
 if (ROOT/'results.json').exists():raise ValueError('already scored; no result-driven rerun')
 data,index=prepare();write(ROOT/'frame_index.json',index);np.savez_compressed(ROOT/'prepared.npz',**data);pred=predict(data);np.savez_compressed(ROOT/'predictions.npz',**pred);result,arrays=summarize(index,pred);np.savez_compressed(ROOT/'decisions.npz',**arrays);write(ROOT/'results.json',result);print(result['decision'],result['gates'],flush=True)
if __name__=='__main__':score()
