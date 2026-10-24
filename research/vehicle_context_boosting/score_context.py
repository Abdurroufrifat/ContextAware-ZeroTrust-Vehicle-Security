"""Fixed development comparison, retaining earlier rejections and all costs."""
from context_core import *
from score_masked import request_summary
from availability_core import serve,request_times,scores

def check_model():
 lock=read(ROOT/'model_lock.json')
 for key,file in [('analysis_lock_sha256','analysis_lock.json'),('model_sha256','model.json'),('training_trace_sha256','training_trace.json'),('fit_export_check_sha256','fit_export_check.npz')]:
  if lock[key]!=digest(ROOT/file):raise ValueError('boosted model lock differs')
 return read(ROOT/'model.json')

def compute():
 freeze();model=check_model();index=read(MASK/'frame_index.json');ready=np.array([r['ready'] for r in index]);events=np.array([r['time_ns'] for r in index],dtype=np.int64);times=request_times(int(events[0]),int(events[-1]),50000000)
 with np.load(MASK/'prepared.npz',allow_pickle=False) as z:x=z['X'].copy();median=z['median_feature'].copy()
 with np.load(NIGHT/'predictions.npz',allow_pickle=False) as z:old=z['day_plus_night__causal_residual_tcn'].copy();peer=z['day_plus_night__masked_cnn_4ch'].copy();anchor=z['anchor'].copy()
 with np.load(NIGHT/'decisions.npz',allow_pickle=False) as z:source_quality=z['day_plus_night__quality'].copy()
 context=features(x,anchor,median);new=predict(model,context,anchor);fullpeer=np.full(len(index),np.nan);fullpeer[ready]=peer
 rawtargets=read(ROOT.parent/'vehicle_continuous_radar/reference_targets.json');requesttargets=read(ROOT.parent/'vehicle_availability_audit/request_reference.json')
 if [int(t) for t in times]!=[r['time_ns'] for r in requesttargets] or [int(t) for t in events]!=[r['time_ns'] for r in rawtargets]:raise ValueError('reference schedule differs')
 yr=np.array([np.nan if r['reference_mps'] is None else r['reference_mps'] for r in rawtargets]);yq=np.array([np.nan if r['reference_mps'] is None else r['reference_mps'] for r in requesttargets]);lookup={int(t):i for i,t in enumerate(events)};blocks={}
 for row in read(OLD/'prepared/calibration/index.json'):
  if row['ready']:blocks.setdefault(row['window_id'],[]).append(lookup[row['time_ns']])
 methods={};arrays={'context_features':context,'context_prediction_mps':new};replays={}
 for name,vector in [('frozen_night_tcn',old),('context_boosting',new)]:
  p=np.full(len(index),np.nan);p[ready]=vector;keep=source_quality.copy() if name=='frozen_night_tcn' else quality(p,fullpeer,source_quality);replay=serve(times,events,ready,p,keep,100000000);q=request_summary(replay,yq);replays[name]=replay;methods[name]={'endpoints':c.account(yr[ready],p[ready],keep[ready],len(index)),'requests':q,'previous_supported_blocks':{k:c.account(yr[ids],p[ids],keep[ids],len(ids)) for k,ids in blocks.items()},'additional_rejected_endpoints':int((source_quality&~keep).sum())};arrays[name+'__quality']=keep
  for key,value in replay.items():arrays[name+'__request__'+key]=value
 both=(replays['frozen_night_tcn']['status']==0)&(replays['context_boosting']['status']==0);segment=(times-times[0])//300000000000
 def comparison(take):return {'shared_requests':int(take.sum()),**{name:scores(yq,rec['prediction_mps'],take) for name,rec in replays.items()}}
 primary=methods['context_boosting'];groups=[primary['requests']['overall']]+list(primary['requests']['segments_300s'].values());b=list(primary['previous_supported_blocks'].values());gates={'request_usable_fraction_ge95pct_every_segment_and_overall':all(r['usable_fraction']>=.95 for r in groups),'request_accepted_reference_coverage_ge95pct_every_segment_and_overall':all(r['accepted']['count']>0 and r['accepted']['count']/(r['accepted']['count']+r['accepted']['missing_truth'])>=.95 for r in groups),'request_accepted_p95_le_half_mps_every_segment_and_overall':all(r['accepted']['count']>0 and r['accepted']['p95_mps']<=.5 for r in groups),'zero_accepted_recorded_request_errors_above2mps':primary['requests']['overall']['accepted']['count']>0 and primary['requests']['overall']['accepted']['above_2_mps']==0,'old_supported_block_quality_retention_ge95pct':all(v['whole_acceptance_fraction']>=.95 for v in b),'old_supported_block_accepted_p95_le_half_mps':all(v['accepted']['count']>0 and v['accepted']['p95_mps']<=.5 for v in b)}
 result={'scope':'One fixed classical baseline on repeatedly inspected development data; no final validation','raw_endpoints':len(index),'ready_endpoints':int(ready.sum()),'fixed_requests':len(times),'methods':methods,'common_accepted_requests':{'overall':comparison(both),'segments_300s':{str(int(k)):comparison(both&(segment==k)) for k in np.unique(segment)}},'gates':gates,'all_gates_pass':all(gates.values()),'decision':'CONTEXT_DEVELOPMENT_GATES_PASS_NO_PROMOTION' if all(gates.values()) else 'STOP_AFTER_CONTEXT_DEVELOPMENT_FAILURE','primary':'context_boosting','no_formerly_rejected_endpoint_revived':bool(not np.any(arrays['context_boosting__quality']&~source_quality)),'reserved_final_opened':False,'witness_enabled':False,'production_policy_changed':False,'journal_readiness_established':False}
 return result,arrays

def score():
 if (ROOT/'results.json').exists():raise ValueError('already scored; no result-based retry')
 result,arrays=compute();np.savez_compressed(ROOT/'replay.npz',**arrays);write(ROOT/'results.json',result);print(result['decision'],result['gates'],flush=True)
if __name__=='__main__':score()
