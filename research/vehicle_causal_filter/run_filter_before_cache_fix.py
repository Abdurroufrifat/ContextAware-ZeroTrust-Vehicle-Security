"""One predeclared causal-filter comparison on three immutable estimator streams."""
import numpy as np
from filter_core import ROOT,m,read,write,freeze,filter_events,fit_noise,serve,request_times
from score_masked import request_summary
def sources():
 base=ROOT.parent;out={}
 for name,folder,pkey,ckey in [('original','vehicle_continuous_radar','causal_residual_tcn','paper_inspired_cnn'),('clean_control','vehicle_masked_history','clean_control__causal_residual_tcn','clean_control__masked_cnn_4ch'),('masked_training','vehicle_masked_history','masked_training__causal_residual_tcn','masked_training__masked_cnn_4ch')]:
  folder=base/folder;index=read(folder/'frame_index.json');ready=np.array([p['ready'] for p in index],dtype=bool);pred=np.full(len(index),np.nan);peer=pred.copy()
  with np.load(folder/'predictions.npz',allow_pickle=False) as z:
   for i,p in enumerate(index):
    if p['ready']:pred[i]=z[pkey][p['array_index']];peer[i]=z[ckey][p['array_index']]
  file=folder/('quality_masks.npz' if name=='original' else 'decisions.npz')
  with np.load(file,allow_pickle=False) as z:
   keep=np.zeros(len(index),dtype=bool)
   if name=='original':keep[ready]=z['guard']
   else:keep=z[name+'__quality'].copy()
  out[name]={'index':index,'ready':ready,'pred':pred,'peer':peer,'quality':keep}
 return out
def noise():
 old=ROOT.parent/'vehicle_radar_multidrive'
 with np.load(old/'prepared/fit/data.npz',allow_pickle=False) as d,np.load(old/'fit_predictions.npz',allow_pickle=False) as p:return fit_noise(d['y'],p['causal_residual_tcn'],'fit')
def diagnose(streams,yr,blocks,events):
 out={}
 for name,s in streams.items():
  out[name]={}
  for key,all_ids in blocks.items():
   ids=np.asarray([i for i in all_ids if s['quality'][i] and np.isfinite(yr[i])],dtype=int);error=s['pred'][ids]-yr[ids];pair=np.diff(events[ids])<100000000
   out[name][key]={'count':len(ids),'bias_mps':float(error.mean()),'residual_sd_mps':float(error.std()),'p95_mps':float(np.quantile(abs(error),.95)),'adjacent_pairs_under100ms':int(pair.sum()),'lag1_correlation':float(np.corrcoef(error[:-1][pair],error[1:][pair])[0,1]) if pair.sum()>1 else None,'mean_absolute_adjacent_error_step_mps':float(np.mean(abs(np.diff(error)[pair]))) if pair.any() else None}
 return out
def compute():
 protocol=freeze();streams=sources();events=np.array([p['time_ns'] for p in streams['original']['index']],dtype=np.int64);times=request_times(int(events[0]),int(events[-1]),50000000);base=ROOT.parent;refs=read(base/'vehicle_continuous_radar/reference_targets.json');rrefs=read(base/'vehicle_availability_audit/request_reference.json');yr=np.array([np.nan if p['reference_mps'] is None else p['reference_mps'] for p in refs]);yq=np.array([np.nan if p['reference_mps'] is None else p['reference_mps'] for p in rrefs]);lookup={int(t):i for i,t in enumerate(events)};blocks={}
 if [int(t) for t in times]!=[p['time_ns'] for p in rrefs]:raise ValueError('scheduled request population changed')
 for p in read(base/'vehicle_radar_multidrive/prepared/calibration/index.json'):
  if p['ready']:blocks.setdefault(p['window_id'],[]).append(lookup[p['time_ns']])
 nf=noise();results={};arrays={}
 for name,s in streams.items():
  if [p['time_ns'] for p in s['index']]!=events.tolist():raise ValueError('source endpoint population changed')
  f=filter_events(events,s['pred'],s['quality'],nf['variance_mps2'],protocol['process_jerk_density'],protocol['reset_gap_ns'],s['peer']);prediction=np.where(np.isfinite(f['estimate_mps']),f['estimate_mps'],s['pred']);keep=f['quality'];ready=s['ready']
  if np.any(keep&~s['quality']):raise ValueError('filter revived a previously rejected source endpoint')
  original=serve(times,events,ready,s['pred'],s['quality'],100000000);replay=serve(times,events,ready,prediction,keep,100000000);common=(original['status']==0)&(replay['status']==0);endpoint=m.c.account(yr[ready],prediction[ready],keep[ready],len(events));block={key:m.c.account(yr[ids],prediction[ids],keep[ids],len(ids)) for key,ids in blocks.items()};common_metrics={'requests':int(common.sum()),'unfiltered':m.c.q.error_metrics(yq[common],original['prediction_mps'][common]),'filtered':m.c.q.error_metrics(yq[common],replay['prediction_mps'][common])}
  results[name]={'unfiltered_requests':request_summary(original,yq),'filtered_requests':request_summary(replay,yq),'filtered_endpoints':endpoint,'previous_supported_blocks':block,'common_accepted_request_cohort':common_metrics,'filter_reset_count':int(f['reset'].sum()),'source_quality_accepted':int(s['quality'].sum()),'additional_quality_withheld':int((s['quality']&~keep).sum())}
  for key,value in f.items():arrays[name+'__filter__'+key]=value
  for key,value in replay.items():arrays[name+'__request__'+key]=value
 primary=results['masked_training'];groups=[primary['filtered_requests']['overall']]+list(primary['filtered_requests']['segments_300s'].values());b=list(primary['previous_supported_blocks'].values());gates={'request_usable_fraction_ge95pct_each_segment_and_overall':all(x['usable_fraction']>=.95 for x in groups),'request_reference_coverage_ge95pct_each_segment_and_overall':all(x['accepted']['count']>0 and x['accepted']['count']/(x['accepted']['count']+x['accepted']['missing_truth'])>=.95 for x in groups),'request_accepted_p95_le_half_mps_each_segment_and_overall':all(x['accepted']['count']>0 and x['accepted']['p95_mps']<=.5 for x in groups),'zero_accepted_request_errors_above2mps':groups[0]['accepted']['above_2_mps']==0 and groups[0]['accepted']['count']>0,'previous_supported_block_retention_ge95pct':all(x['whole_acceptance_fraction']>=.95 for x in b),'previous_supported_block_accepted_p95_le_half_mps':all(x['accepted']['count']>0 and x['accepted']['p95_mps']<=.5 for x in b)}
 return {'scope':'Single same-recording development causal Kalman baseline; no new neural fitting','noise_fit':nf,'diagnostic_before_filter':diagnose(streams,yr,blocks,events),'methods':results,'primary':'masked_training','fixed_requests':len(times),'raw_endpoints':len(events),'gates':gates,'all_gates_pass':all(gates.values()),'decision':'CAUSAL_FILTER_DEVELOPMENT_GATES_PASS_NO_PROMOTION' if all(gates.values()) else 'STOP_AFTER_CAUSAL_FILTER_DEVELOPMENT_FAILURE','neural_models_changed':False,'reserved_final_opened':False,'witness_enabled':False},arrays
def run():
 if (ROOT/'results.json').exists():raise ValueError('already scored; use verifier')
 result,arrays=compute();np.savez_compressed(ROOT/'filter_replay.npz',**arrays);write(ROOT/'results.json',result);print(result['decision']);print(result['gates'])
 for name,stats in result['methods'].items():print(name,stats['filtered_requests']['overall'],flush=True)
if __name__=='__main__':run()
