"""Score one predeclared 20Hz latest-attempt schedule; no training or networking."""
import numpy as np
from availability_core import ROOT,PREVIOUS,STATUS,c,read,write,freeze,request_times,serve,endpoint_reasons,scores,episodes
def inputs():
 index=read(PREVIOUS/'frame_index.json');n=len(index);ready=np.array([x['ready'] for x in index],dtype=bool);values=np.full(n,np.nan);quality=np.zeros(n,dtype=bool)
 with np.load(PREVIOUS/'predictions.npz',allow_pickle=False) as p,np.load(PREVIOUS/'quality_masks.npz',allow_pickle=False) as m:
  for i,x in enumerate(index):
   if x['ready']:values[i]=p['causal_residual_tcn'][x['array_index']];quality[i]=m['guard'][x['array_index']]
 return index,np.array([x['time_ns'] for x in index],dtype=np.int64),ready,values,quality
def reference(times):
 return c.reference_targets([{'time_ns':int(t)} for t in times])
def evaluate(replay,targets,reasons,events):
 protocol=read(ROOT/'protocol.json');times=replay['request_times_ns'];status=replay['status'];y=np.array([np.nan if x['reference_mps'] is None else x['reference_mps'] for x in targets]);start=int(times[0]);segment=(times-start)//300000000000
 def summary(ids):
  st=status[ids];quality=st==0;timing=quality|(st==4);count=len(st)
  return {'requests':count,'status_counts':{label:int((st==code).sum()) for code,label in STATUS.items()},'usable_fraction':float(quality.mean()) if count else 0.,'timing_ready_fraction':float(timing.mean()) if count else 0.,'reference_missing_all_requests':int((~np.isfinite(y[ids])).sum()),'accepted':scores(y[ids],replay['prediction_mps'][ids],quality),'timing_only':scores(y[ids],replay['timing_prediction_mps'][ids],timing)}
 overall=summary(np.ones(len(times),dtype=bool));segments={str(int(k)):summary(segment==k) for k in np.unique(segment)};reason_counts={name:sum(x['reason']==name for x in reasons) for name in sorted({x['reason'] for x in reasons})};gaps=np.diff(events);large=np.flatnonzero(gaps>300000000)
 gap_records=[]
 for i in large:
  affected=[x['time_ns'] for x in reasons if int(events[i]) in x['stale_after_valid_ns']]
  gap_records.append({'before_ns':int(events[i]),'after_ns':int(events[i+1]),'gap_ns':int(gaps[i]),'elapsed_from_start_s':int(events[i+1]-events[0])/1e9,'affected_later_endpoints':len(affected),'last_affected_endpoint_ns':max(affected) if affected else None,'last_affected_after_gap_s':(max(affected)-int(events[i+1]))/1e9 if affected else None})
 gates={'usable_fraction_ge_95pct_overall_and_each_segment':overall['usable_fraction']>=.95 and all(x['usable_fraction']>=.95 for x in segments.values()),'reference_coverage_on_accepted_ge_95pct_overall_and_each_segment':all(x['accepted']['count']>0 and x['accepted']['count']/(x['accepted']['count']+x['accepted']['missing_truth'])>=.95 for x in [overall]+list(segments.values())),'accepted_p95_le_half_mps_overall_and_each_segment':all(x['accepted']['count']>0 and x['accepted']['p95_mps']<=.5 for x in [overall]+list(segments.values())),'zero_accepted_recorded_errors_above_2_mps':overall['accepted']['count']>0 and overall['accepted']['above_2_mps']==0}
 return {'scope':'Fixed-rate retrospective development request replay; not independent validation or live deadline evidence','schedule_hz':20,'estimate_ttl_ns':protocol['estimate_ttl_ns'],'scheduled_requests':len(times),'first_request_ns':int(times[0]),'last_request_ns':int(times[-1]),'original_endpoint_reasons':reason_counts,'inter_record_gaps_above_300ms':gap_records,'overall':overall,'segments_300s':segments,'unavailable_episodes':episodes(times,status,protocol['period_ns']),'gates':gates,'all_gates_pass':all(gates.values()),'decision':'REQUEST_DEVELOPMENT_GATES_PASS_NO_PROMOTION' if all(gates.values()) else 'STOP_AFTER_REQUEST_AVAILABILITY_FAILURE','models_refitted':False,'quality_rule_changed':False,'fresh_final_opened':False,'witness_enabled':False}
def rebuild():
 protocol=freeze();raw,_=c.decode_radar();reasons=endpoint_reasons(raw);index,events,ready,pred,quality=inputs()
 if [x['reason']=='ready' for x in reasons]!=ready.tolist():raise ValueError('diagnostic does not reproduce original endpoint readiness')
 times=request_times(int(events[0]),int(events[-1]),protocol['period_ns']);replay=serve(times,events,ready,pred,quality,protocol['estimate_ttl_ns']);targets=reference(times);result=evaluate(replay,targets,reasons,events)
 return replay,targets,reasons,result
def run():
 if (ROOT/'results.json').exists():raise ValueError('already scored; use verifier')
 replay,targets,reasons,result=rebuild();np.savez_compressed(ROOT/'request_replay.npz',**replay);write(ROOT/'request_reference.json',targets);write(ROOT/'endpoint_reasons.json',reasons);write(ROOT/'results.json',result);print(result['decision']);print(result['overall'])
if __name__=='__main__':run()
