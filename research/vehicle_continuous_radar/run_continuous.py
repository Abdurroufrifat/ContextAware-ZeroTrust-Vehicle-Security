"""Replay frozen radar estimators across a complete already known development drive."""
import numpy as np
from continuous_core import ROOT,read,write,freeze,decode_radar,prepare,reference_targets,account
import estimator_core as e
import quality_core as q
MODEL=ROOT.parent/'vehicle_radar_multidrive'
def predict(data):
 anchor=e.affine(read(MODEL/'classical_models.json')['reference_median_affine'],data['median_feature']);pred={'median_anchor':anchor}
 for kind in ('causal_residual_tcn','paper_inspired_cnn'):
  models=[read(MODEL/f'models/{kind}_{s}.json') for s in (141,142,143)];pred[kind],seeds=e.ensemble(models,data['X'],anchor)
  for seed,value in zip((141,142,143),seeds):pred[kind+'_seed_'+str(seed)]=value
  print('CONTINUOUS MODEL REPLAY',kind,len(anchor),'endpoints',flush=True)
 return pred
def summarize(index,targets,pred):
 ready=[p for p in index if p['ready']];y=np.array([np.nan if targets[p['ordinal']]['reference_mps'] is None else targets[p['ordinal']]['reference_mps'] for p in ready]);p=pred['causal_residual_tcn'];cnn=pred['paper_inspired_cnn'];keep=np.asarray([q.usable(float(a),float(b),True) for a,b in zip(p,cnn)],dtype=bool);random=np.zeros(len(p),dtype=bool);segments={}
 for number in sorted(set(x['segment'] for x in index)):
  raw=[x for x in index if x['segment']==number];ids=np.asarray([i for i,x in enumerate(ready) if x['segment']==number],dtype=int);random[ids]=q.matched_random(keep[ids],'141:continuous_'+str(number));m=account(y[ids],p[ids],keep[ids],len(raw));m.update(first_ns=raw[0]['time_ns'],last_ns=raw[-1]['time_ns'],startup_unready=sum(x['warmup'] and not x['ready'] for x in raw));segments[str(number)]=m
 methods={name:account(y,p,mask,len(index)) for name,mask in [('timing_only',np.ones(len(p),dtype=bool)),('disagreement_guard',keep),('matched_random',random)]};primary=methods['disagreement_guard'];valid=lambda m:m['accepted']['count']>0
 gates={'reference_coverage_on_ready_ge_95pct_each_segment_and_overall':primary['reference_coverage_on_ready']>=.95 and all(s['reference_coverage_on_ready']>=.95 for s in segments.values()),'whole_record_and_segment_usable_fraction_ge_95pct':primary['whole_acceptance_fraction']>=.95 and all(s['whole_acceptance_fraction']>=.95 for s in segments.values()),'accepted_p95_le_half_mps_each_segment_and_overall':valid(primary) and primary['accepted']['p95_mps']<=.5 and all(valid(s) and s['accepted']['p95_mps']<=.5 for s in segments.values()),'zero_accepted_recorded_errors_above_2_mps':valid(primary) and primary['accepted']['above_2_mps']==0}
 old_data=dict(np.load(MODEL/'prepared/calibration/data.npz',allow_pickle=False));old_index=read(MODEL/'prepared/calibration/index.json');old_pred=dict(np.load(MODEL/'calibration_predictions.npz',allow_pickle=False));lookup={x['time_ns']:i for i,x in enumerate(ready)};pairs=[]
 for old in old_index:
  if not old['ready']:continue
  if old['time_ns'] not in lookup:raise ValueError('previously supported endpoint lost continuous support')
  j=lookup[old['time_ns']];i=old['array_index'];pairs.append((i,j))
 for kind in ('causal_residual_tcn','paper_inspired_cnn'):
  np.testing.assert_allclose(np.asarray([pred[kind][j] for i,j in pairs]),np.asarray([old_pred[kind][i] for i,j in pairs]),rtol=1e-7,atol=1e-7)
 np.testing.assert_allclose(np.asarray([y[j] for i,j in pairs]),np.asarray([old_data['y'][i] for i,j in pairs]),rtol=0,atol=1e-10)
 overlap=np.zeros(len(p),dtype=bool)
 for i,j in pairs:overlap[j]=True
 diagnostic={name:account(y[ids],p[ids],keep[ids],int(ids.sum())) for name,ids in [('previous_supported_overlap',overlap),('other_ready_endpoints',~overlap)]}
 result={'scope':'Complete Urban01 night expanded development recording; not a held-out final test','old_selected_blocks_were_previously_inspected':True,'model_refitted':False,'quality_threshold_changed':False,'fresh_final_test_opened':False,'production_policy_changed':False,'witness_enabled':False,'raw_radar_endpoints':len(index),'first_ns':index[0]['time_ns'],'last_ns':index[-1]['time_ns'],'duration_s':(index[-1]['time_ns']-index[0]['time_ns'])/1e9,'startup_unready_endpoints':sum(x['warmup'] and not x['ready'] for x in index),'later_unready_endpoints':sum(not x['warmup'] and not x['ready'] for x in index),'methods':methods,'segments_300s_no_history_reset':segments,'overlap_proof':{'previous_supported_points':len(pairs),'both_frozen_models_reproduce_previous_predictions':True,'reference_targets_reproduce_previous_targets':True,'quality_accuracy_by_overlap':diagnostic},'gates':gates,'all_gates_pass':all(gates.values()),'decision':'CONTINUOUS_DEVELOPMENT_GATES_PASS_NO_PROMOTION' if all(gates.values()) else 'STOP_AFTER_CONTINUOUS_DEVELOPMENT_FAILURE'}
 return result,{'guard':keep,'matched_random':random,'reference_available':np.isfinite(y),'previous_supported_overlap':overlap}
def run():
 freeze()
 if (ROOT/'results.json').exists():raise ValueError('continuous study already scored; use verifier')
 raw,profile=decode_radar();write(ROOT/'source_profile.json',profile);data,index=prepare(raw);write(ROOT/'frame_index.json',index);np.savez_compressed(ROOT/'prepared.npz',**data);print('CONTINUOUS PREPARED',len(index),'raw;',len(data['X']),'ready',flush=True)
 pred=predict(data);np.savez_compressed(ROOT/'predictions.npz',**pred);targets=reference_targets(index);write(ROOT/'reference_targets.json',targets);result,masks=summarize(index,targets,pred);write(ROOT/'results.json',result);np.savez_compressed(ROOT/'quality_masks.npz',**masks);print(result['decision'],result['methods']['disagreement_guard'],flush=True)
if __name__=='__main__':run()
