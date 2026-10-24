"""One locked development evaluation; never opens the reserved test."""
from pathlib import Path
import hashlib,json
import numpy as np
import quality_core as q
ROOT=Path(__file__).resolve().parent
MODEL=ROOT.parent/'vehicle_radar_multidrive'
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def freeze(create=False):
 protocol=read(ROOT/'protocol.json')
 for name,h in protocol['source_sha256'].items():
  p=(ROOT/name).resolve()
  if not p.is_relative_to(ROOT.parent) or hashlib.sha256(p.read_bytes()).hexdigest()!=h:raise ValueError('quality source changed: '+name)
 lock={'protocol_sha256':hashlib.sha256((ROOT/'protocol.json').read_bytes()).hexdigest(),'code_sha256':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ('quality_core.py','run_quality.py')},'threshold_sweep':False,'model_refit':False,'calibration_previously_inspected':True}
 p=ROOT/'analysis_lock.json'
 if p.exists() and read(p)!=lock:raise ValueError('quality experiment implementation changed')
 if create and not p.exists():write(p,lock)
 return lock
def compute():
 freeze();result={'scope':'Development research usability flags, not production permissions','calibration_previously_inspected':True,'rule_motivated_by_posthoc_failure_audit':True,'rule_threshold_mps':1.,'threshold_sweep':False,'model_refit':False,'fresh_test_opened':False,'production_policy_changed':False,'witness_enabled':False,'roles':{}};masks={}
 for role in ('fit','calibration'):
  with np.load(MODEL/'prepared'/role/'data.npz',allow_pickle=False) as data:y=data['y'].copy()
  with np.load(MODEL/(role+'_predictions.npz'),allow_pickle=False) as data:tcn=data['causal_residual_tcn'].copy();cnn=data['paper_inspired_cnn'].copy();median=data['reference_median_affine'].copy()
  if tcn.shape!=cnn.shape or tcn.shape!=y.shape:raise ValueError('model output shape mismatch')
  index=read(MODEL/'prepared'/role/'index.json');ready=[r for r in index if r['ready']]
  if [r['array_index'] for r in ready]!=list(range(len(y))):raise ValueError('support index is not aligned')
  windows=read(MODEL/'inputs'/role/'windows.json');keep=np.asarray([q.usable(float(a),float(b),True) for a,b in zip(tcn,cnn)],dtype=bool);random=np.zeros(len(y),dtype=bool);blocks={}
  for window in windows:
   name=window['id'];ids=np.asarray([i for i,r in enumerate(ready) if r['window_id']==name],dtype=int);random[ids]=q.matched_random(keep[ids],'141:'+name);m=q.metrics(y[ids],tcn[ids],keep[ids]);raw=window['raw_radar_in_block'];m.update(raw_endpoints=raw,history_or_missing_withheld=raw-len(ids),quality_withheld=int((~keep[ids]).sum()),whole_block_acceptance_fraction=int(keep[ids].sum())/raw,median_on_accepted=q.error_metrics(y[ids][keep[ids]],median[ids][keep[ids]]));blocks[name]=m
  raw=sum(w['raw_radar_in_block'] for w in windows);stats={}
  for name,mask in [('timing_only',np.ones(len(y),dtype=bool)),('disagreement_guard',keep),('matched_random',random),('withhold_all',np.zeros(len(y),dtype=bool))]:
   m=q.metrics(y,tcn,mask);m.update(raw_endpoints=raw,history_or_missing_withheld=raw-len(y),quality_withheld=int((~mask).sum()),whole_block_acceptance_fraction=int(mask.sum())/raw,whole_block_withheld_fraction=(raw-int(mask.sum()))/raw);stats[name]=m
  candidate=stats['disagreement_guard'];accepted=candidate['accepted'];rmse_control=q.error_metrics(y[keep],median[keep]);candidate['median_on_accepted']=rmse_control
  valid=lambda m:m['count']>0
  gates={'supported_acceptance_ge_95pct_each_block_and_overall':candidate['supported_acceptance_fraction']>=.95 and all(m['supported_acceptance_fraction']>=.95 for m in blocks.values()),'accepted_p95_le_half_mps_each_block_and_overall':valid(accepted) and accepted['p95_mps']<=.5 and all(valid(m['accepted']) and m['accepted']['p95_mps']<=.5 for m in blocks.values()),'zero_accepted_disagreements_above_2_mps':valid(accepted) and accepted['above_2_mps']==0,'whole_block_acceptance_ge_95pct_each_block_and_overall':candidate['whole_block_acceptance_fraction']>=.95 and all(m['whole_block_acceptance_fraction']>=.95 for m in blocks.values()),'accepted_rmse_at_least_15pct_better_than_same_subset_median':valid(accepted) and accepted['rmse_mps']<=.85*rmse_control['rmse_mps']}
  result['roles'][role]={'supported_endpoints':len(y),'methods':stats,'blocks':blocks,'gates':gates,'all_gates_pass':all(gates.values())};masks[role+'_guard']=keep;masks[role+'_random']=random
 passed=result['roles']['calibration']['all_gates_pass'];result['decision']='READY_FOR_SEPARATELY_FROZEN_QUALITY_TEST' if passed else 'STOP_AFTER_QUALITY_DEVELOPMENT_FAILURE';result['promotion_allowed']=False
 result['synthetic_shared_error_counterexample']={'measured_dataset_result':False,'tcn_prediction_mps':0.,'cnn_prediction_mps':0.,'hypothetical_true_speed_mps':10.,'quality_flag':q.usable(0.,0.,True),'error_mps':10.,'production_permission_issued':False,'meaning':'Agreement of two models using the same radar is not an independent physical witness.'}
 return result,masks
if __name__=='__main__':
 if __import__('sys').argv[1:]==['freeze']:freeze(True);print('Quality rule, sources and development protocol frozen')
 else:
  if (ROOT/'results.json').exists():raise ValueError('experiment already scored; use verify_quality.py to replay')
  result,masks=compute();write(ROOT/'results.json',result);np.savez_compressed(ROOT/'quality_masks.npz',**masks);print(json.dumps(result['roles']['calibration'],indent=2),flush=True);print(result['decision'])
