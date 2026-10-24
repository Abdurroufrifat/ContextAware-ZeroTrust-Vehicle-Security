"""Score the predeclared ensemble; failed development gates prohibit fresh evaluation."""
import collections
import numpy as np
from study_io import ROOT,read,write,digest,freeze
import estimator_core as e
def validate_models():
 freeze();lock=read('model_lock.json')
 for name,expected in {**lock['models_sha256'],'design_lock.json':lock['design_lock_sha256'],'classical_models.json':lock['classical_sha256'],'training_trace.json':lock['training_trace_sha256'],'prepared/fit/data.npz':lock['fit_prepared_sha256']}.items():
  if digest(name)!=expected:raise ValueError('frozen model artifact changed: '+name)
 return lock
def metric(y,p):
 z=p-y
 return {'pairs':len(y),'mae_mps':float(np.mean(np.abs(z))),'rmse_mps':float(np.sqrt(np.mean(z*z))),'p95_disagreement_mps':float(np.quantile(np.abs(z),.95)),'maximum_disagreement_mps':float(np.max(np.abs(z))),'bias_mps':float(np.mean(z))}
def score(role,save=True):
 lock=validate_models();data=dict(np.load(ROOT/'prepared'/role/'data.npz',allow_pickle=False));index=read('prepared/'+role+'/index.json');classical=read('classical_models.json');anchor=e.affine(classical['reference_median_affine'],data['median_feature']);pred={'frozen_median':data['frozen_prediction'],'reference_median_affine':anchor,'causal_savgol_affine':e.affine(classical['causal_savgol_affine'],data['sg_feature'])};seed_predictions={}
 for kind in ('paper_inspired_cnn','causal_residual_tcn'):
  models=[read(f'models/{kind}_{seed}.json') for seed in (141,142,143)];pred[kind],seed_predictions[kind]=e.ensemble(models,data['X'],anchor)
  for seed,p in zip((141,142,143),seed_predictions[kind]):pred[kind+'_seed_'+str(seed)]=p
 y=data['y'];metrics={name:metric(y,p) for name,p in pred.items()};blockmetrics={};groups=collections.defaultdict(list)
 for row in index:groups[row['window_id']].append(row)
 for block,rows in groups.items():
  ids=[q['array_index'] for q in rows if q['ready']];eligible=sum(not q['warmup'] for q in rows);blockmetrics[block]={'scored_endpoints':len(rows),'additional_history_warmup':sum(q['warmup'] for q in rows),'supported_after_prefix':len(ids),'support_fraction_after_prefix':len(ids)/eligible if eligible else 0.,'unsupported_after_prefix':eligible-len(ids),'metrics':{name:metric(y[ids],p[ids]) for name,p in pred.items()}}
 windows=read('inputs/'+role+'/windows.json');rawtotal=sum(w['raw_radar_in_block'] for w in windows);original_prefix=sum(w['warmup_denied_endpoints'] for w in windows);warm=original_prefix+sum(q['warmup'] for q in index);eligible=sum(not q['warmup'] for q in index);coverage=len(y)/eligible if eligible else 0.
 candidate=metrics['causal_residual_tcn'];control=metrics['reference_median_affine'];gates={'support_ge_95pct_each_block_and_overall':coverage>=.95 and all(b['support_fraction_after_prefix']>=.95 for b in blockmetrics.values()),'p95_le_half_mps_each_block_and_overall':candidate['p95_disagreement_mps']<=.5 and all(b['metrics']['causal_residual_tcn']['p95_disagreement_mps']<=.5 for b in blockmetrics.values()),'rmse_at_least_15pct_better_than_refit_median':candidate['rmse_mps']<=.85*control['rmse_mps']}
 result={'role':role,'supported_endpoints':len(y),'raw_radar_in_fixed_blocks':rawtotal,'warmup_denied_endpoints':warm,'unsupported_after_history_prefix':eligible-len(y),'warmup_and_unsupported_fraction':(rawtotal-len(y))/rawtotal,'support_fraction_after_prefix':coverage,'metrics':metrics,'blocks':blockmetrics,'gates':gates,'all_gates_pass':all(gates.values()),'primary':'causal_residual_tcn','paper_cnn_is_exact_reproduction':False,'model_or_seed_selected_using_scores':False,'witness_enabled':False,'journal_readiness_established':False}
 if save:
  write(role+'_results.json',result);np.savez_compressed(ROOT/(role+'_predictions.npz'),**pred);print(role,{k:round(v['p95_disagreement_mps'],4) for k,v in metrics.items() if '_seed_' not in k},'gates',gates,flush=True)
 return result,pred
if __name__=='__main__':
 for role in ('fit','calibration'):score(role)
 development=read('calibration_results.json');write('decision.json',{'development_gate_pass':development['all_gates_pass'],'decision':'READY_FOR_FROZEN_PROSPECTIVE_ACCURACY_TEST' if development['all_gates_pass'] else 'STOP_AFTER_DEVELOPMENT_FAILURE','fresh_evaluation_opened':False,'production_policy_changed':False,'witness_enabled':False,'journal_readiness_established':False})
