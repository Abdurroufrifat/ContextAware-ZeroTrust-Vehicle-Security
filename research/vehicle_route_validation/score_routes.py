"""One retrospective out-of-fold score on every supported fitting row."""
from route_core import *
def check_models():
 lock=read(ROOT/'model_lock.json')
 for key,name in [('analysis_lock_sha256','analysis_lock.json'),('training_trace_sha256','training_trace.json'),('membership_sha256','membership.json'),('support_costs_sha256','support_costs.json')]:
  if lock[key]!=digest(ROOT/name):raise ValueError('route model ancestry differs')
 if not lock['all_fold_models_locked_before_scoring'] or lock['held_scores_created'] or len(lock['files_sha256'])!=9:raise ValueError('fold selection or score ordering differs')
 for name,h in lock['files_sha256'].items():
  if digest(ROOT/name)!=h:raise ValueError('route fold artifact changed: '+name)
 return lock
def summarize(membership,y,pred,baseline,assignment,support):
 if np.any(assignment<0) or len(y)!=len(membership) or not np.isfinite(pred).all() or not np.isfinite(baseline).all():raise ValueError('incomplete held row accounting')
 def stats(ids):return {'supported_rows':int(np.sum(ids)),'context_boosting':c.c.q.error_metrics(y[ids],pred[ids]),'training_only_affine_median':c.c.q.error_metrics(y[ids],baseline[ids])}
 groups=np.array([r['group'] for r in membership]);routes=np.array([r['route'] for r in membership]);windows=np.array([r['window_id'] for r in membership]);bygroup={k:stats(groups==k) for k in GROUPS};byrecord={k:stats(routes==k) for k in ROUTES};bywindow={k:stats(windows==k) for k in sorted(set(windows.tolist()))};pooled=stats(np.ones(len(y),dtype=bool));allcontexts=[pooled]+list(bygroup.values())+list(byrecord.values())+list(bywindow.values());target=all(s['context_boosting']['count']>0 and s['context_boosting']['p95_mps']<=.5 for s in allcontexts)
 macro={name:{key:float(np.mean([v[name][key] for v in bygroup.values()])) for key in ['mae_mps','rmse_mps','p95_mps']} for name in ['context_boosting','training_only_affine_median']}
 result={'scope':'Retrospective named-route cross-validation on previously used fitting data; not untouched final evaluation','supported_rows':len(y),'folds':3,'fold_groups':GROUPS,'pooled_sample_weighted':pooled,'mean_of_route_metrics_not_pooled':macro,'by_group':bygroup,'by_recording':byrecord,'by_window':bywindow,'raw_support_costs':support,'p95_le_half_mps_every_group_recording_window_and_pooled':target,'decision':'ROUTE_VALIDATION_TARGETS_PASS_NO_PROMOTION' if target else 'STOP_AFTER_ROUTE_VALIDATION_FAILURE','no_quality_filter_applied':True,'all_supported_rows_scored_once':bool(np.array_equal(np.sort(assignment),np.sort(np.array([GROUPS.index(g) for g in groups])))),'parent_development_gate_still_failed':True,'reserved_final_opened':False,'urban01_night_development_labels_used':False,'witness_enabled':False,'production_policy_changed':False,'journal_readiness_established':False}
 return result

def compute():
 freeze();check_models();data,membership,groups,day,support=load_data();pred=np.full(len(groups),np.nan);baseline=np.full(len(groups),np.nan);assignment=np.full(len(groups),-1,dtype=np.int16)
 for fold,held in enumerate(GROUPS):
  cal=read(ROOT/'folds'/held/'calibration.json');ids=np.array(cal['held_ids']);model=read(ROOT/'folds'/held/'model.json');anchor=c.e.affine(cal['anchor'],data['median_feature'][ids]);x=c.features(data['X'][ids],anchor,data['median_feature'][ids])
  if np.any(assignment[ids]!=-1) or not np.all(groups[ids]==held):raise ValueError('held assignment duplicated or wrong group')
  pred[ids]=c.predict(model,x,anchor);baseline[ids]=anchor;assignment[ids]=fold
 return summarize(membership,data['y'],pred,baseline,assignment,support),{'prediction_mps':pred,'baseline_mps':baseline,'target_mps':data['y'],'fold_id':assignment}
def score():
 if (ROOT/'results.json').exists():raise ValueError('held results already exist; no result-driven retry')
 result,arrays=compute();np.savez_compressed(ROOT/'oof_predictions.npz',**arrays);write(ROOT/'results.json',result);print(result['decision'],flush=True)
 for name,stats in result['by_group'].items():print('ROUTE HELD',name,'n',stats['supported_rows'],'context P95',stats['context_boosting']['p95_mps'],'affine P95',stats['training_only_affine_median']['p95_mps'],flush=True)
if __name__=='__main__':score()
