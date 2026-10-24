from nonlinear_core import *
def check_models():
 check_lock();lock=read(ROOT/'model_lock.json')
 if lock['analysis_lock_sha256']!=digest(ROOT/'analysis_lock.json') or lock['training_trace_sha256']!=digest(ROOT/'training_trace.json') or lock['all_eight_locked_before_new_scores'] is not True:raise ValueError('model lock differs')
 expected={f'models/{scope}/{kind}{suffix}' for scope in SCOPES for kind in METHODS for suffix in ['.json','_training.npz']}
 if set(lock['files_sha256'])!=expected:raise ValueError('model membership differs')
 for n,h in lock['files_sha256'].items():
  if digest(ROOT/n)!=h:raise ValueError('model changed: '+n)
def compute():
 check_models();data=inputs();y=data['oof']['target_mps'];parent=data['oof']['radar_imu__prediction_mps'];pred={};full={};arrays={}
 old=load(RESEARCH/'vehicle_signed_axis_correction/oof_predictions.npz');pred['signed6_previous']=old['signed_axis_corrected__prediction_mps'];pred['parent_radar_imu']=parent;pred['original_context']=old['original_context__prediction_mps']
 for kind in METHODS:
  p=np.full(len(y),np.nan);seen=np.zeros(len(y),int)
  for scope in SCOPES[:-1]:
   cal=read(RESEARCH/'vehicle_multisource_imu/models'/scope/'calibration.json');ids=np.array(cal['held_ids']);m=read(ROOT/'models'/scope/(kind+'.json'));p[ids]=np.maximum(0.,parent[ids]+predict(m,head_features(parent[ids],data['axes']['fit_axes'][ids]),kind));seen[ids]+=1
  if not np.all(seen==1):raise ValueError('held rows not assigned exactly once')
  pred[kind]=p;m=read(ROOT/'models/all_fit'/(kind+'.json'));ready=data['ready'];values=np.full(len(ready),np.nan);values[ready]=np.maximum(0.,data['parent_full'][ready]+predict(m,head_features(data['parent_full'][ready],data['axes']['full_axes'][ready]),kind));full[kind],keep,q=full_metrics(data,values);arrays[kind+'__ready_prediction_mps']=values[ready];arrays[kind+'__quality']=keep
  for k,v in q.items():arrays[kind+'__request__'+k]=v
 previous=load(RESEARCH/'vehicle_signed_axis_correction/development_replay.npz');values=np.full(len(data['ready']),np.nan);values[data['ready']]=previous['signed_axis_corrected__ready_prediction_mps'];full['signed6_previous'],keep,q=full_metrics(data,values);np.testing.assert_array_equal(keep,previous['signed_axis_corrected__quality']);np.testing.assert_array_equal(q['status'],previous['signed_axis_corrected__request__status'])
 def cell(mask):return {name:metrics(y,v,mask) for name,v in pred.items()}
 route={'pooled':cell(np.ones(len(y),bool))}
 for name,key in [('by_group','group'),('by_recording','route'),('by_window','window_id')]:
  labels=np.array([v[key] for v in data['members']]);route[name]={k:cell(labels==k) for k in sorted(set(labels.tolist()))}
 failures={n:{cat:[k for k,v in route[cat].items() if v[n]['p95_mps']>.5] for cat in ['by_group','by_recording','by_window']} for n in pred}
 gates={n:route['pooled'][n]['p95_mps']<=.5 and all(not v for v in failures[n].values()) and full[n]['all_gates_pass'] for n in [*METHODS,'signed6_previous']}
 result={'decision':'FIXED_NONLINEAR_EXPERIMENT_COMPLETE_NO_PROMOTION','primary_method':'nonlinear7','primary_all_accuracy_availability_gates_pass':gates['nonlinear7'],'all_accuracy_availability_gates_pass':gates,'route':route,'route_failures':failures,'full_development':full,'scope':'Retrospective development; routes and Urban01 NIGHT previously inspected','held_scores_used_in_fitting':False,'in_sample_parent_residual_training':True,'thresholds_changed':False,'production_policy_changed':False,'reserved_final_opened':False,'promotion_allowed':False,'journal_readiness_established':False}
 return result,{'target_mps':y,'fold_id':data['fold'],**{n+'__prediction_mps':v for n,v in pred.items()}},arrays
def score():
 if any((ROOT/n).exists() for n in ['results.json','oof_predictions.npz','development_replay.npz']):raise ValueError('already scored; no result-driven retry')
 result,route,full=compute();write(ROOT/'results.json',result);np.savez_compressed(ROOT/'oof_predictions.npz',**route);np.savez_compressed(ROOT/'development_replay.npz',**full)
 for n,v in result['route']['pooled'].items():print('ROUTE',n,'MAE',v['mae_mps'],'P95',v['p95_mps'],'failed windows',result['route_failures'][n]['by_window'],flush=True)
 for n,v in result['full_development'].items():print('FULL',n,'accepted P95',v['overall']['accepted']['p95_mps'],'availability',v['overall']['usable_fraction'],'gates',v['gates'],flush=True)
 print(result['decision'],'primary passes:',result['primary_all_accuracy_availability_gates_pass'],flush=True)
if __name__=='__main__':score()
