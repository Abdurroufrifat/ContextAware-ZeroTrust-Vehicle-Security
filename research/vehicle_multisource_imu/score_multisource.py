from multisource_core import *
def compute():
 check_models();arrays=cache();data,members,groups,day,support=r.load_data();base=c.features(data['X'],np.zeros(len(members)),np.zeros(len(members)));pred={n:np.full(len(members),np.nan) for n in NEW_METHODS};assignment=np.full(len(members),-1,dtype=np.int16)
 for fold,scope in enumerate(r.GROUPS):
  cal=read(ROOT/'models'/scope/'calibration.json');ids=np.array(cal['held_ids']);anchor=c.e.affine(cal['anchor'],data['median_feature'][ids])
  if not np.all(groups[ids]==scope) or np.any(assignment[ids]!=-1):raise ValueError('multi-source group assignment differs')
  assignment[ids]=fold
  for method in NEW_METHODS:
   x=assemble(base[ids],arrays['fit_imu'][ids],anchor,data['median_feature'][ids],method);used=anchor if method=='radar_imu' else np.zeros(len(ids));model=read(ROOT/'models'/scope/(method+'.json'))
   if model['feature_names']!=FEATURE_NAMES[method]:raise ValueError('multi-source feature schema differs')
   pred[method][ids]=c.predict(model,x,used)
 with np.load(r.ROOT/'oof_predictions.npz',allow_pickle=False) as z:
  if not np.array_equal(z['target_mps'],data['y']) or not np.array_equal(z['fold_id'],assignment):raise ValueError('original OOF cohort differs')
  old=z['prediction_mps'].copy()
 route=route_summary(members,data['y'],{'original_context':old,**pred},support);index,ready,x,median,anchor,*_=f.load_full();fullradar=c.features(x,anchor,median);new={}
 for method in NEW_METHODS:
  model=read(ROOT/'models/all_fit'/(method+'.json'));features=assemble(fullradar,arrays['full_imu'][ready],anchor,median,method);new[method]=c.predict(model,features,anchor if method=='radar_imu' else np.zeros(len(anchor)))
 full,replay=full_summary(new,arrays);result={'scope':'One fixed physical-source addition and IMU-only control; retrospective named routes and already inspected full development','route':route,'full_development':full,'full_urban01_night_development_reused':True,'all_eight_models_locked_before_new_scoring':True,'thresholds_changed':False,'production_policy_changed':False,'reserved_final_opened':False,'witness_enabled':False,'promotion_allowed':False,'journal_readiness_established':False,'decision':'MULTISOURCE_IMU_COMPARISON_COMPLETE_NO_PROMOTION'}
 return result,{'target_mps':data['y'],'fold_id':assignment,'original_context__prediction_mps':old,**{n+'__prediction_mps':v for n,v in pred.items()}},replay
def score():
 if any((ROOT/n).exists() for n in ['results.json','oof_predictions.npz','development_replay.npz']):raise ValueError('multi-source already scored; no result-driven retry')
 result,route,full=compute();write(ROOT/'results.json',result);np.savez_compressed(ROOT/'oof_predictions.npz',**route);np.savez_compressed(ROOT/'development_replay.npz',**full)
 for name,v in result['route']['pooled']['models'].items():print('ROUTE',name,'P95',v['p95_mps'],'MAE',v['mae_mps'],'RMSE',v['rmse_mps'],'failed windows',len(result['route']['diagnostic_failures'][name]['windows']),flush=True)
 for name,v in result['full_development']['methods'].items():print('FULL',name,'accepted P95',v['requests']['overall']['accepted']['p95_mps'],'usable',v['requests']['overall']['usable_fraction'],'binding rejected',v['imu_binding_additional_rejected_requests'],'gates',result['full_development']['gates'][name],flush=True)
 print(result['decision'],flush=True)
if __name__=='__main__':score()
