from fusion_core import *
def compute():
 freeze();check_models();data,members,groups,day,support=r.load_data();base=original_base(data);new=np.full(len(groups),np.nan);assignment=np.full(len(groups),-1,dtype=np.int16)
 for fold,scope in enumerate(r.GROUPS):
  cal=read(ROOT/'models'/scope/'calibration.json');ids=np.array(cal['held_ids']);a=c.e.affine(cal['anchor'],data['median_feature'][ids]);x=assemble(base[ids],a,data['median_feature'][ids]);model=read(ROOT/'models'/scope/'model.json')
  if model['feature_names']!=FEATURE_NAMES or not np.all(groups[ids]==scope) or np.any(assignment[ids]!=-1):raise ValueError('fusion route schema/assignment differs')
  new[ids]=c.predict(model,x,a);assignment[ids]=fold
 with np.load(r.ROOT/'oof_predictions.npz',allow_pickle=False) as z:
  if not np.array_equal(z['target_mps'],data['y']) or not np.array_equal(z['fold_id'],assignment):raise ValueError('fusion/control OOF alignment differs')
  old=z['prediction_mps'].copy()
 route=route_summary(members,data['y'],dict(zip(METHODS,[old,new])),support);index,ready,x,median,anchor,peer,source,oldfull=load_full();model=read(ROOT/'models/all_fit/model.json')
 if model['feature_names']!=FEATURE_NAMES:raise ValueError('fusion full model schema differs')
 prediction=c.predict(model,features(x,anchor,median),anchor);full,arrays=full_summary(prediction)
 result={'scope':'One fixed information-addition ablation; retrospective routes and already inspected full development; no selection/promotion','route':route,'full_development':full,'reserved_final_opened':False,'full_urban01_night_development_reused':True,'production_policy_changed':False,'thresholds_changed':False,'witness_enabled':False,'promotion_allowed':False,'journal_readiness_established':False,'decision':'FUSION_ABLATION_COMPLETE_NO_PROMOTION'}
 return result,{'target_mps':data['y'],'fold_id':assignment,'original_context__prediction_mps':old,'signed_magnitude_fusion__prediction_mps':new},arrays

def score():
 if any((ROOT/n).exists() for n in ['results.json','oof_predictions.npz','development_replay.npz']):raise ValueError('fusion already scored; no result-driven retries')
 result,route,full=compute();write(ROOT/'results.json',result);np.savez_compressed(ROOT/'oof_predictions.npz',**route);np.savez_compressed(ROOT/'development_replay.npz',**full)
 for n,s in result['route']['pooled']['models'].items():print('ROUTE',n,'P95',s['p95_mps'],'failed windows',len(result['route']['diagnostic_failures'][n]['windows']),flush=True)
 for n,s in result['full_development']['methods'].items():print('FULL',n,'accepted P95',s['requests']['overall']['accepted']['p95_mps'],'usable',s['requests']['overall']['usable_fraction'],'gates',result['full_development']['gates'][n],flush=True)
 print(result['decision'],flush=True)
if __name__=='__main__':score()
