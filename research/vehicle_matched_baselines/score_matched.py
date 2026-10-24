"""Reproduce all paired outcomes; no filtering or after-score model selection."""
from matched_core import *

def compute():
 freeze();check_models();data,members,groups,day,support=r.load_data();pred={n:np.full(len(groups),np.nan) for n in BASELINES};assignment=np.full(len(groups),-1,dtype=np.int16)
 for fold,held in enumerate(r.GROUPS):
  cal=read(r.ROOT/'folds'/held/'calibration.json');ids=np.array(cal['held_ids']);anchor=r.c.e.affine(cal['anchor'],data['median_feature'][ids]);x=r.c.features(data['X'][ids],anchor,data['median_feature'][ids]);models=read(ROOT/'folds'/held/'models.json')
  if set(models)!=set(BASELINES) or np.any(assignment[ids]!=-1) or not np.all(groups[ids]==held):raise ValueError('selected model or duplicate held fold')
  for name in BASELINES:pred[name][ids]=predict(models[name],x,anchor)
  assignment[ids]=fold
 with np.load(r.ROOT/'oof_predictions.npz',allow_pickle=False) as z:
  if not np.array_equal(z['target_mps'],data['y']) or not np.array_equal(z['fold_id'],assignment):raise ValueError('HGB and comparator held rows differ')
  pred['context_boosting']=z['prediction_mps'].copy()
 result=summarize(members,data['y'],pred,assignment,support);arrays={n+'__prediction_mps':v for n,v in pred.items()};arrays.update({'target_mps':data['y'],'fold_id':assignment})
 return result,arrays

def score():
 if (ROOT/'results.json').exists() or (ROOT/'oof_predictions.npz').exists():raise ValueError('matched held results already exist; no retries')
 result,arrays=compute();np.savez_compressed(ROOT/'oof_predictions.npz',**arrays);write(ROOT/'results.json',result)
 for n,v in result['pooled']['models'].items():print(n,'P95',v['p95_mps'],'RMSE',v['rmse_mps'],'maximum',v['maximum_mps'],flush=True)
 print(result['decision'],flush=True)
if __name__=='__main__':score()
