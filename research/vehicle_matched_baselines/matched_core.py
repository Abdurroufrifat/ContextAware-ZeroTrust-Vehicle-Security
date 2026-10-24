"""Fixed classical baselines; portable numeric inference and complete accounting."""
from pathlib import Path
import hashlib,sys
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'vehicle_route_validation'))
import route_core as r
from score_routes import check_models as check_parent_models
read=r.read;write=r.write;digest=r.digest
BASELINES=('affine_median_matched','ridge_context','huber_context')
MODELS=('context_boosting',)+BASELINES

def inputs(x,w):
 x=np.asarray(x,dtype=float);w=np.asarray(w,dtype=float)
 if x.ndim!=2 or not x.size or w.shape!=(len(x),) or not np.isfinite(x).all() or not np.isfinite(w).all() or np.any(w<0) or w.sum()<=0:raise ValueError('invalid finite weighted training matrix')
 return x,w

def scaler(x,w):
 x,w=inputs(x,w);mean=np.sum(x*w[:,None],axis=0)/w.sum();var=np.sum((x-mean)**2*w[:,None],axis=0)/w.sum()
 return {'mean':mean.tolist(),'scale':np.where(var<=1e-12,1.,np.sqrt(var)).tolist()}

def transform(x,s):
 x=np.asarray(x,dtype=float);mean=np.asarray(s['mean'],dtype=float);scale=np.asarray(s['scale'],dtype=float)
 if x.ndim!=2 or mean.shape!=(x.shape[1],) or scale.shape!=mean.shape or not np.isfinite(x).all() or not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale<=0):raise ValueError('invalid saved scaler')
 return (x-mean)/scale

def least_squares(x,y,w,alpha):
 x,w=inputs(x,w);y=np.asarray(y,dtype=float)
 if y.shape!=(len(x),) or not np.isfinite(y).all() or not np.isfinite(alpha) or alpha<0:raise ValueError('invalid least-squares target or penalty')
 a=np.column_stack([x,np.ones(len(x))]);penalty=np.zeros((x.shape[1],x.shape[1]+1));penalty[:,:-1]=np.eye(x.shape[1])*np.sqrt(alpha)
 coef=np.linalg.lstsq(np.vstack([a*np.sqrt(w[:,None]),penalty]),np.concatenate([y*np.sqrt(w),np.zeros(x.shape[1])]),rcond=None)[0]
 return coef[:-1],float(coef[-1])

def predict(model,x,anchor):
 x=np.asarray(x,dtype=float);anchor=np.asarray(anchor,dtype=float);ids=np.asarray(model['indices']);coef=np.asarray(model['coef'],dtype=float);intercept=float(model['intercept'])
 if x.ndim!=2 or ids.ndim!=1 or ids.dtype.kind not in 'iu' or len(ids)==0 or len(set(ids.tolist()))!=len(ids) or np.any(ids<0) or np.any(ids>=x.shape[1]) or anchor.shape!=(len(x),) or not np.isfinite(anchor).all() or coef.shape!=(len(ids),) or not np.isfinite(coef).all() or not np.isfinite(intercept):raise ValueError('invalid numeric linear model')
 return np.maximum(0.,anchor+transform(x[:,ids],model['scaler'])@coef+intercept)

def array_digest(x):return hashlib.sha256(np.asarray(x,dtype='<f8').tobytes()).hexdigest()

def freeze(create=False):
 r.freeze();check_parent_models();p=read(ROOT/'protocol.json')
 if p['baselines']!=list(BASELINES) or p['groups']!=list(r.GROUPS) or p['ridge']!={'alpha':1.0,'fit_intercept':True,'solver':'svd'} or p['huber']!={'epsilon':1.35,'alpha':0.0001,'max_iter':500,'tol':0.000001,'fit_intercept':True,'warm_start':False}:raise ValueError('fixed comparison protocol changed')
 for name,h in p['input_sha256'].items():
  path=(ROOT/name).resolve()
  if not path.is_relative_to(ROOT.parent) or digest(path)!=h:raise ValueError('matched input changed: '+name)
 lock={'protocol_sha256':digest(ROOT/'protocol.json'),'code_sha256':{n:digest(ROOT/n) for n in ['matched_core.py','train_matched.py','score_matched.py']}}
 dest=ROOT/'analysis_lock.json'
 if dest.exists() and read(dest)!=lock:raise ValueError('comparison implementation changed after freeze')
 if create and not dest.exists():write(dest,lock)
 return p

def check_models():
 lock=read(ROOT/'model_lock.json')
 if not lock['all_nine_models_locked_before_new_held_scoring'] or lock['new_held_scores_created'] or len(lock['files_sha256'])!=6:raise ValueError('incomplete or selected baseline models')
 for key,name in [('analysis_lock_sha256','analysis_lock.json'),('training_trace_sha256','training_trace.json')]:
  if lock[key]!=digest(ROOT/name):raise ValueError('matched model ancestry changed')
 for name,h in lock['files_sha256'].items():
  if digest(ROOT/name)!=h:raise ValueError('matched model artifact changed: '+name)
 return lock

def summarize(membership,y,predictions,assignment,support):
 y=np.asarray(y,dtype=float);assignment=np.asarray(assignment)
 if y.ndim!=1 or len(y)!=len(membership) or not np.isfinite(y).all() or set(predictions)!=set(MODELS):raise ValueError('incomplete matched population or model list')
 groups=np.array([v['group'] for v in membership]);routes=np.array([v['route'] for v in membership]);windows=np.array([v['window_id'] for v in membership])
 if set(groups.tolist())!=set(r.GROUPS) or assignment.shape!=y.shape or not np.array_equal(assignment,np.array([r.GROUPS.index(g) for g in groups])):raise ValueError('held-fold assignment differs')
 p={n:np.asarray(predictions[n],dtype=float) for n in MODELS}
 if any(v.shape!=y.shape or not np.isfinite(v).all() for v in p.values()):raise ValueError('missing, nonfinite or misaligned model predictions')
 def stats(ids):
  metrics={n:r.c.c.q.error_metrics(y[ids],v[ids]) for n,v in p.items()}
  return {'supported_rows':int(np.sum(ids)),'models':metrics,'paired_hgb_minus_baseline':{n:{k:metrics['context_boosting'][k]-metrics[n][k] for k in ['mae_mps','rmse_mps','p95_mps','maximum_mps']} for n in BASELINES}}
 bygroup={g:stats(groups==g) for g in r.GROUPS};byrecord={k:stats(routes==k) for k in sorted(set(routes.tolist()))};bywindow={k:stats(windows==k) for k in sorted(set(windows.tolist()))};pooled=stats(np.ones(len(y),dtype=bool))
 failures={n:{'pooled':pooled['models'][n]['p95_mps']>.5,'groups':[k for k,s in bygroup.items() if s['models'][n]['p95_mps']>.5],'recordings':[k for k,s in byrecord.items() if s['models'][n]['p95_mps']>.5],'windows':[k for k,s in bywindow.items() if s['models'][n]['p95_mps']>.5]} for n in MODELS}
 macro={n:{k:float(np.mean([s['models'][n][k] for s in bygroup.values()])) for k in ['mae_mps','rmse_mps','p95_mps']} for n in MODELS}
 return {'scope':'Retrospective fixed-configuration established-algorithm comparison; existing HGB outcomes already inspected; not untouched evaluation','supported_rows':len(y),'folds':3,'models':list(MODELS),'pooled':pooled,'mean_route_metrics_not_pooled_quantiles':macro,'by_group':bygroup,'by_recording':byrecord,'by_window':bywindow,'diagnostic_failures':failures,'raw_support_costs':support,'same_held_rows_for_all_models':True,'no_quality_filter':True,'significance_claim':False,'vehicle_paper_reproduction_claim':False,'algorithmic_novelty_established':False,'parent_development_gate_still_failed':True,'reserved_final_opened':False,'witness_enabled':False,'production_policy_changed':False,'promotion_allowed':False,'decision':'MATCHED_COMPARISON_COMPLETE_NO_PROMOTION'}
