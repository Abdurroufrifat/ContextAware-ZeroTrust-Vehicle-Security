"""Fit nine fixed models once; no new held-group scores until all are locked."""
import platform,time,warnings
from matched_core import *

def train():
 protocol=freeze()
 if (ROOT/'folds').exists() or (ROOT/'model_lock.json').exists():raise ValueError('matched training already started; no retries')
 import sklearn
 from sklearn.linear_model import Ridge,HuberRegressor
 if sklearn.__version__!='1.8.0':raise ValueError('locked sklearn version differs')
 data,members,groups,day,support=r.load_data();(ROOT/'folds').mkdir();runs=[];files={};started=time.monotonic()
 for held in r.GROUPS:
  cal=read(r.ROOT/'folds'/held/'calibration.json');rebuilt=r.calibrate(data,groups,day,held)
  from verify_continuous import close
  close(cal,rebuilt)
  x,y,w,anchor=r.fold_inputs(data,cal);parent=next(v for v in read(r.ROOT/'training_trace.json')['runs'] if v['held_group']==held)
  hashes={name:array_digest(v) for name,v in [('feature',x),('target',y),('weight',w)]}
  if any(hashes[k]!=parent[k+'_sha256'] for k in hashes):raise ValueError('training data differs from parent HGB')
  models={};fits={};records=[]
  for name in BASELINES:
   ids=[2] if name=='affine_median_matched' else list(range(x.shape[1]));s=scaler(x[:,ids],w);z=transform(x[:,ids],s);notes=[]
   if name=='affine_median_matched':
    beta,intercept=least_squares(z,y,w,0.);native=np.maximum(0.,anchor+z@beta+intercept);metadata={'method':'weighted least squares','alpha':0.,'optimizer_warning':False}
   else:
    est=Ridge(**protocol['ridge']) if name=='ridge_context' else HuberRegressor(**protocol['huber'])
    with warnings.catch_warnings(record=True) as caught:
     warnings.simplefilter('always');est.fit(z,y,sample_weight=w)
    notes=[{'category':v.category.__name__,'message':str(v.message)} for v in caught];beta=est.coef_;intercept=float(est.intercept_);native=np.maximum(0.,anchor+est.predict(z));metadata={'method':type(est).__name__,'parameters':est.get_params(),'optimizer_warning':any(v['category']=='ConvergenceWarning' for v in notes)}
    if name=='huber_context':metadata.update({'iterations':int(est.n_iter_),'residual_scale':float(est.scale_),'training_outliers':int(est.outliers_.sum()),'iteration_budget_reached':bool(est.n_iter_>=protocol['huber']['max_iter'])})
   model={'name':name,'held_group':held,'indices':ids,'scaler':s,'coef':np.asarray(beta).tolist(),'intercept':intercept,'metadata':metadata,'warnings':notes};portable=predict(model,x,anchor);difference=float(np.max(abs(native-portable)))
   if difference>1e-8:raise ValueError('linear native-export predictions differ')
   models[name]=model;fits[name+'__native']=native;fits[name+'__portable']=portable;records.append({'name':name,'features':len(ids),'export_max_difference_mps':difference,'metadata':metadata,'warnings':notes});print('MATCHED MODEL FROZEN',held,name,'warnings',len(notes),flush=True)
  folder=ROOT/'folds'/held;folder.mkdir();write(folder/'models.json',models);np.savez_compressed(folder/'fit_export.npz',**fits)
  runs.append({'held_group':held,'train_samples':len(cal['train_ids']),'held_samples':len(cal['held_ids']),'fit_rows_with_masked_copy':len(x),'feature_sha256':hashes['feature'],'target_sha256':hashes['target'],'weight_sha256':hashes['weight'],'models':records})
  for name in ['models.json','fit_export.npz']:files[f'folds/{held}/{name}']=digest(folder/name)
  write(ROOT/'training_trace.json',{'runs':runs,'sklearn':sklearn.__version__,'numpy':np.__version__,'python':platform.python_version(),'selection':'All fixed models retained; no result-based selection or refit'})
 write(ROOT/'model_lock.json',{'analysis_lock_sha256':digest(ROOT/'analysis_lock.json'),'training_trace_sha256':digest(ROOT/'training_trace.json'),'files_sha256':files,'all_nine_models_locked_before_new_held_scoring':True,'new_held_scores_created':False,'existing_hgb_held_scores_already_inspected':True})
 print('ALL NINE BASELINES LOCKED; seconds',round(time.monotonic()-started,1),flush=True)
if __name__=='__main__':train()
