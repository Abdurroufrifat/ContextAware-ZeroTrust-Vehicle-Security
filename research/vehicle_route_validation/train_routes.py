"""Train every route-fold model before any held-group scoring."""
import platform,time
from route_core import *
def train():
 protocol=freeze()
 if (ROOT/'folds').exists() or (ROOT/'model_lock.json').exists():raise ValueError('route validation training already started; no retries')
 import sklearn
 from sklearn.ensemble import HistGradientBoostingRegressor
 if sklearn.__version__!='1.8.0':raise ValueError('locked training library version differs')
 data,membership,groups,day,support=load_data();write(ROOT/'membership.json',membership);write(ROOT/'support_costs.json',support);(ROOT/'folds').mkdir();runs=[];files={};started=time.monotonic()
 for held in GROUPS:
  cal=calibrate(data,groups,day,held);x,y,w,anchor=fold_inputs(data,cal);model=HistGradientBoostingRegressor(**protocol['model']);model.fit(x,y,sample_weight=w);artifact=export(model)
  if model.n_iter_!=200:raise ValueError('fixed tree budget differs')
  native=np.maximum(0.,anchor+model.predict(x));portable=c.predict(artifact,x,anchor);difference=float(np.max(abs(native-portable)))
  if difference>1e-8:raise ValueError('native-export training predictions differ')
  folder=ROOT/'folds'/held;folder.mkdir();write(folder/'calibration.json',cal);write(folder/'model.json',artifact);np.savez_compressed(folder/'fit_export.npz',native=native,portable=portable)
  run={'held_group':held,'train_groups':sorted(set(groups[cal['train_ids']].tolist())),'train_samples':len(cal['train_ids']),'held_samples':len(cal['held_ids']),'training_day_calibration_samples':len(cal['calibration_ids']),'fit_rows_including_masked_copy':len(x),'iterations':model.n_iter_,'features':x.shape[1],'export_max_difference_mps':difference,'feature_sha256':hashlib.sha256(x.astype('<f8').tobytes()).hexdigest(),'target_sha256':hashlib.sha256(y.astype('<f8').tobytes()).hexdigest(),'weight_sha256':hashlib.sha256(w.astype('<f8').tobytes()).hexdigest(),'parameters':model.get_params()};runs.append(run)
  for name in ['calibration.json','model.json','fit_export.npz']:
   rel=f'folds/{held}/{name}';files[rel]=digest(ROOT/rel)
  write(ROOT/'training_trace.json',{'runs':runs,'sklearn':sklearn.__version__,'numpy':np.__version__,'python':platform.python_version(),'selection':'all3 predeclared final200-tree models; no held scores used'});print('ROUTE MODEL FROZEN',held,'training',len(cal['train_ids']),'held',len(cal['held_ids']),'export difference',difference,flush=True)
 write(ROOT/'model_lock.json',{'analysis_lock_sha256':digest(ROOT/'analysis_lock.json'),'files_sha256':files,'training_trace_sha256':digest(ROOT/'training_trace.json'),'membership_sha256':digest(ROOT/'membership.json'),'support_costs_sha256':digest(ROOT/'support_costs.json'),'held_scores_created':False,'all_fold_models_locked_before_scoring':True})
 print('ALL THREE MODELS LOCKED BEFORE HELD SCORING;seconds',round(time.monotonic()-started,1),flush=True)
if __name__=='__main__':train()
