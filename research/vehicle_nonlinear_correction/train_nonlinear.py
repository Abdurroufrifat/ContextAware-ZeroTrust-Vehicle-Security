import platform,time
from nonlinear_core import *
def train():
 protocol=check_lock()
 if (ROOT/'models').exists() or (ROOT/'model_lock.json').exists():raise ValueError('fitting already attempted; no retry')
 import sklearn
 from sklearn.ensemble import HistGradientBoostingRegressor
 if sklearn.__version__!='1.8.0':raise ValueError('fixed sklearn 1.8.0 required for fitting')
 data=inputs();y=data['oof']['target_mps'];groups=np.array([v['group'] for v in data['members']]);(ROOT/'models').mkdir();runs=[];started=time.monotonic()
 for scope in SCOPES:
  cal=read(RESEARCH/'vehicle_multisource_imu/models'/scope/'calibration.json');ids=np.array(cal['train_ids'],int);held=np.array(cal['held_ids'],int);validate_split(ids,held,len(y))
  if scope!='all_fit' and (np.any(groups[ids]==scope) or not np.all(groups[held]==scope)):raise ValueError('held route in training')
  parent=load(RESEARCH/'vehicle_multisource_imu/models'/scope/'radar_imu_export.npz')['native'];x=head_features(parent,np.tile(data['axes']['fit_axes'][ids],(2,1)));target=np.tile(y[ids],2)-parent;bins=np.digitize(y[ids],[1,5,10]);counts=np.array(cal['bin_counts']);w=np.tile((len(cal['calibration_ids'])/4)/counts[bins]*.5,2)
  for kind in METHODS:
   if kind=='nonlinear7':
    model=HistGradientBoostingRegressor(**protocol['model']);model.fit(x,target,sample_weight=w);artifact=export_tree(model);native=model.predict(x)
    if model.n_iter_!=200:raise ValueError('iteration count differs')
   else:artifact=fit_ridge(x,target,w);native=ridge_predict(artifact,x)
   portable=predict(artifact,x,kind);delta=float(np.max(abs(native-portable)))
   if delta>1e-8:raise ValueError('native export differs')
   artifact.update(feature_names=FEATURES,method=kind,scope=scope,output='clip(parent + residual, minimum=0)');write(ROOT/'models'/scope/(kind+'.json'),artifact);np.savez_compressed(ROOT/'models'/scope/(kind+'_training.npz'),native=native,portable=portable)
   ah=lambda a:hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
   runs.append({'scope':scope,'method':kind,'training_original_rows':len(ids),'held_rows':len(held),'augmented_rows':len(x),'feature_sha256':ah(x),'residual_target_sha256':ah(target),'weight_sha256':ah(w),'native_export_max_difference_mps':delta})
   print('FROZEN',scope,kind,'rows',len(x),'export difference',delta,flush=True)
 write(ROOT/'training_trace.json',{'runs':runs,'sklearn':sklearn.__version__,'numpy':np.__version__,'python':platform.python_version(),'seconds':time.monotonic()-started,'fitting':'In-sample parent residual refinement; not independent calibration'})
 files={p.relative_to(ROOT).as_posix():digest(p) for p in (ROOT/'models').rglob('*') if p.is_file()};write(ROOT/'model_lock.json',{'analysis_lock_sha256':digest(ROOT/'analysis_lock.json'),'training_trace_sha256':digest(ROOT/'training_trace.json'),'files_sha256':files,'all_eight_locked_before_new_scores':True});print('ALL EIGHT MODELS LOCKED',flush=True)
if __name__=='__main__':train()
