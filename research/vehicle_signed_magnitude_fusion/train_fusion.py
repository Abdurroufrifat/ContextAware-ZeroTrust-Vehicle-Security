import platform,time
from fusion_core import *
def train():
 p=freeze()
 if (ROOT/'models').exists() or (ROOT/'model_lock.json').exists():raise ValueError('fusion fitting already started; no retries')
 import sklearn
 from sklearn.ensemble import HistGradientBoostingRegressor
 if sklearn.__version__!='1.8.0':raise ValueError('locked sklearn differs')
 data,members,groups,day,support=r.load_data();base=original_base(data);(ROOT/'models').mkdir();runs=[];files={};started=time.monotonic()
 for scope in SCOPES:
  cal=calibration(data,groups,day,scope);x,y,w,a,aug=fit_inputs(data,base,cal);check_parent_inputs(scope,x,y,w);model=HistGradientBoostingRegressor(**p['model']);model.fit(x,y,sample_weight=w)
  if model.n_iter_!=200:raise ValueError('fusion iteration budget differs')
  artifact=r.export(model);artifact['feature_names']=FEATURE_NAMES;artifact['output']='clip(unchanged signed-median affine anchor + signed/magnitude fused residual,minimum0)';native=np.maximum(0.,a+model.predict(x));portable=c.predict(artifact,x,a);difference=float(np.max(abs(native-portable)))
  if difference>1e-8:raise ValueError('fusion native/export predictions differ')
  folder=ROOT/'models'/scope;folder.mkdir();write(folder/'model.json',artifact);write(folder/'calibration.json',cal);np.savez_compressed(folder/'fit_export.npz',native=native,portable=portable);runs.append({'scope':scope,'train_samples':len(cal['train_ids']),'held_samples':len(cal['held_ids']),'features':148,'iterations':model.n_iter_,'parameters':model.get_params(),'feature_sha256':g.array_digest(x),'target_sha256':g.array_digest(y),'weight_sha256':g.array_digest(w),'signed_prefix_sha256':g.array_digest(x[:,:75]),'mask_sha256':g.array_digest(aug[:,3]),'export_max_difference_mps':difference})
  for n in ['model.json','calibration.json','fit_export.npz']:files[f'models/{scope}/{n}']=digest(folder/n)
  print('FUSION MODEL FROZEN',scope,'fit rows',len(x),'native/export difference',difference,flush=True)
 write(ROOT/'training_trace.json',{'runs':runs,'sklearn':sklearn.__version__,'numpy':np.__version__,'python':platform.python_version(),'selection':'One fixed148-feature model in4 scopes; no retries','seconds':time.monotonic()-started});write(ROOT/'model_lock.json',{'analysis_lock_sha256':digest(ROOT/'analysis_lock.json'),'training_trace_sha256':digest(ROOT/'training_trace.json'),'files_sha256':files,'all_four_locked_before_new_scoring':True,'new_scores_created':False});print('ALL FOUR FUSION MODELS LOCKED',flush=True)
if __name__=='__main__':train()
