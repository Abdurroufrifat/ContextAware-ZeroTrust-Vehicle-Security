import platform,time
from multisource_core import *
def train():
 p=freeze();arrays=cache()
 if (ROOT/'models').exists() or (ROOT/'model_lock.json').exists():raise ValueError('multi-source fitting already started; no retries')
 import sklearn
 from sklearn.ensemble import HistGradientBoostingRegressor
 if sklearn.__version__!='1.8.0':raise ValueError('locked sklearn differs')
 data,members,groups,day,support=r.load_data();base=c.features(data['X'],np.zeros(len(members)),np.zeros(len(members)));(ROOT/'models').mkdir();runs=[];files={};started=time.monotonic()
 for scope in SCOPES:
  cal=f.calibration(data,groups,day,scope);folder=ROOT/'models'/scope;folder.mkdir();write(folder/'calibration.json',cal);files[f'models/{scope}/calibration.json']=digest(folder/'calibration.json')
  for method in NEW_METHODS:
   x,y,w,a,aug=training_arrays(data,base,arrays['fit_imu'],cal,method)
   if method=='radar_imu':f.check_parent_inputs(scope,x,y,w)
   model=HistGradientBoostingRegressor(**p['model']);model.fit(x,y,sample_weight=w)
   if model.n_iter_!=200:raise ValueError('fixed multi-source iteration budget differs')
   artifact=r.export(model);artifact['feature_names']=FEATURE_NAMES[method];artifact['output']='clip(day-only radar affine anchor + residual,minimum0)' if method=='radar_imu' else 'clip(IMU-only direct speed,minimum0); zero radar anchor';native=np.maximum(0.,a+model.predict(x));portable=c.predict(artifact,x,a);difference=float(np.max(np.abs(native-portable)))
   if difference>1e-8:raise ValueError('multi-source native/export mismatch')
   write(folder/(method+'.json'),artifact);np.savez_compressed(folder/(method+'_export.npz'),native=native,portable=portable);runs.append({'scope':scope,'method':method,'train_samples':len(cal['train_ids']),'held_samples':len(cal['held_ids']),'feature_count':len(FEATURE_NAMES[method]),'iterations':model.n_iter_,'parameters':model.get_params(),'feature_sha256':g.array_digest(x),'target_sha256':g.array_digest(y),'weight_sha256':g.array_digest(w),'used_anchor_sha256':g.array_digest(a),'mask_sha256':g.array_digest(aug[:,3]),'export_max_difference_mps':difference})
   for n in [method+'.json',method+'_export.npz']:files[f'models/{scope}/{n}']=digest(folder/n)
   print('MULTI-SOURCE MODEL FROZEN',scope,method,len(x),'rows;native/export difference',difference,flush=True)
 write(ROOT/'training_trace.json',{'runs':runs,'sklearn':sklearn.__version__,'numpy':np.__version__,'python':platform.python_version(),'selection':'Two prespecified methods in four scopes; eight fixed200-tree fits, no retries','seconds':time.monotonic()-started});write(ROOT/'model_lock.json',{'analysis_lock_sha256':digest(ROOT/'analysis_lock.json'),'data_lock_sha256':digest(ROOT/'data_lock.json'),'training_trace_sha256':digest(ROOT/'training_trace.json'),'files_sha256':files,'all_eight_locked_before_new_scoring':True,'new_scores_created':False});print('ALL EIGHT MULTI-SOURCE MODELS LOCKED BEFORE NEW SCORING',flush=True)
if __name__=='__main__':train()
