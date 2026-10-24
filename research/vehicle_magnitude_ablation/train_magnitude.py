import platform,time
from magnitude_core import *
def train():
 p=freeze()
 if (ROOT/'folds').exists() or (ROOT/'model_lock.json').exists():raise ValueError('magnitude fitting already started; no retries')
 import sklearn
 from sklearn.ensemble import HistGradientBoostingRegressor
 if sklearn.__version__!='1.8.0':raise ValueError('locked fitting library differs')
 data,members,groups,day,support,median_abs=load_data();base=original_base(data);(ROOT/'folds').mkdir();runs=[];files={};started=time.monotonic()
 for held in r.GROUPS:
  parent=read(r.ROOT/'folds'/held/'calibration.json');a=validated_calibration(r.calibrate(data,groups,day,held),parent);b=new_anchor_calibration(data,groups,day,held,median_abs);calibrations={VARIANTS[0]:a,VARIANTS[1]:b}
  if any(a[k]!=b[k] for k in ['train_ids','held_ids','calibration_ids','bin_counts']):raise ValueError('ablation split/weight membership differs')
  foldbase,aug=fold_bases(data,a,base);folder=ROOT/'folds'/held;folder.mkdir();write(folder/'calibrations.json',calibrations);fits={};records=[];parent_run=next(v for v in read(r.ROOT/'training_trace.json')['runs'] if v['held_group']==held)
  for variant in VARIANTS:
   cal=calibrations[variant];x,y,w,anchor=training_inputs(data,median_abs,foldbase,cal,variant);hashes={k:array_digest(v) for k,v in [('feature',x),('target',y),('weight',w)]}
   if hashes['weight']!=parent_run['weight_sha256'] or variant==VARIANTS[0] and hashes['target']!=parent_run['target_sha256']:raise ValueError('fixed parent weights/targets differ')
   model=HistGradientBoostingRegressor(**p['model']);model.fit(x,y,sample_weight=w)
   if model.n_iter_!=200:raise ValueError('iteration budget differs')
   artifact=r.export(model);artifact['feature_names']=feature_names(variant);artifact['output']='clip(fold-specific anchor + boosted residual,minimum0)';native=np.maximum(0.,anchor+model.predict(x));portable=c.predict(artifact,x,anchor);difference=float(np.max(abs(native-portable)))
   if difference>1e-8:raise ValueError('native/export ablation predictions differ')
   write(folder/(variant+'.json'),artifact);fits[variant+'__native']=native;fits[variant+'__portable']=portable;records.append({'variant':variant,'features':75,'iterations':model.n_iter_,'feature_sha256':hashes['feature'],'target_sha256':hashes['target'],'weight_sha256':hashes['weight'],'original_target_sha256':array_digest(np.tile(data['y'][cal['train_ids']],2)),'parameters':model.get_params(),'export_max_difference_mps':difference});print('MAGNITUDE MODEL FROZEN',held,variant,'export difference',difference,flush=True)
  np.savez_compressed(folder/'fit_export.npz',**fits);runs.append({'held_group':held,'train_samples':len(a['train_ids']),'held_samples':len(a['held_ids']),'augmented_mask_sha256':array_digest(aug[:,3]),'models':records})
  for name in ['calibrations.json',*(v+'.json' for v in VARIANTS),'fit_export.npz']:files[f'folds/{held}/{name}']=digest(folder/name)
  write(ROOT/'training_trace.json',{'runs':runs,'sklearn':sklearn.__version__,'numpy':np.__version__,'python':platform.python_version(),'selection':'Both fixed final200-tree variants retained in all3 folds; no retries','base_magnitude_feature_sha256':array_digest(base)})
 write(ROOT/'model_lock.json',{'analysis_lock_sha256':digest(ROOT/'analysis_lock.json'),'training_trace_sha256':digest(ROOT/'training_trace.json'),'files_sha256':files,'all_six_models_locked_before_new_held_scoring':True,'new_held_scores_created':False,'prior_original_held_scores_already_inspected':True})
 print('ALL SIX MAGNITUDE MODELS LOCKED; seconds',round(time.monotonic()-started,1),flush=True)
if __name__=='__main__':train()
