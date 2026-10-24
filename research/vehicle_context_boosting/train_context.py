"""One fixed classical baseline; no development-target fitting or model search."""
import platform,time
from context_core import *
def export(model):
 trees=[]
 for group in model._predictors:
  if len(group)!=1:raise ValueError('unexpected regression tree grouping')
  nodes=[]
  for q in group[0].nodes:
   if q['is_categorical']:raise ValueError('only numeric splits may be exported')
   nodes.append({'leaf':True,'value':float(q['value'])} if q['is_leaf'] else {'leaf':False,'feature':int(q['feature_idx']),'threshold':float(q['num_threshold']),'left':int(q['left']),'right':int(q['right'])})
  trees.append(nodes)
 return {'feature_names':FEATURE_NAMES,'baseline':float(model._baseline_prediction[0,0]),'trees':trees,'output':'clip(old fit-only affine anchor + boosted residual,minimum0)'}
def fit_inputs():
 d,day_y=fitting_data();anchor=e.affine(read(OLD/'classical_models.json')['reference_median_affine'],d['median_feature']);aug=m.augment(d['X'],np.random.default_rng(20261019));a=features(d['X'],anchor,d['median_feature']);b=features(aug,anchor,d['median_feature']);w=n.fixed_weights(day_y,d['y']);return d,anchor,a,b,w

def train():
 p=freeze()
 if (ROOT/'model.json').exists() or (ROOT/'model_lock.json').exists():raise ValueError('context training already started; no retries')
 import sklearn
 from sklearn.ensemble import HistGradientBoostingRegressor
 if sklearn.__version__!='1.8.0':raise ValueError('fit requires locked sklearn1.8.0')
 seen={r['source_route'] for r in read(OLD/'inputs/fit/windows.json')}|{r['route'] for r in read(NIGHT/'night_index.json')}
 if seen!=set(FIT_ROUTES):raise ValueError('fitting source ancestry differs')
 started=time.monotonic();d,anchor,a,b,w=fit_inputs();x=np.concatenate([a,b]);y=np.tile(d['y']-anchor,2);weights=np.tile(w*.5,2);model=HistGradientBoostingRegressor(**p['model']);model.fit(x,y,sample_weight=weights);artifact=export(model)
 if model.n_iter_!=200:raise ValueError('fixed iteration budget differs')
 native=np.maximum(0.,np.tile(anchor,2)+model.predict(x));portable=predict(artifact,x,np.tile(anchor,2));difference=float(np.max(abs(native-portable)))
 if difference>1e-8:raise ValueError('native versus exported model differs')
 write(ROOT/'model.json',artifact);np.savez_compressed(ROOT/'fit_export_check.npz',native=native,portable=portable)
 trace={'samples_original':len(d['y']),'day_samples':16169,'night_samples':11009,'samples_with_one_masked_copy':len(x),'features':len(FEATURE_NAMES),'iterations':model.n_iter_,'parameters':model.get_params(),'source_routes':FIT_ROUTES,'masked_seed':20261019,'export_max_difference_mps':difference,'native_fit_mae_mps':float(np.mean(abs(native[:len(anchor)]-d['y']))),'original_feature_sha256':hashlib.sha256(a.astype('<f8').tobytes()).hexdigest(),'masked_feature_sha256':hashlib.sha256(b.astype('<f8').tobytes()).hexdigest(),'sklearn':sklearn.__version__,'numpy':np.__version__,'python':platform.python_version()};write(ROOT/'training_trace.json',trace)
 write(ROOT/'model_lock.json',{'analysis_lock_sha256':digest(ROOT/'analysis_lock.json'),'model_sha256':digest(ROOT/'model.json'),'training_trace_sha256':digest(ROOT/'training_trace.json'),'fit_export_check_sha256':digest(ROOT/'fit_export_check.npz'),'selection':'one final200-tree model; no selection or retries'})
 print('CONTEXT MODEL FROZEN',len(x),'fit examples;',len(FEATURE_NAMES),'features;200 trees;export difference',difference,'seconds',round(time.monotonic()-started,2),flush=True)
if __name__=='__main__':train()
