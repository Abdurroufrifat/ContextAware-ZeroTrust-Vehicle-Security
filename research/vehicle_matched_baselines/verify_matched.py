"""NumPy-only audit of all models, fold isolation, paired rows and saved results."""
import io,unittest
from matched_core import *
from score_matched import compute
from verify_context import independent_features,scalar_predict as scalar_tree
from verify_continuous import close
import test_matched

def scalar_linear(model,row,anchor):
 value=float(anchor)+float(model['intercept'])
 for k,j in enumerate(model['indices']):value+=(float(row[j])-model['scaler']['mean'][k])/model['scaler']['scale'][k]*model['coef'][k]
 return max(0.,value)

def audit():
 project=ROOT.parent.parent
 for name,h in read(ROOT/'prior_snapshot_manifest.json').items():
  if name in ['FINAL_PROJECT_README.md','scripts/check_coact_project.py']:continue
  p=(project/name).resolve()
  if not p.is_relative_to(project) or digest(p)!=h:raise ValueError('prior project payload changed: '+name)
 for name,h in read(ROOT/'sha256.json').items():
  p=(ROOT/name).resolve()
  if not p.is_relative_to(ROOT) or digest(p)!=h:raise ValueError('matched payload changed: '+name)
 freeze();check_models();data,members,groups,day,support=r.load_data();close(members,read(r.ROOT/'membership.json'));close(support,read(r.ROOT/'support_costs.json'));trace=read(ROOT/'training_trace.json');coverage=np.zeros(len(groups),dtype=int);probes=0;fits=0;warnings_seen=[]
 if [v['held_group'] for v in trace['runs']]!=list(r.GROUPS):raise ValueError('fixed fold list changed')
 for held,run in zip(r.GROUPS,trace['runs']):
  cal=read(r.ROOT/'folds'/held/'calibration.json');close(cal,r.calibrate(data,groups,day,held));x,y,w,anchor=r.fold_inputs(data,cal);ids=np.array(cal['held_ids']);train=np.array(cal['train_ids'])
  if np.any(groups[train]==held) or set(train.tolist())&set(ids.tolist()) or len(train)+len(ids)!=len(groups):raise ValueError('fold leakage or population mismatch')
  if run['train_samples']!=len(train) or run['held_samples']!=len(ids) or run['fit_rows_with_masked_copy']!=len(x):raise ValueError('training trace counts differ')
  parent=next(v for v in read(r.ROOT/'training_trace.json')['runs'] if v['held_group']==held)
  for name,v in [('feature',x),('target',y),('weight',w)]:
   if array_digest(v)!=run[name+'_sha256'] or run[name+'_sha256']!=parent[name+'_sha256']:raise ValueError('HGB and baseline training matrices differ')
  changed={k:v.copy() for k,v in data.items()};changed['y'][ids]+=12345.;changed['median_feature'][ids]+=12345.;changed['X'][ids]+=12345.
  close(cal,r.calibrate(changed,groups,day,held))
  for before,after in zip((x,y,w,anchor),r.fold_inputs(changed,cal)):
   if not np.array_equal(before,after):raise ValueError('held data changed fitted inputs')
  models=read(ROOT/'folds'/held/'models.json')
  if set(models)!=set(BASELINES) or [v['name'] for v in run['models']]!=list(BASELINES):raise ValueError('model omitted or reordered')
  with np.load(ROOT/'folds'/held/'fit_export.npz',allow_pickle=False) as stored:
   for name in BASELINES:
    m=models[name];rec=next(v for v in run['models'] if v['name']==name);jj=m['indices'];z=transform(x[:,jj],m['scaler']);close(m['scaler'],scaler(x[:,jj],w));close(rec['metadata'],m['metadata']);close(rec['warnings'],m['warnings'])
    if m['held_group']!=held or jj!=([2] if name==BASELINES[0] else list(range(75))):raise ValueError('wrong model fold or feature subset')
    # A weighted-average oracle separately reconstructs scaling parameters.
    mean=np.average(x[:,jj],axis=0,weights=w);variance=np.average((x[:,jj]-mean)**2,axis=0,weights=w);close(m['scaler'],{'mean':mean.tolist(),'scale':np.where(variance<=1e-12,1.,np.sqrt(variance)).tolist()})
    if name==BASELINES[0]:
     q=z[:,0];qm=np.average(q,weights=w);ym=np.average(y,weights=w);beta=np.sum(w*(q-qm)*(y-ym))/np.sum(w*(q-qm)**2);close(m['coef'],[float(beta)]);close(m['intercept'],float(ym-beta*qm))
    elif name=='ridge_context':
     beta,intercept=least_squares(z,y,w,1.);close(m['coef'],beta.tolist());close(m['intercept'],intercept)
    pred=predict(m,x,anchor);np.testing.assert_allclose(pred,stored[name+'__portable'],rtol=1e-10,atol=1e-10);np.testing.assert_allclose(pred,stored[name+'__native'],rtol=1e-10,atol=1e-10);fits+=len(pred)
    if m['metadata']['optimizer_warning'] or m['metadata'].get('iteration_budget_reached',False):warnings_seen.append(held+'/'+name)
  held_anchor=r.c.e.affine(cal['anchor'],data['median_feature'][ids]);held_features=r.c.features(data['X'][ids],held_anchor,data['median_feature'][ids]);tree=read(r.ROOT/'folds'/held/'model.json');parent_prediction=r.c.predict(tree,held_features,held_anchor)
  with np.load(r.ROOT/'oof_predictions.npz',allow_pickle=False) as z:np.testing.assert_allclose(parent_prediction,z['prediction_mps'][ids],rtol=1e-10,atol=1e-10)
  for local in np.unique(np.linspace(0,len(ids)-1,65,dtype=int)):
   index=ids[local];feature=independent_features(data['X'][index],held_anchor[local],data['median_feature'][index]);np.testing.assert_allclose(feature,held_features[local],rtol=1e-9,atol=1e-9)
   for name in BASELINES:
    np.testing.assert_allclose(scalar_linear(models[name],feature,held_anchor[local]),predict(models[name],held_features[local:local+1],held_anchor[local:local+1])[0],rtol=1e-9,atol=1e-9);probes+=1
   np.testing.assert_allclose(scalar_tree(tree,feature,held_anchor[local]),parent_prediction[local],rtol=1e-9,atol=1e-9);probes+=1
  coverage[ids]+=1
 if not np.all(coverage==1):raise ValueError('supported held rows not scored exactly once')
 result,arrays=compute();close(result,read(ROOT/'results.json'))
 with np.load(ROOT/'oof_predictions.npz',allow_pickle=False) as stored:
  if set(stored.files)!=set(arrays):raise ValueError('saved prediction schema differs')
  for name,v in arrays.items():np.testing.assert_allclose(v,stored[name],rtol=1e-10,atol=1e-10)
 # Independently recompute every reported error cell from its same-row subset.
 y=arrays['target_mps'];labels={'by_group':'group','by_recording':'route','by_window':'window_id'}
 contexts=[(np.ones(len(y),dtype=bool),result['pooled'])]
 for category,key in labels.items():
  v=np.array([m[key] for m in members]);contexts.extend((v==name,s) for name,s in result[category].items())
 for selected,cell in contexts:
  for name in MODELS:
   errors=np.abs(y[selected]-arrays[name+'__prediction_mps'][selected]);metric={'count':len(errors),'mae_mps':float(errors.mean()),'rmse_mps':float(np.sqrt(np.mean(errors**2))),'p95_mps':float(np.quantile(errors,.95)),'maximum_mps':float(errors.max()),'above_2_mps':int(np.sum(errors>2.))};close(metric,cell['models'][name])
 suite=unittest.defaultTestLoader.loadTestsFromModule(test_matched);buf=io.StringIO();tests=unittest.TextTestRunner(stream=buf,verbosity=1).run(suite)
 if not tests.wasSuccessful():raise ValueError(buf.getvalue())
 print(f'MATCHED BASELINE AUDIT: PASS ({tests.testsRun} tests;9 fixed fits;{len(y)} rows per model;{fits} native/export fit comparisons;{probes} independent scalar probes)',flush=True)
 for name,s in result['pooled']['models'].items():print(name,'P95',round(s['p95_mps'],6),'m/s; failed windows',len(result['diagnostic_failures'][name]['windows']),flush=True)
 print('OPTIMIZER WARNING/BUDGET FLAGS:',warnings_seen,flush=True)
 print(result['decision']+'; RETROSPECTIVE. PARENT GATE STILL FAILED. WITNESS DISABLED.',flush=True)
if __name__=='__main__':audit()
