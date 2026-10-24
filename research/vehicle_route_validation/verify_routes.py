"""Audit fold isolation, fresh calibration, every exported fit and held prediction."""
import io,unittest
from route_core import *
from score_routes import check_models,compute
from verify_context import independent_features,scalar_predict
from verify_continuous import close
import test_routes

def validated_calibration(rebuilt,saved):
 close(rebuilt,saved)
 return saved

def independent_anchor(data,groups,day,held):
 ids=np.flatnonzero((groups!=held)&day);y=data['y'][ids];x=data['median_feature'][ids];bins=np.searchsorted(np.array([1.,5.,10.]),y,side='right');counts=np.array([np.sum(bins==b) for b in range(4)]);w=len(ids)/(4*counts[bins]);design=np.column_stack([x,np.ones(len(x))]);slope,intercept=np.linalg.lstsq(design*np.sqrt(w[:,None]),y*np.sqrt(w),rcond=None)[0]
 return {'slope':float(slope),'intercept':float(intercept)},counts.tolist(),ids

def audit():
 project=ROOT.parent.parent
 for name,h in read(ROOT/'prior_snapshot_manifest.json').items():
  if name in ['FINAL_PROJECT_README.md','scripts/check_coact_project.py']:continue
  p=(project/name).resolve()
  if not p.is_relative_to(project) or digest(p)!=h:raise ValueError('preceding project changed: '+name)
 for name,h in read(ROOT/'sha256.json').items():
  p=(ROOT/name).resolve()
  if not p.is_relative_to(ROOT) or digest(p)!=h:raise ValueError('route-validation payload changed: '+name)
 freeze();check_models();data,membership,groups,day,support=load_data();close(membership,read(ROOT/'membership.json'));close(support,read(ROOT/'support_costs.json'));trace=read(ROOT/'training_trace.json')
 if len(trace['runs'])!=3 or [v['held_group'] for v in trace['runs']]!=list(GROUPS):raise ValueError('all fixed folds not retained')
 coverage=np.zeros(len(groups),dtype=int);probes=0
 for held,run in zip(GROUPS,trace['runs']):
  folder=ROOT/'folds'/held;cal=validated_calibration(calibrate(data,groups,day,held),read(folder/'calibration.json'));train=np.array(cal['train_ids']);test=np.array(cal['held_ids']);coverage[test]+=1
  if np.intersect1d(train,test).size or set(groups[train])&set(groups[test]) or len(train)+len(test)!=27178:raise ValueError('fold leakage or omitted rows')
  changed={**data,'y':data['y'].copy()};changed['y'][test]+=12345.;close(calibrate(changed,groups,day,held),cal);np.testing.assert_array_equal(changed['y'][train],data['y'][train])
  ref,count,ids=independent_anchor(data,groups,day,held);close(ref,cal['anchor']);np.testing.assert_array_equal(ids,cal['calibration_ids']);np.testing.assert_array_equal(count,cal['bin_counts'])
  x,y,w,a=fold_inputs(data,cal)
  for array,key in [(x,'feature_sha256'),(y,'target_sha256'),(w,'weight_sha256')]:
   if hashlib.sha256(array.astype('<f8').tobytes()).hexdigest()!=run[key]:raise ValueError('fold fitting data changed')
  if run['train_samples']!=len(train) or run['held_samples']!=len(test) or run['training_day_calibration_samples']!=len(ids) or run['fit_rows_including_masked_copy']!=2*len(train) or run['iterations']!=200 or run['features']!=75 or run['export_max_difference_mps']>1e-8:raise ValueError('fold fit budget differs')
  p=read(ROOT/'protocol.json')['model']
  if any(run['parameters'].get(k)!=v for k,v in p.items()):raise ValueError('fixed learner parameters differ')
  model=read(folder/'model.json');native_rebuilt=c.predict(model,x,a)
  with np.load(folder/'fit_export.npz',allow_pickle=False) as z:np.testing.assert_allclose(native_rebuilt,z['native'],rtol=0,atol=1e-8);np.testing.assert_allclose(native_rebuilt,z['portable'],rtol=0,atol=1e-9)
  probe_ids=test[np.unique(np.linspace(0,len(test)-1,129,dtype=int))];anchor=c.e.affine(cal['anchor'],data['median_feature'][probe_ids]);features=c.features(data['X'][probe_ids],anchor,data['median_feature'][probe_ids]);prediction=c.predict(model,features,anchor)
  for j,i in enumerate(probe_ids):
   row=independent_features(data['X'][i],anchor[j],data['median_feature'][i]);np.testing.assert_allclose(row,features[j],rtol=0,atol=1e-10);np.testing.assert_allclose(scalar_predict(model,row,anchor[j]),prediction[j],rtol=0,atol=1e-9);probes+=1
 if not np.all(coverage==1):raise ValueError('held rows not scored exactly once')
 result,arrays=compute();close(result,read(ROOT/'results.json'))
 with np.load(ROOT/'oof_predictions.npz',allow_pickle=False) as z:
  if set(z.files)!=set(arrays):raise ValueError('held prediction keys differ')
  for key,value in arrays.items():np.testing.assert_allclose(value,z[key],rtol=0,atol=1e-9)
 np.testing.assert_array_equal(arrays['fold_id'],np.array([GROUPS.index(g) for g in groups]));log=io.StringIO();tests=unittest.TextTestRunner(stream=log).run(unittest.defaultTestLoader.loadTestsFromModule(test_routes))
 if not tests.wasSuccessful():raise ValueError(log.getvalue())
 print(f'ROUTE VALIDATION AUDIT: PASS ({tests.testsRun} tests;3 isolated route folds;27178 rows scored once;{probes} independent feature/tree probes;training-only anchor and weights)')
 for held,stats in result['by_group'].items():print('ROUTE',held,'P95',round(stats['context_boosting']['p95_mps'],6),'m/s; supported rows',stats['supported_rows'])
 print('ROUTE VALIDATION DECISION:',result['decision']);print('RETROSPECTIVE VALIDATION;NOT UNTOUCHED FINAL TEST. PARENT DEVELOPMENT GATE STILL FAILED. WITNESS DISABLED.')
if __name__=='__main__':audit()
