"""Portable replay of fixed ablations: no sklearn, refitting, or final data."""
import io,unittest
from magnitude_core import *
from score_magnitude import compute
from verify_diagnosis import independent_metric
import test_magnitude

def independent_features(row,anchor,median):
 row=np.asarray(row,dtype=np.float64);v=row[0]*10;valid=row[3]==1;out=[row[0,-1],anchor,median]
 for width in c.SPANS:
  vals=v[-width:][valid[-width:]];times=np.arange(-(width-1),1)[valid[-width:]]*.05
  if not len(vals):out.extend([0.]*10+[0.,1.]);continue
  ordered=sorted(vals);trim=int(len(vals)*.1);trimmed=ordered[trim:len(vals)-trim];med=float(np.median(vals));centered=times-times.mean();slope=float(np.dot(centered,vals-vals.mean())/np.dot(centered,centered)) if len(vals)>1 and np.ptp(times)>0 else 0.;half=width//2;left=v[-width:][:half][valid[-width:][:half]];right=v[-width:][half:][valid[-width:][half:]];halfslope=(np.median(right)-np.median(left))/(width*.025) if len(left) and len(right) else 0.
  out.extend([med,float(np.mean(vals)),float(np.mean(trimmed)),float(np.quantile(vals,.1)),float(np.quantile(vals,.9)),float(np.median(abs(vals-med))),float(np.std(vals)),float(max(vals)-min(vals)),slope,float(halfslope),len(vals)/width,float(np.mean(row[2,-width:][valid[-width:]]))])
 return np.array(out)

def scalar_predict(model,row,anchor):
 total=float(model['baseline'])
 for tree in model['trees']:
  j=0
  while not tree[j]['leaf']:
   node=tree[j];j=node['left'] if row[node['feature']]<=node['threshold'] else node['right']
  total+=tree[j]['value']
 return max(0.,anchor+total)

def independent_anchor(data,median,groups,day,held):
 ids=np.flatnonzero((groups!=held)&day);y=data['y'][ids];bins=np.searchsorted([1.,5.,10.],y,side='right');counts=np.bincount(bins,minlength=4);w=len(ids)/(4*counts[bins]);a=np.column_stack([median[ids],np.ones(len(ids))]);slope,intercept=np.linalg.lstsq(a*np.sqrt(w[:,None]),y*np.sqrt(w),rcond=None)[0]
 return {'slope':float(slope),'intercept':float(intercept)},counts.tolist(),ids

def audit():
 freeze();check_models();project=ROOT.parent.parent;available=0;prior=read(ROOT/'prior_snapshot_manifest.json')
 for name,h in prior.items():
  if name in ['FINAL_PROJECT_README.md','scripts/check_coact_project.py']:continue
  f=(project/name).resolve()
  if not f.is_relative_to(project):raise ValueError('unsafe prior artifact path')
  if f.is_file():
   if digest(f)!=h:raise ValueError('preceding payload changed: '+name)
   available+=1
 if (ROOT/'sha256.json').exists():
  for name,h in read(ROOT/'sha256.json').items():
   f=(ROOT/name).resolve()
   if not f.is_relative_to(ROOT) or digest(f)!=h:raise ValueError('ablation payload changed: '+name)
 data,members,groups,day,support,median_abs=load_data();base=original_base(data);trace=read(ROOT/'training_trace.json');p=read(ROOT/'protocol.json')
 if trace['base_magnitude_feature_sha256']!=array_digest(base) or len(trace['runs'])!=3:raise ValueError('feature base or fold count changed')
 coverage=np.zeros(len(groups),int);fits=0;probes=0;mask_checks=0
 for held,run in zip(r.GROUPS,trace['runs']):
  if run['held_group']!=held or [v['variant'] for v in run['models']]!=list(VARIANTS):raise ValueError('selected/missing fold or variant')
  folder=ROOT/'folds'/held;saved=read(folder/'calibrations.json');original=read(r.ROOT/'folds'/held/'calibration.json');close(saved[VARIANTS[0]],original);validated_calibration(r.calibrate(data,groups,day,held),original)
  cal=saved[VARIANTS[0]];train=np.array(cal['train_ids']);test=np.array(cal['held_ids']);coverage[test]+=1
  if len(train)+len(test)!=len(groups) or np.intersect1d(train,test).size or set(groups[train])&set(groups[test]) or run['train_samples']!=len(train) or run['held_samples']!=len(test):raise ValueError('fold leakage or omitted rows')
  close(new_anchor_calibration(data,groups,day,held,median_abs),saved[VARIANTS[1]])
  changed={**data,'y':data['y'].copy()};changed['y'][test]+=12345.;modified_median=median_abs.copy();modified_median[test]+=54321.
  close(new_anchor_calibration(changed,groups,day,held,modified_median),saved[VARIANTS[1]])
  foldbase,aug=fold_bases(data,cal,base);signed_aug=c.m.augment(data['X'][train],np.random.default_rng(20261019));np.testing.assert_array_equal(aug,magnitude_grid(signed_aug));mask_checks+=len(train)
  if run['augmented_mask_sha256']!=array_digest(aug[:,3]):raise ValueError('mask seed/policy changed')
  parent_run=next(v for v in read(r.ROOT/'training_trace.json')['runs'] if v['held_group']==held)
  with np.load(folder/'fit_export.npz',allow_pickle=False) as z:
   if set(z.files)!={v+'__'+kind for v in VARIANTS for kind in ['native','portable']}:raise ValueError('export comparison keys differ')
   for variant,record in zip(VARIANTS,run['models']):
    vcal=saved[variant]
    for key in ['train_ids','held_ids','calibration_ids','bin_counts']:close(vcal[key],cal[key])
    median=data['median_feature'] if variant==VARIANTS[0] else median_abs
    affine,counts,ids=independent_anchor(data,median,groups,day,held);close(affine,vcal['anchor']);close(counts,vcal['bin_counts']);np.testing.assert_array_equal(ids,vcal['calibration_ids'])
    x,y,w,a=training_inputs(data,median_abs,foldbase,vcal,variant)
    for arr,key in [(x,'feature_sha256'),(y,'target_sha256'),(w,'weight_sha256')]:
     if array_digest(arr)!=record[key]:raise ValueError('training data hash differs: '+key)
    if record['weight_sha256']!=parent_run['weight_sha256'] or (variant==VARIANTS[0] and record['target_sha256']!=parent_run['target_sha256']) or record['original_target_sha256']!=array_digest(np.tile(data['y'][train],2)):raise ValueError('parent target/weight identity differs')
    if record['iterations']!=200 or record['features']!=75 or any(record['parameters'].get(k)!=v for k,v in p['model'].items()):raise ValueError('model budget differs')
    model=read(folder/(variant+'.json'))
    if model['feature_names']!=feature_names(variant) or len(model['trees'])!=200:raise ValueError('model schema differs')
    portable=c.predict(model,x,a);np.testing.assert_allclose(portable,z[variant+'__native'],atol=1e-8,rtol=0);np.testing.assert_allclose(portable,z[variant+'__portable'],atol=1e-9,rtol=0);fits+=len(x)
    # Check both held original rows and training masked rows with an independent scalar feature/tree oracle.
    held_ids=test[np.unique(np.linspace(0,len(test)-1,65,dtype=int))];mx=magnitude_grid(data['X'][held_ids]);ha=c.e.affine(vcal['anchor'],median[held_ids]);hf=assemble(base[held_ids],ha,median[held_ids]);hp=c.predict(model,hf,ha)
    for j,i in enumerate(held_ids):
     row=independent_features(mx[j],ha[j],median[i]);np.testing.assert_allclose(row,hf[j],rtol=0,atol=1e-10);np.testing.assert_allclose(scalar_predict(model,row,ha[j]),hp[j],rtol=0,atol=1e-9);probes+=1
    for j in np.unique(np.linspace(0,len(train)-1,17,dtype=int)):
     index=len(train)+j;row=independent_features(aug[j],a[index],median[train[j]]);np.testing.assert_allclose(row,x[index],rtol=0,atol=1e-10);np.testing.assert_allclose(scalar_predict(model,row,a[index]),portable[index],rtol=0,atol=1e-9);probes+=1
 if not np.all(coverage==1):raise ValueError('held rows not covered exactly once')
 result,arrays=compute();close(result,read(ROOT/'results.json'))
 with np.load(ROOT/'oof_predictions.npz',allow_pickle=False) as z:
  if set(z.files)!=set(arrays):raise ValueError('OOF keys differ')
  for key,value in arrays.items():np.testing.assert_allclose(value,z[key],rtol=0,atol=1e-9)
 y=data['y'];labels={'by_group':groups,'by_recording':np.array([v['route'] for v in members]),'by_window':np.array([v['window_id'] for v in members]),'by_reference_speed':np.array([d.SPEED_NAMES[i] for i in np.searchsorted([1.,5.,10.],y,side='right')]),'by_day_night':np.where(day,'day','night')};cells=[(np.ones(len(y),bool),result['pooled'])]
 for key,values in labels.items():cells.extend((values==name,cell) for name,cell in result[key].items())
 for mask,cell in cells:
  if cell['count']!=int(mask.sum()):raise ValueError('metric denominator changed')
  for name in MODELS:close(cell['models'][name],independent_metric(y[mask],arrays[name+'__prediction_mps'][mask]))
 parent=read(r.ROOT/'results.json')
 for category,old in [('pooled','pooled_sample_weighted'),('by_group','by_group'),('by_recording','by_recording'),('by_window','by_window')]:
  current={'pooled':result['pooled']} if category=='pooled' else result[category];previous={'pooled':parent[old]} if category=='pooled' else parent[old]
  for name,cell in current.items():
   for key,value in previous[name]['context_boosting'].items():close(cell['models']['original_context'][key],value)
 buf=io.StringIO();tests=unittest.TextTestRunner(stream=buf).run(unittest.defaultTestLoader.loadTestsFromModule(test_magnitude))
 if not tests.wasSuccessful():raise ValueError(buf.getvalue())
 print(f'MAGNITUDE ABLATION AUDIT: PASS ({tests.testsRun} tests;6 fixed models;{len(y)} held rows/model;{fits} native/export comparisons;{probes} independent feature/tree probes;{len(cells)*3} independent metric cells;{mask_checks} mask commutation checks)')
 print(f'Available preceding payloads verified: {available}/{len(prior)-2}; unavailable historical files are not claimed as reverified.')
 for name,s in result['pooled']['models'].items():print(name,'P95',round(s['p95_mps'],6),'m/s; failed windows',len(result['diagnostic_failures'][name]['windows']))
 print(result['decision']+'; RETROSPECTIVE. PARENT FULL-RECORDING GATE STILL FAILED. WITNESS DISABLED.')
if __name__=='__main__':audit()
