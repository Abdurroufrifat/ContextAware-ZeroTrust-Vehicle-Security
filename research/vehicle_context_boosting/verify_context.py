"""Preserve preceding evidence; independently check features, trees and request clock."""
import io,unittest
from context_core import *
from score_context import compute,check_model
from train_context import fit_inputs
from verify_continuous import close
import test_context
import diagnose_remaining
import paired_blocks

def independent_features(row,anchor,median):
 row=np.asarray(row,dtype=np.float64)
 v=row[0]*10;valid=row[3]==1;out=[row[0,-1],anchor,median]
 for width in SPANS:
  vals=v[-width:][valid[-width:]];times=np.arange(-(width-1),1)[valid[-width:]]*.05
  if not len(vals):out.extend([0.]*10+[0.,1.]);continue
  ordered=sorted(vals);trim=int(len(vals)*.1);trimmed=ordered[trim:len(vals)-trim];med=float(np.median(vals));slope=float(np.dot(times-times.mean(),vals-vals.mean())/np.dot(times-times.mean(),times-times.mean())) if len(vals)>1 and np.ptp(times)>0 else 0.;half=width//2;left=v[-width:][:half][valid[-width:][:half]];right=v[-width:][half:][valid[-width:][half:]];halfslope=(np.median(right)-np.median(left))/(width*.025) if len(left) and len(right) else 0.
  out.extend([med,float(np.mean(vals)),float(np.mean(trimmed)),float(np.quantile(vals,.1)),float(np.quantile(vals,.9)),float(np.median(abs(vals-med))),float(np.std(vals)),float(max(vals)-min(vals)),slope,float(halfslope),len(vals)/width,float(np.mean(row[2,-width:][valid[-width:]]))])
 return np.array(out)
def scalar_predict(model,row,anchor):
 total=float(model['baseline'])
 for tree in model['trees']:
  j=0
  while not tree[j]['leaf']:
   q=tree[j];j=q['left'] if row[q['feature']]<=q['threshold'] else q['right']
  total+=tree[j]['value']
 return max(0.,anchor+total)
def scalar_clock(index,p,peer,source):
 events=[v['time_ns'] for v in index];i=-1;out=[]
 for t in range(events[0],events[-1]+1,50000000):
  while i+1<len(events) and events[i+1]<=t:i+=1
  status=0;v=float('nan')
  if i<0:status=1
  elif t-events[i]>100000000:status=2
  elif not index[i]['ready']:status=3
  else:
   a=index[i]['array_index'];v=float(p[a])
   if not source[i] or not np.isfinite(v) or v<0 or not np.isfinite(peer[a]) or peer[a]<0 or abs(v-peer[a])>1.:status=4;v=float('nan')
  out.append((t,i,t-events[i] if i>=0 else -1,status,v))
 return out

def audit():
 project=ROOT.parent.parent
 for name,h in read(ROOT/'prior_snapshot_manifest.json').items():
  if name in ['FINAL_PROJECT_README.md','scripts/check_coact_project.py']:continue
  p=(project/name).resolve()
  if not p.is_relative_to(project) or digest(p)!=h:raise ValueError('prior project changed: '+name)
 for name,h in read(ROOT/'sha256.json').items():
  p=(ROOT/name).resolve()
  if not p.is_relative_to(ROOT) or digest(p)!=h:raise ValueError('context payload changed: '+name)
 freeze();model=check_model();close(diagnose_remaining.compute(),read(ROOT/'diagnosis_before_change.json'));close(paired_blocks.compute(),read(ROOT/'paired_block_check.json'));d,anchor,a,b,w=fit_inputs();trace=read(ROOT/'training_trace.json')
 if trace['samples_original']!=27178 or trace['samples_with_one_masked_copy']!=54356 or trace['iterations']!=200 or trace['features']!=75 or trace['parameters']!=read(ROOT/'protocol.json')['model']|{'quantile':None,'monotonic_cst':None,'interaction_cst':None,'warm_start':False,'scoring':'loss','validation_fraction':.1,'n_iter_no_change':10,'tol':1e-7,'verbose':0}:raise ValueError('fixed fit budget or library parameters differ')
 for array,key in [(a,'original_feature_sha256'),(b,'masked_feature_sha256')]:
  if hashlib.sha256(array.astype('<f8').tobytes()).hexdigest()!=trace[key]:raise ValueError('fit context differs')
 portable=predict(model,np.concatenate([a,b]),np.tile(anchor,2))
 with np.load(ROOT/'fit_export_check.npz',allow_pickle=False) as z:
  np.testing.assert_allclose(portable,z['portable'],rtol=0,atol=1e-9);np.testing.assert_allclose(portable,z['native'],rtol=0,atol=1e-8)
 result,arrays=compute();close(result,read(ROOT/'results.json'))
 with np.load(ROOT/'replay.npz',allow_pickle=False) as z:
  if set(z.files)!=set(arrays):raise ValueError('replay keys differ')
  for key,value in arrays.items():np.testing.assert_allclose(value,z[key],rtol=0,atol=1e-9,equal_nan=True)
 with np.load(MASK/'prepared.npz',allow_pickle=False) as z:x=z['X'].copy();median=z['median_feature'].copy()
 with np.load(NIGHT/'predictions.npz',allow_pickle=False) as z:anchor=z['anchor'].copy();peer=z['day_plus_night__masked_cnn_4ch'].copy();old=z['day_plus_night__causal_residual_tcn'].copy()
 with np.load(NIGHT/'decisions.npz',allow_pickle=False) as z:source=z['day_plus_night__quality'].copy()
 ids=np.unique(np.linspace(0,len(x)-1,257,dtype=int))
 for i in ids:
  row=independent_features(x[i],anchor[i],median[i]);np.testing.assert_allclose(row,arrays['context_features'][i],rtol=0,atol=1e-10);np.testing.assert_allclose(scalar_predict(model,row,anchor[i]),arrays['context_prediction_mps'][i],rtol=0,atol=1e-9)
 index=read(MASK/'frame_index.json');steps=0
 for name,p in [('frozen_night_tcn',old),('context_boosting',arrays['context_prediction_mps'])]:
  clock=scalar_clock(index,p,peer,source)
  if len(clock)!=37340:raise ValueError('request schedule length differs')
  for col,key in enumerate(['request_times_ns','latest_endpoint','age_ns','status','prediction_mps']):
   value=np.array([r[col] for r in clock]);expected=arrays[name+'__request__'+key]
   if key=='prediction_mps':np.testing.assert_allclose(value,expected,rtol=0,atol=1e-9,equal_nan=True)
   else:np.testing.assert_array_equal(value,expected)
  steps+=len(clock)
 if np.any(arrays['context_boosting__quality']&~source):raise ValueError('a rejected endpoint was revived')
 log=io.StringIO();tests=unittest.TextTestRunner(stream=log).run(unittest.defaultTestLoader.loadTestsFromModule(test_context))
 if not tests.wasSuccessful():raise ValueError(log.getvalue())
 print(f'CONTEXT BOOSTING AUDIT: PASS ({tests.testsRun} tests;200 trees;75 features;54356 native-export fit comparisons;{len(ids)} independent feature/tree probes;{steps} scalar request steps)')
 for name,stats in result['methods'].items():
  q=stats['requests']['overall'];print(f"CONTEXT {name}: {q['usable_fraction']*100:.3f}% requests usable; accepted P95 {q['accepted']['p95_mps']:.6f}m/s; recorded accepted errors>2m/s {q['accepted']['above_2_mps']}")
 print('CONTEXT DEVELOPMENT DECISION:',result['decision']);print('SAME INSPECTED DEVELOPMENT DATA. FINAL TEST CLOSED. WITNESS DISABLED. NO PROMOTION.')
if __name__=='__main__':audit()
