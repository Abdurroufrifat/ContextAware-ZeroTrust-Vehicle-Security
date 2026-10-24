"""Publisher-source checks, full portable inference and independent request clock."""
import io,unittest
from night_core import *
from prepare_night import build
from score_night import check_models,predict,summarize
from verify_continuous import close
import test_night

def scalar_requests(index,point_predictions,point_peer):
 events=[q['time_ns'] for q in index];t=events[0];end=events[-1];i=-1;out=[]
 while t<=end:
  while i+1<len(events) and events[i+1]<=t:i+=1
  p=float('nan');status=0
  if i<0:status=1
  elif t-events[i]>100000000:status=2
  elif not index[i]['ready']:status=3
  else:
   a=index[i]['array_index'];value=point_predictions[a];peer=point_peer[a]
   if not np.isfinite(value) or value<0 or not np.isfinite(peer) or peer<0 or abs(value-peer)>1.:status=4
   else:p=float(value)
  out.append((t,i,t-events[i] if i>=0 else -1,status,p));t+=50000000
 return out

def audit():
 freeze();project=ROOT.parent.parent
 for name,h in read(ROOT/'prior_snapshot_manifest.json').items():
  if name in ['FINAL_PROJECT_README.md','scripts/check_coact_project.py']:continue
  p=(project/name).resolve()
  if not p.is_relative_to(project) or digest(p)!=h:raise ValueError('prior evidence changed: '+name)
 for name,h in read(ROOT/'sha256.json').items():
  p=(ROOT/name).resolve()
  if not p.is_relative_to(ROOT) or digest(p)!=h:raise ValueError('night-study payload changed: '+name)
 lock=read(ROOT/'data_lock.json')
 if lock['analysis_lock_sha256']!=digest(ROOT/'analysis_lock.json') or lock['source_lock_sha256']!=digest(ROOT/'source_lock.json'):raise ValueError('source/data ancestry differs')
 for name,h in read(ROOT/'source_lock.json').items():
  if digest(ROOT/name)!=h:raise ValueError('raw source changed')
 for name,h in lock['files_sha256'].items():
  if digest(ROOT/name)!=h:raise ValueError('prepared fit data changed')
 rebuilt,index,report=build(False);close(index,read(ROOT/'night_index.json'));close(json.loads(json.dumps(report)),read(ROOT/'data_report.json'))
 with np.load(ROOT/'night_fit.npz',allow_pickle=False) as stored:
  if set(stored.files)!=set(rebuilt):raise ValueError('fit keys differ')
  for key,value in rebuilt.items():np.testing.assert_array_equal(value,stored[key])
 check_models();trace=read(ROOT/'training_trace.json');expected={(a,b,s) for a in CONDITIONS for b in KINDS for s in SEEDS}
 if len(trace['runs'])!=12 or {(q['condition'],q['kind'],q['seed']) for q in trace['runs']}!=expected:raise ValueError('matched training incomplete')
 if trace['day_samples']!=16169 or trace['night_samples']!=len(rebuilt['y']) or trace['expanded_samples']!=16169+len(rebuilt['y']):raise ValueError('training counts differ')
 steps=(trace['expanded_samples']+127)//128
 for q in trace['runs']:
  if q['epochs']!=5 or q['steps_per_epoch']!=steps or q['total_optimizer_updates']!=5*steps or q['batch_size']!=128 or len(q['fit_loss_by_epoch'])!=5 or max(q['numpy_torch_max_difference_mps'],q['warm_start_max_difference_mps'])>.0005:raise ValueError('update budget or export differs')
 pred=predict()
 with np.load(ROOT/'predictions.npz',allow_pickle=False) as saved:
  if set(saved.files)!=set(pred):raise ValueError('prediction keys differ')
  for key,value in pred.items():np.testing.assert_allclose(value,saved[key],rtol=1e-7,atol=1e-7)
 result,arrays=summarize(pred);close(result,read(ROOT/'results.json'))
 with np.load(ROOT/'decisions.npz',allow_pickle=False) as saved:
  if set(saved.files)!=set(arrays):raise ValueError('decision keys differ')
  for key,value in arrays.items():np.testing.assert_array_equal(value,saved[key])
 index=read(MASK/'frame_index.json');clock_count=0
 for condition in CONDITIONS:
  scalar=scalar_requests(index,pred[condition+'__'+KINDS[0]],pred[condition+'__'+KINDS[1]]);prefix=condition+'__request__'
  for col,key in enumerate(['request_times_ns','latest_endpoint','age_ns','status','prediction_mps']):
   values=np.array([q[col] for q in scalar]);expected=arrays[prefix+key]
   if key=='prediction_mps':np.testing.assert_allclose(values,expected,rtol=0,atol=1e-12,equal_nan=True)
   else:np.testing.assert_array_equal(values,expected)
  if len(scalar)!=37340:raise ValueError('scheduled denominator differs')
  clock_count+=len(scalar);stats=result['conditions'][condition]['requests']['overall']
  if sum(stats['status_counts'].values())!=37340:raise ValueError('request cost omitted')
 tests=unittest.TextTestRunner(stream=io.StringIO()).run(unittest.defaultTestLoader.loadTestsFromModule(test_night))
 if not tests.wasSuccessful():raise ValueError('night regression tests failed')
 print(f'NIGHT TRAINING AUDIT: PASS ({tests.testsRun} tests;4 publisher-verified bags;12 fixed models;{clock_count} independent request steps)')
 print('NIGHT FIT SAMPLES:',trace['night_samples'],'from2 recordings and6 fixed windows')
 for condition in CONDITIONS:
  stats=result['conditions'][condition]['requests']['overall'];print(f"NIGHT {condition}: {100*stats['usable_fraction']:.3f}% usable; accepted P95 {stats['accepted']['p95_mps']:.6f}m/s; accepted recorded errors>2m/s {stats['accepted']['above_2_mps']}")
 print('NIGHT DEVELOPMENT DECISION:',result['decision']);print('SAME DEVELOPMENT DATA. FINAL TEST UNOPENED. WITNESS DISABLED. NO PROMOTION.')
if __name__=='__main__':audit()
