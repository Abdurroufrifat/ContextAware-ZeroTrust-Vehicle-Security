"""Verify new features and all12 models while preserving the preceding project."""
import io,unittest
import numpy as np
from masked_core import ROOT,PREVIOUS,read,digest,freeze
from score_masked import prepare,predict,summarize,check_models
from verify_continuous import close
import test_masked
def audit():
 for name,h in read(ROOT/'sha256.json').items():
  p=(ROOT/name).resolve()
  if not p.is_relative_to(ROOT) or not p.is_file() or digest(p)!=h:raise ValueError('masked payload changed: '+name)
 project=ROOT.parent.parent
 for name,h in read(ROOT/'prior_snapshot_manifest.json').items():
  if name in ('FINAL_PROJECT_README.md','scripts/check_coact_project.py'):continue
  p=(project/name).resolve()
  if not p.is_relative_to(project) or digest(p)!=h:raise ValueError('preceding research changed: '+name)
 protocol=freeze();lock=check_models();trace=read(ROOT/'training_trace.json');expected={(condition,kind,seed) for condition in ['clean_control','masked_training'] for kind in ['causal_residual_tcn','masked_cnn_4ch'] for seed in [141,142,143]}
 if {(r['condition'],r['kind'],r['seed']) for r in trace['runs']}!=expected or len(trace['runs'])!=12 or trace['fit_samples']!=16169:raise ValueError('matched training cohort or model selection differs')
 if any(r['epochs']!=protocol['training']['epochs'] or len(r['fit_loss_by_epoch'])!=r['epochs'] or r['masked_numpy_torch_max_difference_mps']>.0005 or r['warm_start_dense_max_difference_mps']>.0005 for r in trace['runs']):raise ValueError('training budget or export validation differs')
 data,index=prepare();close(index,read(ROOT/'frame_index.json'))
 with np.load(ROOT/'prepared.npz',allow_pickle=False) as stored:
  if set(stored.files)!=set(data):raise ValueError('masked prepared arrays differ')
  for name,value in data.items():np.testing.assert_allclose(value,stored[name],rtol=0,atol=1e-10)
 old_index=read(PREVIOUS/'frame_index.json')
 with np.load(PREVIOUS/'prepared.npz',allow_pickle=False) as stored:
  pairs=[(a['array_index'],b['array_index']) for a,b in zip(old_index,index) if a['ready']]
  if any(b is None for a,b in pairs):raise ValueError('originally supported endpoint lost')
  a=np.array([a for a,b in pairs]);b=np.array([b for a,b in pairs]);np.testing.assert_array_equal(data['X'][b],stored['X'][a]);np.testing.assert_array_equal(data['median_feature'][b],stored['median_feature'][a])
 rebuilt=predict(data)
 with np.load(ROOT/'predictions.npz',allow_pickle=False) as stored:
  if set(stored.files)!=set(rebuilt):raise ValueError('masked prediction arrays differ')
  pred={name:stored[name].copy() for name in stored.files}
  for name,value in rebuilt.items():np.testing.assert_allclose(value,pred[name],rtol=1e-7,atol=1e-7)
 result,arrays=summarize(index,pred);close(result,read(ROOT/'results.json'))
 with np.load(ROOT/'decisions.npz',allow_pickle=False) as stored:
  if set(stored.files)!=set(arrays):raise ValueError('masked decision arrays differ')
  for name,value in arrays.items():np.testing.assert_array_equal(value,stored[name])
 for condition,stats in result['conditions'].items():
  m=stats['endpoints'];q=stats['requests'];overall=q['overall']
  if m['usable_total']+m['quality_withheld']+m['unready_raw']!=result['raw_endpoints']:raise ValueError('endpoint denominator lost costs')
  if sum(overall['status_counts'].values())!=result['fixed_requests'] or sum(s['requests'] for s in q['segments_300s'].values())!=result['fixed_requests']:raise ValueError('scheduled request denominator lost costs')
 log=io.StringIO();tests=unittest.TextTestRunner(stream=log).run(unittest.defaultTestLoader.loadTestsFromModule(test_masked))
 if not tests.wasSuccessful():raise ValueError(log.getvalue())
 print(f'MASKED HISTORY SOURCE AND MODEL AUDIT: PASS ({tests.testsRun} tests;12 fixed final-epoch models;{len(pairs)} original feature histories identical)')
 for condition,stats in result['conditions'].items():
  m=stats['requests']['overall'];print(f"MASK COMPARISON {condition}: {100*m['usable_fraction']:.2f}% requests usable; accepted P95 {m['accepted']['p95_mps']:.4f} m/s; {m['accepted']['above_2_mps']} accepted recorded errors above2m/s.")
 print('MASKED DEVELOPMENT DECISION:',result['decision']);print('SAME DEVELOPMENT DATA. FINAL TEST UNOPENED. WITNESS DISABLED.')
if __name__=='__main__':audit()
