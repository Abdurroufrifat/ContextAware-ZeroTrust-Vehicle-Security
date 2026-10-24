"""Replay quality decisions, masks, all-denominator accounting and limits."""
import hashlib,io,math,unittest
import numpy as np
from run_quality import ROOT,read,compute,freeze
import test_quality
def close(a,b,path='result'):
 if isinstance(b,dict):
  if set(a)!=set(b):raise ValueError('keys differ: '+path)
  for key in b:close(a[key],b[key],path+'/'+str(key))
 elif isinstance(b,list):
  if len(a)!=len(b):raise ValueError('length differs: '+path)
  for i,(x,y) in enumerate(zip(a,b)):close(x,y,path+'/'+str(i))
 elif type(b) is float:
  if not math.isclose(a,b,rel_tol=1e-7,abs_tol=1e-7):raise ValueError('numerical replay differs: '+path)
 elif a!=b:raise ValueError('value differs: '+path)
def audit():
 for name,sha in read(ROOT/'sha256.json').items():
  path=(ROOT/name).resolve()
  if not path.is_relative_to(ROOT) or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=sha:raise ValueError('quality payload changed: '+name)
 project=ROOT.parent.parent
 for name,sha in read(ROOT/'prior_snapshot_manifest.json').items():
  if name in ('FINAL_PROJECT_README.md','scripts/check_coact_project.py'):continue
  path=(project/name).resolve()
  if not path.is_relative_to(project) or hashlib.sha256(path.read_bytes()).hexdigest()!=sha:raise ValueError('previous research evidence changed: '+name)
 freeze();result,masks=compute();close(result,read(ROOT/'results.json'))
 with np.load(ROOT/'quality_masks.npz',allow_pickle=False) as saved:
  if set(saved.files)!=set(masks):raise ValueError('mask keys differ')
  for name,mask in masks.items():np.testing.assert_array_equal(saved[name],mask)
 for role,data in result['roles'].items():
  for method,m in data['methods'].items():
   if m['accepted']['count']+m['withheld']['count']!=m['all_supported']['count']:raise ValueError('errors were dropped')
   if m['history_or_missing_withheld']+m['quality_withheld']+m['accepted']['count']!=m['raw_endpoints']:raise ValueError('availability costs were dropped')
  a=data['methods']['disagreement_guard'];b=data['methods']['matched_random']
  if a['quality_withheld']!=b['quality_withheld']:raise ValueError('random control has a different rejection budget')
 if result['promotion_allowed'] or result['fresh_test_opened'] or result['witness_enabled']:raise ValueError('unsupported promotion')
 counter=result['synthetic_shared_error_counterexample']
 if not counter['quality_flag'] or counter['measured_dataset_result'] or counter['production_permission_issued']:raise ValueError('shared-error limitation was hidden')
 stream=io.StringIO();tests=unittest.TextTestRunner(stream=stream).run(unittest.defaultTestLoader.loadTestsFromModule(test_quality))
 if not tests.wasSuccessful():raise ValueError(stream.getvalue())
 data=result['roles']['calibration']['methods']['disagreement_guard']
 print(f'RADAR DISAGREEMENT QUALITY AUDIT: PASS ({tests.testsRun} tests; all errors and availability costs retained)')
 print(f"CALIBRATION: {data['quality_withheld']} quality-withheld; {data['accepted']['above_2_mps']} accepted recorded errors above 2 m/s; {100*data['whole_block_acceptance_fraction']:.2f}% whole-block acceptance.")
 print('QUALITY DEVELOPMENT DECISION:',result['decision'])
 print('SHARED-RADAR MODEL AGREEMENT IS NOT AN INDEPENDENT WITNESS. FINAL TEST UNOPENED. NO PRODUCTION PERMISSION ISSUED.')
if __name__=='__main__':audit()
