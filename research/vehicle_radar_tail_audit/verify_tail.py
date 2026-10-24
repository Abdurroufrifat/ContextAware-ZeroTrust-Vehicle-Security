"""Portable source, join and numerical replay of the post-hoc diagnostic."""
import hashlib,io,json,math,unittest
from tail_analysis import ROOT,read,compute
import test_tail
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
  p=(ROOT/name).resolve()
  if not p.is_relative_to(ROOT) or not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=sha:raise ValueError('tail audit payload changed: '+name)
 result,context=compute();close(result,read(ROOT/'results.json'));close(context,read(ROOT/'worst_event_context.json'))
 project=ROOT.parent.parent
 for name,sha in read(ROOT/'prior_snapshot_manifest.json').items():
  if name in ('FINAL_PROJECT_README.md','scripts/check_coact_project.py'):continue
  p=(project/name).resolve()
  if not p.is_relative_to(project) or hashlib.sha256(p.read_bytes()).hexdigest()!=sha:raise ValueError('preceding research evidence changed: '+name)
 for name in ('core_io.py','reference_core.py'):
  if (ROOT/name).read_bytes()!=(ROOT.parent/'vehicle_reference_check'/name).read_bytes():raise ValueError('reference decoder copy changed')
 log=io.StringIO();tests=unittest.TextTestRunner(stream=log).run(unittest.defaultTestLoader.loadTestsFromModule(test_tail))
 if not tests.wasSuccessful():raise ValueError(log.getvalue())
 print(f'RADAR TAIL SOURCE AND ERROR ATTRIBUTION AUDIT: PASS ({tests.testsRun} tests; no observations excluded)')
 tail=result['roles']['calibration']['methods']['causal_residual_tcn']['tails']['above_2_mps']
 print(f"CALIBRATION TAIL: {tail['count']} endpoints account for {100*tail['squared_error_share']:.2f}% of squared error. This is a post-hoc diagnostic.")
 print('MODEL AND RESEARCH GATES: UNCHANGED. FINAL TEST: UNOPENED. WITNESS: DISABLED.')
if __name__=='__main__':audit()
