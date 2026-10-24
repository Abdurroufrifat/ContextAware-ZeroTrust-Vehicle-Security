"""Rebuild fixed radar features and independently replay exported NumPy models."""
import contextlib, hashlib, io, json, math, unittest
import numpy as np
from study_io import ROOT, read, digest
import estimator_core as e
from make_dataset import build
from score_estimator import score, validate_models
import test_estimator, test_preparation

def close(actual, expected, location='result'):
 if isinstance(expected, dict):
  if set(actual)!=set(expected):raise ValueError('keys changed: '+location)
  for key in expected:close(actual[key],expected[key],location+'/'+str(key))
 elif isinstance(expected,list):
  if len(actual)!=len(expected):raise ValueError('length changed: '+location)
  for i,(a,b) in enumerate(zip(actual,expected)):close(a,b,location+'/'+str(i))
 elif isinstance(expected,float):
  if not math.isclose(actual,expected,rel_tol=1e-7,abs_tol=1e-7):raise ValueError('number changed: '+location)
 elif actual!=expected:raise ValueError('value changed: '+location)

def audit():
 for name,sha in read('sha256.json').items():
  path=(ROOT/name).resolve()
  if not path.is_relative_to(ROOT) or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=sha:raise ValueError('estimator payload changed: '+name)
 validate_models()
 am=read('implementation_history/preprocessing_amendment.json');initial=read('implementation_history/initial_model_lock.json');final=read('model_lock.json')
 for key in initial:
  if key!='design_lock_sha256' and initial[key]!=final[key]:raise ValueError('amendment changed training artifacts')
 old_design=read('implementation_history/initial_design_lock.json');new_design=read('design_lock.json')
 assert initial['design_lock_sha256']==digest('implementation_history/initial_design_lock.json')==am['initial_design_sha256']
 assert final['design_lock_sha256']==am['final_design_sha256']==digest('design_lock.json')
 assert am['initial_model_lock_sha256']==digest('implementation_history/initial_model_lock.json')
 assert am['final_model_lock_sha256']==digest('model_lock.json')
 assert old_design['protocol_sha256']==new_design['protocol_sha256']
 for name,sha in old_design['code_sha256'].items():
  assert sha==(digest('implementation_history/initial_make_dataset.py') if name=='make_dataset.py' else new_design['code_sha256'][name])
 assert am['before_calibration_scoring'] and am['models_and_training_unchanged']
 source=ROOT.parent/'vehicle_reference_check'/'matched_endpoints.json'
 reference=json.loads(source.read_text(encoding='utf-8')) if source.exists() else None
 for role in ('fit','calibration'):
  proof=am['proof'][role]
  assert proof['all_arrays_bit_identical'] and proof['support_index_identical']
  assert digest('prepared/'+role+'/data.npz')==proof['prepared_file_sha256']
  if reference is not None:assert read('inputs/'+role+'/labels.json')==[r for r in reference if r['role']==role]
  prior=ROOT.parent/'vehicle_temporal_policy'/'roles'/role
  if prior.is_dir():
   for file in ('packets.jsonl.gz','windows.json'):
    assert hashlib.sha256((prior/file).read_bytes()).hexdigest()==digest('inputs/'+role+'/'+file)
  arrays,index=build(role,False);assert index==read('prepared/'+role+'/index.json')
  with np.load(ROOT/'prepared'/role/'data.npz',allow_pickle=False) as saved:
   assert set(saved.files)==set(arrays)
   for name in arrays:np.testing.assert_allclose(arrays[name],saved[name],rtol=0,atol=1e-10)
  if role=='fit':
   classical=read('classical_models.json')
   for method,feature in [('reference_median_affine','median_feature'),('causal_savgol_affine','sg_feature')]:close(e.fit_affine(arrays[feature],arrays['y'],'fit'),classical[method])
  result,predictions=score(role,False);close(result,read(role+'_results.json'))
  with np.load(ROOT/(role+'_predictions.npz'),allow_pickle=False) as saved:
   assert set(saved.files)==set(predictions)
   for method,p in predictions.items():np.testing.assert_allclose(p,saved[method],rtol=1e-7,atol=1e-7)
  for kind in ('paper_inspired_cnn','causal_residual_tcn'):
   np.testing.assert_allclose(predictions[kind],np.mean([predictions[kind+'_seed_'+str(s)] for s in (141,142,143)],axis=0),atol=1e-12)
 decision=read('decision.json');assert not read('calibration_results.json')['all_gates_pass']
 assert decision=={'development_gate_pass':False,'decision':'STOP_AFTER_DEVELOPMENT_FAILURE','fresh_evaluation_opened':False,'production_policy_changed':False,'witness_enabled':False,'journal_readiness_established':False}
 assert sorted(p.name for p in (ROOT/'inputs').iterdir() if p.is_dir())==['calibration','fit']
 trace=read('training_trace.json');assert len(trace['runs'])==6
 assert {(r['kind'],r['seed']) for r in trace['runs']}=={(k,s) for k in ('paper_inspired_cnn','causal_residual_tcn') for s in (141,142,143)}
 assert all(r['epochs']==80 and len(r['fit_loss_by_epoch'])==80 and r['numpy_torch_max_abs_difference_mps']<.0005 for r in trace['runs'])
 suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromModule(m) for m in (test_estimator,test_preparation));log=io.StringIO()
 tests=unittest.TextTestRunner(stream=log).run(suite)
 if not tests.wasSuccessful():raise ValueError(log.getvalue())
 print(f'RADAR ESTIMATOR SOURCE, FEATURE AND MODEL REPLAY: PASS ({tests.testsRun} portable tests; 6 frozen final-epoch models)')
 print('ESTIMATOR DEVELOPMENT GATES: NOT MET. FRESH EVALUATION REMAINS UNOPENED.')
 print('WITNESS ENABLED: NO. JOURNAL READINESS: NOT ESTABLISHED.')
if __name__=='__main__':audit()
