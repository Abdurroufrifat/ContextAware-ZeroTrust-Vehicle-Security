"""Replay the multi-recording development revision without PyTorch or network."""
import gzip, hashlib, io, json, math, unittest
import numpy as np
from study_io import ROOT, read, digest
import estimator_core as e
from make_dataset import build
from pool_development import compose
from score_estimator import score, validate_models
import test_estimator, test_preparation, test_pool, test_numerical

def require(condition,message):
 if not condition:raise ValueError(message)

def close(actual,expected,path='result'):
 if isinstance(expected,dict):
  require(set(actual)==set(expected),'keys differ: '+path)
  for key in expected:close(actual[key],expected[key],path+'/'+str(key))
 elif isinstance(expected,list):
  require(len(actual)==len(expected),'length differs: '+path)
  for i,(a,b) in enumerate(zip(actual,expected)):close(a,b,path+'/'+str(i))
 elif isinstance(expected,float):require(math.isclose(actual,expected,rel_tol=1e-7,abs_tol=1e-7),'number differs: '+path)
 else:require(actual==expected,'value differs: '+path)

def check_calibration_cohort(rebuilt,frozen,previous):
 """Keep stored cohorts exact while allowing machine roundoff during replay."""
 require(set(rebuilt)==set(frozen)==set(previous),'calibration array keys differ')
 for key in rebuilt:
  np.testing.assert_array_equal(frozen[key],previous[key])
  require(rebuilt[key].shape==frozen[key].shape,'calibration array shape differs')
  np.testing.assert_allclose(rebuilt[key],frozen[key],rtol=0,atol=1e-10)

def audit():
 for name,sha in read('sha256.json').items():
  path=(ROOT/name).resolve()
  require(path.is_relative_to(ROOT) and path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==sha,'study payload changed: '+name)
 validate_models();outputs,sources=compose();require(sources==read('source_lock.json'),'source lock changed')
 for name,sha in sources.items():
  path=(ROOT/name if name.startswith('../') else ROOT.parent/name).resolve()
  require(path.is_relative_to(ROOT.parent) and hashlib.sha256(path.read_bytes()).hexdigest()==sha,'source changed: '+name)
 project=ROOT.parent.parent
 for name,sha in read('predecessor_manifest.json').items():
  if name in ('FINAL_PROJECT_README.md','scripts/check_coact_project.py'):continue
  path=(project/name).resolve()
  require(path.is_relative_to(project) and path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==sha,'prior project evidence changed: '+name)
 previous=ROOT.parent/'vehicle_radar_estimator'
 for name in ('train_estimator.py','models_torch.py','estimator_core.py','make_dataset.py','score_estimator.py'):
  require((ROOT/name).read_bytes()==(previous/name).read_bytes(),'model/training/scoring implementation changed')
 for role,data in outputs.items():
  require(data['labels']==read('inputs/'+role+'/labels.json'),'source label reconstruction differs')
  require(data['windows']==read('inputs/'+role+'/windows.json'),'window reconstruction differs')
  require(data['packet_text']==gzip.decompress((ROOT/'inputs'/role/'packets.jsonl.gz').read_bytes()),'source radar packets differ')
  arrays,index=build(role,False);require(index==read('prepared/'+role+'/index.json'),'support index differs')
  with np.load(ROOT/'prepared'/role/'data.npz',allow_pickle=False) as saved:
   require(set(saved.files)==set(arrays),'prepared keys differ')
   for key in arrays:np.testing.assert_allclose(saved[key],arrays[key],rtol=0,atol=1e-10)
  if role=='calibration':
   with np.load(ROOT/'prepared/calibration/data.npz',allow_pickle=False) as frozen,np.load(previous/'prepared/calibration/data.npz',allow_pickle=False) as old:
    check_calibration_cohort(arrays,dict(frozen),dict(old))
  else:
   classical=read('classical_models.json')
   for method,key in [('reference_median_affine','median_feature'),('causal_savgol_affine','sg_feature')]:close(e.fit_affine(arrays[key],arrays['y'],'fit'),classical[method])
  result,predictions=score(role,False);close(result,read(role+'_results.json'))
  with np.load(ROOT/(role+'_predictions.npz'),allow_pickle=False) as saved:
   require(set(saved.files)==set(predictions),'prediction keys differ')
   for name,p in predictions.items():np.testing.assert_allclose(p,saved[name],rtol=1e-7,atol=1e-7)
  for kind in ('paper_inspired_cnn','causal_residual_tcn'):
   mean=np.mean([predictions[kind+'_seed_'+str(seed)] for seed in (141,142,143)],axis=0)
   np.testing.assert_allclose(predictions[kind],mean,rtol=0,atol=1e-12)
 result=read('calibration_results.json');passed=result['all_gates_pass'];decision=read('decision.json')
 old=json.loads((previous/'calibration_results.json').read_text(encoding='utf-8'))['metrics']['causal_residual_tcn'];primary=result['metrics']['causal_residual_tcn']
 close(read('revision_comparison.json'),{'scope':'Previously seen development calibration, not a fresh test','same_calibration_arrays':True,'old_primary_metrics':old,'new_primary_metrics':primary,'new_minus_old_rmse_mps':primary['rmse_mps']-old['rmse_mps'],'new_minus_old_p95_mps':primary['p95_disagreement_mps']-old['p95_disagreement_mps'],'all_gates_pass':passed,'historically_evaluated_urban03_and_urban04_reclassified_to_training':True,'fresh_evaluation_opened':False,'production_policy_changed':False,'witness_enabled':False})
 require(decision=={'development_gate_pass':passed,'decision':'READY_FOR_FROZEN_PROSPECTIVE_ACCURACY_TEST' if passed else 'STOP_AFTER_DEVELOPMENT_FAILURE','fresh_evaluation_opened':False,'production_policy_changed':False,'witness_enabled':False,'journal_readiness_established':False},'decision does not follow frozen gates')
 require(sorted(p.name for p in (ROOT/'inputs').iterdir() if p.is_dir())==['calibration','fit'],'unexpected measurement split opened')
 trace=read('training_trace.json');require(len(trace['runs'])==6,'missing training run')
 require({(r['kind'],r['seed']) for r in trace['runs']}=={(k,s) for k in ('paper_inspired_cnn','causal_residual_tcn') for s in (141,142,143)},'seed selection differs')
 require(all(r['epochs']==80 and len(r['fit_loss_by_epoch'])==80 and r['numpy_torch_max_abs_difference_mps']<.0005 for r in trace['runs']),'training/export check differs')
 suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromModule(m) for m in (test_estimator,test_preparation,test_pool,test_numerical));log=io.StringIO();tests=unittest.TextTestRunner(stream=log).run(suite)
 require(tests.wasSuccessful(),log.getvalue())
 print(f'MULTI-RECORDING ESTIMATOR SOURCE AND MODEL REPLAY: PASS ({tests.testsRun} portable tests; 9 training blocks; 6 fixed models)')
 print('MULTI-RECORDING DEVELOPMENT GATES:', 'PASS' if passed else 'NOT MET')
 print('FRESH FINAL TEST: UNOPENED. WITNESS ENABLED: NO. JOURNAL READINESS: NOT ESTABLISHED.')
if __name__=='__main__':audit()
