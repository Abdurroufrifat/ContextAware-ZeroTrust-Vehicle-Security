"""Full source, feature, model and accounting replay without training or network."""
import hashlib,io,math,unittest
import numpy as np
from continuous_core import ROOT,read,digest,freeze,decode_radar,prepare,reference_targets,ranges
from run_continuous import predict,summarize
import test_continuous
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
  if not p.is_relative_to(ROOT) or not p.is_file() or digest(p)!=sha:raise ValueError('continuous payload changed: '+name)
 project=ROOT.parent.parent
 for name,sha in read(ROOT/'prior_snapshot_manifest.json').items():
  if name in ('FINAL_PROJECT_README.md','scripts/check_coact_project.py'):continue
  p=(project/name).resolve()
  if not p.is_relative_to(project) or digest(p)!=sha:raise ValueError('prior research changed: '+name)
 for name,folder in [('core_io.py','vehicle_reference_check'),('reference_core.py','vehicle_reference_check'),('estimator_core.py','vehicle_radar_multidrive'),('quality_core.py','vehicle_radar_quality_guard')]:
  if (ROOT/name).read_bytes()!=(ROOT.parent/folder/name).read_bytes():raise ValueError('frozen decoder/model/quality helper changed')
 freeze();protocol=read(ROOT/'protocol.json');download=read(ROOT/'download_manifest.json');source=ROOT/'source/radars.bag'
 if not download['whole_file_verified'] or download['publisher_sha256']!=protocol['radar_publisher_sha256']:raise ValueError('publisher verification record differs')
 with source.open('rb') as file:
  plan=ranges(protocol['radar_bytes'])
  if len(plan)!=len(download['ranges']):raise ValueError('download range record missing')
  for i,(start,end) in enumerate(plan):
   record=download['ranges'][str(i)];data=file.read(end-start+1)
   if (record['start'],record['end'],record['bytes'],record['sha256'])!=(start,end,end-start+1,hashlib.sha256(data).hexdigest()):raise ValueError('source range replay differs')
 raw,profile=decode_radar();close(profile,read(ROOT/'source_profile.json'));data,index=prepare(raw);close(index,read(ROOT/'frame_index.json'))
 with np.load(ROOT/'prepared.npz',allow_pickle=False) as stored:
  if set(data)!=set(stored.files):raise ValueError('prepared keys differ')
  for name,value in data.items():np.testing.assert_allclose(value,stored[name],rtol=0,atol=1e-10)
 targets=reference_targets(index);close(targets,read(ROOT/'reference_targets.json'));replayed=predict(data)
 with np.load(ROOT/'predictions.npz',allow_pickle=False) as stored:
  if set(stored.files)!=set(replayed):raise ValueError('prediction keys differ')
  pred={name:stored[name].copy() for name in stored.files}
  for name,value in replayed.items():np.testing.assert_allclose(value,pred[name],rtol=1e-7,atol=1e-7)
 result,masks=summarize(index,targets,pred);close(result,read(ROOT/'results.json'))
 with np.load(ROOT/'quality_masks.npz',allow_pickle=False) as stored:
  if set(stored.files)!=set(masks):raise ValueError('quality mask keys differ')
  for name,value in masks.items():np.testing.assert_array_equal(value,stored[name])
 for method,m in result['methods'].items():
  if m['usable_total']+m['quality_withheld']+m['unready_raw']!=m['raw_endpoints']:raise ValueError('full-drive availability denominator lost endpoints')
 if sum(s['raw_endpoints'] for s in result['segments_300s_no_history_reset'].values())!=len(index):raise ValueError('reporting segments lost source endpoints')
 log=io.StringIO();tests=unittest.TextTestRunner(stream=log).run(unittest.defaultTestLoader.loadTestsFromModule(test_continuous))
 if not tests.wasSuccessful():raise ValueError(log.getvalue())
 m=result['methods']['disagreement_guard'];print(f'FULL CONTINUOUS RADAR AUDIT: PASS ({tests.testsRun} tests; publisher-verified complete bag; {len(index)} raw endpoints)')
 print(f"CONTINUOUS DEVELOPMENT: {100*m['whole_acceptance_fraction']:.2f}% whole-drive usable; accepted P95 {m['accepted']['p95_mps']:.4f} m/s; {m['accepted']['above_2_mps']} accepted recorded errors above 2 m/s.")
 print('CONTINUOUS DECISION:',result['decision'])
 print('PREVIOUS STUDIES UNCHANGED. RESERVED FINAL TEST UNOPENED. WITNESS DISABLED.')
if __name__=='__main__':audit()
