"""Offline source, rejection-cause and independent fixed-rate service audit."""
import io,unittest
import numpy as np
from availability_core import ROOT,PREVIOUS,STATUS,read,digest,freeze
from run_availability import rebuild,inputs
from verify_continuous import close
import test_availability
def audit():
 for name,h in read(ROOT/'sha256.json').items():
  p=(ROOT/name).resolve()
  if not p.is_relative_to(ROOT) or not p.is_file() or digest(p)!=h:raise ValueError('availability payload changed: '+name)
 project=ROOT.parent.parent
 for name,h in read(ROOT/'prior_snapshot_manifest.json').items():
  if name in ('FINAL_PROJECT_README.md','scripts/check_coact_project.py'):continue
  p=(project/name).resolve()
  if not p.is_relative_to(project) or digest(p)!=h:raise ValueError('previous research changed: '+name)
 protocol=freeze();replay,targets,reasons,result=rebuild()
 close(result,read(ROOT/'results.json'));close(targets,read(ROOT/'request_reference.json'));close(reasons,read(ROOT/'endpoint_reasons.json'))
 with np.load(ROOT/'request_replay.npz',allow_pickle=False) as stored:
  if set(stored.files)!=set(replay):raise ValueError('request arrays differ')
  for name,value in replay.items():np.testing.assert_array_equal(value,stored[name])
 # Separate scalar advancing-event implementation checks every scheduled request.
 index,events,ready,pred,quality=inputs();cursor=-1;counts={name:0 for name in STATUS.values()}
 for i,t in enumerate(replay['request_times_ns']):
  while cursor+1<len(events) and events[cursor+1]<=t:cursor+=1
  status=1 if cursor<0 else 2 if int(t-events[cursor])>protocol['estimate_ttl_ns'] else 3 if not ready[cursor] else 4 if not quality[cursor] else 0
  if int(replay['latest_endpoint'][i])!=cursor or int(replay['status'][i])!=status:raise ValueError('independent request replay differs')
  expected_age=int(t-events[cursor]) if cursor>=0 else -1
  if int(replay['age_ns'][i])!=expected_age:raise ValueError('request age differs')
  if status==0 and replay['prediction_mps'][i]!=pred[cursor]:raise ValueError('accepted request did not use latest endpoint')
  counts[STATUS[status]]+=1
 if counts!=result['overall']['status_counts'] or sum(counts.values())!=len(replay['status']):raise ValueError('request denominator lost scheduled requests')
 if sum(s['requests'] for s in result['segments_300s'].values())!=len(replay['status']):raise ValueError('reporting segments lost requests')
 if sum(s['requests'] for s in result['unavailable_episodes'])!=len(replay['status'])-counts['usable']:raise ValueError('outage episodes lost requests')
 old=read(PREVIOUS/'results.json');rc=result['original_endpoint_reasons']
 if rc.get('ready',0)!=old['methods']['timing_only']['features_ready'] or sum(v for k,v in rc.items() if k!='ready')!=old['methods']['timing_only']['unready_raw']:raise ValueError('root-cause counts do not reproduce earlier evidence')
 log=io.StringIO();tests=unittest.TextTestRunner(stream=log).run(unittest.defaultTestLoader.loadTestsFromModule(test_availability))
 if not tests.wasSuccessful():raise ValueError(log.getvalue())
 m=result['overall'];print(f'FIXED-RATE AVAILABILITY AUDIT: PASS ({tests.testsRun} tests; {len(replay["status"])} scheduled requests; full source and scalar replay)')
 print(f"REQUEST DEVELOPMENT: {100*m['usable_fraction']:.2f}% usable; accepted P95 {m['accepted']['p95_mps']:.4f} m/s; first segment {100*result['segments_300s']['0']['usable_fraction']:.2f}% usable.")
 print('REQUEST DECISION:',result['decision']);print('NO MODEL CHANGE. SAME DEVELOPMENT RECORDING. FINAL UNOPENED. WITNESS DISABLED.')
if __name__=='__main__':audit()
