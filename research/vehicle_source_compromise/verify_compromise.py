"""Independent scalar attack, endpoint clock, admission, routing and metric check."""
import bisect,csv,io,math,sys,unittest
import numpy as np
from compromise_core import ROOT,RESEARCH,PROJECT,CASES,MODELS,METHODS,read,load,digest,freeze,permitted
sys.path.insert(0,str(RESEARCH/'vehicle_radar_fallback'));import verify_fallback as independent

def verify():
 freeze();protocol=read(ROOT/'protocol.json');results=read(ROOT/'results.json');prior_results=read(RESEARCH/'vehicle_matched_head_comparison/results.json')
 frames=read(RESEARCH/'vehicle_masked_history/frame_index.json');events=[int(v['time_ns']) for v in frames];ready=[bool(v['ready']) for v in frames];ids=[i for i,v in enumerate(ready) if v]
 refs=read(RESEARCH/'vehicle_availability_audit/request_reference.json');times=[int(v['time_ns']) for v in refs];truth=[math.nan if v['reference_mps'] is None else float(v['reference_mps']) for v in refs]
 if len(events)!=40690 or len(ids)!=40678 or len(times)!=37340 or any(b<=a for a,b in zip(events,events[1:])) or any(b<=a for a,b in zip(times,times[1:])):raise AssertionError('fixed schedule changed')
 source=load(RESEARCH/'vehicle_night_training/decisions.npz')['day_plus_night__quality'];peer=load(RESEARCH/'vehicle_night_training/predictions.npz')['day_plus_night__masked_cnn_4ch'];duration=events[-1]-events[0];start=events[0]+duration*2//5;end=events[0]+duration*3//5
 names=['all','before','during','after']+['segment_'+str(k) for k in sorted(set((t-times[0])//300_000_000_000 for t in times))]
 groups=[['all','before' if t<start else 'during' if t<end else 'after','segment_'+str((t-times[0])//300_000_000_000)] for t in times]
 checks=cells=control_cells=0
 if list(results['cases'])!=list(CASES) and set(results['cases'])!=set(CASES):raise AssertionError('case omitted')
 csv_rows=[]
 for case in CASES:
  basecase='dropout' if case.startswith('dropout') else 'nominal';base=load(RESEARCH/'vehicle_sensor_faults/replays'/(basecase+'.npz'));guard=load(RESEARCH/'vehicle_deadband_guard/replays'/(basecase+'__deadband.npz'));heads=load(RESEARCH/'vehicle_matched_head_comparison/replays'/(basecase+'.npz'));saved=independent.load(ROOT/'replays'/(case+'.npz'))
  preds={m:(base if m in MODELS[:2] else heads)[m+'__ready_prediction_mps'] for m in MODELS};radar=base['radar_only__ready_prediction_mps'];tag=case.removeprefix('dropout_')
  move_r=tag in ('radar_plus3','radar_peer_plus3','all_peer_plus3');move_p=tag in ('fusion_peer_plus3','all_peer_plus3');move_q=tag in ('peer_plus3','radar_peer_plus3','fusion_peer_plus3','all_peer_plus3')
  values={m:[math.nan]*len(events) for m in ('radar_only',*MODELS)};peers=[math.nan]*len(events);active=[]
  for i,t in enumerate(events):active.append(start<=t<end)
  np.testing.assert_array_equal(saved['attack_active_endpoint'],active)
  for j,i in enumerate(ids):
   a=active[i];peers[i]=float(peer[j])+(3. if a and move_q else 0.);values['radar_only'][i]=float(radar[j])+(3. if a and move_r else 0.)
   for m in MODELS:values[m][i]=float(preds[m][j])+(3. if a and move_p else 0.)
  for key,expected in [('radar_ready_prediction_mps',[values['radar_only'][i] for i in ids]),('peer_ready_prediction_mps',[peers[i] for i in ids]),*[(m+'__ready_prediction_mps',[values[m][i] for i in ids]) for m in MODELS]]:np.testing.assert_allclose(saved[key],expected,rtol=0,atol=0,equal_nan=True)
  branches={}
  for model in ('radar_only',*MODELS):
   branch={k:[] for k in ('status','prediction_mps','latest_endpoint','age_ns')}
   for j,t in enumerate(times):
    i=bisect.bisect_right(events,t)-1;age=t-events[i] if i>=0 else -1;status=1
    if i>=0:
     status=2 if age>100_000_000 else 3
     if age<=100_000_000 and ready[i]:
      status=4;p=values[model][i];q=peers[i]
      acceptable=bool(source[i]) and math.isfinite(p) and math.isfinite(q) and p>=0 and q>=0 and abs(p-q)<=1.
      if model!='radar_only':acceptable=acceptable and int(base['current__status'][i])==0 and bool(base['binding'][j]) and bool(guard['request_content_fresh'][j])
      if acceptable:status=0
    branch['status'].append(status);branch['prediction_mps'].append(values[model][i] if status==0 else math.nan);branch['latest_endpoint'].append(i);branch['age_ns'].append(age)
   branches[model]=branch
  if set(results['cases'][case])!=set(METHODS):raise AssertionError('policy omitted')
  expected_keys={'attack_active_endpoint','radar_ready_prediction_mps','peer_ready_prediction_mps',*[m+'__ready_prediction_mps' for m in MODELS]}
  for method in METHODS:
   model='radar_only' if method=='radar_only' else method.split('__')[0];enable=method.endswith('__fallback')
   out,summary=independent.scalar_comparison(branches[model],branches['radar_only'],enable,times,truth,groups,names)
   independent.close(summary,results['cases'][case][method]);cells+=len(summary);checks+=len(times)
   if case in ('nominal','dropout'):independent.close(summary,prior_results['cases'][case][method]);control_cells+=len(summary)
   for key,v in out.items():
    name=method+'__'+key;expected_keys.add(name);np.testing.assert_allclose(saved[name],v,rtol=0,atol=1e-12,equal_nan=True)
   for population,v in summary.items():
    e=v['accepted_errors'];csv_rows.append({'case':case,'method':method,'population':population,'requests':v['requests'],'accepted_requests':v['accepted_requests'],'availability':v['availability'],'p95_mps':e['p95_mps'],'above_2_mps':e['above_2_mps'],'radar_requests':v['route_counts']['2'],'joint_gate_pass':v['joint_gate_pass']})
  if set(saved)!=expected_keys:raise AssertionError('replay fields changed')
 if results['decision']!='SOURCE_COMPROMISE_BOUNDARY_COMPLETE_NO_PROMOTION':raise AssertionError('decision changed')
 for flag in ('models_refitted','labels_changed','thresholds_changed','production_policy_changed','reserved_final_opened','promotion_allowed','raw_sensor_attack_simulated','authenticated_provenance_established'):
  if results[flag] is not False:raise AssertionError('unsupported claim')
 csv_saved=list(csv.DictReader((ROOT/'comparison.csv').open(newline='',encoding='utf-8')))
 if len(csv_saved)!=len(csv_rows):raise AssertionError('CSV count')
 indexed={tuple(row[k] for k in ('case','method','population')):row for row in csv_saved}
 if len(indexed)!=len(csv_rows):raise AssertionError('duplicate CSV row')
 for row in csv_rows:
  got=indexed[tuple(row[k] for k in ('case','method','population'))]
  for k,v in row.items():
   if isinstance(v,float):
    if not math.isclose(float(got[k]),v,rel_tol=0,abs_tol=1e-9):raise AssertionError('CSV numeric value')
   elif got[k]!=('' if v is None else str(v)):raise AssertionError('CSV value')
 for name,h in read(ROOT/'sha256.json').items():
  if not permitted(name):raise AssertionError('protected artifact')
  if digest(ROOT/name)!=h:raise AssertionError('artifact changed '+name)
 old=read(ROOT/'prior_snapshot_manifest.json');manifest=read(PROJECT/'PROJECT_MANIFEST.json');verified=missing=protected=0
 for name,h in old.items():
  if name!='FINAL_PROJECT_README.md' and manifest.get(name)!=h:raise AssertionError('prior hash changed '+name)
 for name,h in manifest.items():
  if not permitted(name):protected+=1;continue
  p=PROJECT/name
  if not p.is_file():missing+=1;continue
  if digest(p)!=h:raise AssertionError('project hash '+name)
  verified+=1
 import test_compromise
 stream=io.StringIO();tests=unittest.TextTestRunner(stream=stream).run(unittest.defaultTestLoader.loadTestsFromModule(test_compromise))
 if not tests.wasSuccessful():raise AssertionError(stream.getvalue())
 print('SOURCE COMPROMISE VERIFICATION PASS:',checks,'request-policy checks;',cells,'metric cells;',control_cells,'historical control cells;',len(csv_rows),'CSV rows;',tests.testsRun,'tests')
 print('PROJECT:',verified,'verified;',missing,'missing;',protected,'protected paths not opened')
 print('SOURCE_COMPROMISE_BOUNDARY_COMPLETE_NO_PROMOTION')
if __name__=='__main__':verify()
