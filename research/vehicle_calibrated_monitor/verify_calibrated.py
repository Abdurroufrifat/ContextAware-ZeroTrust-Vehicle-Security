"""Independent pair enumeration, source state machine, routing and scalar metrics."""
import bisect,csv,io,math,unittest,zipfile
import numpy as np
from calibrated_core import ROOT,RESEARCH,PROJECT,CASES,MODELS,SOURCES,METHODS,VARIANTS,FIELDS,read,load,digest,blocked,freeze
from scalar_metrics import close,scalar_summary

def independently_calibrate():
 members=read(RESEARCH/'vehicle_route_validation/membership.json');parent=load(RESEARCH/'vehicle_multisource_imu/oof_predictions.npz');heads=load(RESEARCH/'vehicle_matched_head_comparison/held_predictions.npz');signed=load(RESEARCH/'vehicle_signed_axis_correction/oof_predictions.npz');nonlinear=load(RESEARCH/'vehicle_nonlinear_correction/oof_predictions.npz')
 if len(members)!=27178 or [v['id'] for v in members]!=list(range(len(members))):raise AssertionError('fitting identity')
 for other in (heads,signed,nonlinear):
  np.testing.assert_array_equal(parent['target_mps'],other['target_mps']);np.testing.assert_array_equal(parent['fold_id'],other['fold_id'])
 values={'radar_only':parent['original_context__prediction_mps'],'radar_imu_parent':parent['radar_imu__prediction_mps'],**{s:heads[s+'__prediction_mps'] for s in MODELS[1:]}}
 expected_aliases=[(values['radar_only'],signed['original_context__prediction_mps']),(values['radar_only'],nonlinear['original_context__prediction_mps']),(values['radar_imu_parent'],heads['radar_imu_parent__prediction_mps']),(values['radar_imu_parent'],nonlinear['parent_radar_imu__prediction_mps']),*[(values[s],nonlinear[s+'__prediction_mps']) for s in ('linear7','nonlinear7')]]
 for a,b in expected_aliases:np.testing.assert_array_equal(a,b)
 grouped={}
 for i,row in enumerate(members):
  if type(row['time_ns']) is not int:raise AssertionError('integer fitting times')
  grouped.setdefault(row['route'],[]).append(i)
 if len(grouped)!=5 or {v['group'] for v in members}!={'Urban01','Urban03','Urban04'}:raise AssertionError('fitting recordings')
 pairs=[]
 for route,ids in sorted(grouped.items()):
  ids=sorted(ids,key=lambda i:members[i]['time_ns']);times=[members[i]['time_ns'] for i in ids]
  if any(b<=a for a,b in zip(times,times[1:])) or len(set(int(parent['fold_id'][i]) for i in ids))!=1:raise AssertionError('clock/fold identity')
  for j,left in enumerate(ids):
   end=bisect.bisect_right(times,times[j]+100_000_000)
   pairs.extend((left,ids[k]) for k in range(j+1,end))
 saved=load(ROOT/'calibration_pairs.npz');np.testing.assert_array_equal(saved['left_id'],[i for i,j in pairs]);np.testing.assert_array_equal(saved['right_id'],[j for i,j in pairs]);p=read(ROOT/'parameters.json');thresholds={};checks=0;trace={}
 close(p['fitting_rows'],27178);close(p['pairs'],len(pairs));close(p['maximum_pair_lag_ns'],100_000_000);close(p['labels_used_for_limits'],False)
 for source,pred in values.items():
  if len(pred)!=len(members) or any(not math.isfinite(float(v)) or v<0 for v in pred):raise AssertionError('fitting predictions')
  delta=[abs(float(pred[j])-float(pred[i])) for i,j in pairs];thresholds[source]=max(delta);records={}
  for route in sorted(grouped):
   ids=[k for k,(i,j) in enumerate(pairs) if members[i]['route']==route];best=max(ids,key=lambda k:delta[k]);i,j=pairs[best];records[route]={'pairs':len(ids),'maximum_mps':delta[best],'left_id':i,'right_id':j}
  cross={}
  for g in ('Urban01','Urban03','Urban04'):
   held=[delta[k] for k,(i,j) in enumerate(pairs) if members[i]['group']==g];training=[delta[k] for k,(i,j) in enumerate(pairs) if members[i]['group']!=g];maximum=max(training);cross[g]={'calibration_pairs':len(training),'held_pairs':len(held),'other_groups_limit_mps':maximum,'held_above_limit':sum(d>maximum for d in held)}
  trace[source]={'all_pair_maximum_mps':thresholds[source],'by_recording':records,'leave_group_out_diagnostic':cross};checks+=len(pairs)
 close(p['source_limits_mps'],thresholds);close(p['trace'],trace)
 return thresholds,checks

def scalar_bank(times,pred,eligible,endpoint,threshold,kind):
 allowed=[];reason=[];alarms=[];recoveries=[];trusted=None;trusted_stamp=None;quarantine=False;confirmations=[]
 for q,v,ok,s in zip(times,pred,eligible,endpoint):
  q=int(q);s=int(s);v=float(v);permit=False;code=1;alarm=False;recovery=False
  if ok:
   if not math.isfinite(v) or v<0 or not 0<=q-s<=100_000_000:raise AssertionError('eligible source invalid')
   if quarantine:
    code=3
    if kind=='return3' and abs(v-trusted)<=threshold:
     if confirmations and s-confirmations[-1]>100_000_000:confirmations=[]
     if not confirmations or s>confirmations[-1]:confirmations.append(s)
     code=4
     if len(confirmations)>=3:quarantine=False;permit=True;code=5;recovery=True;trusted=v;trusted_stamp=s;confirmations=[]
    else:confirmations=[]
   elif trusted_stamp is None or s-trusted_stamp>100_000_000:
    permit=True;code=6 if trusted_stamp is not None else 0;trusted=v;trusted_stamp=s
   elif abs(v-trusted)>threshold:
    quarantine=True;alarm=True;code=2;confirmations=[]
   else:permit=True;code=0;trusted=v;trusted_stamp=s
  else:confirmations=[]
  allowed.append(permit);reason.append(code);alarms.append(alarm);recoveries.append(recovery)
 return {'allowed':allowed,'reason':reason,'alarm':alarms,'recovery':recoveries}

def sources_from_original(old,times,events):
 sources={}
 for source in SOURCES:
  method='radar_only' if source=='radar_only' else source+'__deadband__withhold';out={k:old[method+'__'+k] for k in FIELDS};eligible=[int(s)==0 for s in out['status']];stamps=[]
  for i,t in enumerate(times):
   if eligible[i]:
    j=bisect.bisect_right(events,int(t))-1
    if j<0 or j!=int(out['latest_endpoint'][i]) or int(t)-events[j]!=int(out['age_ns'][i]) or int(out['route'][i])!=1 or not 0<=int(out['age_ns'][i])<=100_000_000:raise AssertionError('source schedule')
    stamps.append(events[j])
   else:
    if int(out['route'][i])!=0:raise AssertionError('source denial')
    stamps.append(-1)
  sources[source]={'out':out,'eligible':eligible,'stamps':stamps}
 return sources

def select(sources,method,banks):
 model='radar_only' if method=='radar_only' else method.split('__')[0];p=sources[model];r=sources['radar_only'];fallback=method.endswith('__fallback');out={k:[] for k in FIELDS}
 for i in range(len(p['eligible'])):
  p_ok=p['eligible'][i] and (banks is None or banks[model]['allowed'][i]);r_ok=r['eligible'][i] and (banks is None or banks['radar_only']['allowed'][i]);route=1 if p_ok else 2 if fallback and r_ok else 0;chosen=p if route==1 else r if route==2 else None
  out['route'].append(route);out['status'].append(0 if chosen is not None else 4 if p['eligible'][i] else int(p['out']['status'][i]))
  out['prediction_mps'].append(float(chosen['out']['prediction_mps'][i]) if chosen else math.nan);out['latest_endpoint'].append(int(chosen['out']['latest_endpoint'][i]) if chosen else -1);out['age_ns'].append(int(chosen['out']['age_ns'][i]) if chosen else -1);out['primary_status'].append(int(p['out']['status'][i]));out['radar_status'].append(int(r['out']['status'][i]))
 return {k:np.array(v) for k,v in out.items()}

def fixture_verify(saved):
 if set(saved)!={'abrupt_return','persistent_step','slow_bias','startup_bias','gap_hidden_bias','benign_fast_change','anchor_collision'}:raise AssertionError('counterexample missing')
 for name,methods in saved.items():
  times=list(range(0,61*50_000_000,50_000_000));truth=[10.]*61
  if name in ('abrupt_return','benign_fast_change'):pred=[13. if 10<=i<30 else 10. for i in range(61)];truth=pred.copy() if name=='benign_fast_change' else truth
  elif name=='persistent_step':pred=[13. if i>=10 else 10. for i in range(61)]
  elif name=='slow_bias':pred=[10.+min(max(i-9,0)*.1,3.) for i in range(61)]
  elif name=='startup_bias':pred=[13.]*61
  elif name=='gap_hidden_bias':times=[0,100_000_001];pred=[10.,13.];truth=[10.,10.]
  elif name=='anchor_collision':pred=[13. if 10<=i<20 else 10. for i in range(61)];truth=[7. if i>=20 else 10. for i in range(61)]
  for kind in ('latch','return3'):
   bank=scalar_bank(times,pred,[True]*len(times),times,1.,kind);error=[abs(a-b) for a,b,ok in zip(pred,truth,bank['allowed']) if ok]
   close(methods[kind],{'samples':len(times),'allowed':bank['allowed'],'reason':bank['reason'],'alarm_rows':[i for i,v in enumerate(bank['alarm']) if v],'recovery_rows':[i for i,v in enumerate(bank['recovery']) if v],'accepted':sum(bank['allowed']),'accepted_error_above_2':sum(v>2. for v in error),'maximum_accepted_error_mps':max(error) if error else None})

def verify():
 freeze();lock=read(ROOT/'calibration_lock.json');close(lock,{'parameters_sha256':digest(ROOT/'parameters.json'),'pairs_sha256':digest(ROOT/'calibration_pairs.npz'),'analysis_lock_sha256':digest(ROOT/'analysis_lock.json')});thresholds,calibration_checks=independently_calibrate();r=read(ROOT/'results.json');fixture_verify(r['synthetic_counterexamples'])
 frames=read(RESEARCH/'vehicle_masked_history/frame_index.json');events=[int(v['time_ns']) for v in frames];refs=read(RESEARCH/'vehicle_availability_audit/request_reference.json');times=np.array([int(v['time_ns']) for v in refs],np.int64);truth=np.array([math.nan if v['reference_mps'] is None else float(v['reference_mps']) for v in refs]);duration=events[-1]-events[0];start=events[0]+duration*2//5;end=events[0]+duration*3//5
 if len(times)!=37340 or len(events)!=40690:raise AssertionError('cohort')
 names=['all','before','during','after']+['segment_'+str(k) for k in sorted(set((int(t)-int(times[0]))//300_000_000_000 for t in times))];masks={name:np.array([True if name=='all' else t<start if name=='before' else start<=t<end if name=='during' else t>=end if name=='after' else (int(t)-int(times[0]))//300_000_000_000==int(name[8:]) for t in times]) for name in names}
 previous=read(RESEARCH/'vehicle_source_compromise/results.json');branch_checks=output_checks=metric_cells=baseline_cells=0;csv_rows=[]
 if set(r['cases'])!=set(CASES) or set(r['source_diagnostics'])!=set(CASES):raise AssertionError('cases')
 for case in CASES:
  old=load(RESEARCH/'vehicle_source_compromise/replays'/(case+'.npz'));sources=sources_from_original(old,times,events);path=ROOT/'replays'/(case+'.npz')
  with zipfile.ZipFile(path) as z:
   if z.testzip() is not None:raise AssertionError('CRC')
  saved=load(path);banks={};keys=set()
  for kind in ('latch','return3'):
   banks[kind]={}
   for source,s in sources.items():
    v=scalar_bank(times,s['out']['prediction_mps'],s['eligible'],s['stamps'],thresholds[source],kind);banks[kind][source]=v;branch_checks+=len(times)
    for k,x in v.items():key=kind+'__'+source+'__'+k;keys.add(key);np.testing.assert_array_equal(saved[key],x)
    alarms=[i for i,x in enumerate(v['alarm']) if x];recoveries=[i for i,x in enumerate(v['recovery']) if x];pops={}
    for name,mask in masks.items():
     ids=np.flatnonzero(mask);pops[name]={'eligible':sum(bool(s['eligible'][i]) for i in ids),'allowed':sum(v['allowed'][i] for i in ids),'alarms':sum(v['alarm'][i] for i in ids),'recoveries':sum(v['recovery'][i] for i in ids),'gap_reanchors':sum(v['reason'][i]==6 for i in ids)}
    close(r['source_diagnostics'][case][kind][source],{'alarm_rows':alarms,'recovery_rows':recoveries,'first_alarm_time_ns':int(times[alarms[0]]) if alarms else None,'by_population':pops})
  if set(r['cases'][case])!=set(METHODS):raise AssertionError('methods')
  for method in METHODS:
   baseline=select(sources,method,None)
   for k,v in baseline.items():np.testing.assert_allclose(v,old[method+'__'+k],rtol=0,atol=0,equal_nan=True)
   close(r['cases'][case][method]['baseline'],previous['cases'][case][method]);baseline_cells+=len(masks)
   for kind in ('latch','return3'):
    out=select(sources,method,banks[kind]);key=kind+'__'+method+'__route';keys.add(key);np.testing.assert_array_equal(saved[key],out['route']);metric=scalar_summary(out,out['route']>0,times,truth,masks);close(metric,r['cases'][case][method][kind]);output_checks+=len(times);metric_cells+=len(masks)
   if set(r['cases'][case][method])!=set(VARIANTS):raise AssertionError('variants')
   for kind in VARIANTS:
    for pop,v in r['cases'][case][method][kind].items():csv_rows.append({'case':case,'method':method,'variant':kind,'population':pop,'requests':v['requests'],'accepted_requests':v['accepted_requests'],'availability':v['availability'],'p95_mps':v['accepted_errors']['p95_mps'],'above_2_mps':v['accepted_errors']['above_2_mps'],'joint_gate_pass':v['joint_gate_pass']})
  if set(saved)!=keys:raise AssertionError('replay coverage')
  print('VERIFIED',case,flush=True)
 csv_saved=list(csv.DictReader((ROOT/'comparison.csv').open(newline='',encoding='utf-8')));idx={tuple(v[k] for k in ('case','method','variant','population')):v for v in csv_saved}
 if len(idx)!=len(csv_saved) or len(idx)!=len(csv_rows):raise AssertionError('CSV coverage')
 for row in csv_rows:
  got=idx[tuple(row[k] for k in ('case','method','variant','population'))]
  for k,v in row.items():
   if isinstance(v,float):close(float(got[k]),v)
   else:close(got[k],'' if v is None else str(v))
 close(r['decision'],'SOURCE_CALIBRATED_MONITOR_COMPARISON_COMPLETE_NO_PROMOTION')
 for flag in ('models_refitted','labels_changed','accuracy_or_availability_thresholds_changed','production_policy_changed','reserved_final_opened','promotion_allowed','independent_source_trust_established','recovery_certified_safe'):close(r[flag],False)
 for name,h in read(ROOT/'sha256.json').items():
  if blocked(name) or digest(ROOT/name)!=h:raise AssertionError('artifact hash '+name)
 prior=read(ROOT/'prior_snapshot_manifest.json');manifest=read(PROJECT/'PROJECT_MANIFEST.json');verified=missing=protected=0
 for name,h in prior.items():
  if name!='FINAL_PROJECT_README.md' and manifest.get(name)!=h:raise AssertionError('prior hash changed '+name)
 for name,h in manifest.items():
  if blocked(name):protected+=1;continue
  p=PROJECT/name
  if not p.is_file():missing+=1;continue
  if digest(p)!=h:raise AssertionError('project hash '+name)
  verified+=1
 import test_calibrated
 s=io.StringIO();tests=unittest.TextTestRunner(stream=s).run(unittest.defaultTestLoader.loadTestsFromModule(test_calibrated))
 if not tests.wasSuccessful():raise AssertionError(s.getvalue())
 print('CALIBRATED MONITOR VERIFICATION PASS:',calibration_checks,'fitting pair checks;',branch_checks,'source-state checks;',output_checks,'output checks;',metric_cells,'scalar metric cells;',baseline_cells,'historical cells;',len(csv_rows),'CSV rows;',tests.testsRun,'tests')
 print('PROJECT:',verified,'verified;',missing,'missing;',protected,'protected paths not opened')
 print('SOURCE_CALIBRATED_MONITOR_COMPARISON_COMPLETE_NO_PROMOTION')
if __name__=='__main__':verify()
