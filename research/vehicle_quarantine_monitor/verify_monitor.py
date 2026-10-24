"""Separate state machine and scalar metrics; no monitor/reducer calls."""
import csv,io,math,sys,unittest,zipfile
import numpy as np
from monitor_core import ROOT,PROJECT,CASES,METHODS,FIELDS,read,load,digest,check_bundle,prior_inputs,proposal

def close(a,b,path=''):
 if isinstance(a,dict):
  if not isinstance(b,dict) or set(a)!=set(b):raise AssertionError('keys '+path)
  for k in a:close(a[k],b[k],path+'/'+str(k))
 elif isinstance(a,list):
  if len(a)!=len(b):raise AssertionError('length '+path)
  for i,(x,y) in enumerate(zip(a,b)):close(x,y,path+'/'+str(i))
 elif isinstance(a,float):
  if not isinstance(b,(int,float)) or not math.isclose(a,b,rel_tol=0,abs_tol=1e-9):raise AssertionError('numeric '+path)
 elif a!=b:raise AssertionError('value '+path)

def scalar_monitor(times,prediction,base):
 selected=[];reasons=[];anchor_value=anchor_time=None;quarantined=False;alarm=None
 for row,(t,p,keep) in enumerate(zip(times,prediction,base)):
  t=int(t);p=float(p)
  if not keep:selected.append(False);reasons.append(1);continue
  if quarantined:selected.append(False);reasons.append(3);continue
  comparable=anchor_time is not None and t-anchor_time<=100_000_000
  if comparable and abs(p-anchor_value)>1.:
   alarm=row;quarantined=True;selected.append(False);reasons.append(2);continue
  selected.append(True);reasons.append(4 if anchor_time is not None and not comparable else 0);anchor_value=p;anchor_time=t
 return selected,reasons,alarm

def quantile(values):
 values=sorted(values);at=(len(values)-1)*.95;i=int(at);return values[i]+(values[min(i+1,len(values)-1)]-values[i])*(at-i)

def scalar_summary(out,allowed,times,truth,masks):
 cells={name:{'requests':0,'route':[0]*3,'primary':[0]*5,'radar':[0]*5,'errors':[],'missing':0,'steps':[],'switches':0} for name in masks};previous=None
 for i,t in enumerate(times):
  route=int(out['route'][i]) if allowed[i] else 0;p=float(out['prediction_mps'][i]) if route else math.nan;groups=[name for name,mask in masks.items() if mask[i]]
  for name in groups:
   c=cells[name];c['requests']+=1;c['route'][route]+=1;c['primary'][int(out['primary_status'][i])]+=1;c['radar'][int(out['radar_status'][i])]+=1
   if route:
    if math.isfinite(float(truth[i])):c['errors'].append(abs(p-float(truth[i])))
    else:c['missing']+=1
  if previous is not None:
   old_t,old_route,old_p,old_groups=previous
   if route and old_route and int(t)-old_t<=50_000_000:
    for name in groups:
     if name in old_groups:cells[name]['steps'].append(abs(p-old_p));cells[name]['switches']+=int(route!=old_route)
  previous=int(t),route,p,groups
 result={}
 for name,c in cells.items():
  e=c['errors'];n=len(e);a=c['route'][1]+c['route'][2];availability=a/c['requests'];p95=quantile(e) if n else None;above=sum(v>2. for v in e)
  errors={'count':n,'missing_truth':c['missing'],'mae_mps':math.fsum(e)/n if n else None,'rmse_mps':math.sqrt(math.fsum(v*v for v in e)/n) if n else None,'p95_mps':p95,'maximum_mps':max(e) if n else None,'above_2_mps':above}
  gates={'availability_ge_95pct':availability>=.95,'p95_le_0_5_mps':p95 is not None and p95<=.5,'no_missing_accepted_truth':c['missing']==0,'no_accepted_error_above_2_mps':above==0}
  result[name]={'requests':c['requests'],'accepted_requests':a,'availability':availability,'route_counts':{str(k):v for k,v in enumerate(c['route'])},'primary_status_counts':{str(k):v for k,v in enumerate(c['primary'])},'radar_status_counts':{str(k):v for k,v in enumerate(c['radar'])},'accepted_errors':errors,'gates':gates,'joint_gate_pass':all(gates.values()),'adjacent_accepted_pairs':len(c['steps']),'source_transitions':c['switches'],'adjacent_prediction_step_p95_mps':quantile(c['steps']) if c['steps'] else None,'adjacent_prediction_step_maximum_mps':max(c['steps']) if c['steps'] else None}
 return result

def fixture_check(saved):
 for name,v in saved.items():
  t=list(range(0,61*50_000_000,50_000_000));truth=[10.]*61
  if name=='gap_plus3':t=[0,100_000_001];p=[10.,13.];truth=[10.,10.]
  elif name=='abrupt_plus3':p=[10.+(3. if i>=10 else 0.) for i in range(61)]
  elif name=='slow_plus3':p=[10.+min(max(i-9,0)*.1,3.) for i in range(61)]
  elif name=='subthreshold_ramp':p=[10.+min(max(i-9,0)*.9,3.) for i in range(61)]
  elif name=='startup_plus3':p=[13.]*61
  elif name=='benign_fast_change':p=[10.+(1.01 if i>=10 else 0.) for i in range(61)];truth=p.copy()
  else:raise AssertionError('unknown fixture')
  allowed,reason,first=scalar_monitor(t,p,[True]*len(t));errors=[abs(x-y) for x,y,k in zip(p,truth,allowed) if k]
  close({'samples':len(t),'accepted':sum(allowed),'first_alarm':first,'gap_reanchors':reason.count(4),'accepted_error_above_2':sum(v>2. for v in errors),'maximum_accepted_error_mps':max(errors) if errors else None,'allowed':allowed,'reason':reason},v)
 if set(saved)!={'gap_plus3','abrupt_plus3','slow_plus3','subthreshold_ramp','startup_plus3','benign_fast_change'}:raise AssertionError('counterexample missing')

def tests():
 import test_monitor
 s=io.StringIO();r=unittest.TextTestRunner(stream=s).run(unittest.defaultTestLoader.loadTestsFromModule(test_monitor))
 if not r.wasSuccessful():raise AssertionError(s.getvalue())
 return r.testsRun

def verify(self_test=False):
 check_bundle();fixture_check(read(ROOT/'synthetic_counterexamples.json'));count=tests()
 if self_test:
  print('MONITOR SELF-TEST PASS:',count,'tests; 6 independently reconstructed synthetic fixtures. Real project comparison NOT RUN.');return
 old,hashes,times,truth,events,masks,prior=prior_inputs();lock=read(ROOT/'local_run_lock.json');result=read(ROOT/'local_results.json')
 close(lock['prior_sha256'],hashes);close(lock['monitor_lock'],read(ROOT/'analysis_lock.json'))
 if lock['truth_used_by_monitor'] is not False:raise AssertionError('truth-dependent rule')
 if lock['project_manifest_sha256']!=digest(PROJECT/'PROJECT_MANIFEST.json'):raise AssertionError('project manifest changed after local run')
 fixture_check(result['synthetic_counterexamples']);checks=cells=baseline_cells=0;csv_expected=[]
 if set(result['cases'])!=set(CASES):raise AssertionError('case missing')
 for case in CASES:
  original=load(old/'replays'/(case+'.npz'));path=ROOT/'local_replays'/(case+'.npz')
  with zipfile.ZipFile(path) as z:
   if z.testzip() is not None:raise AssertionError('replay CRC')
  saved=load(path)
  if set(saved)!={m+'__'+k for m in METHODS for k in ('allowed','reason')}:raise AssertionError('monitor fields missing')
  if set(result['cases'][case])!=set(METHODS):raise AssertionError('method missing')
  for method in METHODS:
   out=proposal(original,method,times,events);allowed,reasons,first=scalar_monitor(times,out['prediction_mps'],out['route']>0)
   np.testing.assert_array_equal(saved[method+'__allowed'],allowed);np.testing.assert_array_equal(saved[method+'__reason'],reasons)
   if saved[method+'__allowed'].dtype.kind!='b' or saved[method+'__reason'].dtype!=np.uint8:raise AssertionError('guard dtype')
   baseline=scalar_summary(out,out['route']>0,times,truth,masks);guarded=scalar_summary(out,allowed,times,truth,masks);cell=result['cases'][case][method]
   close(baseline,prior['cases'][case][method]);close(baseline,cell['baseline']);close(guarded,cell['quarantine']);checks+=len(times);cells+=2*len(masks);baseline_cells+=len(masks)
   close({'first_alarm_index':first,'first_alarm_time_ns':None if first is None else int(times[first]),'additional_withheld_requests':sum(bool(a) and not bool(b) for a,b in zip(out['route']>0,allowed)),'gap_reanchors':reasons.count(4)},cell['monitor'])
   for variant,populations in [('baseline',baseline),('quarantine',guarded)]:
    for pop,v in populations.items():csv_expected.append({'case':case,'method':method,'variant':variant,'population':pop,'requests':v['requests'],'accepted_requests':v['accepted_requests'],'availability':v['availability'],'p95_mps':v['accepted_errors']['p95_mps'],'above_2_mps':v['accepted_errors']['above_2_mps'],'joint_gate_pass':v['joint_gate_pass']})
 rows=list(csv.DictReader((ROOT/'local_comparison.csv').open(newline='',encoding='utf-8')));keys=('case','method','variant','population');indexed={tuple(row[k] for k in keys):row for row in rows}
 if len(rows)!=len(csv_expected) or len(indexed)!=len(rows):raise AssertionError('CSV coverage')
 for row in csv_expected:
  got=indexed[tuple(row[k] for k in keys)]
  for k,v in row.items():
   if isinstance(v,float):close(float(got[k]),v)
   else:close(got[k],'' if v is None else str(v))
 if result['execution_status']!='QUARANTINE_COMPARISON_COMPLETE_NO_PROMOTION':raise AssertionError('execution status')
 for flag in ('models_refitted','labels_changed','source_policy_changed','production_policy_changed','reserved_final_opened','promotion_allowed','independent_physical_witness_established'):
  if result[flag] is not False:raise AssertionError('unsupported claim')
 expected={'local_run_lock.json','local_results.json','local_comparison.csv',*['local_replays/'+c+'.npz' for c in CASES]};sha=read(ROOT/'local_outputs_sha256.json')
 if set(sha)!=expected:raise AssertionError('local output list changed')
 for name,h in sha.items():
  if digest(ROOT/name)!=h:raise AssertionError('local output changed '+name)
 report={'status':'VERIFIED_EXECUTION_NOT_SECURITY_CERTIFICATION','request_monitor_checks':checks,'metric_cells':cells,'historical_metric_cells':baseline_cells,'csv_rows':len(rows),'tests':count,'synthetic_counterexamples':6,'real_project_comparison_executed':True,'promotion_allowed':False}
 from monitor_core import write
 write(ROOT/'local_verification_summary.json',report)
 print('QUARANTINE MONITOR VERIFICATION PASS:',checks,'request-monitor checks;',cells,'metric cells;',baseline_cells,'historical cells;',len(rows),'CSV rows;',count,'tests')
 print('QUARANTINE_COMPARISON_COMPLETE_NO_PROMOTION')
if __name__=='__main__':verify('--self-test' in sys.argv[1:])
