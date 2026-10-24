"""Independent scalar verification of fixed-output paired robustness."""
import io,math,unittest
from paired_core import *
import test_paired

def close(a,b):
 if isinstance(a,dict):
  if not isinstance(b,dict) or set(a)!=set(b):raise ValueError('audit dictionary fields differ')
  for k in a:close(a[k],b[k])
 elif isinstance(a,list):
  if not isinstance(b,list) or len(a)!=len(b):raise ValueError('audit list population differs')
  for x,y in zip(a,b):close(x,y)
 elif isinstance(a,float):
  if not isinstance(b,(int,float)) or not math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10):raise ValueError('independent numeric audit differs')
 elif a!=b:raise ValueError('independent audit value differs')

def independent_metric(y,p):
 residual=[float(b)-float(a) for a,b in zip(y,p)];e=sorted(abs(v) for v in residual);n=len(e);at=(n-1)*.95;lo=math.floor(at);hi=math.ceil(at)
 return {'count':n,'mae_mps':math.fsum(e)/n,'rmse_mps':math.sqrt(math.fsum(v*v for v in e)/n),'p95_mps':e[lo]+(at-lo)*(e[hi]-e[lo]),'maximum_mps':e[-1],'bias_prediction_minus_reference_mps':math.fsum(residual)/n,'above_half_mps':sum(v>.5 for v in e),'above_2_mps':sum(v>2. for v in e)}

def independent_wins(deltas):
 return {'control_lower':sum(v < -1e-12 for v in deltas),'tied':sum(abs(v)<=1e-12 for v in deltas),'control_higher':sum(v>1e-12 for v in deltas)}

def audit():
 freeze();available=0;project=ROOT.parent.parent;prior=read(ROOT/'prior_snapshot_manifest.json')
 for name,h in prior.items():
  if name in ['FINAL_PROJECT_README.md','scripts/check_coact_project.py']:continue
  f=(project/name).resolve()
  if not f.is_relative_to(project):raise ValueError('unsafe preceding artifact path')
  if f.is_file():
   if digest(f)!=h:raise ValueError('preceding payload changed: '+name)
   available+=1
 if (ROOT/'sha256.json').exists():
  for name,h in read(ROOT/'sha256.json').items():
   f=(ROOT/name).resolve()
   if not f.is_relative_to(ROOT) or digest(f)!=h:raise ValueError('paired study payload changed: '+name)
 members,y,pred,support=load_inputs();result=read(ROOT/'results.json');close(result,build(members,y,pred,support));cells=0;pair_checks=0;unit_checks=0;omission_checks=0
 def verify_cell(ids,cell):
  nonlocal cells
  if cell['count']!=len(ids) or not len(ids):raise ValueError('metric cell denominator differs')
  metrics={n:independent_metric([y[i] for i in ids],[pred[n][i] for i in ids]) for n in MODELS}
  for n in MODELS:close(metrics[n],cell['models'][n]);cells+=1
  for n in COMPARATORS:
   for k in KEYS:close(float(metrics[CONTROL][k]-metrics[n][k]),cell['control_minus_comparator'][n][k])
 verify_cell(list(range(len(y))),result['pooled'])
 for family,key in FAMILIES.items():
  names=sorted({m[key] for m in members});byunit=result['by_unit'][family]
  if set(names)!=set(byunit) or set(names)!=set(result['diagnostic_omissions'][family]):raise ValueError('unit list differs')
  scalar={}
  for name in names:
   ids=[i for i,m in enumerate(members) if m[key]==name];verify_cell(ids,byunit[name]);scalar[name]={n:independent_metric([y[i] for i in ids],[pred[n][i] for i in ids]) for n in MODELS}
   retained=[i for i,m in enumerate(members) if m[key]!=name];omitted=result['diagnostic_omissions'][family][name]
   if omitted['omitted_rows']!=len(ids) or len(ids)+len(retained)!=len(y):raise ValueError('omission population accounting differs')
   verify_cell(retained,omitted['retained']);omission_checks+=1
  for n in MODELS:
   # Equal-unit weighting checked from independent unit errors, not cached balanced().
   means=[scalar[u][n]['mae_mps'] for u in names];mses=[scalar[u][n]['rmse_mps']**2 for u in names];p95=[scalar[u][n]['p95_mps'] for u in names]
   close({'units':len(names),'equal_unit_mae_mps':math.fsum(means)/len(names),'equal_unit_rmse_mps':math.sqrt(math.fsum(mses)/len(names)),'mean_unit_p95_mps':math.fsum(p95)/len(names)},result['equal_unit_summaries'][family][n]);unit_checks+=1
  for n in COMPARATORS:
   for k in KEYS:
    deltas=[scalar[u][CONTROL][k]-scalar[u][n][k] for u in names];close(independent_wins(deltas),result['unit_metric_wins'][family][n][k])
    omission_deltas=[result['diagnostic_omissions'][family][u]['retained']['control_minus_comparator'][n][k] for u in names];close({**independent_wins(omission_deltas),'min_control_minus_comparator_mps':float(min(omission_deltas)),'max_control_minus_comparator_mps':float(max(omission_deltas))},result['omission_sensitivity'][family][n][k])
 for n in COMPARATORS:
  lower=tied=higher=both=aonly=bonly=neither=0
  for target,a,b in zip(y,pred[CONTROL],pred[n]):
   ea=abs(float(a)-float(target));eb=abs(float(b)-float(target));delta=ea-eb
   if delta < -1e-12:lower+=1
   elif delta>1e-12:higher+=1
   else:tied+=1
   if ea>.5 and eb>.5:both+=1
   elif ea>.5:aonly+=1
   elif eb>.5:bonly+=1
   else:neither+=1
  close({'count':len(y),'control_lower_abs_error':lower,'tied_abs_error':tied,'control_higher_abs_error':higher,'above_half_pairs':{'both':both,'control_only':aonly,'comparator_only':bonly,'neither':neither}},result['paired_row_counts'][n])
  if lower+tied+higher!=len(y) or both+aonly+bonly+neither!=len(y):raise ValueError('paired rows lost')
  pair_checks+=len(y)
 parent=read(ROOT.parent/'vehicle_matched_baselines/results.json');mag=read(ROOT.parent/'vehicle_magnitude_ablation/results.json');parent_cells=0
 for family,old in [(None,'pooled'),('route_group','by_group'),('recording','by_recording'),('window','by_window')]:
  current={'pooled':result['pooled']} if family is None else result['by_unit'][family];previous={'pooled':parent[old]} if family is None else parent[old];previous_mag={'pooled':mag[old]} if family is None else mag[old]
  for unit,cell in current.items():
   for n in MODELS:
    saved=previous[unit]['models'][n] if n in MODELS[:4] else previous_mag[unit]['models'][n]
    for k,v in saved.items():close(cell['models'][n][k],v)
    parent_cells+=1
 for n in MODELS:
  old=parent['diagnostic_failures'][n] if n in MODELS[:4] else mag['diagnostic_failures'][n];new=result['diagnostic_failures'][n]
  for a,b in [('pooled','pooled'),('route_groups','groups'),('recordings','recordings'),('windows','windows')]:close(new[a],old[b])
 trace=read(ROOT.parent/'vehicle_matched_baselines/training_trace.json');hubers=[m for run in trace['runs'] for m in run['models'] if m['name']=='huber_context']
 if len(hubers)!=3 or any(not m['metadata']['iteration_budget_reached'] or not m['metadata']['optimizer_warning'] or m['metadata']['iterations']!=500 for m in hubers):raise ValueError('parent Huber optimizer caveat differs')
 if result['model_refitting'] or result['new_inference'] or result['promotion_allowed'] or result['omission_changes_main_cohort'] or result['p_values_computed'] or result['population_confidence_intervals_computed'] or result['independent_replications_claimed']:raise ValueError('unsupported promotion/inference claim')
 buffer=io.StringIO();tests=unittest.TextTestRunner(stream=buffer).run(unittest.defaultTestLoader.loadTestsFromModule(test_paired))
 if not tests.wasSuccessful():raise ValueError(buffer.getvalue())
 print(f'PAIRED ROBUSTNESS AUDIT: PASS ({tests.testsRun} tests;6 saved methods;{len(y)} rows/model;{cells} independent metric cells;{unit_checks} equal-unit summaries;{pair_checks} scalar paired-row comparisons;{omission_checks} omission subsets;{parent_cells} parent metric cells retained)')
 print(f'Available preceding payloads verified: {available}/{len(prior)-2}; unavailable files are not claimed as reverified.')
 for n in COMPARATORS:
  s=result['unit_metric_wins']['window'][n]['p95_mps'];print('WINDOW P95 ORIGINAL vs',n,':',s['control_lower'],'lower,',s['tied'],'tied,',s['control_higher'],'higher')
 print(result['decision']+'; DESCRIPTIVE RETROSPECTIVE AUDIT. FOUR ORIGINAL WINDOW FAILURES REMAIN. WITNESS DISABLED.')
if __name__=='__main__':audit()
