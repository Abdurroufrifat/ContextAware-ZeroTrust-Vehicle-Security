"""Audit complete descriptive output without refitting or reopening raw data."""
import io,math,unittest
from diagnosis_core import *
import test_diagnosis

def close(a,b):
 if isinstance(a,dict):
  if not isinstance(b,dict) or set(a)!=set(b):raise ValueError('dictionary fields differ')
  for k in a:close(a[k],b[k])
 elif isinstance(a,list):
  if not isinstance(b,(list,tuple)) or len(a)!=len(b):raise ValueError('list population differs')
  for x,y in zip(a,b):close(x,y)
 elif isinstance(a,float):
  if not isinstance(b,(float,int)) or not math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10):raise ValueError('numeric audit mismatch')
 elif a!=b:raise ValueError('audit value differs')

def independent_metric(y,p):
 residual=[float(b)-float(a) for a,b in zip(y,p)];e=sorted(abs(x) for x in residual);n=len(e)
 if not n:return {'count':0,**{k:None for k in ['mae_mps','rmse_mps','p95_mps','maximum_mps','bias_prediction_minus_reference_mps']},'above_half_mps':0,'above_2_mps':0}
 at=(n-1)*.95;lo=int(math.floor(at));hi=int(math.ceil(at));q=e[lo]+(at-lo)*(e[hi]-e[lo])
 return {'count':n,'mae_mps':math.fsum(e)/n,'rmse_mps':math.sqrt(math.fsum(v*v for v in e)/n),'p95_mps':q,'maximum_mps':e[-1],'bias_prediction_minus_reference_mps':math.fsum(residual)/n,'above_half_mps':sum(v>.5 for v in e),'above_2_mps':sum(v>2. for v in e)}

def independent_runs(members,y,p):
 records=[];windows=sorted({v['window_id'] for v in members})
 for name in windows:
  ids=sorted([i for i,v in enumerate(members) if v['window_id']==name],key=lambda i:members[i]['time_ns'])
  for model in MODELS:
   errors=[abs(float(p[model][i])-float(y[i])) for i in ids];positions=[j for j,e in enumerate(errors) if e>.5];chunks=[]
   for j in positions:
    if not chunks or j!=chunks[-1][-1]+1 or members[ids[j]]['time_ns']-members[ids[chunks[-1][-1]]]['time_ns']>GAP_NS:chunks.append([j])
    else:chunks[-1].append(j)
   for chunk in chunks:
    first,last=chunk[0],chunk[-1];a,b=ids[first],ids[last];records.append({'model':model,'window_id':name,'route':members[a]['route'],'first_row_id':a,'last_row_id':b,'first_local_index':first,'last_local_index':last,'first_time_ns':members[a]['time_ns'],'last_time_ns':members[b]['time_ns'],'rows':len(chunk),'span_ns':members[b]['time_ns']-members[a]['time_ns'],'peak_absolute_error_mps':max(errors[j] for j in chunk)})
 return records

def audit():
 freeze()
 for name,h in read(ROOT/'sha256.json').items():
  f=(ROOT/name).resolve()
  if not f.is_relative_to(ROOT) or digest(f)!=h:raise ValueError('diagnostic payload changed: '+name)
 members,y,p,support=load_inputs();result,arrays,runs=build(members,y,p);result['raw_support_costs']=support;close(result,read(ROOT/'results.json'));close(runs,read(ROOT/'error_runs.json'));close(runs,independent_runs(members,y,p))
 with np.load(ROOT/'row_assignments.npz',allow_pickle=False) as z:
  if set(z.files)!=set(arrays):raise ValueError('row-assignment schema differs')
  for name,v in arrays.items():
   if not np.array_equal(z[name],v):raise ValueError('row-assignment differs')
 bins=arrays['reference_speed_bin'];third=arrays['within_window_supported_third'];window=np.array([v['window_id'] for v in members]);group=np.array([v['group'] for v in members]);route=np.array([v['route'] for v in members]);day=np.array([v['day'] for v in members]);allrows=np.ones(len(y),dtype=bool);cells=[(allrows,result['pooled'])]
 for key,labels in [('by_group',group),('by_recording',route),('by_window',window)]:cells.extend((labels==name,cell) for name,cell in result[key].items())
 cells.extend((bins==i,result['by_speed'][label]) for i,label in enumerate(SPEED_NAMES));cells.extend([(day,result['by_day_night']['day']),(~day,result['by_day_night']['night'])])
 failed=np.isin(window,result['failed_hgb_windows'])
 for name in result['by_window']:
  ids=np.flatnonzero(window==name);count=0
  for k,cell in result['by_window_third'][name].items():cells.append(((window==name)&(third==int(k)),cell));count+=cell['count']
  if count!=len(ids):raise ValueError('time thirds lost rows')
  count=0
  for i,label in enumerate(SPEED_NAMES):cell=result['by_window_speed'][name][label];cells.append(((window==name)&(bins==i),cell));count+=cell['count']
  if count!=len(ids):raise ValueError('speed bins lost rows')
  # Integer oracle for every supported-time assignment, with last timestamp included.
  first=min(members[i]['time_ns'] for i in ids);last=max(members[i]['time_ns'] for i in ids)
  for i in ids:
   expected=min(2,(members[i]['time_ns']-first)*3//(last-first)) if last>first else 0
   if int(third[i])!=expected:raise ValueError('time-stratum integer oracle differs')
 for prefix,mask in [('failed_hgb_windows',failed),('other_hgb_windows',~failed)]:cells.extend((mask&(bins==i),result['failed_vs_other_by_speed'][prefix][label]) for i,label in enumerate(SPEED_NAMES))
 for mask,cell in cells:
  if cell['count']!=int(mask.sum()):raise ValueError('metric cell denominator differs')
  for model in MODELS:close(cell['models'][model],independent_metric(y[mask],p[model][mask]))
 parent=read(ROOT.parent/'vehicle_matched_baselines/results.json');route_parent=read(ROOT.parent/'vehicle_route_validation/results.json');metric_cells=0
 for category,new_key,old_key in [('pooled','pooled','pooled_sample_weighted'),('by_group','by_group','by_group'),('by_recording','by_recording','by_recording'),('by_window','by_window','by_window')]:
  current={'pooled':result['pooled']} if category=='pooled' else result[category];old={'pooled':parent['pooled']} if category=='pooled' else parent[new_key];oldroute={'pooled':route_parent[old_key]} if category=='pooled' else route_parent[old_key]
  if set(current)!=set(old) or set(current)!=set(oldroute):raise ValueError('parent context list differs')
  for name,cell in current.items():
   for model in MODELS:
    for key,value in old[name]['models'][model].items():close(cell['models'][model][key],value)
    metric_cells+=1
   for key,value in oldroute[name]['context_boosting'].items():close(cell['models']['context_boosting'][key],value)
 if len(result['failed_hgb_windows'])!=4 or result['failed_hgb_windows']!=sorted(parent['diagnostic_failures']['context_boosting']['windows']):raise ValueError('parent failures changed')
 if sum(v['count'] for v in result['by_speed'].values())!=len(y):raise ValueError('pooled speed strata lost rows')
 for model in MODELS:
  if result['sample_adjacent_run_summary'][model]['exceedance_rows']!=result['pooled']['models'][model]['above_half_mps']:raise ValueError('exceedance runs lost rows')
 suite=unittest.defaultTestLoader.loadTestsFromModule(test_diagnosis);buf=io.StringIO();tests=unittest.TextTestRunner(stream=buf).run(suite)
 if not tests.wasSuccessful():raise ValueError(buf.getvalue())
 print(f'FAILURE DIAGNOSIS AUDIT: PASS ({tests.testsRun} tests;{len(y)} rows/model;{len(cells)*len(MODELS)} independently checked metric cells;{len(runs)} independently rebuilt runs;{metric_cells} parent metric cells retained)')
 print('FAILED WINDOWS RETAINED:',', '.join(result['failed_hgb_windows']))
 print(result['decision']+'; ASSOCIATIONS ONLY. WITNESS DISABLED.')
if __name__=='__main__':audit()
