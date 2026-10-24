"""Independent row, clock, scope, metric, stratum and run checks; no training."""
import bisect,io,math,unittest
from remaining_core import *
import test_remaining
close=a.m.sa.close
def scalar_quantile(v,q):
 v=sorted(map(float,v));at=(len(v)-1)*q;lo=math.floor(at);hi=math.ceil(at);return v[lo]+(at-lo)*(v[hi]-v[lo])
def scalar_stats(x):
 values=list(map(float,x));v=[z for z in values if math.isfinite(z)];return {'total':len(values),'finite':len(v),'min':min(v) if v else None,'median':scalar_quantile(v,.5) if v else None,'p95':scalar_quantile(v,.95) if v else None,'max':max(v) if v else None}
def scalar_metrics(y,p):
 y=list(map(float,y));p=list(map(float,p));r=[x-v for x,v in zip(p,y)];e=list(map(abs,r));n=len(e);tail=[x for x in r if abs(x)>.5]
 return {'count':n,'mae_mps':math.fsum(e)/n if n else None,'rmse_mps':math.sqrt(math.fsum(x*x for x in e)/n) if n else None,'p95_mps':scalar_quantile(e,.95) if n else None,'maximum_mps':max(e) if n else None,'bias_prediction_minus_reference_mps':math.fsum(r)/n if n else None,'above_half_mps':len(tail),'above_2_mps':sum(x>2. for x in e),'tail_overprediction_fraction':sum(x>0 for x in tail)/len(tail) if tail else None}
def arraycheck(x,y,atol=1e-9):
 x=np.asarray(x);y=np.asarray(y)
 if y.dtype.kind in 'iubUS':
  if (y.dtype.kind in 'iu' and x.dtype.kind not in 'iu') or (y.dtype.kind=='b' and x.dtype.kind!='b'):raise ValueError('lost exact clock/boolean dtype')
  np.testing.assert_array_equal(x,y)
 else:np.testing.assert_allclose(x,y,rtol=0,atol=atol,equal_nan=True)
def scalar_runs(t,e):
 groups=[];active=[]
 for i,(time,error) in enumerate(zip(t,e)):
  if error<=.5:
   if active:groups.append(active);active=[]
  else:
   if active and int(time)-int(t[active[-1]])>150_000_000:groups.append(active);active=[]
   active.append(i)
 if active:groups.append(active)
 return [{'first_local_index':v[0],'last_local_index':v[-1],'first_time_ns':int(t[v[0]]),'last_time_ns':int(t[v[-1]]),'rows':len(v),'span_ns':int(t[v[-1]])-int(t[v[0]]),'peak_absolute_error_mps':max(float(e[i]) for i in v)} for v in groups]
def audit():
 freeze();prior=read(ROOT/'prior_snapshot_manifest.json');available=0
 for n,h in prior.items():
  if n in ['FINAL_PROJECT_README.md','scripts/check_coact_project.py']:continue
  path=(RESEARCH.parent/n).resolve()
  if not path.is_relative_to(RESEARCH.parent):raise ValueError('unsafe prior path')
  if path.is_file():
   if digest(path)!=h:raise ValueError('previous payload changed '+n)
   available+=1
 if (ROOT/'sha256.json').exists():
  for n,h in read(ROOT/'sha256.json').items():
   path=(ROOT/n).resolve()
   if not path.is_relative_to(ROOT) or digest(path)!=h:raise ValueError('diagnostic artifact changed '+n)
 with np.load(ROOT/'row_diagnostics.npz',allow_pickle=False) as z:rows={scope:{n[len(scope)+2:]:z[n].copy() for n in z.files if n.startswith(scope+'__')} for scope in ['fit','full']}
 members=read(a.m.r.ROOT/'membership.json');index,ready,fx,median,anchor,peer,source,old=a.m.f.load_full();data,*_=a.m.r.load_data();axis=a.cache();reference=read(RESEARCH/'vehicle_continuous_radar/reference_targets.json');allready=np.flatnonzero(ready)
 arraycheck(rows['fit']['row_id'],np.arange(len(members),dtype=np.int64));arraycheck(rows['fit']['time_ns'],np.array([v['time_ns'] for v in members],dtype=np.int64));arraycheck(rows['fit']['window_id'],np.array([v['window_id'] for v in members]));arraycheck(rows['fit']['target_mps'],data['y']);arraycheck(rows['full']['row_id'],allready);arraycheck(rows['full']['time_ns'],np.array([index[i]['time_ns'] for i in allready],dtype=np.int64));arraycheck(rows['full']['target_mps'],np.array([reference[i]['reference_mps'] for i in allready]));arraycheck(rows['full']['reference_age_ns'],np.array([index[i]['time_ns']-reference[i]['reference_header_ns'] for i in allready],dtype=np.int64))
 blocklookup={v['time_ns']:v['window_id'] for v in read(a.m.c.OLD/'prepared/calibration/index.json') if v['ready']};arraycheck(rows['full']['window_id'],np.array([blocklookup.get(int(t),'outside_supported_blocks') for t in rows['full']['time_ns']]))
 with np.load(RESEARCH/'vehicle_trace_audit/row_traces.npz',allow_pickle=False) as z:arraycheck(rows['fit']['reference_age_ns'],z['reference_age_ns'])
 with np.load(a.ROOT/'oof_predictions.npz',allow_pickle=False) as z:
  for n in METHODS:arraycheck(rows['fit'][n+'__prediction_mps'],z[n+'__prediction_mps']);arraycheck(rows['fit'][n+'__accepted'],np.ones(len(members),bool))
 with np.load(a.ROOT/'development_replay.npz',allow_pickle=False) as z:
  for n in METHODS:arraycheck(rows['full'][n+'__prediction_mps'],z[n+'__ready_prediction_mps']);arraycheck(rows['full'][n+'__accepted'],z[n+'__quality'][ready])
 clocks=0
 for key,route in a.m.sa.l.ROUTES.items():
  _,src=a.m.sa.load(key);arrivals=src['bag_ns'].tolist();q=rows['full'] if key=='Urban01_night' else rows['fit'];ids=np.arange(len(q['row_id'])) if key=='Urban01_night' else np.array([i for i,v in enumerate(members) if v['route']==route])
  for i in ids:
   t=int(q['time_ns'][i]);j=bisect.bisect_right(arrivals,t)-1;ha=t-int(src['header_ns'][j]) if j>=0 else -1;ba=t-int(src['bag_ns'][j]) if j>=0 else -1;status=1 if j<0 else 3 if ha<0 else 2 if max(ha,ba)>100_000_000 else 4 if not src['usable'][j] else 0
   if int(q['imu_status'][i])!=status or int(q['imu_header_age_ns'][i])!=ha or int(q['imu_arrival_age_ns'][i])!=ba:raise ValueError('independent source clock mismatch')
   clocks+=1
 for scope,q in rows.items():
  axes=axis['fit_axes'] if scope=='fit' else axis['full_axes'][ready];tensors=data['X'] if scope=='fit' else fx;arraycheck(q['signed_axes'],axes);arraycheck(q['imu_valid_tick_fraction'],axis['fit_coverage'] if scope=='fit' else axis['full_coverage'][ready]);valid=tensors[:,3,:]==1;arraycheck(q['radar_valid_fraction'],np.mean(valid,axis=1));arraycheck(q['radar_mixed_sign_grid'],np.any(valid&(tensors[:,0,:]<0),axis=1)&np.any(valid&(tensors[:,0,:]>0),axis=1));ages=np.array([math.fsum(float(v) for v,ok in zip(row[2],row[3]) if ok)/max(1,int(sum(row[3])))*300 for row in tensors]);arraycheck(q['radar_mean_recorded_age_ms'],ages,atol=1e-4)
  covered=np.zeros(len(axes),int)
  for headscope in (a.m.r.GROUPS if scope=='fit' else ['all_fit']):
   cal=read(a.m.ROOT/'models'/headscope/'calibration.json');train=np.array(cal['train_ids']);ids=np.array(cal['held_ids']) if scope=='fit' else np.arange(len(axes));training=axis['fit_axes'][train];h=read(a.ROOT/'heads'/headscope/'head.json');outside=np.column_stack([(axes[ids,j]<min(training[:,j]))|(axes[ids,j]>max(training[:,j])) for j in range(6)]);arraycheck(q['outside_training_axis'][ids],outside)
   if scope=='fit' and np.intersect1d(train,ids).size:raise ValueError('held feature support leakage')
   terms=np.column_stack([(axes[ids,j]-h['mean'][j])/h['scale'][j]*h['coefficient'][j] for j in range(6)]);delta=np.array([math.fsum([h['intercept'],*row]) for row in terms]);arraycheck(q['head_terms'][ids],terms);arraycheck(q['raw_head_delta_mps'][ids],delta);arraycheck(q['axis_standardized_abs_max'][ids],np.array([max(abs(float(v)-mu)/scale for v,mu,scale in zip(row,h['mean'],h['scale'])) for row in axes[ids]]));arraycheck(q[PRIMARY+'__prediction_mps'][ids],np.clip(q[BASE+'__prediction_mps'][ids]+delta,0.,None));covered[ids]+=1
  arraycheck(covered,np.ones(len(axes),int));arraycheck(q['applied_delta_mps'],q[PRIMARY+'__prediction_mps']-q[BASE+'__prediction_mps'])
 result=read(ROOT/'results.json');records=read(ROOT/'error_runs.json');expectedrecords=[];metriccells=statcells=0
 def check_cell(q,ids,saved,details=False):
  nonlocal metriccells,statcells
  close(saved['count'],len(ids));y=q['target_mps'][ids];bad=abs(q[PRIMARY+'__prediction_mps'][ids]-y)>.5;basebad=abs(q[BASE+'__prediction_mps'][ids]-y)>.5
  for n in METHODS:close(saved['models'][n],scalar_metrics(y,q[n+'__prediction_mps'][ids]));close(saved['accepted_counts_on_same_rows'][n],int(q[n+'__accepted'][ids].sum()));metriccells+=1
  close(saved['base_corrected_tail_overlap'],{'both':int((bad&basebad).sum()),'base_only':int((basebad&~bad).sum()),'corrected_only':int((bad&~basebad).sum()),'neither':int((~bad&~basebad).sum())})
  if details:
   for field in COVARIATES:
    for name,flag in [('tail',bad),('other',~bad)]:close(saved['tail_vs_other_covariates'][field][name],scalar_stats(q[field][ids][flag]));statcells+=1
   for name,flag in [('tail',bad),('other',~bad)]:
    which=ids[flag];s=saved[name+'_support'];expected={'rows':len(which),'outside_any_training_axis':sum(any(row) for row in q['outside_training_axis'][which]),'outside_training_axis_counts':[sum(row[j] for row in q['outside_training_axis'][which]) for j in range(6)],'mixed_sign_radar_grid':int(q['radar_mixed_sign_grid'][which].sum()),'invalid_latest_imu':int((q['imu_status'][which]!=0).sum()),'reference_age_above100ms':int((q['reference_age_ns'][which]>100_000_000).sum()),'negative_reference_age':int((q['reference_age_ns'][which]<0).sum()),'head_term_stats':{field:scalar_stats(q['head_terms'][which,j]) for j,field in enumerate(a.FEATURE_NAMES)}};close(s,expected);statcells+=6
 for scope,q in rows.items():
  pop=result['populations'][scope];close(pop['rows'],len(q['row_id']));close(pop['corrected_accepted'],int(q[PRIMARY+'__accepted'].sum()));check_cell(q,np.arange(len(q['row_id'])),pop['all_rows'])
  for field in ['reference_age_ns','imu_header_age_ns','imu_arrival_age_ns']:close(pop[field.replace('_ns','_max_ns')],int(max(q[field])))
  names=sorted(set(q['window_id'].tolist()))
  if set(pop['windows'])!=set(names):raise ValueError('missing diagnostic windows')
  for name in names:
   ids=np.array(sorted(np.flatnonzero(q['window_id']==name),key=lambda i:int(q['time_ns'][i])));sel=ids[q[PRIMARY+'__accepted'][ids]];common=np.array([i for i in ids if all(q[n+'__accepted'][i] for n in METHODS)],dtype=int);w=pop['windows'][name];check_cell(q,ids,w['all_supported_or_ready_rows']);check_cell(q,sel,w['corrected_accepted_rows'],True);check_cell(q,common,w['all_three_common_accepted_rows']);origin=int(q['time_ns'][ids[0]])
   speed=np.array([0 if y<1 else 1 if y<5 else 2 if y<10 else 3 for y in q['target_mps'][sel]])
   for i,label in enumerate(a.m.d.SPEED_NAMES):check_cell(q,sel[speed==i],w['by_reference_speed'][label])
   buckets={str((int(t)-origin)//20_000_000_000) for t in q['time_ns'][ids]}
   if set(w['by_elapsed_20s'])!=buckets:raise ValueError('missing or changed time buckets')
   for b in buckets:check_cell(q,np.array([i for i in sel if (int(q['time_ns'][i])-origin)//20_000_000_000==int(b)],dtype=int),w['by_elapsed_20s'][b])
   for run in scalar_runs(q['time_ns'][sel],abs(q[PRIMARY+'__prediction_mps'][sel]-q['target_mps'][sel])):expectedrecords.append({'population':scope,'window_id':name,'first_row_id':int(q['row_id'][sel[run['first_local_index']]]),'last_row_id':int(q['row_id'][sel[run['last_local_index']]]),**run})
 close(records,expectedrecords);parent=read(a.ROOT/'results.json');close(result['remaining_failed_fit_windows'],parent['route']['diagnostic_failures'][PRIMARY]['windows']);close(result['remaining_failed_old_development_blocks'],[n for n,v in parent['full_development']['methods'][PRIMARY]['previous_supported_blocks'].items() if v['accepted']['p95_mps']>.5])
 for n,v in result.items():
  if n in ['root_cause_established','reference_independence_established','new_model_fitted','thresholds_changed','prediction_or_timestamp_changed','reserved_final_opened','promotion_allowed','witness_enabled'] and v is not False:raise ValueError('unsupported diagnostic claim '+n)
 if result['decision']!='REMAINING_FAILURE_DIAGNOSIS_COMPLETE_NO_PROMOTION' or result['associations_are_not_causes'] is not True:raise ValueError('unsupported decision')
 suite=unittest.defaultTestLoader.loadTestsFromModule(test_remaining);buffer=io.StringIO();tests=unittest.TextTestRunner(stream=buffer).run(suite)
 if not tests.wasSuccessful():raise ValueError(buffer.getvalue())
 print(f'REMAINING FAILURE AUDIT: PASS ({tests.testsRun} tests;{clocks} independent source-clock rows;{sum(len(v["row_id"]) for v in rows.values())} contribution/support rows;{metriccells} scalar metric cells;{statcells} scalar statistic cells;{len(records)} error runs)',flush=True)
 print(f'Available preceding payloads verified: {available}/{len(prior)-2}; unavailable historical files not claimed as reverified.',flush=True)
 print('Remaining failures:',result['remaining_failed_fit_windows'],result['remaining_failed_old_development_blocks'],flush=True);print(result['decision']+'; MODEL AND THRESHOLDS UNCHANGED; RESERVED FINAL CLOSED; WITNESS DISABLED.',flush=True)
if __name__=='__main__':audit()
