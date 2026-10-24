"""Portable audit: independent signed features, ridge equations, metrics and clocks."""
import bisect,io,math,unittest
from axis_core import *
import verify_multisource as vm
import test_axis
close=m.sa.close
def scalar_axes(source,time):
 arrivals=source['bag_ns'].tolist();rows=[]
 for offset in range(-10,1):
  t=int(time)+offset*50_000_000;i=bisect.bisect_right(arrivals,t)-1
  if i>=0 and source['usable'][i] and 0<=t-int(source['header_ns'][i])<=100_000_000 and 0<=t-int(source['bag_ns'][i])<=100_000_000:rows.append([*source['acceleration'][i],*source['angular_velocity'][i]])
 return np.array([math.fsum(row[j] for row in rows)/len(rows) if rows else 0. for j in range(6)]),len(rows)/11
def feature_audit(values,members):
 probes=rawprobes=0;index=read(m.c.MASK/'frame_index.json')
 for key,route in m.sa.l.ROUTES.items():
  raw,source=m.sa.load(key)
  for i in np.unique(np.linspace(0,len(source['bag_ns'])-1,65,dtype=int)):
   decoded=m.sa.scalar_record(raw['payload_bytes'][i,:int(raw['payload_length'][i])].tobytes(),int(source['bag_ns'][i]));np.testing.assert_allclose(source['acceleration'][i],decoded[3],rtol=0,atol=1e-10);np.testing.assert_array_equal(source['angular_velocity'][i],decoded[4]);rawprobes+=1
  if key=='Urban01_night':times=np.array([v['time_ns'] for v in index],dtype=np.int64);x=values['full_axes'];coverage=values['full_coverage'];count=129
  else:
   ids=np.array([i for i,v in enumerate(members) if v['route']==route]);times=np.array([members[i]['time_ns'] for i in ids],dtype=np.int64);x=values['fit_axes'][ids];coverage=values['fit_coverage'][ids];count=33
  rebuilt,cov=axis_features(source,times);np.testing.assert_allclose(rebuilt,x,rtol=0,atol=1e-10);np.testing.assert_array_equal(cov,coverage)
  for i in np.unique(np.linspace(0,len(times)-1,count,dtype=int)):
   row,cv=scalar_axes(source,int(times[i]));np.testing.assert_allclose(row,x[i],rtol=0,atol=1e-10);close(cv,float(coverage[i]));probes+=1
 return probes,rawprobes
def reverse(value):
 mapping={v:k for k,v in SLOT_NAMES.items()}
 if isinstance(value,dict):return {mapping.get(k,k):reverse(v) for k,v in value.items()}
 if isinstance(value,list):return [reverse(v) for v in value]
 return value
def validate_claims(result):
 if result.get('decision')!='SIGNED_AXIS_CORRECTION_COMPLETE_NO_PROMOTION':raise ValueError('unsupported correction decision')
 for n in ['thresholds_changed','promotion_allowed','production_policy_changed','reserved_final_opened','witness_enabled','journal_readiness_established']:
  if result.get(n) is not False:raise ValueError('unsupported claim '+n)
 for n in ['parent_models_unchanged','all_four_heads_locked_before_new_scoring','full_development_reused']:
  if result.get(n) is not True:raise ValueError('missing claim '+n)
 if result['full_development']['no_formerly_rejected_source_endpoint_revived'] is not True:raise ValueError('revived source rejection')
def audit():
 check_heads();result=read(ROOT/'results.json');validate_claims(result);prior=read(ROOT/'prior_snapshot_manifest.json');available=0
 for name,h in prior.items():
  if name in ['FINAL_PROJECT_README.md','scripts/check_coact_project.py']:continue
  path=(RESEARCH.parent/name).resolve()
  if not path.is_relative_to(RESEARCH.parent):raise ValueError('unsafe prior path')
  if path.is_file():
   if digest(path)!=h:raise ValueError('preceding payload changed '+name)
   available+=1
 if (ROOT/'sha256.json').exists():
  for name,h in read(ROOT/'sha256.json').items():
   path=(ROOT/name).resolve()
   if not path.is_relative_to(ROOT) or digest(path)!=h:raise ValueError('correction artifact changed '+name)
 protocol=read(ROOT/'protocol.json');required=[m.ROOT/'oof_predictions.npz',m.ROOT/'development_replay.npz',m.ROOT/'results.json',ROOT/'paired_error_diagnosis.json']
 for path in required:
  import os
  if os.path.relpath(path,ROOT).replace(os.sep,'/') not in protocol['input_sha256']:raise ValueError('direct scoring input not frozen')
 close(diagnosis(),read(ROOT/'paired_error_diagnosis.json'));values=cache();data,members,groups,day,support=m.r.load_data();probes,rawprobes=feature_audit(values,members);trace=read(ROOT/'training_trace.json');coverage=np.zeros(len(members),int);trainingrows=0;independent_heads={}
 if [v['scope'] for v in trace['runs']]!=list(SCOPES):raise ValueError('missing/selected correction scopes')
 for scope,run in zip(SCOPES,trace['runs']):
  x,y,w,ids,native=training_inputs(scope,data,values['fit_axes']);cal=read(m.ROOT/'models'/scope/'calibration.json');held=np.array(cal['held_ids'],dtype=int);np.testing.assert_array_equal(ids,run['training_ids'])
  if scope!='all_fit':
   coverage[held]+=1
   if np.intersect1d(ids,held).size or set(groups[ids])&set(groups[held]):raise ValueError('held-group leakage')
   yy=data['y'].copy();xx=values['fit_axes'].copy();yy[held]+=123456;xx[held]+=98765
   changed=correction_inputs(xx,yy,ids,native)
   for a,b in zip(changed,[x,y]):np.testing.assert_array_equal(a,b)
  _,counts,anchorids=vm.vf.independent_anchor(data,data['median_feature'],groups,day,scope if scope!='all_fit' else 'NOT_A_FITTING_ROUTE');bins=np.searchsorted([1.,5.,10.],data['y'][ids],side='right');counts=np.bincount(np.searchsorted([1.,5.,10.],data['y'][anchorids],side='right'),minlength=4);np.testing.assert_array_equal(w,np.tile(len(anchorids)/(4*counts[bins])*.5,2))
  np.testing.assert_array_equal(x[:len(ids)],x[len(ids):]);np.testing.assert_array_equal(y,np.tile(data['y'][ids],2)-native)
  for arr,key in [(x,'feature_sha256'),(y,'residual_sha256'),(w,'weight_sha256'),(native,'parent_native_sha256')]:
   if m.g.array_digest(arr)!=run[key]:raise ValueError('training trace differs '+key)
  if run['training_examples']!=len(x) or run['alpha']!=1. or run['feature_count']!=6:raise ValueError('training scope differs')
  mean=np.average(x,axis=0,weights=w);scale=np.sqrt(np.average((x-mean)**2,axis=0,weights=w));scale[scale==0]=1.;z=(x-mean)/scale;q=np.column_stack([np.ones(len(x)),z]);penalty=np.eye(7);penalty[0,0]=0.;beta=np.linalg.solve(q.T@(q*w[:,None])+penalty,q.T@(w*y));head=read(ROOT/'heads'/scope/'head.json');np.testing.assert_allclose(head['mean'],mean,rtol=0,atol=1e-10);np.testing.assert_allclose(head['scale'],scale,rtol=0,atol=1e-10);np.testing.assert_allclose([head['intercept'],*head['coefficient']],beta,rtol=0,atol=1e-9)
  with np.load(ROOT/'heads'/scope/'training_check.npz',allow_pickle=False) as check:np.testing.assert_array_equal(check['parent_native'],native);np.testing.assert_allclose(check['correction'],q@beta,rtol=0,atol=1e-9)
  independent_heads[scope]=(mean,scale,beta)
  trainingrows+=len(x)
 if not np.all(coverage==1):raise ValueError('held rows not covered once')
 from score_axis import compute
 rebuilt,route,full=compute();close(rebuilt,result)
 for name,expected in [('oof_predictions.npz',route),('development_replay.npz',full)]:
  with np.load(ROOT/name,allow_pickle=False) as saved:
   if set(saved.files)!=set(expected):raise ValueError('prediction array schema differs')
   for key,value in expected.items():vm.compare_replay(saved[key],value)
 # Apply independently solved coefficients to exact held/full feature indices.
 with np.load(m.ROOT/'oof_predictions.npz',allow_pickle=False) as z:parent_oof=z['radar_imu__prediction_mps'].copy()
 oracle_oof=np.full(len(members),np.nan)
 for scope in m.r.GROUPS:
  ids=np.array(read(m.ROOT/'models'/scope/'calibration.json')['held_ids'],dtype=int);mean,scale,beta=independent_heads[scope];z=(values['fit_axes'][ids]-mean)/scale;correction=np.array([math.fsum([float(beta[0]),*[float(a)*float(b) for a,b in zip(row,beta[1:])]]) for row in z]);oracle_oof[ids]=np.clip(parent_oof[ids]+correction,0.,None)
 np.testing.assert_allclose(route['signed_axis_corrected__prediction_mps'],oracle_oof,rtol=0,atol=1e-9)
 index,ready,*_=m.f.load_full();mean,scale,beta=independent_heads['all_fit'];z=(values['full_axes'][np.flatnonzero(ready)]-mean)/scale;correction=np.array([math.fsum([float(beta[0]),*[float(a)*float(b) for a,b in zip(row,beta[1:])]]) for row in z])
 with np.load(m.ROOT/'development_replay.npz',allow_pickle=False) as saved:oracle_full=np.clip(saved['radar_imu__ready_prediction_mps']+correction,0.,None)
 np.testing.assert_allclose(full['signed_axis_corrected__ready_prediction_mps'],oracle_full,rtol=0,atol=1e-9)
 parent=read(m.ROOT/'results.json')
 with np.load(m.ROOT/'oof_predictions.npz',allow_pickle=False) as saved:
  for old,new in [('original_context','original_context'),('radar_imu','uncorrected_radar_imu')]:np.testing.assert_array_equal(saved[old+'__prediction_mps'],route[new+'__prediction_mps'])
 with np.load(m.ROOT/'development_replay.npz',allow_pickle=False) as saved:
  for key in saved.files:
   old,suffix=key.split('__',1)
   if old in ['original_context','radar_imu']:vm.compare_replay(saved[key],full[SLOT_NAMES[old]+'__'+suffix])
 cells=[(np.ones(len(members),bool),result['route']['pooled'])]
 for category,key in [('by_group','group'),('by_recording','route'),('by_window','window_id')]:
  labels=np.array([v[key] for v in members]);cells.extend((labels==label,cell) for label,cell in result['route'][category].items())
 bins=np.searchsorted([1.,5.,10.],data['y'],side='right');cells.extend((bins==i,result['route']['by_reference_speed'][label]) for i,label in enumerate(m.d.SPEED_NAMES));cells.extend([(day,result['route']['by_day_night']['day']),(~day,result['route']['by_day_night']['night'])])
 for mask,cell in cells:
  close(cell['count'],int(mask.sum()))
  for method in METHODS:vm.metric_check(data['y'][mask],route[method+'__prediction_mps'][mask],cell['models'][method])
 # Rebuild the oracle with scalar interval geometry, independent of oracle_errors.
 y=route['target_mps'];a=route['original_context__prediction_mps'];b=route['uncorrected_radar_imu__prediction_mps'];floor=np.array([0. if min(p,q)<=v<=max(p,q) else min(abs(v-p),abs(v-q)) for v,p,q in zip(y,a,b)]);labels=np.array([v['window_id'] for v in members]);diag=read(ROOT/'paired_error_diagnosis.json')
 for key,cell in diag['by_window'].items():close(cell['oracle_interval_p95_lower_bound_mps'],float(np.quantile(floor[labels==key],.95)))
 index,ready,x,median,anchor,peer,source,old=m.f.load_full();current={k:m.cache()['full_current__'+k] for k in ['endpoint_ns','status','bag_ns','header_ns']};slotfull={}
 for key,value in full.items():prefix,suffix=key.split('__',1);slotfull[{v:k for k,v in SLOT_NAMES.items()}[prefix]+'__'+suffix]=value
 steps,fullcells=vm.independent_full(reverse(result),slotfull,index,ready,peer,source,current)
 for old,new in [('original_context','original_context'),('radar_imu','uncorrected_radar_imu')]:close(result['full_development']['methods'][new],parent['full_development']['methods'][old]);close(result['full_development']['gates'][new],parent['full_development']['gates'][old])
 times=full['signed_axis_corrected__request__request_times_ns'];segments=(times-times[0])//300_000_000_000;truth=np.array([np.nan if v['reference_mps'] is None else v['reference_mps'] for v in read(RESEARCH/'vehicle_availability_audit/request_reference.json')]);pair=['uncorrected_radar_imu','signed_axis_corrected'];common=(full[pair[0]+'__request__status']==0)&(full[pair[1]+'__request__status']==0);cells2=result['full_development']['correction_common_accepted_requests']
 for mask,cell in [(common,cells2['overall'])]+[(common&(segments==int(k)),v) for k,v in cells2['segments_300s'].items()]:
  close(cell['shared_requests'],int(mask.sum()))
  for method in pair:vm.metric_check(truth[mask&np.isfinite(truth)],full[method+'__request__prediction_mps'][mask&np.isfinite(truth)],cell['models'][method]);fullcells+=1
 buffer=io.StringIO();suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromModule(v) for v in [test_axis,vm.test_imu,vm.test_multisource]);tests=unittest.TextTestRunner(stream=buffer).run(suite)
 if not tests.wasSuccessful():raise ValueError(buffer.getvalue())
 print(f'SIGNED-AXIS AUDIT: PASS ({tests.testsRun} tests;4 locked heads;{trainingrows} ridge-equation predictions;{probes} scalar feature probes;{rawprobes} raw-vector probes;{len(cells)*3} route metric cells;{fullcells} full metric cells;{steps} scalar request steps)',flush=True)
 print(f'Available preceding payloads verified: {available}/{len(prior)-2}; unavailable historical files not claimed as reverified.',flush=True)
 for name in METHODS:print(name,'route P95',result['route']['pooled']['models'][name]['p95_mps'],'failed windows',len(result['route']['diagnostic_failures'][name]['windows']),'full gates',result['full_development']['all_gates_pass'][name],flush=True)
 print(result['decision']+'; INSPECTED DEVELOPMENT REUSED; RESERVED FINAL CLOSED; WITNESS DISABLED.',flush=True)
if __name__=='__main__':audit()
