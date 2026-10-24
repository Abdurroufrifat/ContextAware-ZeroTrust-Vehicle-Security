"""Independent raw-source, feature, tree, metric and serving replay; no fitting."""
import bisect,io,math,unittest
from multisource_core import *
import verify_fusion as vf
import test_imu,test_multisource
close=sa.close
def compare_replay(actual,expected):
 actual=np.asarray(actual);expected=np.asarray(expected)
 if expected.dtype.kind in 'iub':
  if actual.dtype.kind not in ('b' if expected.dtype.kind=='b' else 'iu'):raise AssertionError('integer/boolean replay must retain integer/boolean dtype')
  np.testing.assert_array_equal(actual,expected)
 else:np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-9,equal_nan=True)
def validate_claims(result):
 if result.get('decision')!='MULTISOURCE_IMU_COMPARISON_COMPLETE_NO_PROMOTION':raise ValueError('unexpected multi-source decision/promotion')
 for name in ['thresholds_changed','production_policy_changed','reserved_final_opened','witness_enabled','promotion_allowed','journal_readiness_established']:
  if result.get(name) is not False:raise ValueError('unsupported multi-source claim: '+name)
 if result.get('full_urban01_night_development_reused') is not True or result.get('all_eight_models_locked_before_new_scoring') is not True or result.get('full_development',{}).get('no_formerly_rejected_source_endpoint_revived') is not True:raise ValueError('invalid reuse/lock/rejection-preservation claim')
def scalar_imu(source,time):
 arrivals=source['bag_ns'].tolist();values=[]
 for offset in range(-127,1):
  t=int(time)+offset*50_000_000;i=bisect.bisect_right(arrivals,t)-1;good=i>=0 and source['usable'][i] and 0<=t-int(source['header_ns'][i])<=100_000_000 and 0<=t-int(source['bag_ns'][i])<=100_000_000
  values.append((bool(good),float(source['acceleration_norm'][i]) if good else 0.,float(source['angular_velocity_norm'][i]) if good else 0.))
 result=[values[-1][1],values[-1][2],float(values[-1][0])]
 for width in [5,11,21,41,81,128]:
  selected=[(j-(width-1),a,b) for j,(ok,a,b) in enumerate(values[-width:]) if ok];n=len(selected)
  for channel in [1,2]:
   if not n:result.extend([0.,0.,0.]);continue
   mean=math.fsum(row[channel] for row in selected)/n;sd=math.sqrt(math.fsum((row[channel]-mean)**2 for row in selected)/n);tm=math.fsum(row[0]*.05 for row in selected)/n;den=math.fsum((row[0]*.05-tm)**2 for row in selected);slope=math.fsum((row[0]*.05-tm)*(row[channel]-mean) for row in selected)/den if den else 0.;result.extend([mean,sd,slope])
  result.append(n/width)
 return np.array(result)
def feature_replay(arrays,members):
 sources={};probes=0;full_index=read(c.MASK/'frame_index.json')
 for key,route in sa.l.ROUTES.items():
  raw,source=sa.load(key);sources[route]=source
  if key=='Urban01_night':ids=np.arange(len(full_index));times=np.array([v['time_ns'] for v in full_index],dtype=np.int64);expected=arrays['full_imu'];count=129
  else:ids=np.array([i for i,v in enumerate(members) if v['route']==route]);times=np.array([members[i]['time_ns'] for i in ids],dtype=np.int64);expected=arrays['fit_imu'][ids];count=33
  rebuilt,current=im.features(source,times);np.testing.assert_allclose(rebuilt,expected,rtol=0,atol=1e-10)
  if key=='Urban01_night':
   for name,value in current.items():np.testing.assert_array_equal(value,arrays['full_current__'+name])
  for j in np.unique(np.linspace(0,len(times)-1,count,dtype=int)):
   np.testing.assert_allclose(scalar_imu(source,int(times[j])),expected[j],rtol=0,atol=1e-10);probes+=1
 return sources,probes
def metric_check(y,p,expected):
 rebuilt=vf.independent_metric(y,p)
 for key,value in expected.items():
  if key!='missing_truth':close(rebuilt[key],value)
def independent_full(result,arrays,index,ready,peer,source,current):
 events=np.array([v['time_ns'] for v in index],dtype=np.int64);targets=read(RESEARCH/'vehicle_availability_audit/request_reference.json');times=np.array([v['time_ns'] for v in targets],dtype=np.int64);yq=np.array([np.nan if v['reference_mps'] is None else v['reference_mps'] for v in targets]);raw=read(RESEARCH/'vehicle_continuous_radar/reference_targets.json');yr=np.array([np.nan if v['reference_mps'] is None else v['reference_mps'] for v in raw]);peerfull=np.full(len(index),np.nan);peerfull[ready]=peer;segments=(times-times[0])//300_000_000_000;full=result['full_development'];steps=cells=0
 lookup={int(t):i for i,t in enumerate(events)};blocks={}
 for v in read(c.OLD/'prepared/calibration/index.json'):
  if v['ready']:blocks.setdefault(v['window_id'],[]).append(lookup[v['time_ns']])
 def score(truth,pred,mask,expected):
  nonlocal cells
  known=mask&np.isfinite(truth);metric_check(truth[known],pred[known],expected)
  if 'missing_truth' in expected:close(expected['missing_truth'],int(np.sum(mask&~np.isfinite(truth))))
  cells+=1
 def account(truth,pred,keep,raw_count,expected):
  nonlocal cells
  known=np.isfinite(truth);counts={'raw_endpoints':raw_count,'features_ready':len(truth),'unready_raw':raw_count-len(truth),'ready_reference_missing':int((~known).sum()),'reference_coverage_on_ready':float(known.mean()),'usable_total':int(keep.sum()),'usable_reference_missing':int(np.sum(keep&~known)),'quality_withheld':int((~keep).sum()),'whole_acceptance_fraction':float(keep.sum()/raw_count)}
  for k,v in counts.items():close(expected[k],v)
  for name,mask in [('all_ready_reference_scored',known),('accepted',keep&known),('withheld',~keep&known)]:metric_check(truth[mask],pred[mask],expected[name]);cells+=1
 def request_cell(name,mask,expected):
  st=arrays[name+'__request__status'][mask];truth=yq[mask];pred=arrays[name+'__request__prediction_mps'][mask];good=st==0;close(expected['requests'],len(st));close(expected['usable_fraction'],float(good.mean()));close(expected['reference_missing_all_requests'],int(np.sum(~np.isfinite(truth))))
  for code,label in av.STATUS.items():close(expected['status_counts'][label],int(np.sum(st==code)))
  score(truth,pred,good,expected['accepted'])
 for name in METHODS:
  values=arrays[name+'__ready_prediction_mps'];pred=np.full(len(index),np.nan);pred[ready]=values;keep=np.array([bool(ok) and math.isfinite(float(a)) and math.isfinite(float(b)) and a>=0 and b>=0 and abs(a-b)<=1. and (name=='original_context' or current['status'][j]==0) for j,(a,b,ok) in enumerate(zip(pred,peerfull,source))],dtype=bool);np.testing.assert_array_equal(keep,arrays[name+'__quality']);clock={k:[] for k in ['request_times_ns','latest_endpoint','age_ns','status','prediction_mps','timing_prediction_mps']};bound=[];i=-1;blocked=0
  for t in times:
   while i+1<len(events) and events[i+1]<=t:i+=1
   age=int(t)-int(events[i]) if i>=0 else -1;status=1 if i<0 else 2 if age>100_000_000 else 3 if not ready[i] else 4 if not keep[i] else 0;valid=True
   if name!='original_context':
    valid=i>=0 and current['status'][i]==0 and t>=current['endpoint_ns'][i] and 0<=int(t)-int(current['bag_ns'][i])<=100_000_000 and 0<=int(t)-int(current['header_ns'][i])<=100_000_000
    if status==0 and not valid:status=4;blocked+=1
    bound.append(valid)
   pv=float(pred[i]) if status==0 and valid else np.nan;timing=float(pred[i]) if status in (0,4) and valid else np.nan
   for k,v in zip(clock,[int(t),i,age,status,pv,timing]):clock[k].append(v)
  for k,v in clock.items():compare_replay(arrays[name+'__request__'+k],np.array(v))
  if name!='original_context':np.testing.assert_array_equal(np.array(bound),arrays[name+'__request__imu_binding_fresh'])
  method=full['methods'][name];close(method['imu_binding_additional_rejected_requests'],blocked);close(method['additional_rejected_endpoints'],int(np.sum(source&~keep)));account(yr[ready],values,keep[ready],len(index),method['endpoints'])
  for key,ids in blocks.items():account(yr[ids],pred[ids],keep[ids],len(ids),method['previous_supported_blocks'][key])
  score(yr,pred,ready,method['ungated_ready_endpoints']);request_cell(name,np.ones(len(times),bool),method['requests']['overall'])
  for key,v in method['requests']['segments_300s'].items():request_cell(name,segments==int(key),v)
  requests=[method['requests']['overall']]+list(method['requests']['segments_300s'].values());oldblocks=list(method['previous_supported_blocks'].values());flags=[all(v['usable_fraction']>=.95 for v in requests),all(v['accepted']['count']>0 and v['accepted']['count']/(v['accepted']['count']+v['accepted']['missing_truth'])>=.95 for v in requests),all(v['accepted']['count']>0 and v['accepted']['p95_mps']<=.5 for v in requests),method['requests']['overall']['accepted']['count']>0 and method['requests']['overall']['accepted']['above_2_mps']==0,all(v['whole_acceptance_fraction']>=.95 for v in oldblocks),all(v['accepted']['count']>0 and v['accepted']['p95_mps']<=.5 for v in oldblocks)];close(dict(zip(full['gates'][name],flags)),full['gates'][name]);close(full['all_gates_pass'][name],all(flags));steps+=len(times)
 for name in NEW_METHODS:
  common=(arrays['original_context__request__status']==0)&(arrays[name+'__request__status']==0);paired=full['common_accepted_requests'][name]
  for mask,cell in [(common,paired['overall'])]+[(common&(segments==int(k)),v) for k,v in paired['segments_300s'].items()]:
   close(cell['shared_requests'],int(mask.sum()))
   for method in ['original_context',name]:score(yq,arrays[method+'__request__prediction_mps'],mask,cell['models'][method])
 return steps,cells
def audit():
 check_models();result=read(ROOT/'results.json');validate_claims(result);prior=read(ROOT/'prior_snapshot_manifest.json');available=0
 for name,h in prior.items():
  if name in ['FINAL_PROJECT_README.md','scripts/check_coact_project.py']:continue
  path=(RESEARCH.parent/name).resolve()
  if not path.is_relative_to(RESEARCH.parent):raise ValueError('unsafe prior project path')
  if path.is_file():
   if digest(path)!=h:raise ValueError('preceding payload changed: '+name)
   available+=1
 if (ROOT/'sha256.json').exists():
  for name,h in read(ROOT/'sha256.json').items():
   path=(ROOT/name).resolve()
   if not path.is_relative_to(ROOT) or digest(path)!=h:raise ValueError('multi-source artifact changed: '+name)
 audited=sa.audit_sources();close(audited,read(ROOT/'source_audit.json'));arrays=cache();data,members,groups,day,support=r.load_data();sources,featureprobes=feature_replay(arrays,members);base=c.features(data['X'],np.zeros(len(members)),np.zeros(len(members)));trace=read(ROOT/'training_trace.json');p=read(ROOT/'protocol.json');fits=treeprobes=0;coverage=np.zeros(len(groups),int)
 if len(trace['runs'])!=8 or [(v['scope'],v['method']) for v in trace['runs']]!=[(s,n) for s in SCOPES for n in NEW_METHODS]:raise ValueError('missing/selected training scopes')
 for scope in SCOPES:
  cal=read(ROOT/'models'/scope/'calibration.json');close(cal,f.calibration(data,groups,day,scope));ids=np.array(cal['train_ids']);held=np.array(cal['held_ids'],dtype=int)
  if scope!='all_fit':
   coverage[held]+=1
   if np.intersect1d(ids,held).size or set(groups[ids])&set(groups[held]):raise ValueError('held group leakage')
   changed={**data,'y':data['y'].copy(),'median_feature':data['median_feature'].copy()};changed['y'][held]+=12345;changed['median_feature'][held]+=1000;close(r.calibrate(changed,groups,day,scope),cal)
  oracle,counts,anchorids=vf.independent_anchor(data,data['median_feature'],groups,day,scope if scope!='all_fit' else 'NOT_A_FITTING_ROUTE');close(oracle,cal['anchor']);close(counts,cal['bin_counts']);np.testing.assert_array_equal(anchorids,cal['calibration_ids']);counts=np.bincount(np.searchsorted([1.,5.,10.],data['y'][anchorids],side='right'),minlength=4);oraclew=np.tile((len(anchorids)/(4*counts[np.searchsorted([1.,5.,10.],data['y'][ids],side='right')]))*.5,2)
  for method in NEW_METHODS:
   run=next(v for v in trace['runs'] if v['scope']==scope and v['method']==method);x,y,w,a,aug=training_arrays(data,base,arrays['fit_imu'],cal,method);np.testing.assert_array_equal(w,oraclew);originalanchor=c.e.affine(cal['anchor'],data['median_feature'][ids]);np.testing.assert_array_equal(a,np.tile(originalanchor if method=='radar_imu' else np.zeros(len(ids)),2));np.testing.assert_array_equal(y,np.tile(data['y'][ids]-originalanchor if method=='radar_imu' else data['y'][ids],2))
   if method=='radar_imu':f.check_parent_inputs(scope,x,y,w);np.testing.assert_array_equal(x[:len(ids),75:],x[len(ids):,75:])
   else:np.testing.assert_array_equal(x[:len(ids)],x[len(ids):]);np.testing.assert_array_equal(x[:len(ids)],arrays['fit_imu'][ids])
   for v,key in [(x,'feature_sha256'),(y,'target_sha256'),(w,'weight_sha256'),(a,'used_anchor_sha256'),(aug[:,3],'mask_sha256')]:
    if g.array_digest(v)!=run[key]:raise ValueError('frozen training array changed: '+key)
   if run['train_samples']!=len(ids) or run['held_samples']!=len(held) or run['feature_count']!=len(FEATURE_NAMES[method]) or run['iterations']!=200 or any(run['parameters'].get(k)!=v for k,v in p['model'].items()):raise ValueError('training budget/parameters differ')
   model=read(ROOT/'models'/scope/(method+'.json'))
   if model['feature_names']!=FEATURE_NAMES[method] or len(model['trees'])!=200:raise ValueError('model feature/tree schema differs')
   pred=c.predict(model,x,a)
   with np.load(ROOT/'models'/scope/(method+'_export.npz'),allow_pickle=False) as z:np.testing.assert_allclose(pred,z['native'],rtol=0,atol=1e-8);np.testing.assert_allclose(pred,z['portable'],rtol=0,atol=1e-9)
   fits+=len(x)
   for j in np.unique(np.linspace(0,len(ids)-1,17,dtype=int)):
    at=len(ids)+j;row=scalar_imu(sources[members[ids[j]]['route']],members[ids[j]]['time_ns'])
    if method=='radar_imu':row=np.concatenate([vf.independent_features(aug[j],a[at],data['median_feature'][ids[j]]),row])
    np.testing.assert_allclose(row,x[at],rtol=0,atol=1e-10);np.testing.assert_allclose(vf.scalar_predict(model,row,a[at]),pred[at],rtol=0,atol=1e-9);treeprobes+=1
   if scope!='all_fit':
    anchor=c.e.affine(cal['anchor'],data['median_feature'][held]) if method=='radar_imu' else np.zeros(len(held));xf=assemble(base[held],arrays['fit_imu'][held],anchor,data['median_feature'][held],method);pv=c.predict(model,xf,anchor)
    for j in np.unique(np.linspace(0,len(held)-1,65,dtype=int)):
     row=scalar_imu(sources[members[held[j]]['route']],members[held[j]]['time_ns'])
     if method=='radar_imu':row=np.concatenate([vf.independent_features(data['X'][held[j]],anchor[j],data['median_feature'][held[j]]),row])
     np.testing.assert_allclose(row,xf[j],rtol=0,atol=1e-10);np.testing.assert_allclose(vf.scalar_predict(model,row,anchor[j]),pv[j],rtol=0,atol=1e-9);treeprobes+=1
 if not np.all(coverage==1):raise ValueError('held rows not scored exactly once')
 from score_multisource import compute
 rebuilt,route,fullarrays=compute();close(rebuilt,result)
 for name,expected in [('oof_predictions.npz',route),('development_replay.npz',fullarrays)]:
  with np.load(ROOT/name,allow_pickle=False) as z:
   if set(z.files)!=set(expected):raise ValueError('replay array schemas differ')
   for key,value in expected.items():compare_replay(z[key],value)
 with np.load(r.ROOT/'oof_predictions.npz',allow_pickle=False) as z:np.testing.assert_array_equal(route['original_context__prediction_mps'],z['prediction_mps'])
 cells=[(np.ones(len(members),bool),result['route']['pooled'])]
 for category,key in [('by_group','group'),('by_recording','route'),('by_window','window_id')]:
  labels=np.array([v[key] for v in members]);cells.extend((labels==label,cell) for label,cell in result['route'][category].items())
 bins=np.searchsorted([1.,5.,10.],data['y'],side='right');cells.extend((bins==i,result['route']['by_reference_speed'][label]) for i,label in enumerate(d.SPEED_NAMES));cells.extend([(day,result['route']['by_day_night']['day']),(~day,result['route']['by_day_night']['night'])])
 for mask,cell in cells:
  close(cell['count'],int(mask.sum()))
  for method in METHODS:metric_check(data['y'][mask],route[method+'__prediction_mps'][mask],cell['models'][method])
 index,ready,x,median,anchor,peer,source,old=f.load_full();fullbase=c.features(x,anchor,median)
 with np.load(c.ROOT/'replay.npz',allow_pickle=False) as z:np.testing.assert_allclose(fullbase,z['context_features'],rtol=0,atol=1e-10)
 np.testing.assert_allclose(c.predict(read(c.ROOT/'model.json'),fullbase,anchor),old,rtol=0,atol=1e-9)
 for method in NEW_METHODS:
  model=read(ROOT/'models/all_fit'/(method+'.json'));used=anchor if method=='radar_imu' else np.zeros(len(anchor));values=fullarrays[method+'__ready_prediction_mps']
  for j in np.unique(np.linspace(0,len(x)-1,129,dtype=int)):
   rawid=np.flatnonzero(ready)[j];row=scalar_imu(sources[sa.l.ROUTES['Urban01_night']],index[rawid]['time_ns'])
   if method=='radar_imu':row=np.concatenate([vf.independent_features(x[j],anchor[j],median[j]),row])
   np.testing.assert_allclose(vf.scalar_predict(model,row,used[j]),values[j],rtol=0,atol=1e-9);treeprobes+=1
 current={k:arrays['full_current__'+k] for k in ['endpoint_ns','status','bag_ns','header_ns']};steps,fullcells=independent_full(result,fullarrays,index,ready,peer,source,current);parent=read(c.ROOT/'results.json');original=result['full_development']['methods']['original_context'];close({k:v for k,v in original.items() if k not in ['ungated_ready_endpoints','imu_binding_additional_rejected_requests']},parent['methods']['context_boosting']);close(result['full_development']['gates']['original_context'],parent['gates'])
 buffer=io.StringIO();suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(test_imu),unittest.defaultTestLoader.loadTestsFromModule(test_multisource)]);tests=unittest.TextTestRunner(stream=buffer).run(suite)
 if not tests.wasSuccessful():raise ValueError(buffer.getvalue())
 print(f'MULTI-SOURCE IMU AUDIT: PASS ({tests.testsRun} tests;8 fixed models;{fits} native/export comparisons;{featureprobes} independent IMU feature probes;{treeprobes} independent feature/tree probes;{len(cells)*3} route metric cells;{fullcells} full metric cells;{steps} scalar request steps)',flush=True)
 print(f'Available preceding payloads verified: {available}/{len(prior)-2}; unavailable historical files not claimed as reverified.',flush=True)
 for name in METHODS:print(name,'route P95',result['route']['pooled']['models'][name]['p95_mps'],'failed windows',len(result['route']['diagnostic_failures'][name]['windows']),'full gates',result['full_development']['all_gates_pass'][name],flush=True)
 print(result['decision']+'; INSPECTED DEVELOPMENT REUSED; RESERVED FINAL CLOSED; WITNESS DISABLED.',flush=True)
if __name__=='__main__':audit()
