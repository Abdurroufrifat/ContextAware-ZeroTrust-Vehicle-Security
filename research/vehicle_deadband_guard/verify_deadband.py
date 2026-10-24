"""Separate calibration quantiles, scalar anchor state, histories, inference and metrics."""
import bisect,io,math,unittest,zipfile
import numpy as np
from deadband_core import ROOT,RESEARCH,PROJECT,FITS,CASES,POLICIES,read,load,digest,check_parameters,f,g
import verify_guard as vg
vf=vg.vf;close=vf.close;stats=vf.stats
def reference_stamps(source,scale,epsilon=.001):
 out=np.full(len(source['bag_ns']),-1,np.int64);anchor=None;stamp=-1
 for i,t in enumerate(source['bag_ns']):
  row=tuple(float(v) for v in [*source['acceleration'][i],*source['angular_velocity'][i]])
  if not source['usable'][i] or not all(math.isfinite(v) for v in row):continue
  changed=anchor is None or any(abs(a-b)>epsilon*float(s) for a,b,s in zip(row,anchor,scale))
  if changed:anchor=row;stamp=int(t)
  out[i]=stamp
 return out
def quantile(v,q):
 v=sorted(map(float,v));at=(len(v)-1)*q;i=int(at);return v[i]+(v[min(i+1,len(v)-1)]-v[i])*(at-i)
def reference_source(data,case,scale):
 if case in f.CASES:return vf.source_copy(data['source_imu'],case,data['start'],data['end'])
 s=vf.source_copy(data['source_imu'],'frozen_values',data['start'],data['end']);amplitude=1e-12 if case=='tiny_jitter' else 2*.001*float(scale[0]);counter=0
 for i,t in enumerate(s['bag_ns']):
  if data['start']<=int(t)<data['end']:s['acceleration'][i,0]+=counter%2*amplitude;counter+=1
 s['acceleration_norm']=np.hypot.reduce(s['acceleration'],axis=1);return s
def verify_new_base(data,source,base):
 features,current=f.im.features(source,data['events']);a=vf.signed_features(source,data['events']);ready=data['ready'];parent=np.maximum(0.,data['anchor']+vf.tree_infer(data['parent_model'],np.column_stack([data['radar'],features[ready]])));candidate=np.maximum(0.,parent+vf.tree_infer(data['head'],np.column_stack([parent,a[ready]])));pred={'radar_only':data['radar_only'],'radar_imu_parent':parent,'nonlinear7':candidate};events=data['events'].tolist();arrivals=source['bag_ns'].tolist();feature_probes=0
 for i in np.unique(np.linspace(0,len(events)-1,75,dtype=int)):
  ff,aa=vf.scalar_features(source,arrivals,events[i]);np.testing.assert_allclose(ff,features[i],rtol=0,atol=1e-9);np.testing.assert_allclose(aa,a[i],rtol=0,atol=1e-9);feature_probes+=1
 for i,t in enumerate(events):
  j,status=vf.lookup(source,arrivals,t)
  if status!=int(base['current__status'][i]) or (arrivals[j] if j>=0 else -1)!=int(base['current__bag_ns'][i]) or (int(source['header_ns'][j]) if j>=0 else -1)!=int(base['current__header_ns'][i]):raise AssertionError('new-case source clock differs')
 for method,values in pred.items():
  np.testing.assert_allclose(values,base[method+'__ready_prediction_mps'],rtol=0,atol=1e-9);p=np.full(len(events),np.nan);p[ready]=values;keep=np.array([bool(s) and math.isfinite(float(v)) and math.isfinite(float(peer)) and v>=0 and peer>=0 and abs(v-peer)<=1. and (method=='radar_only' or int(current['status'][i])==0) for i,(v,peer,s) in enumerate(zip(p,data['peer'],data['source']))]);np.testing.assert_array_equal(keep,base[method+'__quality']);statuses=[];served=[];latest=[];ages=[]
  for k,t in enumerate(data['times']):
   t=int(t);i=bisect.bisect_right(events,t)-1;age=t-events[i] if i>=0 else -1;bound=i>=0 and int(current['status'][i])==0 and t>=events[i] and 0<=t-int(current['bag_ns'][i])<=100_000_000 and 0<=t-int(current['header_ns'][i])<=100_000_000
   if bound!=bool(base['binding'][k]):raise AssertionError('new-case source binding differs')
   status=1 if i<0 else 2 if age>100_000_000 else 3 if not ready[i] else 4
   if status==4 and keep[i] and (method=='radar_only' or bound):status=0
   statuses.append(status);served.append(float(p[i]) if status==0 else math.nan);latest.append(i);ages.append(age)
  for key,v in [('status',statuses),('prediction_mps',served),('latest_endpoint',latest),('age_ns',ages)]:np.testing.assert_allclose(v,base[method+'__request__'+key],rtol=0,atol=1e-9,equal_nan=True)
 return feature_probes
def verify():
 parameters=check_parameters();scale=np.array(parameters['scale']);fit=[];counts={}
 for key in FITS:
  s=f.im.decode_payloads(load(RESEARCH/'vehicle_livox_collection/runtime_export'/(key+'.npz')));v=np.column_stack([s['acceleration'],s['angular_velocity']]);v=v[s['usable']&np.isfinite(v).all(1)];fit.extend(v.tolist());counts[key]=len(v)
 lo=[quantile([v[j] for v in fit],.25) for j in range(6)];hi=[quantile([v[j] for v in fit],.75) for j in range(6)];raw=[b-a for a,b in zip(lo,hi)];close(lo,parameters['q25']);close(hi,parameters['q75']);close([s if s>0 else 1. for s in raw],parameters['scale']);close([i for i,s in enumerate(raw) if s==0],parameters['fallback_axes']);close(counts,parameters['fitting_record_counts']);close(len(fit),parameters['fitting_rows']);del fit
 data=f.inputs();result=read(ROOT/'results.json');events=data['events'];times=data['times'];segment=(times-times[0])//300_000_000_000;masks={'all':np.ones(len(times),bool),'before':times<data['start'],'during':(times>=data['start'])&(times<data['end']),'after':times>=data['end'],**{'segment_'+str(int(k)):segment==k for k in np.unique(segment)}};trials=cells=history_probes=feature_probes=0
 if set(result['cases'])!=set(CASES):raise AssertionError('case omitted')
 for case in CASES:
  source=reference_source(data,case,scale);base=load(RESEARCH/'vehicle_sensor_faults/replays'/(case+'.npz')) if case in f.CASES else load(ROOT/'replays'/(case+'__base.npz'))
  if case not in f.CASES:feature_probes+=verify_new_base(data,source,base)
  if set(result['cases'][case])!=set(POLICIES):raise AssertionError('policy omitted')
  for policy in POLICIES:
   stamp=reference_stamps(source,scale) if policy=='deadband' else vg.scalar_stamps(source);clean,current=vg.independent_history(source,events,stamp);endfresh=(current>=0)&(events>=current)&(events-current<=100_000_000);latest=base['nonlinear7__request__latest_endpoint'];fresh=np.array([i>=0 and bool(clean[i]) and int(current[i])>=0 and 0<=int(t)-int(current[i])<=100_000_000 for t,i in zip(times,latest)],bool)
   actual=None
   if policy=='deadband' or (policy=='exact' and case not in f.CASES):path=ROOT/'replays'/(case+'__'+policy+'.npz');actual=load(path)
   elif policy=='exact':path=RESEARCH/'vehicle_content_guard/replays'/(case+'.npz');actual=load(path)
   if actual is not None:
    with zipfile.ZipFile(path) as z:
     if z.testzip() is not None:raise AssertionError('bad overlay archive')
    for key,v in [('message_content_stamp_ns',stamp),('history_clean',clean),('endpoint_content_stamp_ns',current),('endpoint_content_fresh',endfresh),('request_content_fresh',fresh)]:np.testing.assert_array_equal(v,actual[key])
   if policy=='deadband':
    arrivals=source['bag_ns'].tolist();selected=set(np.linspace(0,len(events)-1,65,dtype=int).tolist())
    for boundary in [data['start'],data['end']]:
     j=int(np.searchsorted(events,boundary));selected.update(i for i in range(j-2,j+3) if 0<=i<len(events))
    for i in selected:
     good=True
     for offset in range(-127,1):
      t=int(events[i])+offset*50_000_000;j,status=vf.lookup(source,arrivals,t)
      if status==0 and not (int(stamp[j])>=0 and 0<=t-int(stamp[j])<=100_000_000):good=False
     if bool(clean[i])!=good:raise AssertionError('scalar history differs')
     history_probes+=1
   if set(result['cases'][case][policy])!=set(f.METHODS):raise AssertionError('model omitted')
   for method in f.METHODS:
    enabled=policy!='none' and method!='radar_only';status=base[method+'__request__status'].copy();reject=(status==0)&~fresh if enabled else np.zeros(len(times),bool);status[reject]=4;quality=base[method+'__quality']&clean&endfresh if enabled else base[method+'__quality'];prediction=np.array([float(v) if s==0 else math.nan for v,s in zip(base[method+'__request__prediction_mps'],status)])
    if np.any((status==0)&(base[method+'__request__status']!=0)) or np.any(quality&~base[method+'__quality']):raise AssertionError('prior denial revived')
    if actual is not None:
     for key,v in [('status',status),('prediction_mps',prediction),('content_guard_rejected',reject)]:np.testing.assert_allclose(v,actual[method+'__request__'+key],rtol=0,atol=0,equal_nan=True)
     np.testing.assert_array_equal(quality,actual[method+'__quality'])
     for key in ['latest_endpoint','age_ns']:np.testing.assert_array_equal(base[method+'__request__'+key],actual[method+'__request__'+key])
    summaries={}
    for name,mask in masks.items():
     accepted=mask&(status==0);summaries[name]={'requests':int(mask.sum()),'accepted_requests':int(accepted.sum()),'availability':float(accepted.sum()/mask.sum()),'statuses':{str(k):int(np.sum(mask&(status==k))) for k in range(5)},'accepted_errors':stats(data['req_y'],prediction,accepted)};cells+=1
    p=np.full(len(events),np.nan);p[data['ready']]=base[method+'__ready_prediction_mps'];v={'requests':summaries,'ungated_ready_errors':stats(data['raw_y'],p,data['ready']),'endpoint_usable':int(quality.sum()),'additional_withheld_requests':int(reject.sum()),'nominal_prediction_values_changed':False};close(v,result['cases'][case][policy][method]);cells+=1;trials+=len(times)
 sources={}
 for key in g.SOURCES:
  source=f.im.decode_payloads(load(RESEARCH/'vehicle_livox_collection/runtime_export'/(key+'.npz')));stamp=reference_stamps(source,scale);valid=stamp>=0;ages=source['bag_ns'][valid]-stamp[valid];sources[key]={'captured_records':len(stamp),'valid_records':int(valid.sum()),'content_withheld_records':int(np.sum(ages>100_000_000)),'calibration_source':key in FITS}
 close(sources,result['source_checks']);control_results={};t=1_000_000_000+np.arange(501,dtype=np.int64)*10_000_000
 for case,amplitude in [('constant_legitimate_possible',0.),('tiny_jitter',1e-12),('threshold_aware_pulses',2*.001*float(scale[0]))]:
  a=np.zeros((501,3));a[:,2]=9.8;a[:,0]=np.arange(501)%2*amplitude;s={'bag_ns':t,'usable':np.ones(501,bool),'acceleration':a,'angular_velocity':np.zeros((501,3))};control_results[case]={'records':501}
  for policy,stamp in [('exact',vg.scalar_stamps(s)),('deadband',reference_stamps(s,scale))]:control_results[case][policy]=sum(int(v)>=0 and 0<=int(q)-int(v)<=100_000_000 for q,v in zip(t,stamp))
 close(control_results,result['controls'])
 if result['decision']!='FIXED_DEADBAND_CONTENT_GUARD_COMPARISON_COMPLETE_NO_PROMOTION':raise AssertionError('decision differs')
 for flag in ['models_retrained','labels_changed','threshold_search','production_policy_changed','reserved_final_opened','promotion_allowed']:
  if result[flag] is not False:raise AssertionError('unsupported claim')
 for name,h in read(ROOT/'sha256.json').items():
  if digest(ROOT/name)!=h:raise AssertionError('artifact changed '+name)
 for name,h in read(ROOT/'verification_dependencies.json').items():
  if digest(RESEARCH/name)!=h:raise AssertionError('verification code changed '+name)
 prior=read(ROOT/'prior_snapshot_manifest.json');manifest=read(PROJECT/'PROJECT_MANIFEST.json');verified=missing=protected=0
 for name,h in prior.items():
  if name!='FINAL_PROJECT_README.md' and manifest.get(name)!=h:raise AssertionError('prior manifest changed '+name)
 for name,h in manifest.items():
  if 'urban02' in name.lower() or 'rav4' in name.lower() or name.startswith('research/vehicle_final_evaluation/'):protected+=1;continue
  p=PROJECT/name
  if not p.is_file():missing+=1;continue
  if digest(p)!=h:raise AssertionError('project artifact changed '+name)
  verified+=1
 import test_deadband
 stream=io.StringIO();tests=unittest.TextTestRunner(stream=stream).run(unittest.defaultTestLoader.loadTestsFromModule(test_deadband))
 if not tests.wasSuccessful():raise AssertionError(stream.getvalue())
 print('DEADBAND GUARD VERIFICATION PASS:',trials,'policy-request trials;',cells,'metric cells;',history_probes,'history probes;',feature_probes,'new-case feature probes;',parameters['fitting_rows'],'fitting-only vectors;',tests.testsRun,'tests')
 print('PROJECT:',verified,'verified;',missing,'missing;',protected,'protected paths not opened')
 print(result['decision']);print('ADAPTIVE-PULSE EVASION AND CONSTANT-CONTENT FALSE-DENIAL CONTROLS RETAINED')
if __name__=='__main__':verify()
