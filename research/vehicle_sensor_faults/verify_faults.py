"""Independent fault construction, tree traversal, source-clock replay and metrics."""
import bisect,io,math,unittest,zipfile
import numpy as np
from fault_core import ROOT,RESEARCH,PROJECT,CASES,METHODS,read,load,digest,freeze,inputs,im
from verify_nonlinear import close,stats
def source_copy(base,case,start,end):
 ids=[i for i,t in enumerate(base['bag_ns']) if case!='dropout' or not start<=int(t)<end];out={k:base[k][ids].copy() for k in ['bag_ns','header_ns','usable','acceleration','angular_velocity','acceleration_norm','angular_velocity_norm']};previous=bisect.bisect_left(base['bag_ns'].tolist(),start)-1
 for i,t in enumerate(out['bag_ns']):
  if not start<=int(t)<end:continue
  if case=='stale_header':out['header_ns'][i]=int(out['header_ns'][i])-150_000_000
  elif case=='future_header':out['header_ns'][i]=int(out['header_ns'][i])+150_000_000
  elif case=='accel_bias':out['acceleration'][i,0]=float(out['acceleration'][i,0])+.5
  elif case=='gyro_bias':out['angular_velocity'][i,2]=float(out['angular_velocity'][i,2])+.1
  elif case=='frozen_values':
   out['acceleration'][i]=base['acceleration'][previous];out['angular_velocity'][i]=base['angular_velocity'][previous]
 out['acceleration_norm']=np.hypot.reduce(out['acceleration'],axis=1);out['angular_velocity_norm']=np.hypot.reduce(out['angular_velocity'],axis=1)
 return out
def lookup(source,arrivals,t):
 i=bisect.bisect_right(arrivals,t)-1
 if i<0:return i,1
 h=t-int(source['header_ns'][i]);b=t-arrivals[i]
 if h<0:return i,3
 if h>100_000_000 or b>100_000_000:return i,2
 if not source['usable'][i] or not math.isfinite(float(source['acceleration_norm'][i])) or not math.isfinite(float(source['angular_velocity_norm'][i])):return i,4
 return i,0
def scalar_features(source,arrivals,t):
 rows=[]
 for offset in range(-127,1):
  i,status=lookup(source,arrivals,t+offset*50_000_000);rows.append((status==0,i))
 ok,i=rows[-1];result=[float(source['acceleration_norm'][i]) if ok else 0.,float(source['angular_velocity_norm'][i]) if ok else 0.,float(ok)]
 for width in [5,11,21,41,81,128]:
  selected=[(j-(width-1),i) for j,(valid,i) in enumerate(rows[-width:]) if valid];count=len(selected)
  for key in ['acceleration_norm','angular_velocity_norm']:
   if not count:result.extend([0.,0.,0.]);continue
   mean=math.fsum(float(source[key][i]) for j,i in selected)/count;sd=math.sqrt(math.fsum((float(source[key][i])-mean)**2 for j,i in selected)/count);tm=math.fsum(j*.05 for j,i in selected)/count;den=math.fsum((j*.05-tm)**2 for j,i in selected);slope=math.fsum((j*.05-tm)*(float(source[key][i])-mean) for j,i in selected)/den if den else 0.;result.extend([mean,sd,slope])
  result.append(count/width)
 selected=[i for valid,i in rows[-11:] if valid];signed=[math.fsum(float(source[key][i,k]) for i in selected)/len(selected) if selected else 0. for key in ['acceleration','angular_velocity'] for k in range(3)]
 return result,signed
def signed_features(source,events):
 grid=events[:,None]+np.arange(-10,1,dtype=np.int64)*50_000_000;idx=np.searchsorted(source['bag_ns'],grid,side='right')-1;j=np.maximum(idx,0);valid=(idx>=0)&source['usable'][j]&(grid>=source['header_ns'][j])&(grid-source['header_ns'][j]<=100_000_000)&(grid-source['bag_ns'][j]<=100_000_000);count=np.maximum(valid.sum(1),1)
 return np.column_stack([np.sum(np.where(valid[:,:,None],source[key][j],0.),axis=1)/count[:,None] for key in ['acceleration','angular_velocity']])
def tree_infer(model,x):
 out=np.full(len(x),float(model['baseline']))
 for nodes in model['trees']:
  values=np.zeros(len(x));pending=[(0,np.arange(len(x)))]
  while pending:
   node,ids=pending.pop();v=nodes[node];old='leaf' in v
   if v['leaf' if old else 'is_leaf']:values[ids]=v['value'];continue
   split=x[ids,v['feature' if old else 'feature_idx']]<=v['threshold' if old else 'num_threshold']
   if split.any():pending.append((v['left'],ids[split]))
   if (~split).any():pending.append((v['right'],ids[~split]))
  out+=values
 return out
def verify():
 freeze();data=inputs();result=read(ROOT/'results.json');previous=read(ROOT/'recorded_previous_outcomes.json');close(result['fault_interval_ns'],previous['fault_interval_ns']);events=data['events'].tolist();times=data['times'];ready=data['ready'];trials=cells=probes=0
 if set(result['cases'])!=set(CASES):raise AssertionError('case omitted')
 for case in CASES:
  path=ROOT/'replays'/(case+'.npz')
  with zipfile.ZipFile(path) as z:
   if z.testzip() is not None or len(z.namelist())!=22:raise AssertionError('incomplete replay archive')
  saved=load(path);source=source_copy(data['source_imu'],case,data['start'],data['end']);arrivals=source['bag_ns'].tolist();features,current=im.features(source,data['events']);signed=signed_features(source,data['events']);parent=np.maximum(0.,data['anchor']+tree_infer(data['parent_model'],np.column_stack([data['radar'],features[ready]])));candidate=np.maximum(0.,parent+tree_infer(data['head'],np.column_stack([parent,signed[ready]])));pred={'radar_only':data['radar_only'],'radar_imu_parent':parent,'nonlinear7':candidate}
  selected=set(np.linspace(0,len(events)-1,65,dtype=int).tolist())
  for boundary in [data['start'],data['end']]:
   j=bisect.bisect_left(events,boundary);selected.update(i for i in range(j-2,j+3) if 0<=i<len(events))
  for i in sorted(selected):
   f,a=scalar_features(source,arrivals,events[i]);np.testing.assert_allclose(f,features[i],rtol=0,atol=1e-9);np.testing.assert_allclose(a,signed[i],rtol=0,atol=1e-9);probes+=1
  for i,t in enumerate(events):
   j,status=lookup(source,arrivals,t)
   if status!=int(saved['current__status'][i]) or (arrivals[j] if j>=0 else -1)!=int(saved['current__bag_ns'][i]) or (int(source['header_ns'][j]) if j>=0 else -1)!=int(saved['current__header_ns'][i]):raise AssertionError('source clocks or status differ')
  segment=(times-times[0])//300_000_000_000;masks={'all':np.ones(len(times),bool),'before':times<data['start'],'during':(times>=data['start'])&(times<data['end']),'after':times>=data['end'],**{'segment_'+str(int(k)):segment==k for k in np.unique(segment)}}
  if set(result['cases'][case])!=set(METHODS):raise AssertionError('method omitted')
  for method,values in pred.items():
   np.testing.assert_allclose(values,saved[method+'__ready_prediction_mps'],rtol=0,atol=1e-9);p=np.full(len(events),np.nan);p[ready]=values;keep=np.array([bool(s) and math.isfinite(float(v)) and math.isfinite(float(peer)) and v>=0 and peer>=0 and abs(v-peer)<=1. and (method=='radar_only' or int(current['status'][i])==0) for i,(v,peer,s) in enumerate(zip(p,data['peer'],data['source']))]);np.testing.assert_array_equal(keep,saved[method+'__quality']);statuses=[];served=[];latest=[];ages=[]
   for k,time in enumerate(times):
    time=int(time);i=bisect.bisect_right(events,time)-1;age=time-events[i] if i>=0 else -1;bound=i>=0 and int(current['status'][i])==0 and time>=events[i] and 0<=time-int(current['bag_ns'][i])<=100_000_000 and 0<=time-int(current['header_ns'][i])<=100_000_000
    if bound!=bool(saved['binding'][k]):raise AssertionError('request source binding differs')
    status=1 if i<0 else 2 if age>100_000_000 else 3 if not ready[i] else 4
    if status==4 and keep[i] and (method=='radar_only' or bound):status=0
    latest.append(i);ages.append(age);statuses.append(status);served.append(float(p[i]) if status==0 else math.nan)
   statuses=np.array(statuses);served=np.array(served)
   for name,v in [('latest_endpoint',latest),('age_ns',ages),('status',statuses),('prediction_mps',served)]:np.testing.assert_allclose(v,saved[method+'__request__'+name],rtol=0,atol=1e-9,equal_nan=True)
   summary={}
   for name,mask in masks.items():
    accepted=mask&(statuses==0);summary[name]={'requests':int(mask.sum()),'accepted_requests':int(accepted.sum()),'availability':float(accepted.sum()/mask.sum()),'statuses':{str(k):int(np.sum(mask&(statuses==k))) for k in range(5)},'accepted_errors':stats(data['req_y'],served,accepted)};cells+=1
   close({'requests':summary,'ungated_ready_errors':stats(data['raw_y'],p,ready),'endpoint_usable':int(keep.sum())},result['cases'][case][method]);cells+=1;trials+=len(times)
  actual=result['cases'][case]['nonlinear7']['requests']['during'];expected=previous['candidate_during'][case];close(actual['accepted_requests'],expected[0]);close(actual['accepted_errors']['p95_mps'],expected[1])
 if result['decision']!='FROZEN_MODEL_SENSOR_FAULT_REPLAY_COMPLETE_NO_PROMOTION':raise AssertionError('decision differs')
 for flag in ['thresholds_changed','model_retrained','reserved_final_opened','production_policy_changed','promotion_allowed','physical_security_validated']:
  if result[flag] is not False:raise AssertionError('unsupported claim')
 for name,h in read(ROOT/'sha256.json').items():
  if digest(ROOT/name)!=h:raise AssertionError('delivery artifact changed '+name)
 prior=read(ROOT/'prior_snapshot_manifest.json');manifest=read(PROJECT/'PROJECT_MANIFEST.json');verified=missing=protected=0
 for name,h in prior.items():
  if name!='FINAL_PROJECT_README.md' and manifest.get(name)!=h:raise AssertionError('prior manifest entry changed '+name)
 for name,h in manifest.items():
  if 'urban02' in name.lower() or 'rav4' in name.lower() or name.startswith('research/vehicle_final_evaluation/'):protected+=1;continue
  p=PROJECT/name
  if not p.is_file():missing+=1;continue
  if digest(p)!=h:raise AssertionError('project artifact changed '+name)
  verified+=1
 import test_faults
 stream=io.StringIO();tests=unittest.TextTestRunner(stream=stream).run(unittest.defaultTestLoader.loadTestsFromModule(test_faults))
 if not tests.wasSuccessful():raise AssertionError(stream.getvalue())
 print('SENSOR FAULT REPLAY VERIFICATION PASS:',len(CASES),'cases;',len(METHODS),'models;',trials,'request trials;',cells,'metric cells;',probes,'scalar feature probes;',tests.testsRun,'tests')
 print('PROJECT:',verified,'verified;',missing,'missing;',protected,'protected paths not opened')
 print('RECORDED PRE-CLEANUP CANDIDATE OUTCOMES MATCH ALL SEVEN CASES')
 print(result['decision'])
if __name__=='__main__':verify()
