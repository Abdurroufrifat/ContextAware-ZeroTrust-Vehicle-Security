"""Scalar change tracking, independent history sweep, serving and metric checks."""
import io,math,unittest,zipfile
import numpy as np
from guard_core import ROOT,RESEARCH,PROJECT,SOURCES,read,load,digest,freeze,f
import verify_faults as vf
close=vf.close;stats=vf.stats
def scalar_stamps(source):
 out=np.full(len(source['bag_ns']),-1,dtype=np.int64);previous=None;stamp=-1
 for i,t in enumerate(source['bag_ns']):
  values=tuple(float(x) for x in [*source['acceleration'][i],*source['angular_velocity'][i]])
  if not source['usable'][i] or not all(math.isfinite(x) for x in values):continue
  if values!=previous:stamp=int(t);previous=values
  out[i]=stamp
 return out
def independent_history(source,events,stamp):
 clean=np.ones(len(events),bool)
 for offset in range(-127,1):
  query=events+offset*50_000_000;i=np.searchsorted(source['bag_ns'],query,side='right')-1;j=np.maximum(i,0);s=np.where(i>=0,stamp[j],-1);used=(i>=0)&source['usable'][j]&np.isfinite(source['acceleration_norm'][j])&np.isfinite(source['angular_velocity_norm'][j])&(query>=source['header_ns'][j])&(query-source['header_ns'][j]<=100_000_000)&(query-source['bag_ns'][j]<=100_000_000);fresh=(s>=0)&(query>=s)&(query-s<=100_000_000);clean&=~(used&~fresh)
 return clean,s
def verify():
 freeze();data=f.inputs();result=read(ROOT/'results.json');baseline=read(RESEARCH/'vehicle_sensor_faults/results.json')['cases'];close(result['baseline'],baseline);events=data['events'];times=data['times'];cells=trials=probes=0;gates={'no_prior_denial_revived':True};all_source_rows=0
 if set(result['guarded'])!=set(f.CASES):raise AssertionError('case omitted')
 for case in f.CASES:
  path=ROOT/'replays'/(case+'.npz')
  with zipfile.ZipFile(path) as z:
   if z.testzip() is not None:raise AssertionError('bad overlay archive')
  actual=load(path);old=load(RESEARCH/'vehicle_sensor_faults/replays'/(case+'.npz'));source=vf.source_copy(data['source_imu'],case,data['start'],data['end']);stamp=scalar_stamps(source);np.testing.assert_array_equal(stamp,actual['message_content_stamp_ns']);clean,current_stamp=independent_history(source,events,stamp);fresh=(current_stamp>=0)&(events>=current_stamp)&(events-current_stamp<=100_000_000);np.testing.assert_array_equal(clean,actual['history_clean']);np.testing.assert_array_equal(current_stamp,actual['endpoint_content_stamp_ns']);np.testing.assert_array_equal(fresh,actual['endpoint_content_fresh']);arrivals=source['bag_ns'].tolist()
  selected=set(np.linspace(0,len(events)-1,65,dtype=int).tolist())
  for boundary in [data['start'],data['end']]:
   j=int(np.searchsorted(events,boundary));selected.update(i for i in range(j-2,j+3) if 0<=i<len(events))
  for i in sorted(selected):
   good=True
   for offset in range(-127,1):
    t=int(events[i])+offset*50_000_000;j,status=vf.lookup(source,arrivals,t)
    if status==0 and not (int(stamp[j])>=0 and 0<=t-int(stamp[j])<=100_000_000):good=False
   if bool(clean[i])!=good:raise AssertionError('scalar consumed-history guard differs')
   probes+=1
  latest=old['nonlinear7__request__latest_endpoint'];request_fresh=[]
  for t,i in zip(times,latest):request_fresh.append(i>=0 and bool(clean[i]) and int(current_stamp[i])>=0 and 0<=int(t)-int(current_stamp[i])<=100_000_000)
  request_fresh=np.array(request_fresh,bool);np.testing.assert_array_equal(request_fresh,actual['request_content_fresh']);segment=(times-times[0])//300_000_000_000;masks={'all':np.ones(len(times),bool),'before':times<data['start'],'during':(times>=data['start'])&(times<data['end']),'after':times>=data['end'],**{'segment_'+str(int(k)):segment==k for k in np.unique(segment)}}
  if set(result['guarded'][case])!=set(f.METHODS):raise AssertionError('method omitted')
  added={}
  for method in f.METHODS:
   status=old[method+'__request__status'].copy();quality=old[method+'__quality'].copy();reject=(status==0)&~request_fresh if method!='radar_only' else np.zeros(len(times),bool);status[reject]=4
   if method!='radar_only':quality=quality&clean&fresh
   prediction=np.array([float(p) if s==0 else math.nan for p,s in zip(old[method+'__request__prediction_mps'],status)]);np.testing.assert_array_equal(status,actual[method+'__request__status']);np.testing.assert_array_equal(reject,actual[method+'__request__content_guard_rejected']);np.testing.assert_array_equal(quality,actual[method+'__quality']);np.testing.assert_allclose(prediction,actual[method+'__request__prediction_mps'],rtol=0,atol=0,equal_nan=True)
   for key in ['latest_endpoint','age_ns']:np.testing.assert_array_equal(old[method+'__request__'+key],actual[method+'__request__'+key])
   if np.any((status==0)&(old[method+'__request__status']!=0)) or np.any(quality&~old[method+'__quality']):raise AssertionError('old denial revived')
   summaries={}
   for name,mask in masks.items():
    accepted=mask&(status==0);summaries[name]={'requests':int(mask.sum()),'accepted_requests':int(accepted.sum()),'availability':float(accepted.sum()/mask.sum()),'statuses':{str(k):int(np.sum(mask&(status==k))) for k in range(5)},'accepted_errors':stats(data['req_y'],prediction,accepted)};cells+=1
   p=np.full(len(events),np.nan);p[data['ready']]=old[method+'__ready_prediction_mps'];added[method]=int(reject.sum());v={'requests':summaries,'ungated_ready_errors':stats(data['raw_y'],p,data['ready']),'endpoint_usable':int(quality.sum()),'additional_withheld_requests':added[method],'nominal_prediction_values_changed':False};close(v,result['guarded'][case][method]);cells+=1;trials+=len(times)
  if case=='nominal':gates['no_additional_nominal_request_denials']=all(v==0 for v in added.values())
  if case=='frozen_values':
   mask=(times>=data['start']+100_000_000)&(times<data['end']);gates['zero_multisource_grants_in_sustained_frozen_interval']=all(not np.any(mask&(actual[m+'__request__status']==0)) for m in ['radar_imu_parent','nonlinear7'])
 sources={}
 for key in SOURCES:
  source=f.im.decode_payloads(load(RESEARCH/'vehicle_livox_collection/runtime_export'/(key+'.npz')));stamp=scalar_stamps(source);valid=[i for i,s in enumerate(stamp) if int(s)>=0];ages=[int(source['bag_ns'][i])-int(stamp[i]) for i in valid];sources[key]={'captured_records':len(stamp),'valid_vector_records':len(valid),'unchanged_content_withheld':sum(a>100_000_000 for a in ages),'maximum_content_age_ns':max(ages) if ages else None};all_source_rows+=len(stamp)
 close(sources,result['sources']);gates['no_extra_content_denials_in_six_captured_sources']=all(v['unchanged_content_withheld']==0 for v in sources.values());close(gates,result['observed_contracts']);close(result['controls'],{'constant_legitimate_possible':{'records':501,'fresh':11,'withheld':490,'meaning':'Possible benign constant content is withheld'},'tiny_alternating_jitter':{'records':501,'fresh':501,'withheld':0,'meaning':'Tiny value changes evade exact-equality guard'}})
 if result['decision']!='OFFLINE_CONTENT_FRESHNESS_GUARD_COMPLETE_NO_PROMOTION':raise AssertionError('decision differs')
 for flag in ['models_retrained','prediction_values_changed','thresholds_tuned','production_policy_changed','reserved_final_opened','promotion_allowed','physical_security_validated']:
  if result[flag] is not False:raise AssertionError('unsupported guard claim')
 for name,h in read(ROOT/'sha256.json').items():
  if digest(ROOT/name)!=h:raise AssertionError('delivery artifact changed '+name)
 for name,h in read(ROOT/'verification_dependencies.json').items():
  if digest(RESEARCH/name)!=h:raise AssertionError('verification helper changed '+name)
 prior=read(ROOT/'prior_snapshot_manifest.json');manifest=read(PROJECT/'PROJECT_MANIFEST.json');verified=missing=protected=0
 for name,h in prior.items():
  if name!='FINAL_PROJECT_README.md' and manifest.get(name)!=h:raise AssertionError('prior manifest changed '+name)
 for name,h in manifest.items():
  if 'urban02' in name.lower() or 'rav4' in name.lower() or name.startswith('research/vehicle_final_evaluation/'):protected+=1;continue
  p=PROJECT/name
  if not p.is_file():missing+=1;continue
  if digest(p)!=h:raise AssertionError('project hash differs '+name)
  verified+=1
 import test_guard
 stream=io.StringIO();tests=unittest.TextTestRunner(stream=stream).run(unittest.defaultTestLoader.loadTestsFromModule(test_guard))
 if not tests.wasSuccessful():raise AssertionError(stream.getvalue())
 print('CONTENT GUARD VERIFICATION PASS:',trials,'request trials;',cells,'guarded metric cells;',probes,'scalar history probes;',all_source_rows,'captured source records;',tests.testsRun,'tests')
 print('PROJECT:',verified,'verified;',missing,'missing;',protected,'protected paths not opened')
 print(result['decision']);print('CONSTANT-CONTENT FALSE DENIAL AND TINY-JITTER EVASION CONTROLS RETAINED')
if __name__=='__main__':verify()
