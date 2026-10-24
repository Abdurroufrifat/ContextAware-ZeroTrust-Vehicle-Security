"""Portable independent fusion replay; no refitting or reserved-final access."""
import io,unittest
from fusion_core import *
from verify_magnitude import independent_features,scalar_predict,independent_anchor
from verify_diagnosis import independent_metric
import test_fusion

def validate_claims(result):
 if result.get('decision')!='FUSION_ABLATION_COMPLETE_NO_PROMOTION':raise ValueError('unexpected fusion promotion/decision')
 for key in ['reserved_final_opened','production_policy_changed','thresholds_changed','witness_enabled','promotion_allowed','journal_readiness_established']:
  if result.get(key) is not False:raise ValueError('unsupported fusion claim: '+key)
 if result.get('full_urban01_night_development_reused') is not True or result.get('full_development',{}).get('no_formerly_rejected_endpoint_revived') is not True:raise ValueError('development reuse or rejection preservation claim differs')

def scalar_features(row,anchor,median):
 a=independent_features(row,anchor,median);mag=np.asarray(row).copy();mag[0]=abs(mag[0]);b=independent_features(mag,anchor,median);return np.concatenate([a,b[EXTRA]])

def metric_check(y,p,expected):
 ref=independent_metric(y,p)
 for key,value in expected.items():
  if key!='missing_truth':close(ref[key],value)
 if 'missing_truth' in expected:close(expected['missing_truth'],0)

def scalar_clock(times,events,ready,p,keep):
 out={k:[] for k in ['request_times_ns','latest_endpoint','age_ns','status','prediction_mps','timing_prediction_mps']};i=-1
 for t in times:
  while i+1<len(events) and events[i+1]<=t:i+=1
  age=int(t)-int(events[i]) if i>=0 else -1
  status=1 if i<0 else 2 if age>100000000 else 3 if not ready[i] else 4 if not keep[i] else 0
  values=[int(t),i,age,status,float(p[i]) if status==0 else np.nan,float(p[i]) if status in (0,4) else np.nan]
  for k,v in zip(out,values):out[k].append(v)
 return {k:np.asarray(v) for k,v in out.items()}

def audit():
 freeze();check_models();validate_claims(read(ROOT/'results.json'));project=ROOT.parent.parent;available=0;prior=read(ROOT/'prior_snapshot_manifest.json')
 for name,h in prior.items():
  if name in ['FINAL_PROJECT_README.md','scripts/check_coact_project.py']:continue
  path=(project/name).resolve()
  if not path.is_relative_to(project):raise ValueError('unsafe prior artifact path')
  if path.is_file():
   if digest(path)!=h:raise ValueError('prior payload changed: '+name)
   available+=1
 if (ROOT/'sha256.json').exists():
  for name,h in read(ROOT/'sha256.json').items():
   path=(ROOT/name).resolve()
   if not path.is_relative_to(ROOT) or digest(path)!=h:raise ValueError('fusion study payload changed: '+name)
 data,members,groups,day,support=r.load_data();base=original_base(data);trace=read(ROOT/'training_trace.json');coverage=np.zeros(len(groups),int);routepred=np.full(len(groups),np.nan);fits=probes=0
 if len(trace['runs'])!=4 or [v['scope'] for v in trace['runs']]!=list(SCOPES):raise ValueError('missing/selected fusion scope')
 p=read(ROOT/'protocol.json')
 for scope,run in zip(SCOPES,trace['runs']):
  folder=ROOT/'models'/scope;cal=g.validated_calibration(calibration(data,groups,day,scope),read(folder/'calibration.json'));ids=np.array(cal['train_ids']);held=np.array(cal['held_ids'],dtype=int)
  if scope!='all_fit':
   coverage[held]+=1
   if np.intersect1d(ids,held).size or set(groups[ids])&set(groups[held]):raise ValueError('fusion held route leakage')
   changed={**data,'y':data['y'].copy(),'median_feature':data['median_feature'].copy()};changed['y'][held]+=12345.;changed['median_feature'][held]+=1000.;close(r.calibrate(changed,groups,day,scope),cal)
  else:
   if ids.tolist()!=list(range(len(groups))) or len(held):raise ValueError('all-fit cohort differs')
  affine,counts,anchorids=independent_anchor(data,data['median_feature'],groups,day,scope if scope!='all_fit' else 'NOT_A_FITTING_ROUTE');close(affine,cal['anchor']);close(counts,cal['bin_counts']);np.testing.assert_array_equal(anchorids,cal['calibration_ids'])
  x,y,w,a,aug=fit_inputs(data,base,cal);check_parent_inputs(scope,x,y,w)
  counts=np.bincount(np.searchsorted([1.,5.,10.],data['y'][anchorids],side='right'),minlength=4);oraclew=np.tile(len(anchorids)/(4*counts[np.searchsorted([1.,5.,10.],data['y'][ids],side='right')])*.5,2);np.testing.assert_array_equal(w,oraclew);np.testing.assert_array_equal(y,np.tile(data['y'][ids]-a[:len(ids)],2))
  for arr,key in [(x,'feature_sha256'),(y,'target_sha256'),(w,'weight_sha256'),(x[:,:75],'signed_prefix_sha256'),(aug[:,3],'mask_sha256')]:
   if g.array_digest(arr)!=run[key]:raise ValueError('fusion training data changed: '+key)
  if run['train_samples']!=len(ids) or run['held_samples']!=len(held) or run['features']!=148 or run['iterations']!=200 or any(run['parameters'].get(k)!=v for k,v in p['model'].items()):raise ValueError('fusion fit budget differs')
  model=read(folder/'model.json')
  if model['feature_names']!=FEATURE_NAMES or len(model['trees'])!=200:raise ValueError('fusion feature/tree schema differs')
  fitpred=c.predict(model,x,a)
  with np.load(folder/'fit_export.npz',allow_pickle=False) as z:np.testing.assert_allclose(fitpred,z['native'],rtol=0,atol=1e-8);np.testing.assert_allclose(fitpred,z['portable'],rtol=0,atol=1e-9)
  fits+=len(x)
  for j in np.unique(np.linspace(0,len(ids)-1,17,dtype=int)):
   at=len(ids)+j;row=scalar_features(aug[j],a[at],data['median_feature'][ids[j]]);np.testing.assert_allclose(row,x[at],rtol=0,atol=1e-10);np.testing.assert_allclose(scalar_predict(model,row,a[at]),fitpred[at],rtol=0,atol=1e-9);probes+=1
  if scope!='all_fit':
   ha=c.e.affine(cal['anchor'],data['median_feature'][held]);hf=assemble(base[held],ha,data['median_feature'][held]);hp=c.predict(model,hf,ha);routepred[held]=hp
   for j in np.unique(np.linspace(0,len(held)-1,65,dtype=int)):
    row=scalar_features(data['X'][held[j]],ha[j],data['median_feature'][held[j]]);np.testing.assert_allclose(row,hf[j],rtol=0,atol=1e-10);np.testing.assert_allclose(scalar_predict(model,row,ha[j]),hp[j],rtol=0,atol=1e-9);probes+=1
 if not np.all(coverage==1):raise ValueError('fusion held rows not covered once')
 result=read(ROOT/'results.json')
 with np.load(ROOT/'oof_predictions.npz',allow_pickle=False) as z:
  np.testing.assert_array_equal(z['target_mps'],data['y']);np.testing.assert_array_equal(z['fold_id'],np.array([r.GROUPS.index(v) for v in groups]));np.testing.assert_allclose(z[METHODS[1]+'__prediction_mps'],routepred,rtol=0,atol=1e-9);old=z[METHODS[0]+'__prediction_mps'].copy()
 with np.load(r.ROOT/'oof_predictions.npz',allow_pickle=False) as z:np.testing.assert_array_equal(old,z['prediction_mps'])
 rebuilt=route_summary(members,data['y'],dict(zip(METHODS,[old,routepred])),support);close(rebuilt,result['route']);yc=data['y'];cells=[(np.ones(len(yc),bool),rebuilt['pooled'])]
 for category,key in [('by_group','group'),('by_recording','route'),('by_window','window_id')]:
  labels=np.array([m[key] for m in members]);cells.extend((labels==name,v) for name,v in rebuilt[category].items())
 speed=np.searchsorted([1.,5.,10.],yc,side='right');cells.extend((speed==i,rebuilt['by_reference_speed'][n]) for i,n in enumerate(d.SPEED_NAMES));cells.extend([(day,rebuilt['by_day_night']['day']),(~day,rebuilt['by_day_night']['night'])])
 for mask,cell in cells:
  if cell['count']!=int(mask.sum()):raise ValueError('fusion route denominator differs')
  for name,values in zip(METHODS,[old,routepred]):close(cell['models'][name],independent_metric(yc[mask],values[mask]))
 parentroute=read(r.ROOT/'results.json')
 for category,oldkey in [('pooled','pooled_sample_weighted'),('by_group','by_group'),('by_recording','by_recording'),('by_window','by_window')]:
  current={'all':rebuilt['pooled']} if category=='pooled' else rebuilt[category];previous={'all':parentroute[oldkey]} if category=='pooled' else parentroute[oldkey]
  for name,cell in current.items():
   for key,value in previous[name]['context_boosting'].items():close(cell['models'][METHODS[0]][key],value)
 index,ready,x,median,anchor,peer,source,oldfull=load_full();cal=read(ROOT/'models/all_fit/calibration.json');np.testing.assert_array_equal(anchor,c.e.affine(cal['anchor'],median));fullfeatures=features(x,anchor,median);model=read(ROOT/'models/all_fit/model.json');newfull=c.predict(model,fullfeatures,anchor);originalmodel=read(c.ROOT/'model.json');np.testing.assert_allclose(c.predict(originalmodel,fullfeatures[:,:75],anchor),oldfull,rtol=0,atol=1e-9)
 with np.load(c.ROOT/'replay.npz',allow_pickle=False) as z:np.testing.assert_allclose(fullfeatures[:,:75],z['context_features'],rtol=0,atol=1e-10)
 controlprobes=0
 for j in np.unique(np.linspace(0,len(x)-1,129,dtype=int)):
  row=scalar_features(x[j],anchor[j],median[j]);np.testing.assert_allclose(row,fullfeatures[j],rtol=0,atol=1e-10);np.testing.assert_allclose(scalar_predict(model,row,anchor[j]),newfull[j],rtol=0,atol=1e-9);np.testing.assert_allclose(scalar_predict(originalmodel,row[:75],anchor[j]),oldfull[j],rtol=0,atol=1e-9);probes+=1;controlprobes+=1
 full,arrays=full_summary(newfull);close(full,result['full_development'])
 with np.load(ROOT/'development_replay.npz',allow_pickle=False) as z:
  if set(z.files)!=set(arrays):raise ValueError('fusion full replay schema differs')
  for key,values in arrays.items():np.testing.assert_allclose(z[key],values,rtol=0,atol=1e-9,equal_nan=True)
 parent=read(c.ROOT/'results.json');original=full['methods'][METHODS[0]];close({k:v for k,v in original.items() if k!='ungated_ready_endpoints'},parent['methods']['context_boosting']);close(full['gates'][METHODS[0]],parent['gates'])
 events=np.array([v['time_ns'] for v in index],dtype=np.int64);times=np.array(read(ROOT.parent/'vehicle_availability_audit/request_reference.json'));targets=read(ROOT.parent/'vehicle_availability_audit/request_reference.json');times=np.array([v['time_ns'] for v in targets],dtype=np.int64);yq=np.array([np.nan if v['reference_mps'] is None else v['reference_mps'] for v in targets]);raw=read(ROOT.parent/'vehicle_continuous_radar/reference_targets.json');yr=np.array([np.nan if v['reference_mps'] is None else v['reference_mps'] for v in raw]);peerfull=np.full(len(index),np.nan);peerfull[ready]=peer;segment=(times-times[0])//300000000000;scalarsteps=fullmetrics=0
 def check_request_cell(name,mask,expected):
  nonlocal fullmetrics
  status=arrays[name+'__request__status'][mask];prediction=arrays[name+'__request__prediction_mps'][mask];truth=yq[mask];keep=status==0;known=keep&np.isfinite(truth);metric_check(truth[known],prediction[known],{k:v for k,v in expected['accepted'].items() if k!='missing_truth'});close(expected['accepted']['missing_truth'],int(np.sum(keep&~np.isfinite(truth))))
  close(expected['requests'],len(status));close(expected['usable_fraction'],float(np.sum(keep))/len(status));close(expected['reference_missing_all_requests'],int(np.sum(~np.isfinite(truth))))
  for code,label in av.STATUS.items():close(expected['status_counts'][label],int(np.sum(status==code)))
  fullmetrics+=1
 def check_account(truth,prediction,keep,rawcount,expected):
  nonlocal fullmetrics
  known=np.isfinite(truth);accepted=keep&known;withheld=~keep&known
  counts={'raw_endpoints':rawcount,'features_ready':len(truth),'unready_raw':rawcount-len(truth),'ready_reference_missing':int(np.sum(~known)),'reference_coverage_on_ready':float(np.sum(known))/len(truth),'usable_total':int(np.sum(keep)),'usable_reference_missing':int(np.sum(keep&~known)),'quality_withheld':int(np.sum(~keep)),'whole_acceptance_fraction':int(np.sum(keep))/rawcount}
  for k,v in counts.items():close(expected[k],v)
  for key,mask in [('all_ready_reference_scored',known),('accepted',accepted),('withheld',withheld)]:metric_check(truth[mask],prediction[mask],expected[key]);fullmetrics+=1
 lookup={int(t):i for i,t in enumerate(events)};blocks={}
 for row in read(c.OLD/'prepared/calibration/index.json'):
  if row['ready']:blocks.setdefault(row['window_id'],[]).append(lookup[row['time_ns']])
 for name,values in zip(METHODS,[oldfull,newfull]):
  pred=np.full(len(index),np.nan);pred[ready]=values;keep=np.array([bool(s) and np.isfinite(a) and np.isfinite(b) and a>=0 and b>=0 and abs(a-b)<=1. for a,b,s in zip(pred,peerfull,source)]);np.testing.assert_array_equal(keep,arrays[name+'__quality'])
  if np.any(keep&~source):raise ValueError('fusion revived a previously rejected endpoint')
  clock=scalar_clock(times,events,ready,pred,keep)
  for k,v in clock.items():np.testing.assert_allclose(v,arrays[name+'__request__'+k],rtol=0,atol=1e-9,equal_nan=True)
  scalarsteps+=len(times);method=full['methods'][name];check_request_cell(name,np.ones(len(times),bool),method['requests']['overall'])
  for key,stats in method['requests']['segments_300s'].items():check_request_cell(name,segment==int(key),stats)
  check_account(yr[ready],values,keep[ready],len(index),method['endpoints'])
  for key,ids in blocks.items():check_account(yr[ids],pred[ids],keep[ids],len(ids),method['previous_supported_blocks'][key])
  known=ready&np.isfinite(yr);metric_check(yr[known],pred[known],{k:v for k,v in method['ungated_ready_endpoints'].items() if k!='missing_truth'});close(method['ungated_ready_endpoints']['missing_truth'],int(np.sum(ready&~np.isfinite(yr))));fullmetrics+=1
  # Gates independently rebuilt from independently verified numerical cells.
  q=[method['requests']['overall']]+list(method['requests']['segments_300s'].values());b=list(method['previous_supported_blocks'].values());expected={}
  names=list(parent['gates']);flags=[all(v['usable_fraction']>=.95 for v in q),all(v['accepted']['count']>0 and v['accepted']['count']/(v['accepted']['count']+v['accepted']['missing_truth'])>=.95 for v in q),all(v['accepted']['count']>0 and v['accepted']['p95_mps']<=.5 for v in q),method['requests']['overall']['accepted']['count']>0 and method['requests']['overall']['accepted']['above_2_mps']==0,all(v['whole_acceptance_fraction']>=.95 for v in b),all(v['accepted']['count']>0 and v['accepted']['p95_mps']<=.5 for v in b)];close(dict(zip(names,flags)),full['gates'][name]);close(full['all_gates_pass'][name],all(flags))
 common=(arrays[METHODS[0]+'__request__status']==0)&(arrays[METHODS[1]+'__request__status']==0)
 for mask,cell in [(common,full['common_accepted_requests']['overall'])]+[(common&(segment==int(k)),v) for k,v in full['common_accepted_requests']['segments_300s'].items()]:
  close(cell['shared_requests'],int(np.sum(mask)))
  for name in METHODS:
   known=mask&np.isfinite(yq);metric_check(yq[known],arrays[name+'__request__prediction_mps'][known],{k:v for k,v in cell['models'][name].items() if k!='missing_truth'});close(cell['models'][name]['missing_truth'],int(np.sum(mask&~np.isfinite(yq))));fullmetrics+=1
 buf=io.StringIO();tests=unittest.TextTestRunner(stream=buf).run(unittest.defaultTestLoader.loadTestsFromModule(test_fusion))
 if not tests.wasSuccessful():raise ValueError(buf.getvalue())
 print(f'SIGNED MAGNITUDE FUSION AUDIT: PASS ({tests.testsRun} tests;4 fixed models;{fits} native/export comparisons;{probes} fusion + {controlprobes} original independent feature/tree probes;{len(cells)*2} route metric cells;{fullmetrics} full-development metric cells;{scalarsteps} scalar request steps)')
 print(f'Available preceding payloads verified: {available}/{len(prior)-2}; unavailable historical files are not claimed as reverified.')
 for name in METHODS:print(name,'route P95',round(result['route']['pooled']['models'][name]['p95_mps'],6),'failed windows',len(result['route']['diagnostic_failures'][name]['windows']),'full gates',full['all_gates_pass'][name])
 print(result['decision']+'; INSPECTED DEVELOPMENT REUSED. FINAL RESERVED DATA CLOSED. WITNESS DISABLED.')
if __name__=='__main__':audit()
