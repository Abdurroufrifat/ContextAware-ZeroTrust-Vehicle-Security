"""Independent scalar inference, gates and causal replay; NumPy/stdlib only, no fitting."""
import bisect,io,math,unittest
from nonlinear_core import ROOT,RESEARCH,SCOPES,METHODS,FEATURES,read,write,digest,load,inputs
from score_nonlinear import check_models
def close(a,b):
 if isinstance(b,dict):
  if set(a)!=set(b):raise AssertionError('dictionary membership differs')
  for k in b:close(a[k],b[k])
 elif isinstance(b,list):
  if len(a)!=len(b):raise AssertionError('list length differs')
  for x,y in zip(a,b):close(x,y)
 elif isinstance(b,float):
  if not math.isclose(float(a),b,rel_tol=0,abs_tol=1e-9):raise AssertionError((a,b))
 elif a!=b:raise AssertionError((a,b))
def scalar_predict(m,x,kind):
 if kind=='linear7':return m['intercept']+math.fsum((float(v)-a)/b*c for v,a,b,c in zip(x,m['mean'],m['scale'],m['coefficient']))
 result=m['baseline']
 for tree in m['trees']:
  n=tree[0]
  while not n['is_leaf']:
   v=float(x[n['feature_idx']]);left=bool(n['missing_go_to_left']) if math.isnan(v) else v<=n['num_threshold'];n=tree[n['left'] if left else n['right']]
  result+=n['value']
 return result
def stats(y,p,mask):
 pairs=[(float(a),float(b)) for a,b,ok in zip(y,p,mask) if ok];errors=sorted(abs(b-a) for a,b in pairs if math.isfinite(a) and math.isfinite(b));n=len(errors)
 def quantile():
  t=(n-1)*.95;i=int(t);return errors[i]+(errors[min(i+1,n-1)]-errors[i])*(t-i)
 return {'count':n,'missing_truth':sum(not math.isfinite(a) for a,b in pairs),'mae_mps':math.fsum(errors)/n if n else None,'rmse_mps':math.sqrt(math.fsum(e*e for e in errors)/n) if n else None,'p95_mps':quantile() if n else None,'maximum_mps':errors[-1] if n else None,'above_2_mps':sum(e>2 for e in errors)}
def independent_full(data,p):
 import numpy as np
 keep=np.array([bool(s) and int(imu)==0 and math.isfinite(float(v)) and math.isfinite(float(peer)) and v>=0 and peer>=0 and abs(v-peer)<=1. for v,peer,s,imu in zip(p,data['peer'],data['source'],data['status'])]);imu=load(RESEARCH/'vehicle_multisource_imu/feature_cache.npz');events=data['events'].tolist();latest=[];ages=[];statuses=[];served=[]
 for t in data['times']:
  t=int(t);i=bisect.bisect_right(events,t)-1;age=t-events[i] if i>=0 else -1;status=1 if i<0 else 2 if age>100_000_000 else 3 if not data['ready'][i] else 4
  bound=i>=0 and int(imu['full_current__status'][i])==0 and t>=events[i] and 0<=t-int(imu['full_current__bag_ns'][i])<=100_000_000 and 0<=t-int(imu['full_current__header_ns'][i])<=100_000_000
  if bound!=bool(data['binding'][len(latest)]):raise AssertionError('source binding differs')
  if status==4 and keep[i] and bound:status=0
  latest.append(i);ages.append(age);statuses.append(status);served.append(float(p[i]) if status==0 else math.nan)
 statuses=np.array(statuses);served=np.array(served);segments=(data['times']-data['times'][0])//300_000_000_000
 def cell(mask):
  accepted=mask&(statuses==0);return {'requests':int(mask.sum()),'usable':int(accepted.sum()),'usable_fraction':float(accepted.sum()/mask.sum()),'accepted':stats(data['req_y'],served,accepted)}
 overall=cell(np.ones(len(segments),bool));seg={str(int(k)):cell(segments==k) for k in np.unique(segments)};blocks={}
 for name in sorted(set(data['full_window'].tolist())-{'outside_supported_blocks'}):
  ids=np.flatnonzero(data['ready'])[data['full_window']==name];blocks[name]={'raw_endpoints':len(ids),'retention':float(sum(bool(keep[i]) for i in ids)/len(ids)),'accepted':stats(data['raw_y'][ids],p[ids],keep[ids])}
 cells=[overall,*seg.values()];gates={'availability':all(v['usable_fraction']>=.95 for v in cells),'truth_coverage':all(v['accepted']['count']>0 and v['accepted']['count']/(v['accepted']['count']+v['accepted']['missing_truth'])>=.95 for v in cells),'request_p95':all(v['accepted']['count']>0 and v['accepted']['p95_mps']<=.5 for v in cells),'zero_outliers':overall['accepted']['count']>0 and overall['accepted']['above_2_mps']==0,'block_retention':len(blocks)==3 and all(v['retention']>=.95 for v in blocks.values()),'block_p95':len(blocks)==3 and all(v['accepted']['count']>0 and v['accepted']['p95_mps']<=.5 for v in blocks.values())}
 return {'overall':overall,'segments_300s':seg,'blocks':blocks,'ungated_ready':stats(data['raw_y'],p,data['ready']),'gates':gates,'all_gates_pass':all(gates.values())},keep,{'latest_endpoint':np.array(latest),'age_ns':np.array(ages),'status':statuses,'prediction_mps':served}
def verify():
 import numpy as np
 check_models();data=inputs();result=read(ROOT/'results.json');route=load(ROOT/'oof_predictions.npz');full=load(ROOT/'development_replay.npz');y=data['oof']['target_mps'];np.testing.assert_array_equal(y,route['target_mps']);np.testing.assert_array_equal(data['fold'],route['fold_id']);pred={};native_probes=0
 for kind in METHODS:
  p=np.full(len(y),np.nan)
  for scope in SCOPES:
   m=read(ROOT/'models'/scope/(kind+'.json'));cal=read(RESEARCH/'vehicle_multisource_imu/models'/scope/'calibration.json');train=np.array(cal['train_ids']);held=np.array(cal['held_ids']);parent=load(RESEARCH/'vehicle_multisource_imu/models'/scope/'radar_imu_export.npz')['native'];axes=np.tile(data['axes']['fit_axes'][train],(2,1));native=load(ROOT/'models'/scope/(kind+'_training.npz'))['native']
   if m['feature_names']!=FEATURES or m['method']!=kind or m['scope']!=scope:raise AssertionError('model schema differs')
   trace=next(v for v in read(ROOT/'training_trace.json')['runs'] if v['scope']==scope and v['method']==kind);xx=np.column_stack([parent,axes]);target=np.tile(y[train],2)-parent;counts=np.array(cal['bin_counts']);weights=np.tile((len(cal['calibration_ids'])/4)/counts[np.digitize(y[train],[1,5,10])]*.5,2)
   import hashlib
   for name,array in [('feature_sha256',xx),('residual_target_sha256',target),('weight_sha256',weights)]:
    if hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()!=trace[name]:raise AssertionError('training array trace differs')
   if set(train)&set(held) or set(train)|set(held)!=set(range(len(y))):raise AssertionError('leaking or incomplete split')
   if scope!='all_fit' and (any(data['members'][i]['group']==scope for i in train) or any(data['members'][i]['group']!=scope for i in held)):raise AssertionError('held route in training')
   for i in np.unique(np.linspace(0,len(parent)-1,65,dtype=int)):
    v=scalar_predict(m,[parent[i],*axes[i]],kind)
    if abs(v-float(native[i]))>1e-8:raise AssertionError('native training inference differs')
    native_probes+=1
   if kind=='linear7':
    z=(xx-np.array(m['mean']))/np.array(m['scale']);res=target-native;gradient=z.T@(weights*res)-np.array(m['coefficient'])
    if np.max(abs(gradient))>1e-7 or abs(np.dot(weights,res))>1e-7:raise AssertionError('weighted ridge stationarity differs')
   if scope=='all_fit':
    ids=np.flatnonzero(data['ready']);fp=np.full(len(data['ready']),np.nan)
    for i in ids:
     par=data['parent_full'][i];fp[i]=max(0.,par+scalar_predict(m,[par,*data['axes']['full_axes'][i]],kind))
    np.testing.assert_allclose(fp[ids],full[kind+'__ready_prediction_mps'],rtol=0,atol=1e-9)
    report,keep,replay=independent_full(data,fp);close(report,result['full_development'][kind]);np.testing.assert_array_equal(keep,full[kind+'__quality'])
    for k,v in replay.items():np.testing.assert_allclose(v,full[kind+'__request__'+k],rtol=0,atol=1e-9,equal_nan=True)
   else:
    for i in held:
     par=data['oof']['radar_imu__prediction_mps'][i];p[i]=max(0.,par+scalar_predict(m,[par,*data['axes']['fit_axes'][i]],kind))
  pred[kind]=p;np.testing.assert_allclose(p,route[kind+'__prediction_mps'],rtol=0,atol=1e-9)
 previous=load(RESEARCH/'vehicle_signed_axis_correction/oof_predictions.npz');pred.update(signed6_previous=previous['signed_axis_corrected__prediction_mps'],original_context=previous['original_context__prediction_mps'],parent_radar_imu=data['oof']['radar_imu__prediction_mps'])
 previous_full=load(RESEARCH/'vehicle_signed_axis_correction/development_replay.npz');fp=np.full(len(data['ready']),np.nan);fp[data['ready']]=previous_full['signed_axis_corrected__ready_prediction_mps'];report,keep,replay=independent_full(data,fp);close(report,result['full_development']['signed6_previous']);np.testing.assert_array_equal(keep,previous_full['signed_axis_corrected__quality'])
 for k,v in replay.items():np.testing.assert_allclose(v,previous_full['signed_axis_corrected__request__'+k],rtol=0,atol=1e-9,equal_nan=True)
 metric_cells=0
 for category,key in [('pooled',None),('by_group','group'),('by_recording','route'),('by_window','window_id')]:
  labels=np.array([v[key] for v in data['members']]) if key else None;names=sorted(set(labels.tolist())) if key else ['all']
  for name in names:
   mask=labels==name if key else np.ones(len(y),bool)
   for kind,p in pred.items():close(stats(y,p,mask),result['route'][category][name][kind] if key else result['route'][category][kind]);np.testing.assert_array_equal(p,route[kind+'__prediction_mps']) if kind not in METHODS else None;metric_cells+=1
 for n in pred:
  failures={cat:[k for k,v in result['route'][cat].items() if v[n]['p95_mps']>.5] for cat in ['by_group','by_recording','by_window']};close(failures,result['route_failures'][n])
 for n in [*METHODS,'signed6_previous']:
  passed=result['route']['pooled'][n]['p95_mps']<=.5 and all(not v for v in result['route_failures'][n].values()) and result['full_development'][n]['all_gates_pass'];close(passed,result['all_accuracy_availability_gates_pass'][n])
 close(result['primary_all_accuracy_availability_gates_pass'],result['all_accuracy_availability_gates_pass']['nonlinear7'])
 if result['decision']!='FIXED_NONLINEAR_EXPERIMENT_COMPLETE_NO_PROMOTION' or result['primary_method']!='nonlinear7':raise AssertionError('decision differs')
 for flag in ['held_scores_used_in_fitting','thresholds_changed','production_policy_changed','reserved_final_opened','promotion_allowed','journal_readiness_established']:
  if result[flag] is not False:raise AssertionError('unsupported claim: '+flag)
 import test_nonlinear
 stream=io.StringIO();out=unittest.TextTestRunner(stream=stream).run(unittest.defaultTestLoader.loadTestsFromModule(test_nonlinear))
 if not out.wasSuccessful():raise AssertionError(stream.getvalue())
 for n,h in read(ROOT/'sha256.json').items():
  if digest(ROOT/n)!=h:raise AssertionError('delivery hash differs: '+n)
 project=RESEARCH.parent;prior=read(ROOT/'prior_snapshot_manifest.json');manifest=read(project/'PROJECT_MANIFEST.json');verified=missing=protected=0
 for n,h in prior.items():
  if n!='FINAL_PROJECT_README.md' and manifest.get(n)!=h:raise AssertionError('prior manifest changed: '+n)
 for n,h in manifest.items():
  if 'urban02' in n.lower() or 'rav4' in n.lower() or n.startswith('research/vehicle_final_evaluation/'):protected+=1;continue
  path=project/n
  if not path.is_file():missing+=1;continue
  if digest(path)!=h:raise AssertionError('project hash differs: '+n)
  verified+=1
 print('INDEPENDENT NONLINEAR EXPERIMENT PASS:',len(y),'held rows;',len(data['times']),'requests;',metric_cells,'route metric cells;',native_probes,'native probes;',out.testsRun,'tests')
 print('PROJECT:',verified,'verified;',missing,'missing;',protected,'protected paths not opened')
 print(result['decision'],'primary all accuracy/availability gates:',result['primary_all_accuracy_availability_gates_pass'])
if __name__=='__main__':verify()
