"""Fitting-only empirical source limits and separate causal source banks."""
from pathlib import Path
import numpy as np
from replay_metrics import read,write,load,digest,blocked,write_npz,summary
ROOT=Path(__file__).resolve().parent;RESEARCH=ROOT.parent;PROJECT=RESEARCH.parent
CASES=('nominal','radar_plus3','peer_plus3','radar_peer_plus3','fusion_peer_plus3','all_peer_plus3','dropout','dropout_radar_plus3','dropout_radar_peer_plus3')
MODELS=('radar_imu_parent','nonlinear7','linear7','huber7','extra_trees7');SOURCES=('radar_only',*MODELS)
METHODS=('radar_only',)+tuple(m+'__deadband__'+s for m in MODELS for s in ('withhold','fallback'))
VARIANTS=('baseline','latch','return3');FIELDS=('route','status','prediction_mps','latest_endpoint','age_ns','primary_status','radar_status');BUDGET=100_000_000

def pairs(members):
 groups={}
 for i,row in enumerate(members):
  if type(row['time_ns']) is not int or row['time_ns']<0:raise ValueError('integer fitting clocks required')
  groups.setdefault(row['route'],[]).append(i)
 left=[];right=[]
 for route,ids in sorted(groups.items()):
  ids.sort(key=lambda i:members[i]['time_ns']);t=[members[i]['time_ns'] for i in ids]
  if any(b<=a for a,b in zip(t,t[1:])):raise ValueError('duplicate/nonincreasing recording clock '+route)
  for j,i in enumerate(ids):
   k=j+1
   while k<len(ids) and t[k]-t[j]<=BUDGET:left.append(i);right.append(ids[k]);k+=1
 return np.array(left,np.int32),np.array(right,np.int32)

def limit(values,left,right):
 p=np.asarray(values)
 if p.ndim!=1 or not np.isfinite(p).all() or np.any(p<0) or not len(left):raise ValueError('finite predictions and nonempty fitting pairs required')
 return float(np.max(abs(p[right]-p[left])))

def bank(times,prediction,eligible,endpoint_ns,step_limit,kind):
 t=np.asarray(times);p=np.asarray(prediction);ok=np.asarray(eligible);e=np.asarray(endpoint_ns)
 if kind not in ('latch','return3') or not np.isfinite(step_limit) or step_limit<0:raise ValueError('fixed monitor mode/finite limit required')
 if t.ndim!=1 or t.dtype.kind not in 'iu' or e.shape!=t.shape or e.dtype.kind not in 'iu' or p.shape!=t.shape or ok.shape!=t.shape or ok.dtype.kind!='b' or np.any(t<0) or np.any(t[1:]<=t[:-1]):raise ValueError('aligned integer clocks and boolean eligibility required')
 if np.any(ok&(~np.isfinite(p)|(p<0)|(e<0)|(e>t)|((t-e)>BUDGET))) or np.any(np.diff(e[ok])<0):raise ValueError('invalid eligible source value/clock')
 allowed=np.zeros(len(t),bool);reason=np.ones(len(t),np.uint8);alarms=np.zeros(len(t),bool);recovery=np.zeros(len(t),bool)
 anchor=stamp=None;q=False;streak=0;confirm_stamp=None
 for i in range(len(t)):
  if not ok[i]:streak=0;confirm_stamp=None;continue
  v=float(p[i]);s=int(e[i])
  if q:
   reason[i]=3
   if kind=='return3' and abs(v-anchor)<=step_limit:
    if confirm_stamp is not None and s-confirm_stamp>BUDGET:streak=0;confirm_stamp=None
    if confirm_stamp is None or s>confirm_stamp:streak+=1;confirm_stamp=s
    reason[i]=4
    if streak>=3:q=False;allowed[i]=True;reason[i]=5;recovery[i]=True;anchor=v;stamp=s;streak=0;confirm_stamp=None
   else:streak=0;confirm_stamp=None
   continue
  gap=stamp is not None and s-stamp>BUDGET
  if stamp is not None and not gap and abs(v-anchor)>step_limit:
   q=True;alarms[i]=True;reason[i]=2;streak=0;confirm_stamp=None;continue
  allowed[i]=True;reason[i]=6 if gap else 0;anchor=v;stamp=s
 return {'allowed':allowed,'reason':reason,'alarm':alarms,'recovery':recovery}

def fitting():
 members=read(RESEARCH/'vehicle_route_validation/membership.json');parent=load(RESEARCH/'vehicle_multisource_imu/oof_predictions.npz');signed=load(RESEARCH/'vehicle_signed_axis_correction/oof_predictions.npz');new=load(RESEARCH/'vehicle_nonlinear_correction/oof_predictions.npz');heads=load(RESEARCH/'vehicle_matched_head_comparison/held_predictions.npz')
 if len(members)!=27178 or [v['id'] for v in members]!=list(range(27178)):raise ValueError('fitting membership changed')
 if set(v['group'] for v in members)!={'Urban01','Urban03','Urban04'} or len(set(v['route'] for v in members))!=5:raise ValueError('fitting source selection changed')
 for a in (signed,new,heads):
  np.testing.assert_array_equal(a['target_mps'],parent['target_mps']);np.testing.assert_array_equal(a['fold_id'],parent['fold_id'])
 for key,archive,key2 in [('original_context__prediction_mps',signed,'original_context__prediction_mps'),('original_context__prediction_mps',new,'original_context__prediction_mps'),('radar_imu__prediction_mps',new,'parent_radar_imu__prediction_mps'),('radar_imu__prediction_mps',heads,'radar_imu_parent__prediction_mps')]:np.testing.assert_array_equal(parent[key],archive[key2])
 for m in ('nonlinear7','linear7'):np.testing.assert_array_equal(new[m+'__prediction_mps'],heads[m+'__prediction_mps'])
 pred={'radar_only':parent['original_context__prediction_mps'],'radar_imu_parent':parent['radar_imu__prediction_mps'],**{m:heads[m+'__prediction_mps'] for m in MODELS[1:]}}
 if any(v.shape!=(len(members),) for v in pred.values()):raise ValueError('fitting vector length')
 for route in set(v['route'] for v in members):
  ids=[i for i,v in enumerate(members) if v['route']==route]
  if len(set(parent['fold_id'][ids].tolist()))!=1:raise ValueError('fold changes within recording')
 return members,pred

def calibrate():
 freeze()
 if (ROOT/'parameters.json').exists():raise ValueError('parameters already frozen; no recalibration')
 members,pred=fitting();a,b=pairs(members);np.testing.assert_array_equal([members[i]['route'] for i in a],[members[i]['route'] for i in b]);row_groups=np.array([v['group'] for v in members]);recordings=sorted({v['route'] for v in members});trace={};thresholds={}
 for name,p in pred.items():
  thresholds[name]=limit(p,a,b);delta=abs(p[b]-p[a]);records={}
  for route in recordings:
   mask=np.array([members[i]['route']==route for i in a]);values=delta[mask];j=np.flatnonzero(mask)[int(np.argmax(values))];records[route]={'pairs':len(values),'maximum_mps':float(values.max()),'left_id':int(a[j]),'right_id':int(b[j])}
  cross={}
  for g in ('Urban01','Urban03','Urban04'):
   held=row_groups[a]==g;train=~held;v=float(delta[train].max());cross[g]={'calibration_pairs':int(train.sum()),'held_pairs':int(held.sum()),'other_groups_limit_mps':v,'held_above_limit':int(np.sum(delta[held]>v))}
  trace[name]={'all_pair_maximum_mps':thresholds[name],'by_recording':records,'leave_group_out_diagnostic':cross}
 write_npz(ROOT/'calibration_pairs.npz',{'left_id':a,'right_id':b});write(ROOT/'parameters.json',{'scope':'Empirical maxima from existing fitting-recording OOF predictions; not independent statistical calibration or a physical bound. No full-development/fault outputs used.','fitting_rows':len(members),'pairs':len(a),'maximum_pair_lag_ns':BUDGET,'source_limits_mps':thresholds,'trace':trace,'labels_used_for_limits':False})
 write(ROOT/'calibration_lock.json',{'parameters_sha256':digest(ROOT/'parameters.json'),'pairs_sha256':digest(ROOT/'calibration_pairs.npz'),'analysis_lock_sha256':digest(ROOT/'analysis_lock.json')})
 print('CALIBRATION FROZEN:',len(a),'same-recording pairs;',thresholds,flush=True)

def freeze(create=False):
 protocol=read(ROOT/'protocol.json')
 if protocol['cases']!=list(CASES) or protocol['methods']!=list(METHODS) or protocol['variants']!=list(VARIANTS) or protocol['max_pair_lag_ns']!=BUDGET:raise ValueError('scope changed')
 for name,h in protocol['input_sha256'].items():
  if blocked(name):raise ValueError('protected input')
  if digest(PROJECT/name)!=h:raise ValueError('input changed '+name)
 code=('calibrated_core.py','run_calibrated.py','verify_calibrated.py','test_calibrated.py','replay_metrics.py','scalar_metrics.py')
 lock={'protocol_sha256':digest(ROOT/'protocol.json'),'code_sha256':{n:digest(ROOT/n) for n in code}}
 if (ROOT/'analysis_lock.json').exists():
  if read(ROOT/'analysis_lock.json')!=lock:raise ValueError('analysis lock changed')
 elif create:write(ROOT/'analysis_lock.json',lock)
 else:raise ValueError('freeze first')

def data():
 frames=read(RESEARCH/'vehicle_masked_history/frame_index.json');events=np.array([v['time_ns'] for v in frames],np.int64);refs=read(RESEARCH/'vehicle_availability_audit/request_reference.json');t=np.array([v['time_ns'] for v in refs],np.int64);truth=np.array([np.nan if v['reference_mps'] is None else v['reference_mps'] for v in refs],float)
 if len(events)!=40690 or len(t)!=37340 or np.any(np.diff(events)<=0) or np.any(np.diff(t)<=0):raise ValueError('schedule changed')
 duration=int(events[-1])-int(events[0]);start=int(events[0])+duration*2//5;end=int(events[0])+duration*3//5;segment=(t-t[0])//300_000_000_000
 masks={'all':np.ones(len(t),bool),'before':t<start,'during':(t>=start)&(t<end),'after':t>=end,**{'segment_'+str(int(k)):segment==k for k in np.unique(segment)}}
 return t,truth,events,masks

def branch_inputs(saved,t,events):
 sources={}
 for source in SOURCES:
  method='radar_only' if source=='radar_only' else source+'__deadband__withhold';out={k:saved[method+'__'+k] for k in FIELDS};ok=out['route']>0
  if any(v.shape!=t.shape for v in out.values()) or not np.array_equal(ok,out['status']==0):raise ValueError('source proposal alignment')
  latest=out['latest_endpoint'];expected=np.searchsorted(events,t,side='right')-1
  if np.any(ok&((latest!=expected)|(latest<0))):raise ValueError('noncausal source')
  stamps=np.where(ok,events[np.maximum(latest,0)],-1)
  if np.any(ok&((out['age_ns']!=t-stamps)|(out['age_ns']<0)|(out['age_ns']>BUDGET))):raise ValueError('source freshness mismatch')
  sources[source]={'original':out,'prediction':out['prediction_mps'],'eligible':ok,'endpoint_ns':stamps}
 return sources

def select_output(sources,method,banks=None):
 model='radar_only' if method=='radar_only' else method.split('__')[0];p=sources[model];r=sources['radar_only'];pk=p['eligible'].copy();rk=r['eligible'].copy()
 if banks is not None:pk&=banks[model]['allowed'];rk&=banks['radar_only']['allowed']
 fallback=method.endswith('__fallback');route=np.where(pk,1,np.where(fallback&rk,2,0)).astype(np.uint8)
 pp=p['original'];rr=r['original'];oldstatus=pp['status'];out={'route':route,'status':np.where(route>0,0,np.where(p['eligible'],4,oldstatus)).astype(np.uint8),'prediction_mps':np.where(route==1,p['prediction'],np.where(route==2,r['prediction'],np.nan)),'latest_endpoint':np.where(route==1,pp['latest_endpoint'],np.where(route==2,rr['latest_endpoint'],-1)),'age_ns':np.where(route==1,pp['age_ns'],np.where(route==2,rr['age_ns'],-1)),'primary_status':pp['status'].copy(),'radar_status':rr['status'].copy()}
 return out
