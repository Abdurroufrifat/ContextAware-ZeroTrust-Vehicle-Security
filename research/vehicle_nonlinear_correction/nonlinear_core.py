"""Fixed seven-input residual experiment on previously inspected development data."""
from pathlib import Path
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parent
RESEARCH=ROOT.parent
SCOPES=('Urban01','Urban03','Urban04','all_fit')
METHODS=('nonlinear7','linear7')
FEATURES=['frozen_parent_speed_mps','mean11_acc_x','mean11_acc_y','mean11_acc_z','mean11_gyro_x','mean11_gyro_y','mean11_gyro_z']
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def head_features(parent,axes):
 p=np.asarray(parent);a=np.asarray(axes)
 if p.ndim!=1 or a.shape!=(len(p),6) or not np.isfinite(p).all() or not np.isfinite(a).all():raise ValueError('invalid seven-input head features')
 return np.column_stack([p,a])
def validate_split(train,held,n):
 a=np.asarray(train,dtype=int);b=np.asarray(held,dtype=int)
 if len(set(a.tolist()))!=len(a) or len(set(b.tolist()))!=len(b) or set(a)&set(b) or set(a)|set(b)!=set(range(n)):raise ValueError('scope partition differs')
def fit_ridge(x,y,w):
 total=w.sum();mean=(x*w[:,None]).sum(0)/total;scale=np.sqrt(((x-mean)**2*w[:,None]).sum(0)/total);scale[scale==0]=1.;z=(x-mean)/scale;intercept=float((y*w).sum()/total);coef=np.linalg.solve(z.T@(z*w[:,None])+np.eye(x.shape[1]),z.T@(w*(y-intercept)))
 return {'mean':mean.tolist(),'scale':scale.tolist(),'intercept':intercept,'coefficient':coef.tolist(),'alpha':1.}
def ridge_predict(m,x):return m['intercept']+((x-np.array(m['mean']))/np.array(m['scale']))@np.array(m['coefficient'])
def export_tree(m):
 trees=[]
 for iteration in m._predictors:
  if len(iteration)!=1:raise ValueError('single-output regression required')
  nodes=iteration[0].nodes
  trees.append([{k:(float(n[k]) if k in ['value','num_threshold'] else int(n[k])) for k in ['value','feature_idx','num_threshold','left','right','is_leaf','missing_go_to_left']} for n in nodes])
 return {'baseline':float(m._baseline_prediction[0,0]),'trees':trees}
def tree_predict(m,x):
 x=np.asarray(x);out=np.full(len(x),m['baseline'],dtype=float)
 for tree in m['trees']:
  ids=np.zeros(len(x),dtype=int)
  while True:
   active=np.array([not tree[i]['is_leaf'] for i in ids])
   if not active.any():break
   rows=np.flatnonzero(active)
   for node in np.unique(ids[rows]):
    ii=rows[ids[rows]==node];n=tree[node];v=x[ii,n['feature_idx']];left=np.where(np.isnan(v),bool(n['missing_go_to_left']),v<=n['num_threshold']);ids[ii]=np.where(left,n['left'],n['right'])
  out+=np.array([tree[i]['value'] for i in ids])
 return out
def predict(m,x,kind):return tree_predict(m,x) if kind=='nonlinear7' else ridge_predict(m,x)
def quality(p,peer,source,status):return source&(status==0)&np.isfinite(p)&np.isfinite(peer)&(p>=0)&(peer>=0)&(abs(p-peer)<=1.)
def serve(times,events,ready,p,keep,binding,ttl=100_000_000):
 latest=np.searchsorted(events,times,side='right')-1;i=np.maximum(latest,0);present=latest>=0;age=np.where(present,times-events[i],-1);fresh=present&(age<=ttl);status=np.ones(len(times),dtype=np.uint8);status[present&~fresh]=2;status[fresh]=3;status[fresh&ready[i]]=4;status[fresh&ready[i]&keep[i]&binding]=0
 return {'latest_endpoint':latest,'age_ns':age,'status':status,'prediction_mps':np.where(status==0,p[i],np.nan)}
def metrics(y,p,mask=None):
 y=np.asarray(y);p=np.asarray(p);mask=np.ones(len(y),bool) if mask is None else np.asarray(mask);valid=mask&np.isfinite(y)&np.isfinite(p);e=abs(p[valid]-y[valid]);n=len(e)
 return {'count':n,'missing_truth':int(np.sum(mask&~np.isfinite(y))),'mae_mps':float(e.mean()) if n else None,'rmse_mps':float(np.sqrt(np.mean(e**2))) if n else None,'p95_mps':float(np.quantile(e,.95)) if n else None,'maximum_mps':float(e.max()) if n else None,'above_2_mps':int(np.sum(e>2.))}
def inputs():
 a=load(RESEARCH/'vehicle_signed_axis_correction/feature_cache.npz');o=load(RESEARCH/'vehicle_multisource_imu/oof_predictions.npz');d=load(RESEARCH/'vehicle_multisource_imu/development_replay.npz');imu=load(RESEARCH/'vehicle_multisource_imu/feature_cache.npz');diag=load(RESEARCH/'vehicle_remaining_failure_audit/row_diagnostics.npz');members=read(RESEARCH/'vehicle_route_validation/membership.json');index=read(RESEARCH/'vehicle_masked_history/frame_index.json');night=load(RESEARCH/'vehicle_night_training/predictions.npz');dec=load(RESEARCH/'vehicle_night_training/decisions.npz')
 ready=np.array([v['ready'] for v in index],bool);events=np.array([v['time_ns'] for v in index],np.int64);ids=np.flatnonzero(ready)
 if len(members)!=27178 or len(index)!=40690 or len(ids)!=40678:raise ValueError('fixed cohort differs')
 np.testing.assert_array_equal(ids,diag['full__row_id']);np.testing.assert_array_equal(events[ids],diag['full__time_ns']);np.testing.assert_array_equal(events,imu['full_current__endpoint_ns']);np.testing.assert_array_equal(o['target_mps'],diag['fit__target_mps']);np.testing.assert_array_equal(a['fit_axes'],diag['fit__signed_axes']);np.testing.assert_array_equal(a['full_axes'][ids],diag['full__signed_axes'])
 raw=read(RESEARCH/'vehicle_continuous_radar/reference_targets.json');req=read(RESEARCH/'vehicle_availability_audit/request_reference.json');times=d['radar_imu__request__request_times_ns'];np.testing.assert_array_equal(events,[v['time_ns'] for v in raw]);np.testing.assert_array_equal(times,[v['time_ns'] for v in req]);np.testing.assert_array_equal(d['radar_imu__request__latest_endpoint'],np.searchsorted(events,times,side='right')-1)
 peer=np.full(len(index),np.nan);peer[ready]=night['day_plus_night__masked_cnn_4ch'];source=dec['day_plus_night__quality'];status=imu['full_current__status'];parent=np.full(len(index),np.nan);parent[ready]=d['radar_imu__ready_prediction_mps'];keep=quality(parent,peer,source,status);np.testing.assert_array_equal(keep,d['radar_imu__quality'])
 binding=d['radar_imu__request__imu_binding_fresh'];replay=serve(times,events,ready,parent,keep,binding)
 for k in ['status','latest_endpoint','age_ns','prediction_mps']:np.testing.assert_allclose(replay[k],d['radar_imu__request__'+k],rtol=0,atol=1e-12,equal_nan=True)
 return {'axes':a,'oof':o,'parent_full':parent,'ready':ready,'events':events,'peer':peer,'source':source,'status':status,'binding':binding,'times':times,'raw_y':np.array([np.nan if v['reference_mps'] is None else v['reference_mps'] for v in raw]),'req_y':np.array([np.nan if v['reference_mps'] is None else v['reference_mps'] for v in req]),'members':members,'full_window':diag['full__window_id'],'fold':o['fold_id']}
def check_lock():
 p=read(ROOT/'protocol.json');lock=read(ROOT/'analysis_lock.json')
 if digest(ROOT/'protocol.json')!=lock['protocol_sha256']:raise ValueError('protocol changed')
 for n,h in lock['code_sha256'].items():
  if digest(ROOT/n)!=h:raise ValueError('frozen code changed: '+n)
 for n,h in p['input_sha256'].items():
  if digest(RESEARCH/n)!=h:raise ValueError('frozen input changed: '+n)
 return p
def full_metrics(data,p):
 keep=quality(p,data['peer'],data['source'],data['status']);q=serve(data['times'],data['events'],data['ready'],p,keep,data['binding']);segments=(data['times']-data['times'][0])//300_000_000_000
 def cell(mask):
  accepted=mask&(q['status']==0)
  return {'requests':int(mask.sum()),'usable':int(accepted.sum()),'usable_fraction':float(accepted.sum()/mask.sum()),'accepted':metrics(data['req_y'],q['prediction_mps'],accepted)}
 overall=cell(np.ones(len(segments),bool));seg={str(int(k)):cell(segments==k) for k in np.unique(segments)};blocks={}
 for name in sorted(set(data['full_window'].tolist())-{'outside_supported_blocks'}):
  ids=np.flatnonzero(data['ready'])[data['full_window']==name];blocks[name]={'raw_endpoints':len(ids),'retention':float(keep[ids].mean()),'accepted':metrics(data['raw_y'][ids],p[ids],keep[ids])}
 cells=[overall,*seg.values()];gates={'availability':all(v['usable_fraction']>=.95 for v in cells),'truth_coverage':all(v['accepted']['count']>0 and v['accepted']['count']/(v['accepted']['count']+v['accepted']['missing_truth'])>=.95 for v in cells),'request_p95':all(v['accepted']['count']>0 and v['accepted']['p95_mps']<=.5 for v in cells),'zero_outliers':overall['accepted']['count']>0 and overall['accepted']['above_2_mps']==0,'block_retention':len(blocks)==3 and all(v['retention']>=.95 for v in blocks.values()),'block_p95':len(blocks)==3 and all(v['accepted']['count']>0 and v['accepted']['p95_mps']<=.5 for v in blocks.values())}
 if np.any(keep&~data['source']):raise ValueError('source rejection revived')
 return {'overall':overall,'segments_300s':seg,'blocks':blocks,'ungated_ready':metrics(data['raw_y'],p,data['ready']),'gates':gates,'all_gates_pass':all(gates.values())},keep,q
