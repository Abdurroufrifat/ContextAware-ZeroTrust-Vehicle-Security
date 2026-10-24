"""Recovered fixed offline sensor-fault replay; no fitting or live policy changes."""
from pathlib import Path
import ast,importlib.util,io,json,os,sys,tempfile
import numpy as np
ROOT=Path(__file__).resolve().parent;RESEARCH=ROOT.parent;PROJECT=RESEARCH.parent
sys.path.insert(0,str(RESEARCH/'vehicle_nonlinear_correction'));import nonlinear_core as n
read=n.read;write=n.write;load=n.load;digest=n.digest
spec=importlib.util.spec_from_file_location('_frozen_imu_core',RESEARCH/'vehicle_multisource_imu/imu_core.py');im=importlib.util.module_from_spec(spec);spec.loader.exec_module(im)
CASES=('nominal','dropout','stale_header','future_header','accel_bias','gyro_bias','frozen_values')
METHODS=('radar_only','radar_imu_parent','nonlinear7')
FIELDS=('bag_ns','header_ns','usable','acceleration','angular_velocity','acceleration_norm','angular_velocity_norm')
def write_replay(path,arrays):
 if any(np.asarray(v).dtype.hasobject for v in arrays.values()):raise ValueError('numeric arrays required')
 path=Path(path);buffer=io.BytesIO();np.savez_compressed(buffer,**arrays);temporary=None
 try:
  with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.replay-',suffix='.tmp',delete=False) as out:
   temporary=Path(out.name);out.write(buffer.getvalue());out.flush();os.fsync(out.fileno())
  os.replace(temporary,path)
 finally:
  if temporary is not None:temporary.unlink(missing_ok=True)
def mutate(source,case,start,end):
 if case not in CASES or not start<end:raise ValueError('unknown case or interval')
 out={k:source[k].copy() for k in FIELDS};inside=(out['bag_ns']>=start)&(out['bag_ns']<end)
 if case=='dropout':return {k:v[~inside].copy() for k,v in out.items()}
 if case=='stale_header':out['header_ns'][inside]-=150_000_000
 elif case=='future_header':out['header_ns'][inside]+=150_000_000
 elif case=='accel_bias':out['acceleration'][inside,0]+=.5
 elif case=='gyro_bias':out['angular_velocity'][inside,2]+=.1
 elif case=='frozen_values':
  previous=np.flatnonzero(out['bag_ns']<start)
  if not len(previous):raise ValueError('prior record required')
  for k in ['acceleration','angular_velocity']:out[k][inside]=out[k][previous[-1]]
 out['acceleration_norm']=np.hypot.reduce(out['acceleration'],axis=1);out['angular_velocity_norm']=np.hypot.reduce(out['angular_velocity'],axis=1)
 return out
def axes(source,times):
 grid=times[:,None]+np.arange(-10,1,dtype=np.int64)*50_000_000;q=im.lookup(source,grid);valid=q['status']==0;ids=np.maximum(q['latest_message'],0);count=np.maximum(valid.sum(1),1)
 return np.column_stack([np.where(valid[:,:,None],source[k][ids],0.).sum(1)/count[:,None] for k in ['acceleration','angular_velocity']])
def binding(times,latest,current,ttl=100_000_000):
 i=np.maximum(latest,0);ba=times-current['bag_ns'][i];ha=times-current['header_ns'][i]
 return (latest>=0)&(current['status'][i]==0)&(times>=current['endpoint_ns'][i])&(ba>=0)&(ha>=0)&(ba<=ttl)&(ha<=ttl)
def infer(model,x):
 x=np.asarray(x,float)
 if x.ndim!=2 or not np.isfinite(x).all():raise ValueError('finite matrix required')
 result=np.full(len(x),model['baseline'],float)
 for nodes in model['trees']:
  old='leaf' in nodes[0];leaf=np.array([v['leaf' if old else 'is_leaf'] for v in nodes],bool);value=np.array([v.get('value',0.) for v in nodes]);feature=np.array([v.get('feature' if old else 'feature_idx',0) for v in nodes],int);threshold=np.array([v.get('threshold' if old else 'num_threshold',0.) for v in nodes]);left=np.array([v.get('left',0) for v in nodes],int);right=np.array([v.get('right',0) for v in nodes],int);at=np.zeros(len(x),int)
  for _ in range(len(nodes)+1):
   active=~leaf[at]
   if not active.any():break
   rows=np.flatnonzero(active);j=at[rows];at[rows]=np.where(x[rows,feature[j]]<=threshold[j],left[j],right[j])
  else:raise ValueError('cyclic tree')
  result+=value[at]
 return result
def radar_features(x,anchor,median):
 tree=ast.parse((RESEARCH/'vehicle_context_boosting/context_core.py').read_text(encoding='utf-8'));fn=next(v for v in tree.body if isinstance(v,ast.FunctionDef) and v.name=='features');namespace={'np':np,'SPANS':(5,11,21,41,81,128)};exec(compile(ast.Module(body=[fn],type_ignores=[]),'frozen_radar_features','exec'),namespace)
 return namespace['features'](x,anchor,median)
def inputs():
 cache=load(RESEARCH/'vehicle_multisource_imu/feature_cache.npz');parent=load(RESEARCH/'vehicle_multisource_imu/development_replay.npz');new=load(RESEARCH/'vehicle_nonlinear_correction/development_replay.npz');signed=load(RESEARCH/'vehicle_signed_axis_correction/feature_cache.npz');prepared=load(RESEARCH/'vehicle_masked_history/prepared.npz');night=load(RESEARCH/'vehicle_night_training/predictions.npz');dec=load(RESEARCH/'vehicle_night_training/decisions.npz');diag=load(RESEARCH/'vehicle_remaining_failure_audit/row_diagnostics.npz');index=read(RESEARCH/'vehicle_masked_history/frame_index.json');ready=np.array([v['ready'] for v in index],bool);events=np.array([v['time_ns'] for v in index],np.int64);times=parent['radar_imu__request__request_times_ns'];req=read(RESEARCH/'vehicle_availability_audit/request_reference.json')
 if len(events)!=40690 or ready.sum()!=40678 or len(times)!=37340:raise ValueError('fixed cohort differs')
 np.testing.assert_array_equal(events,cache['full_current__endpoint_ns']);np.testing.assert_array_equal(np.flatnonzero(ready),diag['full__row_id']);np.testing.assert_array_equal(events[ready],diag['full__time_ns']);np.testing.assert_array_equal(times,[v['time_ns'] for v in req]);np.testing.assert_array_equal([v['array_index'] for v in index if v['ready']],np.arange(40678));y=np.full(len(events),np.nan);y[ready]=diag['full__target_mps'];peer=np.full(len(events),np.nan);peer[ready]=night['day_plus_night__masked_cnn_4ch'];duration=int(events[-1])-int(events[0]);radar=radar_features(prepared['X'],night['anchor'],prepared['median_feature'])
 return {'ready':ready,'events':events,'times':times,'raw_y':y,'req_y':np.array([np.nan if v['reference_mps'] is None else v['reference_mps'] for v in req]),'peer':peer,'source':dec['day_plus_night__quality'],'anchor':night['anchor'],'radar':radar,'source_imu':im.decode_payloads(load(RESEARCH/'vehicle_livox_collection/runtime_export/Urban01_night.npz')),'parent_model':read(RESEARCH/'vehicle_multisource_imu/models/all_fit/radar_imu.json'),'head':read(RESEARCH/'vehicle_nonlinear_correction/models/all_fit/nonlinear7.json'),'radar_only':parent['original_context__ready_prediction_mps'],'parent_nominal':parent,'new_nominal':new,'imu_nominal':cache,'axes_nominal':signed['full_axes'],'start':int(events[0])+duration*2//5,'end':int(events[0])+duration*3//5}
def summarize(data,p,keep,replay):
 t=data['times'];segment=(t-t[0])//300_000_000_000;masks={'all':np.ones(len(t),bool),'before':t<data['start'],'during':(t>=data['start'])&(t<data['end']),'after':t>=data['end'],**{'segment_'+str(int(k)):segment==k for k in np.unique(segment)}}
 def cell(mask):
  accepted=mask&(replay['status']==0);return {'requests':int(mask.sum()),'accepted_requests':int(accepted.sum()),'availability':float(accepted.sum()/mask.sum()),'statuses':{str(k):int(np.sum(mask&(replay['status']==k))) for k in range(5)},'accepted_errors':n.metrics(data['req_y'],replay['prediction_mps'],accepted)}
 return {'requests':{name:cell(mask) for name,mask in masks.items()},'ungated_ready_errors':n.metrics(data['raw_y'],p,data['ready']),'endpoint_usable':int(keep.sum())}
def simulate(data,case):
 source=mutate(data['source_imu'],case,data['start'],data['end']);features,current=im.features(source,data['events']);signed=axes(source,data['events']);ready=data['ready'];parent=np.maximum(0.,data['anchor']+infer(data['parent_model'],np.column_stack([data['radar'],features[ready]])));candidate=np.maximum(0.,parent+infer(data['head'],np.column_stack([parent,signed[ready]])));pred={'radar_only':data['radar_only'],'radar_imu_parent':parent,'nonlinear7':candidate};latest=np.searchsorted(data['events'],data['times'],side='right')-1;bound=binding(data['times'],latest,current);output={};result={}
 if case=='nominal':
  np.testing.assert_allclose(features,data['imu_nominal']['full_imu'],rtol=0,atol=1e-10);np.testing.assert_allclose(signed,data['axes_nominal'],rtol=0,atol=1e-10);np.testing.assert_allclose(parent,data['parent_nominal']['radar_imu__ready_prediction_mps'],rtol=0,atol=1e-9);np.testing.assert_allclose(candidate,data['new_nominal']['nonlinear7__ready_prediction_mps'],rtol=0,atol=1e-9)
 for method,values in pred.items():
  p=np.full(len(ready),np.nan);p[ready]=values;keep=data['source']&np.isfinite(p)&np.isfinite(data['peer'])&(p>=0)&(data['peer']>=0)&(abs(p-data['peer'])<=1.);b=np.ones(len(latest),bool)
  if method!='radar_only':keep&=current['status']==0;b=bound
  replay=n.serve(data['times'],data['events'],ready,p,keep,b)
  if case=='nominal':
   old=data['new_nominal'] if method=='nonlinear7' else data['parent_nominal'];key={'nonlinear7':'nonlinear7','radar_imu_parent':'radar_imu','radar_only':'original_context'}[method];np.testing.assert_array_equal(keep,old[key+'__quality']);np.testing.assert_array_equal(replay['status'],old[key+'__request__status'])
  result[method]=summarize(data,p,keep,replay);output[method+'__ready_prediction_mps']=values;output[method+'__quality']=keep
  for k,v in replay.items():output[method+'__request__'+k]=v
 output['current__status']=current['status'];output['current__bag_ns']=current['bag_ns'];output['current__header_ns']=current['header_ns'];output['binding']=bound
 return result,output
def freeze(create=False):
 p=read(ROOT/'protocol.json')
 if p['cases']!=list(CASES) or p['methods']!=list(METHODS):raise ValueError('fixed matrix differs')
 for name,h in p['input_sha256'].items():
  if 'urban02' in name.lower() or 'rav4' in name.lower() or name.startswith('vehicle_final_evaluation/'):raise ValueError('protected input')
  if digest(RESEARCH/name)!=h:raise ValueError('frozen input changed '+name)
 lock={'protocol_sha256':digest(ROOT/'protocol.json'),'code_sha256':{name:digest(ROOT/name) for name in ['fault_core.py','run_faults.py']}}
 if (ROOT/'analysis_lock.json').exists():
  if read(ROOT/'analysis_lock.json')!=lock:raise ValueError('frozen code changed')
 elif create:write(ROOT/'analysis_lock.json',lock)
 else:raise ValueError('freeze before scoring')
