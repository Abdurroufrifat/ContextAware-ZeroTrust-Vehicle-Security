"""Portable read-only frame and reference audit; no model imports or fitting."""
from pathlib import Path
import hashlib,json,math,xml.etree.ElementTree as ET
import numpy as np
ROOT=Path(__file__).resolve().parent;RESEARCH=ROOT.parent;PROJECT=RESEARCH.parent
METHODS=('original_context','uncorrected_radar_imu','signed_axis_corrected');PRIMARY=METHODS[-1]
RAW=RESEARCH/'vehicle_livox_collection/runtime_export';SOURCES=('Urban01_day','Urban03_day','Urban04_day','Urban03_night','Urban04_night','Urban01_night')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def metrics(y,p):
 y=np.asarray(y,dtype=float);p=np.asarray(p,dtype=float)
 if y.ndim!=1 or p.shape!=y.shape or not np.isfinite(y).all() or not np.isfinite(p).all() or np.any(y<0) or np.any(p<0):raise ValueError('unaligned/nonfinite speed comparison')
 e=p-y;a=abs(e);n=len(y)
 return {'count':n,'mae_mps':float(a.mean()) if n else None,'rmse_mps':float(np.sqrt(np.mean(e*e))) if n else None,'p95_mps':float(np.quantile(a,.95)) if n else None,'maximum_mps':float(a.max()) if n else None,'bias_prediction_minus_reference_mps':float(e.mean()) if n else None,'above_half_mps':int((a>.5).sum()),'above_2_mps':int((a>2.).sum())}
def stats(x):
 x=np.asarray(x);v=x[np.isfinite(x)]
 return {'total':len(x),'finite':len(v),'min':float(v.min()) if len(v) else None,'median':float(np.median(v)) if len(v) else None,'p95':float(np.quantile(v,.95)) if len(v) else None,'max':float(v.max()) if len(v) else None}
def reference_lookup(records):
 lookup={}
 for i,v in enumerate(records):
  t=v['time_ns'];key=(v['window_id'],t)
  if type(t) is not int or t<0 or key in lookup:raise ValueError('duplicate/noninteger exact reference key')
  lookup[key]=i
 return lookup
def join_ids(keys,lookup):return [lookup.get(key,-1) for key in keys]
def sigma_available(row,time):
 t=row.get('uncertainty_header_time_ns');v=[row.get(k) for k in ['north_velocity_stdev','east_velocity_stdev']]
 return type(t) is int and 0<=time-t<=1_500_000_000 and all(type(x) in (int,float) and math.isfinite(x) and x>=0 for x in v)
def orientation_available(q,c):
 q=np.asarray(q,dtype=float);c=np.asarray(c,dtype=float)
 if q.ndim!=2 or q.shape[1]!=4 or c.shape!=(len(q),9):raise ValueError('invalid orientation arrays')
 return np.isfinite(q).all(1)&np.isfinite(c).all(1)&(c[:,0]!=-1)&(abs(np.linalg.norm(q,axis=1)-1.)<=1e-6)
def horizontal_speed(north,east,up):
 if not all(math.isfinite(x) for x in [north,east,up]):raise ValueError('nonfinite reference velocity')
 return math.hypot(north,east)
def paired_metrics(y,p,obd):
 y=np.asarray(y,dtype=float);p=np.asarray(p,dtype=float);obd=np.asarray(obd,dtype=float)
 if y.ndim!=1 or p.shape!=y.shape or obd.shape!=y.shape:raise ValueError('unaligned alternate-reference comparison')
 bad=abs(p-y)>.5;alt=abs(p-obd)>.5
 return {'primary_reference':metrics(y,p),'obd_comparison':metrics(obd,p),'obd_minus_primary_reference':metrics(y,obd),'primary_tail_also_above_half_against_obd':int((bad&alt).sum()),'primary_tail_at_or_below_half_against_obd':int((bad&~alt).sum())}
def launch_geometry(text,captured_frame):
 nodes=[]
 for node in ET.fromstring(text).iter('node'):
  if node.get('pkg')!='tf2_ros' or node.get('type')!='static_transform_publisher':continue
  args=node.get('args','').split()
  if len(args) not in (8,9) or not all(math.isfinite(float(v)) for v in args[:-2]):raise ValueError('unsupported static transform arguments')
  nodes.append({'node_name':node.get('name'),'parent':args[-2],'child':args[-1],'numeric_arguments':list(map(float,args[:-2]))})
 children=[v['child'] for v in nodes];frames=set(children)|{v['parent'] for v in nodes}
 return {'static_publisher_count':len(nodes),'frames':sorted(frames),'captured_frame':captured_frame,'captured_frame_in_transform_tree':captured_frame in frames,'nodes_named_as_captured_frame':[v for v in nodes if v['node_name']==captured_frame],'duplicate_child_frames':sorted({v for v in children if children.count(v)>1}),'nodes':nodes,'node_name_is_not_frame_identity':True,'rotation_applied':False}
def freeze(create=False):
 if not create and not (ROOT/'analysis_lock.json').is_file():raise ValueError('source audit must be frozen')
 p=read(ROOT/'protocol.json')
 if p['new_model_fitted'] or p['labels_changed'] or p['promotion_allowed'] or p['sigma_match_limit_ns']!=1_500_000_000 or p['retrospective_reconstruction'] is not True:raise ValueError('source-audit scope differs')
 for name,h in p['input_sha256'].items():
  f=(ROOT/name).resolve()
  if not f.is_relative_to(RESEARCH) or digest(f)!=h:raise ValueError('source-audit input changed '+name)
 lock={'protocol_sha256':digest(ROOT/'protocol.json'),'code_sha256':{n:digest(ROOT/n) for n in ['frame_reference_core.py','generate_frame_reference.py']}};file=ROOT/'analysis_lock.json'
 if file.exists() and read(file)!=lock:raise ValueError('source-audit implementation changed')
 if create and not file.exists():write(file,lock)
def orientation_profiles():
 profiles={}
 for key in SOURCES:
  with np.load(RAW/(key+'.npz'),allow_pickle=False) as z:
   data=z['payload_bytes'];length=z['payload_length'];head=data[:,:16].copy().view('<u4').reshape(len(data),4);fl=int(head[0,3]);size=16+fl+296
   if np.any(head[:,3]!=fl) or np.any(length!=size) or data.shape[1]<size:raise ValueError('mixed/truncated serialized source')
   frame=data[0,16:16+fl].tobytes().decode('utf-8')
   if not np.all(data[:,16:16+fl]==data[0,16:16+fl]):raise ValueError('mixed source frame')
   v=data[:,16+fl:size].copy().view('<f8').reshape(len(data),37);q=v[:,:4];oc=v[:,4:13];norm=np.linalg.norm(q,axis=1);scaled=np.hypot.reduce(v[:,25:28]*9.8,axis=1);np.testing.assert_allclose(scaled,z['acceleration_norm'],rtol=0,atol=1e-10)
   profiles[key]={'records':len(q),'frame_id':frame,'zero_quaternion_records':int(np.all(q==0,axis=1).sum()),'usable_orientation_records':int(orientation_available(q,oc).sum()),'zero_orientation_covariance_records':int(np.all(oc==0,axis=1).sum()),'acceleration_scale_9p8_reproduced_rows':len(q),'unknown_acceleration_covariance_rows':int(np.all(v[:,28:37]==0,axis=1).sum()),'unknown_gyro_covariance_rows':int(np.all(v[:,16:25]==0,axis=1).sum()),'quaternion_norm':stats(norm)}
 return profiles
PAIR_FIELDS=('row_id','reference_match_id','comparison_present','obd_mps','gnss_mps','sigma_present','uncertainty_age_ns','reported_component_sigma_max_mps','reference_3d_minus_horizontal_mps','wheel_update_code')
def load_pairs():
 records=read(RESEARCH/'vehicle_reference_check/matched_endpoints.json');lookup=reference_lookup(records)
 with np.load(RESEARCH/'vehicle_remaining_failure_audit/row_diagnostics.npz',allow_pickle=False) as z:out={s:{n[len(s)+2:]:z[n].copy() for n in z.files if n.startswith(s+'__')} for s in ['fit','full']}
 for scope,q in out.items():
  ids=np.array(join_ids(zip(q['window_id'].tolist(),map(int,q['time_ns'])),lookup),dtype=np.int64);n=len(ids);q['reference_match_id']=ids;q['comparison_present']=ids>=0
  for name in ['obd_mps','gnss_mps','reported_component_sigma_max_mps','reference_3d_minus_horizontal_mps']:q[name]=np.full(n,np.nan)
  q['sigma_present']=np.zeros(n,bool);q['uncertainty_age_ns']=np.full(n,-1,dtype=np.int64);q['wheel_update_code']=np.full(n,-1,dtype=np.int8)
  for i in np.flatnonzero(ids>=0):
   r=records[ids[i]];t=int(q['time_ns'][i])
   if r['reference_horizontal_mps']!=float(q['target_mps'][i]) or t-r['reference_header_time_ns']!=int(q['reference_age_ns'][i]):raise ValueError('joined target/reference clock changed')
   q['obd_mps'][i]=r['label_mps'];q['gnss_mps'][i]=r['gnss_mps'];q['reference_3d_minus_horizontal_mps'][i]=r['reference_3d_mps']-r['reference_horizontal_mps'];q['sigma_present'][i]=sigma_available(r,t);stamp=r['uncertainty_header_time_ns'];q['uncertainty_age_ns'][i]=t-stamp if stamp is not None else -1;q['wheel_update_code'][i]=-1 if r['wheel_sensor_update_bit'] is None else int(r['wheel_sensor_update_bit'])
   if q['sigma_present'][i]:q['reported_component_sigma_max_mps'][i]=max(r['north_velocity_stdev'],r['east_velocity_stdev'])
 return out
def pair_cell(q,ids):
 y=q['target_mps'][ids];p=q[PRIMARY+'__prediction_mps'][ids];bad=abs(p-y)>.5
 return {'rows':len(ids),'models':{n:paired_metrics(y,q[n+'__prediction_mps'][ids],q['obd_mps'][ids]) for n in METHODS},'gnss_minus_primary_reference':metrics(y,q['gnss_mps'][ids]),'tail_vs_other':{name:{'rows':int(flag.sum()),'obd_minus_primary_reference':metrics(y[flag],q['obd_mps'][ids][flag]),'reported_component_sigma_max_mps':stats(q['reported_component_sigma_max_mps'][ids][flag]),'reference_3d_minus_horizontal_mps':stats(q['reference_3d_minus_horizontal_mps'][ids][flag]),'uncertainty_age_ns':stats(q['uncertainty_age_ns'][ids][flag]),'missing_sigma':int((~q['sigma_present'][ids][flag]).sum()),'wheel_update_bit_set':int((q['wheel_update_code'][ids][flag]==1).sum())} for name,flag in [('tail',bad),('other',~bad)]}}
def compute():
 freeze();rows=load_pairs();pop={}
 for scope,q in rows.items():
  windows={}
  for name in sorted(set(q['window_id'].tolist())):
   ids=np.flatnonzero(q['window_id']==name);matched=ids[q['comparison_present'][ids]];sel=matched[q[PRIMARY+'__accepted'][matched]];windows[name]={'parent_rows':len(ids),'exact_comparison_rows':len(matched),'missing_comparison_rows':len(ids)-len(matched),'corrected_accepted_comparison_rows':len(sel),'all_matched':pair_cell(q,matched),'corrected_accepted_matched':pair_cell(q,sel)}
  pop[scope]={'parent_rows':len(q['row_id']),'comparison_rows':int(q['comparison_present'].sum()),'missing_comparison_rows':int((~q['comparison_present']).sum()),'windows':windows}
 result={'scope':'Retrospective published-frame and saved-reference audit; no new evaluation','populations':pop,'raw_imu_orientation':orientation_profiles(),'published_geometry':read(ROOT/'sources/geometry_audit.json'),'root_cause_established':False,'new_model_fitted':False,'labels_changed':False,'thresholds_changed':False,'geometry_physically_revalidated':False,'reference_threat_independence_established':False,'reported_standard_deviations_are_uniform_bounds':False,'postprocessed_attitude_used_in_predictor':False,'reserved_final_opened':False,'promotion_allowed':False,'witness_enabled':False,'decision':'FRAME_REFERENCE_AUDIT_COMPLETE_NO_PROMOTION'}
 return result,{scope+'__'+field:q[field] for scope,q in rows.items() for field in PAIR_FIELDS}
