"""Offline auxiliary-source qualification; no model or policy changes."""
from pathlib import Path
import hashlib,json,struct,sys,math
import numpy as np
ROOT=Path(__file__).resolve().parent;RESEARCH=ROOT.parent;PROBE=RESEARCH/'witness_calibration_probe'
sys.path.insert(0,str(PROBE));import probe_core as c
TOPICS=('/livox/imu','/xsens/imu');MD5='6a62c6daae103f4ff57a132d6f95cec2';TTL=100_000_000
STATUS={0:'available',1:'no_prior_sample',2:'stale',3:'future_header',4:'invalid_measurement_or_covariance'}
ROLES=('fit','calibration','evaluation_urban03','evaluation_urban04')
PENDING={'Urban03':'Urban03/2023-11-23-19-12-16-Night','Urban04':'Urban04/2023-11-23-19-36-42-Night'}
def read(f):return json.loads(Path(f).read_text(encoding='utf-8'))
def write(f,v):Path(f).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def digest(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def check_metadata_source(route,name):
 if route not in PENDING.values() or name!='imus.bag':raise ValueError('only two pending fitting IMU metadata sources allowed')
def allowed_metadata_range(start,end,size,part):
 if any(type(v) is not int for v in [start,end,size]) or not 0<=start<=end<size:raise ValueError('invalid metadata byte range')
 if part=='header' and start==0 and end==4095:return
 if part=='index' and start>=4096 and end==size-1 and end-start+1<=2_097_152:return
 raise ValueError('only4KiB header and bounded index tail may be downloaded')
def covariance_valid(cov):
 a=np.asarray(cov,dtype=float)
 return bool(np.isfinite(a).all() and a[0]!=-1 and np.all(a[[0,4,8]]>=0))
def decode_imu(blob,bag_ns,topic):
 if topic not in TOPICS:raise ValueError('reference, filtered or unknown predictor channel forbidden')
 if type(bag_ns) is not int or not 0<=bag_ns<=np.iinfo(np.int64).max or len(blob)<16:raise ValueError('invalid IMU record/header')
 seq,seconds,nanos,n=struct.unpack_from('<IIII',blob)
 if nanos>=1_000_000_000 or n==0 or len(blob)!=16+n+296:raise ValueError('invalid IMU timestamp/frame/payload size')
 try:frame=blob[16:16+n].decode('utf-8')
 except UnicodeDecodeError as exc:raise ValueError('invalid IMU frame encoding') from exc
 v=struct.unpack_from('<37d',blob,16+n);gyro=np.array(v[13:16]);acc=np.array(v[25:28]);gc=np.array(v[16:25]);ac=np.array(v[28:37]);scale=9.8 if topic==TOPICS[0] else 1.
 # Preserve the -1 unavailable sentinel instead of scaling it as a variance.
 scaled_acc=np.array([float(x)*scale for x in acc]);scaled_cov=ac.copy() if ac[0]==-1 else np.array([float(x)*scale**2 for x in ac])
 valid=bool(np.isfinite(gyro).all() and np.isfinite(scaled_acc).all() and covariance_valid(gc) and covariance_valid(scaled_cov) and math.isfinite(math.hypot(*gyro)) and math.isfinite(math.hypot(*scaled_acc)))
 return {'sequence':seq,'header_ns':seconds*1_000_000_000+nanos,'bag_ns':bag_ns,'topic':topic,'frame_id':frame,'angular_velocity':gyro.tolist(),'acceleration':scaled_acc.tolist(),'angular_velocity_covariance':gc.tolist(),'acceleration_covariance':scaled_cov.tolist(),'angular_velocity_covariance_unknown':bool(np.all(gc==0)),'acceleration_covariance_unknown':bool(np.all(ac==0)),'usable':valid}

def context(rows,endpoints):
 t=np.asarray(endpoints)
 if t.ndim!=1 or t.dtype.kind!='i' or np.any(t<0) or np.any(t[1:]<=t[:-1]):raise ValueError('strict integer endpoint clock required')
 n=len(rows);bag=np.array([v['bag_ns'] for v in rows],dtype=np.int64);header=np.array([v['header_ns'] for v in rows],dtype=np.int64)
 if n and (np.any(bag[1:]<=bag[:-1]) or len({(v['topic'],v['frame_id']) for v in rows})!=1):raise ValueError('duplicate/reordered arrivals or mixed source frames')
 latest=np.searchsorted(bag,t,side='right')-1;status=np.ones(len(t),dtype=np.uint8);ha=np.full(len(t),-1,dtype=np.int64);ba=ha.copy();acc=np.full(len(t),np.nan);gyro=acc.copy()
 for j,i in enumerate(latest):
  if i<0:continue
  ha[j]=int(t[j])-int(header[i]);ba[j]=int(t[j])-int(bag[i]);row=rows[i]
  if header[i]>t[j]:status[j]=3
  elif ha[j]>TTL or ba[j]>TTL:status[j]=2
  elif not row['usable']:status[j]=4
  else:
   an=math.hypot(*row['acceleration']);gn=math.hypot(*row['angular_velocity'])
   if not math.isfinite(an) or not math.isfinite(gn):status[j]=4
   else:status[j]=0;acc[j]=an;gyro[j]=gn
 return {'endpoint_ns':t.copy(),'latest_row':latest,'status':status,'header_age_ns':ha,'arrival_age_ns':ba,'acceleration_norm':acc,'angular_velocity_norm':gyro}

def descriptive(x):
 x=np.asarray(x,dtype=float);good=x[np.isfinite(x)]
 return {'finite_count':len(good),'missing_or_nonfinite':int(len(x)-len(good)),'minimum':float(good.min()) if len(good) else None,'median':float(np.median(good)) if len(good) else None,'p95':float(np.quantile(good,.95)) if len(good) else None,'maximum':float(good.max()) if len(good) else None}

def context_stats(q):
 good=q['status']==0;n=len(good)
 return {'probe_ticks':n,'available_ticks':int(good.sum()),'available_fraction_on_probe_ticks':float(good.mean()) if n else None,'status_counts':{name:int(np.sum(q['status']==code)) for code,name in STATUS.items()},'available_header_age_ns':descriptive(q['header_age_ns'][good]),'available_acceleration_norm_declared_mps2':descriptive(q['acceleration_norm'][good]),'available_angular_velocity_norm_declared_radps':descriptive(q['angular_velocity_norm'][good])}

def inventory(meta,index):
 chunks=c.chunk_infos(index);parsed={}
 for h,blob in c.records(index):
  if h['op']==b'\x07':
   fields=c.fields(blob);cid=c.u32(h['conn']);value={'topic':h['topic'].decode(),'type':fields['type'].decode(),'md5sum':fields['md5sum'].decode()}
   if cid in parsed and parsed[cid]!=value:raise ValueError('ambiguous metadata connection')
   parsed[cid]=value
 result={}
 for v in meta['connections']:
  observed={k:v[k] for k in ['topic','type','md5sum']}
  if parsed.get(v['conn'])!=observed:raise ValueError('metadata JSON/index connection differs')
  if v['type']=='sensor_msgs/Imu':
   if v['md5sum']!=MD5:raise ValueError('noncanonical IMU schema')
   result[v['topic']]={'indexed_messages':sum(q['counts'].get(v['conn'],0) for q in chunks),'type':v['type'],'md5sum':v['md5sum']}
 return {'source_url':meta['url'],'source_size_bytes':meta['size'],'indexed_chunks':len(chunks),'imu_topics':result,'full_bag_payload_verified':False,'counts_are_usable_samples':False}

def extract():
 plan=read(PROBE/'acquisition_plan.json');meta=read(PROBE/'source_index/imus.bag.json');chunks=c.chunk_infos((PROBE/'source_index/imus.bag.index').read_bytes());conns={v['conn']:v['topic'] for v in meta['connections'] if v['topic'] in TOPICS};rows={t:[] for t in TOPICS};bychunk={};allrecords=0
 for q in plan['ranges']:
  if q['file']!='imus.bag':continue
  name=q['relative_cache'];blob=(PROBE/'evidence/chunks'/name).read_bytes();outer=list(c.records(blob));h,body=outer[0]
  if h['op']!=b'\x05' or h.get('compression')!=b'none' or c.u32(h['size'])!=len(body):raise ValueError('unsupported/corrupt cached IMU chunk')
  counts={};bychunk[q['chunk_index']]={t:[] for t in TOPICS}
  for rh,data in c.records(body):
   if rh['op']!=b'\x02':continue
   cid=c.u32(rh['conn']);counts[cid]=counts.get(cid,0)+1;allrecords+=1
   if cid in conns:
    row=decode_imu(data,c.ros_time(rh['time']),conns[cid]);rows[conns[cid]].append(row);bychunk[q['chunk_index']][conns[cid]].append(row)
  if counts!=chunks[q['chunk_index']]['counts']:raise ValueError('all-connection decoded chunk counts differ from index')
 for values in rows.values():values.sort(key=lambda v:v['bag_ns'])
 return plan,rows,bychunk,allrecords

def compute():
 freeze();plan,rows,bychunk,allrecords=extract();inventories={}
 for role in ROLES:
  folder=RESEARCH/'vehicle_temporal_policy/metadata'/role;inventories[role]=inventory(read(folder/'imus.bag.json'),(folder/'imus.bag.index').read_bytes())
 arrays={};stats={};windows={}
 for topic,values in rows.items():
  name='livox' if topic==TOPICS[0] else 'xsens';gy=np.array([v['angular_velocity'] for v in values]);ac=np.array([v['acceleration'] for v in values]);arrival=np.array([v['bag_ns'] for v in values],dtype=np.int64);header=np.array([v['header_ns'] for v in values],dtype=np.int64)
  for k,v in {'sequence':np.array([v['sequence'] for v in values],dtype=np.uint32),'arrival_ns':arrival,'header_ns':header,'angular_velocity':gy,'acceleration':ac,'gyro_covariance':np.array([v['angular_velocity_covariance'] for v in values]),'accel_covariance':np.array([v['acceleration_covariance'] for v in values]),'usable':np.array([v['usable'] for v in values],dtype=bool),'frame_id':np.array([v['frame_id'] for v in values])}.items():arrays[name+'__'+k]=v
  stats[name]={'topic':topic,'decoded_records':len(values),'frames':sorted({v['frame_id'] for v in values}),'usable_payload_records':sum(v['usable'] for v in values),'unknown_gyro_covariance_records':sum(v['angular_velocity_covariance_unknown'] for v in values),'unknown_accel_covariance_records':sum(v['acceleration_covariance_unknown'] for v in values),'future_header_at_writer_records':int(np.sum(header>arrival)),'header_minus_writer_ns':descriptive(header-arrival),'acceleration_norm_declared_mps2':descriptive([math.hypot(*v) for v in ac]),'gyro_norm_declared_radps':descriptive([math.hypot(*v) for v in gy]),'declared_acceleration_scale':9.8 if name=='livox' else 1.0,'configured_units_independently_verified':False}
  for w in plan['windows']:
   ids=w['imu_chunks'];values=sorted([v for i in ids for v in bychunk[i][topic]],key=lambda v:v['bag_ns']);sourcechunks=c.chunk_infos((PROBE/'source_index/imus.bag.index').read_bytes());start=min(sourcechunks[i]['start_ns'] for i in ids);end=max(sourcechunks[i]['end_ns'] for i in ids);ticks=start+np.arange((end-start)//50_000_000+1,dtype=np.int64)*50_000_000;q=context(values,ticks);windows.setdefault(w['id'],{})[name]=context_stats(q)
   for k,v in q.items():arrays[name+'__'+w['id']+'__'+k]=v
 result={'scope':'Cached Urban01 DAY auxiliary IMU pilot plus four captured source indexes; no full-cohort or accuracy validation','inventory':inventories,'pilot_source_route':read(PROBE/'protocol.json')['route'],'pilot_chunks':len(bychunk),'all_connection_records_checked':allrecords,'pilot_channels':stats,'by_probe_window':windows,'pending_fit_metadata':PENDING,'xsens_missing_full_development':TOPICS[1] not in inventories['calibration']['imu_topics'],'livox_present_in_all_four_captured_indexes':all(TOPICS[0] in v['imu_topics'] for v in inventories.values()),'full_cohort_source_coverage_verified':False,'model_training_started':False,'independent_validation_done':False,'accuracy_failures_resolved':False,'unit_bound_established':False,'live_time_authentication_established':False,'source_partial_bags_only':True,'witness_enabled':False,'production_policy_changed':False,'reserved_final_opened':False,'promotion_allowed':False,'decision':'AUXILIARY_IMU_PILOT_COMPLETE_METADATA_REQUIRED'}
 return result,arrays

def freeze(create=False):
 p=read(ROOT/'protocol.json')
 if p['topics']!=list(TOPICS) or p['arrival_and_header_ttl_ns']!=TTL or p['livox_acceleration_scale']!=9.8 or p['pending_routes']!=PENDING or p['model_training'] or p['promotion_allowed']:raise ValueError('auxiliary qualification protocol changed')
 for n,h in p['input_sha256'].items():
  path=(ROOT/n).resolve()
  if not path.is_relative_to(RESEARCH) or digest(path)!=h:raise ValueError('auxiliary source changed: '+n)
 lock={'protocol_sha256':digest(ROOT/'protocol.json'),'code_sha256':{n:digest(ROOT/n) for n in ['aux_core.py','build_auxiliary.py','acquire_metadata.py']}};path=ROOT/'analysis_lock.json'
 if path.exists() and read(path)!=lock:raise ValueError('auxiliary implementation changed after lock')
 if create and not path.exists():write(path,lock)
 return p
