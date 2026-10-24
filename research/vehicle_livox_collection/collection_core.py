"""Fixed-cohort auxiliary-source collection primitives; no model or permission."""
from pathlib import Path
import collections,hashlib,io,json,math,os,struct,sys
import numpy as np
ROOT=Path(__file__).resolve().parent;RESEARCH=ROOT.parent
sys.path.insert(0,str(RESEARCH/'vehicle_auxiliary_imu'));import aux_core as a
read=a.read;write=a.write;digest=a.digest;c=a.c
ROUTES={'Urban01_day':'Urban01/2023-11-23-15-38-43-Day','Urban03_day':'Urban03/2023-11-03-09-42-41-Day','Urban04_day':'Urban04/2023-11-01-12-16-16-Day','Urban03_night':'Urban03/2023-11-23-19-12-16-Night','Urban04_night':'Urban04/2023-11-23-19-36-42-Night','Urban01_night':'Urban01/2023-11-23-18-24-14-Night'}
PREFIX_NS=6_500_000_000;URL_PREFIX='https://www.frdr-dfdr.ca/repo/files/1/published/publication_1084/submitted_data/'
def allowed_source(route):return route in ROUTES.values()
def atomic(path,blob):
 path.parent.mkdir(parents=True,exist_ok=True);part=path.with_name(path.name+'.part')
 with part.open('wb') as out:out.write(blob);out.flush();os.fsync(out.fileno())
 os.replace(part,path)
def select_chunks(chunks,spans,index_pos,size):
 if not chunks or type(index_pos) is not int or type(size) is not int or not 4096<index_pos<size:raise ValueError('invalid source index bounds')
 if not spans or any(len(s)!=2 or any(type(v) is not int for v in s) or not 0<=s[0]<=s[1] for s in spans):raise ValueError('invalid collection intervals')
 positions=[v['position'] for v in chunks]
 if positions[0]<4096 or positions[-1]>=index_pos or any(x>=y for x,y in zip(positions,positions[1:])):raise ValueError('ambiguous chunk positions')
 result=[]
 for i,v in enumerate(chunks):
  if v['start_ns']>v['end_ns']:raise ValueError('invalid chunk time interval')
  if any(v['start_ns']<=end and v['end_ns']>=start for start,end in spans):
   end=positions[i+1]-1 if i+1<len(chunks) else index_pos-1
   if not 0<end-positions[i]+1<=16_777_216:raise ValueError('unexpected chunk size')
   result.append({'chunk_index':i,'start':positions[i],'end':end,'start_ns':v['start_ns'],'end_ns':v['end_ns'],'counts':{str(k):n for k,n in v['counts'].items()}})
 if not result:raise ValueError('no source chunks cover fixed cohort')
 return result
def decode_chunk(blob,expected,meta):
 connections={v['conn']:v for v in meta['connections']}
 if len(connections)!=len(meta['connections']):raise ValueError('duplicate connection metadata')
 wanted={i for i,v in connections.items() if v['topic']=='/livox/imu'}
 if not wanted or any(connections[i]['type']!='sensor_msgs/Imu' or connections[i]['md5sum']!=a.MD5 for i in wanted):raise ValueError('Livox source/schema missing or changed')
 records=list(c.records(blob))
 if not records:raise ValueError('empty chunk range')
 h,body=records[0]
 if h['op']!=b'\x05' or h.get('compression')!=b'none' or c.u32(h['size'])!=len(body) or any(v[0]['op']!=b'\x04' for v in records[1:]):raise ValueError('unsupported chunk framing/compression')
 counts=collections.Counter();rows=[]
 for ordinal,(h,payload) in enumerate(c.records(body)):
  if h['op']==b'\x07':
   cid=c.u32(h['conn']);declaration=c.fields(payload);known=connections.get(cid)
   if known is None or h.get('topic')!=known['topic'].encode() or declaration.get('type')!=known['type'].encode() or declaration.get('md5sum')!=known['md5sum'].encode():raise ValueError('embedded connection declaration differs from source index')
   continue
  if h['op']!=b'\x02':raise ValueError('unexpected record inside IMU chunk')
  cid=c.u32(h['conn'])
  if cid not in connections:raise ValueError('unknown chunk connection')
  counts[cid]+=1
  if cid in wanted:
   timestamp=c.ros_time(h['time']);a.decode_imu(payload,timestamp,'/livox/imu')
   rows.append({'bag_ns':timestamp,'payload':payload,'chunk_index':expected['chunk_index'],'record_ordinal':ordinal})
 if dict(counts)!={int(k):v for k,v in expected['counts'].items()}:raise ValueError('all-connection chunk count mismatch')
 return rows
def pack_raw(rows):
 if not rows:raise ValueError('no selected Livox messages')
 bag=np.array([v['bag_ns'] for v in rows],dtype=np.int64)
 if np.any(bag<0) or np.any(bag[1:]<=bag[:-1]):raise ValueError('duplicate/reordered selected arrivals')
 lengths=np.array([len(v['payload']) for v in rows],dtype=np.int64)
 if np.any(lengths<=0) or np.any(lengths>4096):raise ValueError('invalid selected payload length')
 raw=np.zeros((len(rows),int(lengths.max())),dtype=np.uint8);header=np.zeros(len(rows),dtype=np.int64);usable=np.zeros(len(rows),dtype=bool);acc=np.zeros(len(rows));gyro=acc.copy();frames=set();unknown_acc=unknown_gyro=0
 for i,row in enumerate(rows):
  data=a.decode_imu(row['payload'],int(bag[i]),'/livox/imu');raw[i,:lengths[i]]=np.frombuffer(row['payload'],dtype=np.uint8);header[i]=data['header_ns'];usable[i]=data['usable'];acc[i]=math.hypot(*data['acceleration']);gyro[i]=math.hypot(*data['angular_velocity']);frames.add(data['frame_id']);unknown_acc+=int(data['acceleration_covariance_unknown']);unknown_gyro+=int(data['angular_velocity_covariance_unknown'])
 if len(frames)!=1:raise ValueError('mixed source frames')
 arrays={'bag_ns':bag,'header_ns':header,'usable':usable,'acceleration_norm':acc,'angular_velocity_norm':gyro,'payload_length':lengths.astype(np.uint16),'payload_bytes':raw,'chunk_index':np.array([v['chunk_index'] for v in rows],dtype=np.int32),'record_ordinal':np.array([v['record_ordinal'] for v in rows],dtype=np.int32)}
 report={'records':len(rows),'usable_records':int(usable.sum()),'frames':sorted(frames),'unknown_acceleration_covariance_records':unknown_acc,'unknown_gyro_covariance_records':unknown_gyro,'future_header_at_writer_records':int(np.sum(header>bag)),'header_minus_writer_ns':a.descriptive(header-bag),'acceleration_norm_declared_mps2':a.descriptive(acc),'angular_velocity_norm_declared_radps':a.descriptive(gyro),'physical_unit_calibration_verified':False,'full_bag_checksum_verified':False,'authenticated_live_time_verified':False}
 return arrays,report
def align(arrays,targets):
 targets=np.asarray(targets);bag=arrays['bag_ns'];header=arrays['header_ns'];usable=arrays['usable']
 if targets.ndim!=1 or targets.dtype.kind!='i' or np.any(targets<0) or np.any(targets[1:]<=targets[:-1]) or not len(bag) or np.any(bag[1:]<=bag[:-1]):raise ValueError('ambiguous IMU/target clocks')
 if any(np.asarray(arrays[k]).shape!=bag.shape for k in ['header_ns','usable','acceleration_norm','angular_velocity_norm']):raise ValueError('unaligned IMU arrays')
 latest=np.searchsorted(bag,targets,side='right')-1;present=latest>=0;ids=np.maximum(latest,0);ha=np.where(present,targets-header[ids],-1);ba=np.where(present,targets-bag[ids],-1);status=np.ones(len(targets),dtype=np.uint8);future=present&(ha<0);stale=present&~future&((ha>a.TTL)|(ba>a.TTL));valid=present&~future&~stale;finite=np.isfinite(arrays['acceleration_norm'][ids])&np.isfinite(arrays['angular_velocity_norm'][ids]);status[future]=3;status[stale]=2;status[valid]=4;status[valid&usable[ids]&finite]=0;good=status==0
 return {'target_ns':targets.copy(),'latest_message':latest,'header_age_ns':ha,'arrival_age_ns':ba,'status':status,'acceleration_norm':np.where(good,arrays['acceleration_norm'][ids],np.nan),'angular_velocity_norm':np.where(good,arrays['angular_velocity_norm'][ids],np.nan)}
def cached_range(path,request):
 receipt=path.with_suffix('.json')
 if not path.is_file() or not receipt.is_file():return None
 value=read(receipt)
 if any(value.get(k)!=request[k] for k in ['url','start','end','total']):raise ValueError('cached range source identity differs')
 blob=path.read_bytes()
 if len(blob)!=request['end']-request['start']+1 or value.get('bytes')!=len(blob) or value.get('sha256')!=hashlib.sha256(blob).hexdigest():return None
 return blob
def inventory(key):
 folder=ROOT/'metadata'/key;meta=read(folder/'source.json');header=(folder/'imus.bag.header').read_bytes();index=(folder/'imus.bag.index').read_bytes();route=ROUTES[key]
 if not allowed_source(route) or meta['url']!=URL_PREFIX+route+'/imus.bag' or not header.startswith(b'#ROSBAG V2.0\n') or len(header) not in (4096,65536):raise ValueError('source identity/header differs')
 h=c.fields(header[17:17+c.u32(header,13)]);position=struct.unpack('<Q',h['index_pos'])[0]
 if position!=meta['index_pos'] or len(index)!=meta['size']-position:raise ValueError('binary header/index length differs')
 result=a.inventory(meta,index)
 if '/livox/imu' not in result['imu_topics']:raise ValueError('required Livox source absent')
 return meta,c.chunk_infos(index),result
def make_plan():
 windows=read(ROOT/'cohort_windows.json');sources={};total=0
 for key,route in ROUTES.items():
  meta,chunks,observed=inventory(key);spans=[[v['first_ns']-PREFIX_NS,v['last_ns']] for v in windows[key]];ranges=select_chunks(chunks,spans,meta['index_pos'],meta['size']);count=sum(q['end']-q['start']+1 for q in ranges);total+=count
  sources[key]={'route':route,'url':meta['url'],'size':meta['size'],'spans_ns':spans,'ranges':ranges,'download_bytes':count,'indexed_livox_messages_in_selected_chunks':sum(int(q['counts'].get(str(v['conn']),0)) for q in ranges for v in meta['connections'] if v['topic']=='/livox/imu'),'inventory':observed}
 return {'scope':'Fixed fitting windows and entire previously inspected Urban01 NIGHT development; no model fitting','history_prefix_ns':PREFIX_NS,'sources':sources,'total_download_bytes':total,'reserved_final_opened':False,'model_training_started':False,'production_policy_changed':False,'witness_enabled':False}
def freeze(create=False):
 if not (ROOT/'analysis_lock.json').is_file() and not create:raise ValueError('persisted collection lock missing; measurements forbidden')
 protocol=read(ROOT/'protocol.json')
 if protocol['routes']!=ROUTES or protocol['history_prefix_ns']!=PREFIX_NS or protocol['ttl_ns']!=a.TTL or protocol['topic']!='/livox/imu' or protocol['model_training'] or protocol['promotion_allowed']:raise ValueError('fixed collection protocol differs')
 for n,h in protocol['input_sha256'].items():
  path=(ROOT/n).resolve()
  if not path.is_relative_to(RESEARCH) or digest(path)!=h:raise ValueError('collection input changed: '+n)
 lock={'protocol_sha256':digest(ROOT/'protocol.json'),'plan_sha256':digest(ROOT/'acquisition_plan.json'),'code_sha256':{n:digest(ROOT/n) for n in ['collection_core.py','collect_livox.py','verify_collection.py']}}
 file=ROOT/'analysis_lock.json'
 if file.exists() and read(file)!=lock:raise ValueError('collection implementation changed after freeze')
 if create and not file.exists():write(file,lock)
 return lock
