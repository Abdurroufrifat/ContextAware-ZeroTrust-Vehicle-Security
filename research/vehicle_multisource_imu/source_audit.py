"""Reconstruct every uploaded IMU payload and fixed causal match offline."""
from pathlib import Path
import bisect,hashlib,json,math,re,struct,sys
import numpy as np
import imu_core as im
ROOT=Path(__file__).resolve().parent;RESEARCH=ROOT.parent
sys.path.insert(0,str(RESEARCH/'vehicle_livox_collection'));import collection_core as l
read=l.read;write=l.write;digest=l.digest;INPUT=l.ROOT/'runtime_export'
def close(x,y):
 if isinstance(x,dict):
  if not isinstance(y,dict) or set(x)!=set(y):raise ValueError('audit dictionary schema differs')
  for k in x:close(x[k],y[k])
 elif isinstance(x,list):
  if not isinstance(y,list) or len(x)!=len(y):raise ValueError('audit list differs')
  for a,b in zip(x,y):close(a,b)
 elif isinstance(x,float):
  if not isinstance(y,(float,int)) or not math.isclose(x,y,rel_tol=1e-10,abs_tol=1e-10):raise ValueError('audit numeric value differs')
 elif x!=y:raise ValueError('audit value differs')
def load(key):
 if key not in l.ROUTES:raise ValueError('unknown/reserved IMU source')
 with np.load(INPUT/(key+'.npz'),allow_pickle=False) as z:raw={n:z[n].copy() for n in z.files}
 return raw,im.decode_payloads(raw)
def norms_equal(x,y):return (math.isnan(float(x)) and math.isnan(float(y))) or math.isclose(float(x),float(y),rel_tol=1e-12,abs_tol=1e-10)
def scalar_record(blob,bag):
 seq,sec,ns,n=struct.unpack_from('<IIII',blob)
 if ns>=1_000_000_000 or not n or len(blob)!=312+n:raise ValueError('invalid independent IMU serialization')
 frame=blob[16:16+n].decode('utf-8');v=struct.unpack_from('<37d',blob,16+n);g=v[13:16];acc=tuple(float(x)*9.8 for x in v[25:28]);gc=v[16:25];ac=v[28:37];scaled=ac if ac[0]==-1 else tuple(float(x)*(9.8**2) for x in ac)
 cov=lambda x:all(math.isfinite(z) for z in x) and x[0]!=-1 and all(x[j]>=0 for j in (0,4,8));an=math.hypot(*acc);gn=math.hypot(*g);usable=all(math.isfinite(z) for z in (*g,*acc)) and cov(gc) and cov(scaled) and math.isfinite(an) and math.isfinite(gn)
 return seq,sec*1_000_000_000+ns,frame,acc,g,an,gn,usable,all(z==0 for z in ac),all(z==0 for z in gc)
def qualify(raw,s):
 return {'records':len(s['bag_ns']),'usable_records':int(s['usable'].sum()),'frames':[s['frame_id']],'unknown_acceleration_covariance_records':int(s['accel_cov_unknown'].sum()),'unknown_gyro_covariance_records':int(s['gyro_cov_unknown'].sum()),'future_header_at_writer_records':int(np.sum(s['header_ns']>s['bag_ns'])),'header_minus_writer_ns':l.a.descriptive(s['header_ns']-s['bag_ns']),'acceleration_norm_declared_mps2':l.a.descriptive(s['acceleration_norm']),'angular_velocity_norm_declared_radps':l.a.descriptive(s['angular_velocity_norm']),'physical_unit_calibration_verified':False,'full_bag_checksum_verified':False,'authenticated_live_time_verified':False}
def audit_sources():
 l.freeze();identity=read(INPUT/'collection_identity.json');plan=read(l.ROOT/'acquisition_plan.json');sources={};payloads=steps=overlap=0
 if identity['analysis_lock']!=read(l.ROOT/'analysis_lock.json') or identity['input_plan_sha256']!=digest(l.ROOT/'acquisition_plan.json') or identity['full_bag_checksums_verified'] is not False or identity['source_http_receipts_are_not_authenticated_live_time'] is not True:raise ValueError('uploaded collection identity differs')
 expected={'quality_summary.json'}|{key+suffix for key in l.ROUTES for suffix in ['.npz','_quality.json','_receipts.json']}
 if set(identity['files_sha256'])!=expected:raise ValueError('upload input member scope differs')
 for name,h in identity['files_sha256'].items():
  if digest(INPUT/name)!=h:raise ValueError('uploaded source hash mismatch: '+name)
 summary=read(INPUT/'quality_summary.json')
 for flag in ['model_training_started','reserved_final_opened','witness_enabled','production_policy_changed']:
  if summary[flag] is not False:raise ValueError('unsupported uploaded source claim')
 with np.load(l.ROOT/'cohort_targets.npz',allow_pickle=False) as z:targets={k:z[k].copy() for k in z.files}
 for key,cfg in plan['sources'].items():
  raw,s=load(key);n=len(s['bag_ns']);payloads+=n;receipts=read(INPUT/(key+'_receipts.json'));ranges=cfg['ranges']
  if len(receipts)!=len(ranges) or n!=cfg['indexed_livox_messages_in_selected_chunks']:raise ValueError('source receipt/selected record count differs')
  meta=read(l.ROOT/'metadata'/key/'source.json');wanted=[v['conn'] for v in meta['connections'] if v['topic']=='/livox/imu'];chunk=np.asarray(raw['chunk_index']);ordinal=np.asarray(raw['record_ordinal'])
  if chunk.shape!=(n,) or ordinal.shape!=(n,) or chunk.dtype.kind!='i' or ordinal.dtype.kind!='i' or np.any(ordinal<0) or set(chunk.tolist())!={q['chunk_index'] for q in ranges}:raise ValueError('invalid chunk/record lineage')
  for q,receipt in zip(ranges,receipts):
   for name,value in {'chunk_index':q['chunk_index'],'url':cfg['url'],'start':q['start'],'end':q['end'],'total':cfg['size'],'bytes':q['end']-q['start']+1}.items():
    if receipt.get(name)!=value:raise ValueError('planned range receipt differs')
   if not re.fullmatch('[0-9a-f]{64}',receipt['sha256']):raise ValueError('invalid recorded chunk digest')
   l.c.validate_range_response(206,receipt['content_range'],str(receipt['bytes']),q['start'],q['end'],cfg['size']);ids=np.flatnonzero(chunk==q['chunk_index']);count=sum(q['counts'].get(str(cid),0) for cid in wanted)
   if len(ids)!=count or len(np.unique(ordinal[ids]))!=len(ids) or np.any(ordinal[ids][1:]<=ordinal[ids][:-1]) or np.any(s['bag_ns'][ids]<q['start_ns']) or np.any(s['bag_ns'][ids]>q['end_ns']):raise ValueError('indexed selected counts/clock/record identity differ')
  for i in range(n):
   length=int(raw['payload_length'][i]);v=scalar_record(raw['payload_bytes'][i,:length].tobytes(),int(s['bag_ns'][i]));seq,t,frame,acc,gyro,an,gn,ok,ac,gc=v
   if int(s['sequence'][i])!=seq or int(s['header_ns'][i])!=t or s['frame_id']!=frame or bool(s['usable'][i])!=ok or bool(s['accel_cov_unknown'][i])!=ac or bool(s['gyro_cov_unknown'][i])!=gc:raise ValueError('independent serialized field/eligibility mismatch')
   if not np.allclose(s['acceleration'][i],acc,rtol=0,atol=1e-10,equal_nan=True) or not np.array_equal(s['angular_velocity'][i],gyro,equal_nan=True) or not norms_equal(s['acceleration_norm'][i],an) or not norms_equal(s['angular_velocity_norm'][i],gn):raise ValueError('independent raw vector/norm mismatch')
  for name in ['header_ns','usable','acceleration_norm','angular_velocity_norm']:
   if name in ['acceleration_norm','angular_velocity_norm']:np.testing.assert_allclose(raw[name],s[name],rtol=0,atol=1e-10,equal_nan=True)
   else:np.testing.assert_array_equal(raw[name],s[name])
  result=qualify(raw,s);result['route']=cfg['route'];result['source_only_alignment']={};arrivals=s['bag_ns'].tolist()
  for name,q in targets.items():
   if not name.startswith(key+'__'):continue
   if not name.endswith('_times_ns'):
    np.testing.assert_array_equal(raw['identity__'+name[len(key)+2:]],q);continue
   scope=name[len(key)+2:-9];match=im.lookup(s,q)
   for field,value in match.items():
    stored=raw['alignment__'+scope+'__'+field]
    if value.dtype.kind=='f':np.testing.assert_allclose(stored,value,rtol=0,atol=1e-10,equal_nan=True)
    else:np.testing.assert_array_equal(stored,value)
   for j,time in enumerate(q):
    i=bisect.bisect_right(arrivals,int(time))-1;st=1;ha=ba=-1
    if i>=0:
     ha=int(time)-int(s['header_ns'][i]);ba=int(time)-int(s['bag_ns'][i]);st=3 if ha<0 else 2 if max(ha,ba)>100_000_000 else 4 if not s['usable'][i] else 0
    if int(match['latest_message'][j])!=i or int(match['status'][j])!=st or int(match['header_age_ns'][j])!=ha or int(match['arrival_age_ns'][j])!=ba:raise ValueError('independent latest-arrival reconstruction differs')
   steps+=len(q);result['source_only_alignment'][scope]={'target_count':len(q),'available_count':int(np.sum(match['status']==0)),'status_counts':{label:int(np.sum(match['status']==code)) for code,label in l.a.STATUS.items()}}
  close(result,read(INPUT/(key+'_quality.json')));close(result,summary['sources'][key]);sources[key]=result
  if key=='Urban01_day':
   probe=l.RESEARCH/'witness_calibration_probe';byidentity={(int(chunk[i]),int(ordinal[i])):i for i in range(n)};old=read(probe/'acquisition_plan.json')
   for q in old['ranges']:
    if q['file']!='imus.bag' or q['chunk_index'] not in set(chunk.tolist()):continue
    blob=(probe/'evidence/chunks'/q['relative_cache']).read_bytes()
    for position,(h,payload) in enumerate(l.c.records(next(l.c.records(blob))[1])):
     if h['op']==b'\x02' and l.c.u32(h['conn']) in wanted:
      i=byidentity[(q['chunk_index'],position)]
      if int(s['bag_ns'][i])!=l.c.ros_time(h['time']) or raw['payload_bytes'][i,:int(raw['payload_length'][i])].tobytes()!=payload:raise ValueError('existing cached raw-source overlap differs')
      overlap+=1
  print('RAW SOURCE VERIFIED',key,n,'serialized records',flush=True)
 return {'sources':sources,'all_serialized_payloads_independently_decoded':payloads,'scalar_causal_matches_reconstructed':steps,'existing_cached_raw_message_overlaps_verified':overlap,'publisher_full_bag_hashes_independently_verified':False,'source_receipts_are_authenticated_live_time':False,'physical_unit_calibration_verified':False,'reserved_final_opened':False,'model_training_started':False}
if __name__=='__main__':
 result=audit_sources();write(ROOT/'source_audit.json',result);print('SOURCE AUDIT PASS',result['all_serialized_payloads_independently_decoded'],'payloads;',result['scalar_causal_matches_reconstructed'],'causal matches;',result['existing_cached_raw_message_overlaps_verified'],'old raw overlaps',flush=True)
