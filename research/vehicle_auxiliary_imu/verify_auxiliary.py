"""Independent source/clock checks; no network, learning or final payload access."""
import bisect,io,math,unittest
from aux_core import *
import test_auxiliary

def close(a,b):
 if isinstance(a,dict):
  if not isinstance(b,dict) or set(a)!=set(b):raise ValueError('auxiliary audit dictionary differs')
  for k in a:close(a[k],b[k])
 elif isinstance(a,list):
  if not isinstance(b,list) or len(a)!=len(b):raise ValueError('auxiliary audit list differs')
  for x,y in zip(a,b):close(x,y)
 elif isinstance(a,float):
  if not isinstance(b,(int,float)) or not math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10):raise ValueError('auxiliary numeric audit differs')
 elif a!=b:raise ValueError('auxiliary audit value differs')

def independent_description(values):
 values=[float(v) for v in values];ordered=sorted(v for v in values if math.isfinite(v));n=len(ordered)
 def quantile(p):
  at=(n-1)*p;lo=math.floor(at);hi=math.ceil(at);return ordered[lo]+(at-lo)*(ordered[hi]-ordered[lo])
 return {'finite_count':n,'missing_or_nonfinite':len(values)-n,'minimum':ordered[0] if n else None,'median':quantile(.5) if n else None,'p95':quantile(.95) if n else None,'maximum':ordered[-1] if n else None}

def scalar_payload(blob,bag_ns,topic):
 seq,sec,ns=struct.unpack_from('<III',blob);length=struct.unpack_from('<I',blob,12)[0];start=16+length;frame=blob[16:start].decode();gyro=list(struct.unpack_from('<ddd',blob,start+13*8));acc=list(struct.unpack_from('<ddd',blob,start+25*8));gc=[struct.unpack_from('<d',blob,start+(16+j)*8)[0] for j in range(9)];ac=[struct.unpack_from('<d',blob,start+(28+j)*8)[0] for j in range(9)];scale=9.8 if topic=='/livox/imu' else 1.;known=lambda v:all(math.isfinite(x) for x in v) and v[0]!=-1 and all(v[j]>=0 for j in [0,4,8]);acc=[v*scale for v in acc];scaled_cov=ac if ac[0]==-1 else [v*scale**2 for v in ac];valid=all(math.isfinite(v) for v in gyro+acc) and known(gc) and known(scaled_cov) and math.isfinite(math.hypot(*gyro)) and math.isfinite(math.hypot(*acc))
 return {'sequence':seq,'header_ns':sec*1_000_000_000+ns,'bag_ns':bag_ns,'topic':topic,'frame_id':frame,'angular_velocity':gyro,'acceleration':acc,'angular_velocity_covariance':gc,'acceleration_covariance':scaled_cov,'angular_velocity_covariance_unknown':all(v==0 for v in gc),'acceleration_covariance_unknown':all(v==0 for v in ac),'usable':valid}

def audit():
 freeze();result=read(ROOT/'results.json');expected_flags={'full_cohort_source_coverage_verified':False,'model_training_started':False,'independent_validation_done':False,'accuracy_failures_resolved':False,'unit_bound_established':False,'live_time_authentication_established':False,'source_partial_bags_only':True,'witness_enabled':False,'production_policy_changed':False,'reserved_final_opened':False,'promotion_allowed':False}
 for k,v in expected_flags.items():
  if result.get(k) is not v:raise ValueError('unsupported auxiliary source claim: '+k)
 if result['decision']!='AUXILIARY_IMU_PILOT_COMPLETE_METADATA_REQUIRED':raise ValueError('unexpected auxiliary qualification decision')
 project=RESEARCH.parent;prior=read(ROOT/'prior_snapshot_manifest.json');available=0
 for name,h in prior.items():
  if name in ['FINAL_PROJECT_README.md','scripts/check_coact_project.py']:continue
  path=(project/name).resolve()
  if not path.is_relative_to(project):raise ValueError('unsafe preceding file path')
  if path.is_file():
   if digest(path)!=h:raise ValueError('preceding payload changed: '+name)
   available+=1
 if (ROOT/'sha256.json').exists():
  for name,h in read(ROOT/'sha256.json').items():
   path=(ROOT/name).resolve()
   if not path.is_relative_to(ROOT) or digest(path)!=h:raise ValueError('auxiliary payload changed: '+name)
 rebuilt,arrays=compute();close(result,rebuilt)
 with np.load(ROOT/'pilot_arrays.npz',allow_pickle=False) as z:
  if set(z.files)!=set(arrays):raise ValueError('auxiliary array keys differ')
  for key,v in arrays.items():
   if v.dtype.kind=='f':np.testing.assert_allclose(z[key],v,rtol=0,atol=1e-10,equal_nan=True)
   else:np.testing.assert_array_equal(z[key],v)
 plan,rows,bychunk,allrecords=extract();meta=read(PROBE/'source_index/imus.bag.json');conns={v['conn']:v['topic'] for v in meta['connections'] if v['topic'] in TOPICS};raw={t:[] for t in TOPICS};receipts=read(PROBE/'evidence/downloads.json')
 for q in plan['ranges']:
  if q['file']!='imus.bag':continue
  path=PROBE/'evidence/chunks'/q['relative_cache'];blob=path.read_bytes()
  if digest(path)!=receipts[q['relative_cache']]['sha256'] or len(blob)!=q['end']-q['start']+1:raise ValueError('cached range receipt mismatch')
  body=next(c.records(blob))[1]
  for h,b in c.records(body):
   if h['op']==b'\x02' and c.u32(h['conn']) in conns:raw[conns[c.u32(h['conn'])]].append((c.ros_time(h['time']),b))
 probes=0;steps=0;descriptions=0
 for topic,values in rows.items():
  name='livox' if topic==TOPICS[0] else 'xsens';payloads=sorted(raw[topic],key=lambda v:v[0])
  if len(payloads)!=len(values):raise ValueError('independent raw decoder population differs')
  for i in np.unique(np.linspace(0,len(values)-1,257,dtype=int)):
   close(values[i],scalar_payload(payloads[i][1],payloads[i][0],topic));probes+=1
  stats=result['pilot_channels'][name]
  for key,seq in [('header_minus_writer_ns',[v['header_ns']-v['bag_ns'] for v in values]),('acceleration_norm_declared_mps2',[math.hypot(*v['acceleration']) for v in values]),('gyro_norm_declared_radps',[math.hypot(*v['angular_velocity']) for v in values])]:close(stats[key],independent_description(seq));descriptions+=1
  if stats['decoded_records']!=len(values) or stats['usable_payload_records']!=sum(v['usable'] for v in values) or stats['unknown_gyro_covariance_records']!=sum(v['angular_velocity_covariance_unknown'] for v in values) or stats['unknown_accel_covariance_records']!=sum(v['acceleration_covariance_unknown'] for v in values):raise ValueError('raw pilot eligibility/covariance ledger differs')
  for w in plan['windows']:
   records=sorted([v for ci in w['imu_chunks'] for v in bychunk[ci][topic]],key=lambda v:v['bag_ns']);arrivals=[v['bag_ns'] for v in records];prefix=name+'__'+w['id']+'__';ticks=arrays[prefix+'endpoint_ns'];computed={k:[] for k in ['latest_row','status','header_age_ns','arrival_age_ns','acceleration_norm','angular_velocity_norm']}
   for t in ticks:
    i=bisect.bisect_right(arrivals,int(t))-1;status=1;ha=ba=-1;acc=gyro=float('nan')
    if i>=0:
     row=records[i];ha=int(t)-row['header_ns'];ba=int(t)-row['bag_ns'];status=3 if ha<0 else 2 if max(ha,ba)>100_000_000 else 4 if not row['usable'] else 0
     if status==0:acc=math.hypot(*row['acceleration']);gyro=math.hypot(*row['angular_velocity'])
    for k,v in zip(computed,[i,status,ha,ba,acc,gyro]):computed[k].append(v)
   for k,v in computed.items():np.testing.assert_allclose(arrays[prefix+k],np.array(v),rtol=0,atol=1e-10,equal_nan=True)
   cell=result['by_probe_window'][w['id']][name];status=computed['status'];good=[i for i,v in enumerate(status) if v==0]
   close(cell['probe_ticks'],len(ticks));close(cell['available_ticks'],len(good));close(cell['available_fraction_on_probe_ticks'],len(good)/len(ticks));close(cell['status_counts'],{label:sum(v==code for v in status) for code,label in STATUS.items()})
   for k,v in [('available_header_age_ns','header_age_ns'),('available_acceleration_norm_declared_mps2','acceleration_norm'),('available_angular_velocity_norm_declared_radps','angular_velocity_norm')]:close(cell[k],independent_description([computed[v][i] for i in good]));descriptions+=1
   steps+=len(ticks)
 if result['pilot_chunks']!=37 or len(result['by_probe_window'])!=9 or allrecords!=result['all_connection_records_checked'] or not result['xsens_missing_full_development'] or not result['livox_present_in_all_four_captured_indexes']:raise ValueError('pilot/cohort inventory conclusion differs')
 # Network helper must retain exactly the two approved fitting source identities.
 for route in PENDING.values():check_metadata_source(route,'imus.bag')
 buffer=io.StringIO();tests=unittest.TextTestRunner(stream=buffer).run(unittest.defaultTestLoader.loadTestsFromModule(test_auxiliary))
 if not tests.wasSuccessful():raise ValueError(buffer.getvalue())
 print(f'AUXILIARY IMU AUDIT: PASS ({tests.testsRun} tests;37 cached chunks;{allrecords} all-connection records;{probes} independent payload probes;{steps} scalar causal steps;{descriptions} independently reconstructed descriptive cells)')
 print(f'Available preceding payloads verified: {available}/{len(prior)-2}; unavailable historical files are not claimed as reverified.')
 print('Xsens absent from captured Urban01 NIGHT index. Livox present in4 captured indexes;2 fitting-night indexes still required.')
 print(result['decision']+'; PILOT ONLY. NO NEW MODEL OR ACCURACY RESULT. WITNESS DISABLED.')
if __name__=='__main__':audit()
