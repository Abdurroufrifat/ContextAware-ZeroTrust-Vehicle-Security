"""Verify uploaded source metadata, fixed cohort and collection software offline."""
import bisect,unittest,zipfile
from collection_core import *

def check_upload():
 path=ROOT/'upload_evidence/Upload_Auxiliary_IMU_Metadata.zip';expected={'auxiliary_imu_metadata/summary.json'}|{f'auxiliary_imu_metadata/{g}/{n}' for g in ['Urban03','Urban04'] for n in ['imus.bag.header','imus.bag.index','source.json','receipt.json']}
 with zipfile.ZipFile(path) as z:
  if len(z.namelist())!=len(expected) or set(z.namelist())!=expected or z.testzip() is not None:raise ValueError('metadata upload member/CRC mismatch')
  summary=json.loads(z.read('auxiliary_imu_metadata/summary.json'))
  if summary['reserved_final_opened'] or summary['model_training_started']:raise ValueError('unsupported upload scope')
  for group in ['Urban03','Urban04']:
   folder=ROOT/'metadata'/(group+'_night');prefix='auxiliary_imu_metadata/'+group+'/';receipt=read(folder/'receipt.json');meta,chunks,result=inventory(group+'_night')
   for name in ['imus.bag.header','imus.bag.index','source.json','receipt.json']:
    if (folder/name).read_bytes()!=z.read(prefix+name):raise ValueError('uploaded metadata bytes changed')
   if receipt['route']!=ROUTES[group+'_night'] or receipt['source_url']!=meta['url'] or receipt['source_size']!=meta['size'] or receipt['header_index_only'] is not True or receipt['measurement_payloads_requested'] is not False or receipt['publisher_full_bag_sha256_verified'] is not False or set(receipt['files_sha256'])!={'imus.bag.header','imus.bag.index','source.json'}:raise ValueError('metadata receipt scope/source differs')
   for name,h in receipt['files_sha256'].items():
    if digest(folder/name)!=h:raise ValueError('uploaded metadata hash differs')
   if len(receipt['ranges'])!=2:raise ValueError('unexpected metadata ranges')
   for name,r,start,end in [('imus.bag.header',receipt['ranges'][0],0,4095),('imus.bag.index',receipt['ranges'][1],meta['index_pos'],meta['size']-1)]:
    if r['start']!=start or r['end']!=end or r['bytes']!=end-start+1 or r['sha256']!=digest(folder/name):raise ValueError('range receipt mismatch')
    c.validate_range_response(206,r['content_range'],str(r['bytes']),start,end,meta['size'])
   if result!=summary['sources'][group]:raise ValueError('uploaded inventory summary differs')

def check_targets():
 members=read(RESEARCH/'vehicle_route_validation/membership.json');index=read(RESEARCH/'vehicle_masked_history/frame_index.json');windows={k:[] for k in ROUTES};expected={}
 if len(members)!=27178 or len(index)!=40690 or [v['id'] for v in members]!=list(range(27178)):raise ValueError('preceding cohort sizes/identity differ')
 for key,route in ROUTES.items():
  if key=='Urban01_night':
   t=np.array([v['time_ns'] for v in index],dtype=np.int64);requests=int(t[0])+np.arange((int(t[-1])-int(t[0]))//50_000_000+1,dtype=np.int64)*50_000_000
   if len(requests)!=37340:raise ValueError('fixed development request count differs')
   expected[key+'__endpoint_times_ns']=t;expected[key+'__request_times_ns']=requests;expected[key+'__endpoint_ready']=np.array([v['ready'] for v in index],dtype=bool);windows[key]=[{'id':'full_inspected_development','first_ns':int(t[0]),'last_ns':int(t[-1]),'endpoint_count':len(t)}]
  else:
   rows=[v for v in members if v['route']==route];expected[key+'__fit_times_ns']=np.array([v['time_ns'] for v in rows],dtype=np.int64);expected[key+'__fit_ids']=np.array([v['id'] for v in rows],dtype=np.int64)
   for window in sorted({v['window_id'] for v in rows}):
    selected=[v for v in rows if v['window_id']==window];windows[key].append({'id':window,'first_ns':min(v['time_ns'] for v in selected),'last_ns':max(v['time_ns'] for v in selected),'endpoint_count':len(selected)})
 with np.load(ROOT/'cohort_targets.npz',allow_pickle=False) as z:
  if set(z.files)!=set(expected):raise ValueError('collection target scopes differ')
  for key,v in expected.items():np.testing.assert_array_equal(z[key],v)
 if windows!=read(ROOT/'cohort_windows.json') or sum(len(v) for k,v in windows.items() if k!='Urban01_night')!=15:raise ValueError('frozen fitting windows changed')
 return expected

def check_cached_extraction(targets):
 probe=RESEARCH/'witness_calibration_probe';old=read(probe/'acquisition_plan.json');meta=read(probe/'source_index/imus.bag.json');chunks=c.chunk_infos((probe/'source_index/imus.bag.index').read_bytes());rows=[];independent=[];all_counts=0
 for q in old['ranges']:
  if q['file']!='imus.bag':continue
  path=probe/'evidence/chunks'/q['relative_cache'];blob=path.read_bytes();receipt=read(probe/'evidence/downloads.json')[q['relative_cache']]
  if digest(path)!=receipt['sha256'] or len(blob)!=q['end']-q['start']+1:raise ValueError('cached range bytes changed')
  expected={'chunk_index':q['chunk_index'],'counts':chunks[q['chunk_index']]['counts']};rows.extend(decode_chunk(blob,expected,meta));all_counts+=sum(expected['counts'].values())
  wanted={v['conn'] for v in meta['connections'] if v['topic']=='/livox/imu'};body=next(c.records(blob))[1]
  for ordinal,(h,payload) in enumerate(c.records(body)):
   if h['op']==b'\x02' and c.u32(h['conn']) in wanted:independent.append((c.ros_time(h['time']),q['chunk_index'],ordinal,payload))
 rows.sort(key=lambda r:r['bag_ns']);independent.sort();arrays,report=pack_raw(rows)
 if len(rows)!=19788 or len(independent)!=len(rows) or report['usable_records']!=len(rows) or all_counts!=115366:raise ValueError('known pilot population differs')
 for i,(bag,chunk,ordinal,payload) in enumerate(independent):
  if int(arrays['bag_ns'][i])!=bag or int(arrays['chunk_index'][i])!=chunk or int(arrays['record_ordinal'][i])!=ordinal or arrays['payload_bytes'][i,:int(arrays['payload_length'][i])].tobytes()!=payload:raise ValueError('serialized payload/lineage changed')
 # Scalar clock matching covers all fitting queries from this source, including missing/stale cases.
 query=targets['Urban01_day__fit_times_ns'];out=align(arrays,query);arrivals=arrays['bag_ns'].tolist()
 for j,t in enumerate(query):
  i=bisect.bisect_right(arrivals,int(t))-1;status=1;ha=ba=-1
  if i>=0:
   ha=int(t)-int(arrays['header_ns'][i]);ba=int(t)-int(arrays['bag_ns'][i]);status=3 if ha<0 else 2 if ha>100_000_000 or ba>100_000_000 else 4 if not arrays['usable'][i] or not math.isfinite(arrays['acceleration_norm'][i]) or not math.isfinite(arrays['angular_velocity_norm'][i]) else 0
  if int(out['latest_message'][j])!=i or int(out['status'][j])!=status or int(out['header_age_ns'][j])!=ha or int(out['arrival_age_ns'][j])!=ba:raise ValueError('scalar clock reconstruction differs')
  if status==0:
   for name in ['acceleration_norm','angular_velocity_norm']:
    if out[name][j]!=arrays[name][i]:raise ValueError('scalar available magnitude differs')
  elif not np.isnan(out['acceleration_norm'][j]) or not np.isnan(out['angular_velocity_norm'][j]):raise ValueError('unavailable context leaked values')
 return len(rows),len(query)

def audit():
 freeze();check_upload();targets=check_targets();plan=read(ROOT/'acquisition_plan.json')
 if make_plan()!=plan:raise ValueError('fixed source acquisition plan differs')
 prior=read(ROOT/'prior_snapshot_manifest.json');available=0
 for name,h in prior.items():
  if name in ['FINAL_PROJECT_README.md','scripts/check_coact_project.py']:continue
  path=(RESEARCH.parent/name).resolve()
  if not path.is_relative_to(RESEARCH.parent):raise ValueError('unsafe prior manifest name')
  if path.is_file():
   if digest(path)!=h:raise ValueError('preceding project payload changed: '+name)
   available+=1
 if (ROOT/'sha256.json').exists():
  for name,h in read(ROOT/'sha256.json').items():
   path=(ROOT/name).resolve()
   if not path.is_relative_to(ROOT) or digest(path)!=h:raise ValueError('collection payload changed: '+name)
 raw,steps=check_cached_extraction(targets);buffer=io.StringIO();suite=unittest.defaultTestLoader.discover(str(ROOT),pattern='test_*.py');result=unittest.TextTestRunner(stream=buffer).run(suite)
 if not result.wasSuccessful():raise ValueError(buffer.getvalue())
 print(f'LIVOX COLLECTION AUDIT: PASS ({result.testsRun} tests;2 uploaded inventories;6 source indexes;{raw} exact serialized pilot payloads;{steps} scalar clock queries)',flush=True)
 print(f'Fixed plan: {sum(len(v["ranges"]) for v in plan["sources"].values())} chunks;{plan["total_download_bytes"]} bytes;15 fitting windows;27178 fitting endpoints;40690 development endpoints;37340 development requests.',flush=True)
 print(f'Available preceding payloads verified: {available}/{len(prior)-2}; unavailable files not claimed as reverified.',flush=True)
 print('COLLECTION SOFTWARE READY; REAL COHORT IMU MEASUREMENTS STILL REQUIRED. NO NEW MODEL/ACCURACY CLAIM; WITNESS DISABLED.',flush=True)
if __name__=='__main__':audit()
