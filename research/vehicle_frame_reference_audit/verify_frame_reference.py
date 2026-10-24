"""Independent scalar verification; no training, inference, network or relabeling."""
import base64,io,re,struct,unittest,xml.dom.minidom
from frame_reference_core import *
import test_frame_reference
def close(a,b):
 if isinstance(a,dict):
  if not isinstance(b,dict) or set(a)!=set(b):raise ValueError('audit dictionary schema differs')
  for k in a:close(a[k],b[k])
 elif isinstance(a,list):
  if not isinstance(b,list) or len(a)!=len(b):raise ValueError('audit list differs')
  for x,y in zip(a,b):close(x,y)
 elif isinstance(a,float):
  if not isinstance(b,(float,int)) or not math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10):raise ValueError('audit numeric mismatch')
 elif a!=b:raise ValueError('audit value mismatch')
def eq(x,y):
 x=np.asarray(x);y=np.asarray(y)
 if y.dtype.kind in 'iubUS':
  if y.dtype.kind in 'iu' and x.dtype.kind not in 'iu':raise ValueError('lost exact integer clock')
  if y.dtype.kind=='b' and x.dtype.kind!='b':raise ValueError('lost boolean flag')
  np.testing.assert_array_equal(x,y)
 else:np.testing.assert_allclose(x,y,rtol=0,atol=1e-10,equal_nan=True)
def quantile(v,q):
 v=sorted(map(float,v));at=(len(v)-1)*q;lo=math.floor(at);hi=math.ceil(at);return v[lo]+(at-lo)*(v[hi]-v[lo])
def scalar_stats(x):
 x=list(map(float,x));v=[a for a in x if math.isfinite(a)]
 return {'total':len(x),'finite':len(v),'min':min(v) if v else None,'median':quantile(v,.5) if v else None,'p95':quantile(v,.95) if v else None,'max':max(v) if v else None}
def scalar_metrics(y,p):
 e=[float(a)-float(b) for a,b in zip(p,y)];v=list(map(abs,e));n=len(v)
 return {'count':n,'mae_mps':math.fsum(v)/n if n else None,'rmse_mps':math.sqrt(math.fsum(a*a for a in e)/n) if n else None,'p95_mps':quantile(v,.95) if n else None,'maximum_mps':max(v) if n else None,'bias_prediction_minus_reference_mps':math.fsum(e)/n if n else None,'above_half_mps':sum(a>.5 for a in v),'above_2_mps':sum(a>2. for a in v)}
def check_geometry(result):
 checks=(ROOT/'sources/frdr-dfdr-checksums.txt').read_text(encoding='utf-8');verified=0
 for receipt in read(ROOT/'sources/retrieval_receipts.json')['files']:
  if not receipt['publisher_path'].startswith('CalibrationData/'):continue
  pattern=r'Filename/Fichier: '+re.escape(receipt['publisher_path'])+r'\s+SHA256 Checksum/Sommes de contrôle: ([0-9a-f]{64})';m=re.search(pattern,checks)
  if m is None or digest(ROOT/'sources'/receipt['file'])!=m[1] or receipt['sha256']!=m[1]:raise ValueError('publisher calibration hash mismatch')
  verified+=1
 geometry=read(ROOT/'sources/geometry_audit.json');close(geometry,result['published_geometry'])
 for filename,saved in geometry['configurations'].items():
  document=xml.dom.minidom.parseString((ROOT/'sources'/filename).read_text(encoding='utf-8'));nodes=[]
  for node in document.getElementsByTagName('node'):
   if node.getAttribute('pkg')!='tf2_ros' or node.getAttribute('type')!='static_transform_publisher':continue
   tokens=node.getAttribute('args').split();nodes.append({'node_name':node.getAttribute('name'),'parent':tokens[-2],'child':tokens[-1],'numeric_arguments':[float(x) for x in tokens[:-2]]})
  frames=sorted(set(v['parent'] for v in nodes)|set(v['child'] for v in nodes));children=[v['child'] for v in nodes];expected={'static_publisher_count':len(nodes),'frames':frames,'captured_frame':'livox_imu_link','captured_frame_in_transform_tree':'livox_imu_link' in frames,'nodes_named_as_captured_frame':[v for v in nodes if v['node_name']=='livox_imu_link'],'duplicate_child_frames':sorted({v for v in children if children.count(v)>1}),'nodes':nodes,'node_name_is_not_frame_identity':True,'rotation_applied':False};close(saved,expected)
 if geometry['applicable_configuration']!='setup_tf_1.launch' or geometry['physical_sensor_geometry_verified'] is not False or geometry['new_rotation_applied'] is not False or geometry['captured_imu_frame_unresolved'] is not True:raise ValueError('unsupported geometry claim')
 if 'Urban01, ..., Urban04' not in (ROOT/'sources/Readme.txt').read_text(encoding='utf-8'):raise ValueError('missing published configuration assignment')
 return verified
def audit():
 freeze();result=read(ROOT/'results.json');prior=read(ROOT/'prior_snapshot_manifest.json');available=0
 for name,h in prior.items():
  if name in ['FINAL_PROJECT_README.md','scripts/check_coact_project.py']:continue
  path=(PROJECT/name).resolve()
  if not path.is_relative_to(PROJECT):raise ValueError('unsafe previous file path')
  if path.is_file():
   if digest(path)!=h:raise ValueError('previous payload changed '+name)
   available+=1
 if (ROOT/'sha256.json').exists():
  for name,h in read(ROOT/'sha256.json').items():
   path=(ROOT/name).resolve()
   if not path.is_relative_to(ROOT) or digest(path)!=h:raise ValueError('source-audit artifact changed '+name)
 published=check_geometry(result);records=read(RESEARCH/'vehicle_reference_check/matched_endpoints.json');lookup={}
 for j,row in enumerate(records):
  key=(row['window_id'],row['time_ns'])
  if key in lookup or type(row['time_ns']) is not int:raise ValueError('invalid exact reference key')
  lookup[key]=j
 with np.load(RESEARCH/'vehicle_remaining_failure_audit/row_diagnostics.npz',allow_pickle=False) as z:parents={s:{n[len(s)+2:]:z[n].copy() for n in z.files if n.startswith(s+'__')} for s in ['fit','full']}
 with np.load(ROOT/'reference_pairs.npz',allow_pickle=False) as z:
  if set(z.files)!={s+'__'+f for s in ['fit','full'] for f in PAIR_FIELDS}:raise ValueError('unexpected pair-array schema')
  arrays={n:z[n].copy() for n in z.files}
 joins=cells=0
 def check_cell(q,ids,saved):
  nonlocal cells
  close(saved['rows'],len(ids));y=q['target_mps'][ids];obd=q['obd_mps'][ids]
  def metric(y,p,saved):
   nonlocal cells
   close(saved,scalar_metrics(y,p));cells+=1
  for name in METHODS:
   p=q[name+'__prediction_mps'][ids];v=saved['models'][name];metric(y,p,v['primary_reference']);metric(obd,p,v['obd_comparison']);metric(y,obd,v['obd_minus_primary_reference']);bad=abs(p-y)>.5;alt=abs(p-obd)>.5;close(v['primary_tail_also_above_half_against_obd'],int((bad&alt).sum()));close(v['primary_tail_at_or_below_half_against_obd'],int((bad&~alt).sum()))
  metric(y,q['gnss_mps'][ids],saved['gnss_minus_primary_reference']);bad=abs(q[PRIMARY+'__prediction_mps'][ids]-y)>.5
  for name,flag in [('tail',bad),('other',~bad)]:
   v=saved['tail_vs_other'][name];which=ids[flag];close(v['rows'],len(which));metric(q['target_mps'][which],q['obd_mps'][which],v['obd_minus_primary_reference'])
   for field in ['reported_component_sigma_max_mps','reference_3d_minus_horizontal_mps','uncertainty_age_ns']:close(v[field],scalar_stats(q[field][which]))
   close(v['missing_sigma'],int((~q['sigma_present'][which]).sum()));close(v['wheel_update_bit_set'],int((q['wheel_update_code'][which]==1).sum()))
 for scope,q in parents.items():
  expected=np.array([lookup.get((str(w),int(t)),-1) for w,t in zip(q['window_id'],q['time_ns'])],dtype=np.int64);eq(arrays[scope+'__reference_match_id'],expected);eq(arrays[scope+'__row_id'],q['row_id']);eq(arrays[scope+'__comparison_present'],expected>=0)
  for field in PAIR_FIELDS:q[field]=arrays[scope+'__'+field]
  for i,j in enumerate(expected):
   if j<0:
    if q['sigma_present'][i] or q['uncertainty_age_ns'][i]!=-1 or q['wheel_update_code'][i]!=-1 or any(np.isfinite(q[f][i]) for f in ['obd_mps','gnss_mps','reported_component_sigma_max_mps','reference_3d_minus_horizontal_mps']):raise ValueError('missing comparison imputed')
    continue
   row=records[j];t=int(q['time_ns'][i]);close(float(q['target_mps'][i]),row['reference_horizontal_mps']);eq(q['reference_age_ns'][i],t-row['reference_header_time_ns']);close(float(q['obd_mps'][i]),row['label_mps']);close(float(q['gnss_mps'][i]),row['gnss_mps']);close(float(q['reference_3d_minus_horizontal_mps'][i]),row['reference_3d_mps']-row['reference_horizontal_mps']);stamp=row['uncertainty_header_time_ns'];age=t-stamp if stamp is not None else -1;good=stamp is not None and 0<=age<=1_500_000_000 and all(row[k] is not None and math.isfinite(row[k]) and row[k]>=0 for k in ['north_velocity_stdev','east_velocity_stdev']);eq(q['uncertainty_age_ns'][i],age);eq(q['sigma_present'][i],good);eq(q['wheel_update_code'][i],-1 if row['wheel_sensor_update_bit'] is None else int(row['wheel_sensor_update_bit']))
   if good:close(float(q['reported_component_sigma_max_mps'][i]),max(row['north_velocity_stdev'],row['east_velocity_stdev']))
   elif np.isfinite(q['reported_component_sigma_max_mps'][i]):raise ValueError('missing sigma filled')
   joins+=1
  pop=result['populations'][scope];close(pop['parent_rows'],len(expected));close(pop['comparison_rows'],int((expected>=0).sum()));close(pop['missing_comparison_rows'],int((expected<0).sum()))
  if set(pop['windows'])!=set(q['window_id'].tolist()):raise ValueError('missing comparison window')
  for name,v in pop['windows'].items():
   ids=np.flatnonzero(q['window_id']==name);matched=ids[expected[ids]>=0];accepted=matched[q[PRIMARY+'__accepted'][matched]];close(v['parent_rows'],len(ids));close(v['exact_comparison_rows'],len(matched));close(v['missing_comparison_rows'],len(ids)-len(matched));close(v['corrected_accepted_comparison_rows'],len(accepted));check_cell(q,matched,v['all_matched']);check_cell(q,accepted,v['corrected_accepted_matched'])
 rawcount=scaleprobes=0
 for key in SOURCES:
  with np.load(RAW/(key+'.npz'),allow_pickle=False) as z:
   data=z['payload_bytes'];n=len(data);fl=int.from_bytes(data[0,12:16].tobytes(),'little');at=16+fl;zeroq=np.all(data[:,at:at+32]==0,axis=1);zeroc=np.all(data[:,at+32:at+104]==0,axis=1);saved=result['raw_imu_orientation'][key]
   if not zeroq.all():raise ValueError('zero-quaternion result no longer supported')
   close(saved['records'],n);close(saved['frame_id'],data[0,16:at].tobytes().decode());close(saved['zero_quaternion_records'],int(zeroq.sum()));close(saved['usable_orientation_records'],0);close(saved['zero_orientation_covariance_records'],int(zeroc.sum()));close(saved['acceleration_scale_9p8_reproduced_rows'],n);close(saved['unknown_acceleration_covariance_rows'],int(np.all(data[:,at+224:at+296]==0,axis=1).sum()));close(saved['unknown_gyro_covariance_rows'],int(np.all(data[:,at+128:at+200]==0,axis=1).sum()));close(saved['quaternion_norm'],scalar_stats(np.zeros(n)))
   for i in np.unique(np.linspace(0,n-1,65,dtype=int)):
    values=struct.unpack_from('<37d',data[i].tobytes(),at);close(float(z['acceleration_norm'][i]),math.hypot(*[v*9.8 for v in values[25:28]]));scaleprobes+=1
   rawcount+=n
 probes=0
 for record in read(RESEARCH/'vehicle_trace_audit/decode_probes.json'):
  if record['message_kind']!='INSPVA':continue
  b=base64.b64decode(record['payload_base64']);seq,sec,ns,fl=struct.unpack_from('<4I',b);at=16+fl;length=struct.unpack_from('<I',b,at)[0];at+=4+length+struct.calcsize('<HBIBHI');v=struct.unpack_from('<9d',b,at);status=struct.unpack_from('<I',b,at+72)[0];saved=record['expected_fields'];close(math.hypot(v[3],v[4]),saved['horizontal_mps']);close(seq,saved['sequence']);close(sec*1_000_000_000+ns,saved['header_time_ns']);close(status,saved['status']);probes+=1
 for name in ['root_cause_established','new_model_fitted','labels_changed','thresholds_changed','geometry_physically_revalidated','reference_threat_independence_established','reported_standard_deviations_are_uniform_bounds','postprocessed_attitude_used_in_predictor','reserved_final_opened','promotion_allowed','witness_enabled']:
  if result.get(name) is not False:raise ValueError('unsupported source-audit claim '+name)
 if result['decision']!='FRAME_REFERENCE_AUDIT_COMPLETE_NO_PROMOTION':raise ValueError('unsupported decision')
 buffer=io.StringIO();tests=unittest.TextTestRunner(stream=buffer).run(unittest.defaultTestLoader.loadTestsFromModule(test_frame_reference))
 if not tests.wasSuccessful():raise ValueError(buffer.getvalue())
 print(f'FRAME/REFERENCE AUDIT: PASS ({tests.testsRun} tests;{published} publisher calibration hashes;{joins} exact reference joins;{cells} scalar metric cells;{rawcount} raw zero-quaternion records;{scaleprobes} independent scale probes;{probes} independent INSPVA probes)',flush=True)
 print(f'Available preceding payloads verified: {available}/{len(prior)-2}; unavailable historical files not claimed as reverified.',flush=True);print('Captured livox_imu_link lacks a published transform edge. Models, labels and thresholds unchanged.',flush=True);print(result['decision']+'; ACCURACY FAILURES REMAIN; RESERVED FINAL CLOSED; WITNESS DISABLED.',flush=True)
if __name__=='__main__':audit()
