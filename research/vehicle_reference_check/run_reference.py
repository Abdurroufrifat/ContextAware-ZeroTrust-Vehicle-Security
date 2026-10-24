"""Reconstruct an offline reference audit without fitting any model or policy."""
from pathlib import Path
import collections,hashlib,json,math,statistics,sys
import core_io as c
import reference_core as r
ROOT=Path(__file__).resolve().parent
def read(path):return json.loads((ROOT/path).read_text())
def digest(path):return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
def percentile(values,p=.95):
 a=sorted(values)
 if not a:return None
 z=(len(a)-1)*p;i=int(z);return a[i]+(a[min(i+1,len(a)-1)]-a[i])*(z-i)
def freeze(write=True):
 protocol=read('protocol.json')
 for role,cfg in protocol['recordings'].items():
  for n,sha in cfg['input_sha256'].items():
   if digest('input_data/'+role+'/'+n)!=sha:raise ValueError('prior endpoint selection or prediction changed')
 code=('core_io.py','reference_core.py','run_reference.py','acquire_reference.py')
 lock={'protocol_sha256':digest('protocol.json'),'code_sha256':{n:digest(n) for n in code},'reference_payloads_used_to_fit_or_tune':False}
 path=ROOT/'analysis_lock.json'
 if path.exists() and read('analysis_lock.json')!=lock:raise ValueError('reference analysis protocol or code changed')
 if write and not path.exists():path.write_text(json.dumps(lock,indent=2)+'\n')
 return lock
def profile(rows):
 delays=[q['bag_time_ns']-q['header_time_ns'] for q in rows];gps=[q['gps_week']*604800000+q['gps_milliseconds'] for q in rows];timing=[abs((b-a)-(v['header_time_ns']-u['header_time_ns'])/1e6) for a,b,u,v in zip(gps,gps[1:],rows,rows[1:])]
 return {'messages':len(rows),'status_counts':dict(collections.Counter(str(q['status']) for q in rows)),'frame_ids':dict(collections.Counter(q['frame_id'] for q in rows)),'nonfinite_velocity_rows':sum(q['horizontal_mps'] is None for q in rows),'time_status_counts':dict(collections.Counter(str(q['time_status']) for q in rows)),'bag_minus_header_ns_min':min(delays) if delays else None,'bag_minus_header_ns_max':max(delays) if delays else None,'bag_minus_header_ns_p95':percentile(delays),'gps_ros_increment_disagreement_ms_max':max(timing) if timing else None,'gps_nonincreasing_steps':sum(a>=b for a,b in zip(gps,gps[1:]))}
def decode_bag(role):
 cfg=read('protocol.json')['recordings'][role];meta=read('metadata/'+role+'/reference.bag.json');blob=(ROOT/'reference_bags'/(role+'.bag')).read_bytes()
 if len(blob)!=meta['size'] or hashlib.sha256(blob).hexdigest()!=cfg['reference_bag_publisher_sha256']:raise ValueError('publisher reference checksum failed')
 if blob[:4096]!=(ROOT/'metadata'/role/'reference.bag.header').read_bytes() or blob[meta['index_pos']:]!=(ROOT/'metadata'/role/'reference.bag.index').read_bytes():raise ValueError('reference index/header mismatch')
 connections={q['conn']:q for q in meta['connections']};indices=c.chunk_infos(blob[meta['index_pos']:]);rows={'INSPVA':[],'INSPVAX':[]};chunk_number=0
 for h,b in c.records(blob[13:meta['index_pos']]):
  if h['op']!=b'\x05':continue
  if h['compression']!=b'none' or c.u32(h['size'])!=len(b):raise ValueError('unsupported reference compression')
  counts=collections.Counter()
  for mh,payload in c.records(b):
   if mh['op']!=b'\x02':continue
   conn=c.u32(mh['conn']);counts[conn]+=1;info=connections[conn];kind=info['type'].split('/')[-1]
   if kind not in rows:raise ValueError('unexpected reference message schema')
   rows[kind].append(r.decode_reference(kind,payload,c.ros_time(mh['time'])))
  if dict(counts)!=indices[chunk_number]['counts']:raise ValueError('reference chunk counts disagree with index')
  chunk_number+=1
 if chunk_number!=len(indices):raise ValueError('reference chunk missing')
 for values in rows.values():r.ReferenceIndex(values)
 stats={kind:profile(values) for kind,values in rows.items()}
 x=rows['INSPVAX'];stats['INSPVAX']['extended_status_counts']=dict(collections.Counter(str(q['extended_status']) for q in x));stats['INSPVAX']['wheel_sensor_update_bit_set']=sum(bool(q['extended_status']&8) for q in x)
 for k in ('north_velocity_stdev','east_velocity_stdev','up_velocity_stdev'):
  v=[q[k] for q in x if r.finite(q[k]) and q[k]>=0];stats['INSPVAX'][k]={'finite_nonnegative':len(v),'p95':percentile(v),'max':max(v) if v else None}
 return rows,stats
def summary(rows):
 matched=[q for q in rows if q['reference_horizontal_mps'] is not None];metrics={}
 for key in ('prediction_mps','label_mps','gnss_mps'):
  errors=[q[key]-q['reference_horizontal_mps'] for q in matched if r.finite(q[key])]
  metrics[key]={'pairs':len(errors),'bias_mps':statistics.mean(errors) if errors else None,'mae_mps':statistics.mean(abs(x) for x in errors) if errors else None,'rmse_mps':math.sqrt(statistics.mean(x*x for x in errors)) if errors else None,'p95_disagreement_mps':percentile([abs(x) for x in errors]),'maximum_disagreement_mps':max(map(abs,errors)) if errors else None}
 p95=metrics['prediction_mps']['p95_disagreement_mps'];coverage=len(matched)/len(rows) if rows else 0.;grade=[q['reference_3d_mps']-q['reference_horizontal_mps'] for q in matched];sigma=[q for q in rows if q['north_velocity_stdev'] is not None]
 return {'scored_endpoints':len(rows),'reference_matched':len(matched),'unmatched_reference':len(rows)-len(matched),'reference_match_fraction':coverage,'prior_missing_or_invalid_endpoints':sum(not q['valid'] for q in rows),'uncertainty_associated':len(sigma),'radar_missing_on_matched_reference':sum(not r.finite(q['prediction_mps']) for q in matched),'reference_age_ns_p95':percentile([q['reference_age_ns'] for q in matched]),'reference_3d_minus_horizontal_p95_mps':percentile(grade),'metrics':metrics,'diagnostic_target_pass':coverage>=.95 and p95 is not None and p95<=.5}
def compute():
 freeze(False);protocol=read('protocol.json');allrows=[];roles={};blocks={};profiles={}
 for role in protocol['recordings']:
  decoded,profiles[role]=decode_bag(role);primary=r.ReferenceIndex(decoded['INSPVA']);uncertainty=r.ReferenceIndex(decoded['INSPVAX']);paired=[]
  for old in read('input_data/'+role+'/stream.json'):
   t=old['time_ns'];p=primary.latest(t,100000000);u=uncertainty.latest(t,1500000000);sigma_ok=u is not None and all(r.finite(u[k]) and u[k]>=0 for k in ('north_velocity_stdev','east_velocity_stdev','up_velocity_stdev'))
   paired.append({**old,'reference_horizontal_mps':p['horizontal_mps'] if p else None,'reference_3d_mps':p['speed_3d_mps'] if p else None,'reference_header_time_ns':p['header_time_ns'] if p else None,'reference_age_ns':t-p['header_time_ns'] if p else None,'reference_bag_time_ns':p['bag_time_ns'] if p else None,'uncertainty_header_time_ns':u['header_time_ns'] if u else None,'north_velocity_stdev':u['north_velocity_stdev'] if sigma_ok else None,'east_velocity_stdev':u['east_velocity_stdev'] if sigma_ok else None,'up_velocity_stdev':u['up_velocity_stdev'] if sigma_ok else None,'wheel_sensor_update_bit':bool(u['extended_status']&8) if u else None})
  role_summary=summary(paired);role_summary['radar_prefix_denied_endpoints']=sum(w['warmup_denied_endpoints'] for w in read('input_data/'+role+'/windows.json'));role_summary['context_bins']={}
  for lo,hi in ((0,1),(1,5),(5,10),(10,math.inf)):
   subset=[q for q in paired if q['reference_horizontal_mps'] is not None and lo<=q['reference_horizontal_mps']<hi]
   role_summary['context_bins'][f'{lo}_to_{hi}']=summary(subset) if subset else None
  roles[role]=role_summary;group=collections.defaultdict(list)
  for q in paired:group[q['window_id']].append(q)
  blocks.update({k:summary(v) for k,v in group.items()});allrows.extend(paired)
 evidence=read('evidence_requirements.json');qualification={k:r.qualify(v) for k,v in evidence.items()};conditional=[{'epsilon_mps':.1,'a_max_mps2':3.,'age_s':age,'hypotheses_disjoint_only_for_shift_strictly_above_mps':r.conditional_limit(.1,3.,age)} for age in (0.,.05,.1,.25)]
 result={'scope':protocol['scope'],'recording_metrics':roles,'block_metrics':blocks,'full_reference_source_profiles':profiles,'scored_endpoints':len(allrows),'reference_matches':sum(q['reference_horizontal_mps'] is not None for q in allrows),'all_reference_diagnostic_targets_pass':all(q['diagnostic_target_pass'] for q in list(roles.values())+list(blocks.values())),'qualification_requirements':qualification,'live_witness_qualified':False,'current_guaranteed_detectable_shift_mps':None,'hypothetical_age_sensitivity':conditional,'hypothetical_values_are_not_measured_bounds':True,'production_policy_changed':False,'witness_enabled':False,'journal_readiness_established':False,'no_model_refit_or_offset_search':True}
 return result,allrows
if __name__=='__main__':
 if len(sys.argv)>1 and sys.argv[1]=='freeze':print(json.dumps(freeze(),indent=2))
 else:
  result,paired=compute()
  for n,v in [('results.json',result),('matched_endpoints.json',paired)]: (ROOT/n).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
  print(json.dumps({'scored':result['scored_endpoints'],'matched':result['reference_matches'],'diagnostic_targets':result['all_reference_diagnostic_targets_pass'],'by_recording':{k:{'coverage':v['reference_match_fraction'],'radar':v['metrics']['prediction_mps'],'obd':v['metrics']['label_mps'],'gnss':v['metrics']['gnss_mps']} for k,v in result['recording_metrics'].items()},'witness_enabled':False},indent=2))
