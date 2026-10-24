"""Source replay on fitting recordings only; no timestamp search or model fit."""
from pathlib import Path
import base64,bisect,collections,gzip,hashlib,json,math,struct,sys
import numpy as np
ROOT=Path(__file__).resolve().parent;RESEARCH=ROOT.parent
sys.path.insert(0,str(RESEARCH/'vehicle_reference_check'))
import core_io as io
import reference_core as refcore
sys.path.insert(0,str(RESEARCH/'vehicle_failure_diagnosis'))
import diagnosis_core as d
ROUTES=('Urban01/2023-11-23-15-38-43-Day','Urban03/2023-11-03-09-42-41-Day','Urban04/2023-11-01-12-16-16-Day','Urban03/2023-11-23-19-12-16-Night','Urban04/2023-11-23-19-36-42-Night')
DAY_ROLES=('fit','evaluation_urban03','evaluation_urban04')
read=d.read;write=d.write;digest=d.digest

def match_reference(rows,times):
 stamps=np.array([q['header_time_ns'] for q in rows],dtype=np.int64);times=np.asarray(times)
 if not len(stamps) or np.any(stamps<0) or np.any(stamps[1:]<=stamps[:-1]) or times.ndim!=1 or times.dtype.kind!='i' or np.any(times<0):raise ValueError('ambiguous reference/endpoint time')
 speed=np.array([q['horizontal_mps'] if q['horizontal_mps'] is not None else np.nan for q in rows]);valid=np.array([q['status']==3 and q['horizontal_mps'] is not None and q['speed_3d_mps'] is not None and math.isfinite(q['horizontal_mps']) and math.isfinite(q['speed_3d_mps']) for q in rows]);j=np.searchsorted(stamps,times,side='right')-1;safe=np.maximum(j,0);ages=times-stamps[safe];present=(j>=0)&(ages>=0)&(ages<=100_000_000)&valid[safe]
 previous=np.maximum(safe-1,0);dt=stamps[safe]-stamps[previous];proxy_good=present&(safe>0)&valid[previous]&(dt>0)&(dt<=100_000_000);proxy=np.full(len(times),np.nan);proxy[proxy_good]=np.abs(speed[safe[proxy_good]]-speed[previous[proxy_good]])*ages[proxy_good]/dt[proxy_good]
 return {'present':present,'reference_header_ns':np.where(present,stamps[safe],-1),'reference_age_ns':np.where(present,ages,-1),'reference_mps':np.where(present,speed[safe],np.nan),'reference_bag_minus_header_ns':np.array([rows[k]['bag_time_ns']-rows[k]['header_time_ns'] for k in safe],dtype=np.int64),'local_age_proxy_mps':proxy}

def age_bins(age):
 age=np.asarray(age)
 if age.ndim!=1 or age.dtype.kind!='i' or np.any(age<0) or np.any(age>100_000_000):raise ValueError('reference age outside supported budget')
 return np.searchsorted(np.array([20_000_000,50_000_000]),age,side='right').astype(np.int8)

def history_summary(values):
 values=np.asarray(values,dtype=float)
 if values.ndim!=1 or not len(values) or not np.isfinite(values).all():raise ValueError('invalid radar history')
 signed=abs(float(np.median(values)));absolute=float(np.median(np.abs(values)))
 return {'abs_signed_median_raw':signed,'median_abs_raw':absolute,'sign_cancellation_raw':max(0.,absolute-signed),'mixed_sign':bool(np.any(values<0) and np.any(values>0)),'raw_history_count':len(values)}

def replay_grid(rows,times,start):
 rawtime=np.array([q['bag_time_ns'] for q in rows],dtype=np.int64);times=np.asarray(times,dtype=np.int64)
 if not len(rawtime) or np.any(rawtime[1:]<=rawtime[:-1]):raise ValueError('ambiguous radar endpoint time')
 good=[q for q in rows if math.isfinite(q['speed_raw']) and 0<=q['bag_time_ns']-q['header_time_ns']<=250_000_000];tt=np.array([q['bag_time_ns'] for q in good],dtype=np.int64);vv=np.array([q['speed_raw'] for q in good],dtype=np.float64)
 if not len(tt):raise ValueError('no valid radar history')
 ticks=times[:,None]-np.arange(127,-1,-1,dtype=np.int64)[None,:]*50_000_000;at=np.searchsorted(tt,ticks,side='right')-1;safe=np.maximum(at,0);age=ticks-tt[safe]
 if np.any(at<0) or np.any(ticks<start) or np.any(tt[safe]<start) or np.any(age<0) or np.any(age>300_000_000) or not np.array_equal(tt[safe[:,-1]],times):raise ValueError('stored complete-history row cannot be reproduced')
 value=vv[safe];x=np.stack([value/10,np.abs(value)/10,age/300_000_000,np.ones_like(value)],axis=1).astype(np.float32);hist={k:[] for k in ['abs_signed_median_raw','median_abs_raw','sign_cancellation_raw','mixed_sign','raw_history_count']};hist.update({'radar_header_age_ns':[],'raw_current_signed':[]})
 for t,current in zip(times,safe[:,-1]):
  lo=np.searchsorted(tt,max(start,int(t)-1_000_000_000),side='left');hi=np.searchsorted(tt,t,side='right')
  if hi-lo<10:raise ValueError('stored anchor has inadequate history')
  for k,v in history_summary(vv[lo:hi]).items():hist[k].append(v)
  hist['radar_header_age_ns'].append(int(t)-good[current]['header_time_ns']);hist['raw_current_signed'].append(good[current]['speed_raw'])
 return x,{k:np.array(v,dtype=bool if k=='mixed_sign' else np.int64 if k in ['raw_history_count','radar_header_age_ns'] else np.float64) for k,v in hist.items()}

def independent_message(kind,b):
 def integer(p):return struct.unpack_from('<I',b,p)[0]
 def skip_string(p):
  end=p+4+integer(p)
  if end>len(b):raise ValueError('truncated independent string')
  return end
 if kind=='radar':
  speed=struct.unpack_from('<f',b,0)[0];p=skip_string(4)+4;p=skip_string(p)+4;seq,sec,ns=struct.unpack_from('<III',b,p)
  if ns>=10**9:raise ValueError('invalid independent nanoseconds')
  return {'speed_raw':float(speed),'sequence':seq,'header_time_ns':sec*10**9+ns}
 if kind!='INSPVA':raise ValueError('unknown independent probe kind')
 seq,sec,ns=struct.unpack_from('<III',b,0)
 if ns>=10**9:raise ValueError('invalid independent nanoseconds')
 p=skip_string(12);p=skip_string(p)+struct.calcsize('<HBIBHI');values=struct.unpack_from('<9d',b,p);status=integer(p+72)
 return {'sequence':seq,'header_time_ns':sec*10**9+ns,'horizontal_mps':math.hypot(values[3],values[4]),'status':status}

def decode_bag_blob(blob,kind):
 if kind not in ['radar','reference'] or not blob.startswith(b'#ROSBAG V2.0\n'):raise ValueError('unsupported trace bag')
 h=io.fields(blob[17:17+io.u32(blob,13)]);pos=struct.unpack('<Q',h['index_pos'])[0]
 if not 4096<=pos<len(blob):raise ValueError('invalid trace bag index')
 connections={}
 for rh,data in io.records(blob[pos:]):
  if rh['op']==b'\x07':
   cid=io.u32(rh['conn']);info=io.fields(data)
   if cid in connections:raise ValueError('duplicate source connection')
   connections[cid]={'topic':rh['topic'].decode(),'type':info['type'].decode()}
 chunks=io.chunk_infos(blob[pos:]);rows=[];payloads=[];chunk=0;messages=0
 for rh,body in io.records(blob[13:pos]):
  if rh['op']!=b'\x05':continue
  if chunk>=len(chunks) or rh['compression']!=b'none' or io.u32(rh['size'])!=len(body):raise ValueError('invalid trace chunk')
  counts=collections.Counter()
  for mh,data in io.records(body):
   if mh['op']!=b'\x02':continue
   cid=io.u32(mh['conn']);info=connections[cid];counts[cid]+=1;messages+=1;bagtime=io.ros_time(mh['time']);selected=kind=='radar' and info['topic']=='/omnipresense/radar/front_bumper' or kind=='reference' and info['type'].split('/')[-1]=='INSPVA'
   if selected:rows.append(io.decode_message('radar',data,bagtime) if kind=='radar' else refcore.decode_reference('INSPVA',data,bagtime));payloads.append(data)
  if dict(counts)!=chunks[chunk]['counts']:raise ValueError('source chunk message counts disagree')
  chunk+=1
 if chunk!=len(chunks) or not rows:raise ValueError('missing selected source messages')
 probes=[{'message_kind':'radar' if kind=='radar' else 'INSPVA','payload_base64':base64.b64encode(payloads[i]).decode(),'expected_fields':{k:rows[i][k] for k in (['speed_raw','sequence','header_time_ns'] if kind=='radar' else ['sequence','header_time_ns','horizontal_mps','status'])}} for i in sorted({0,len(rows)//2,len(rows)-1})]
 return rows,{'selected_messages':len(rows),'all_messages':messages,'chunks':chunk,'connections':{str(k):v for k,v in connections.items()},'bag_minus_header_ns_min':min(q['bag_time_ns']-q['header_time_ns'] for q in rows),'bag_minus_header_ns_max':max(q['bag_time_ns']-q['header_time_ns'] for q in rows),'status_counts':dict(collections.Counter(str(q.get('status','radar')) for q in rows))},probes

def freeze(create=False):
 p=read(ROOT/'protocol.json')
 if p['routes']!=list(ROUTES) or p['reference_max_age_ns']!=100_000_000 or p['reference_age_boundaries_ns']!=[20_000_000,50_000_000] or not p['no_offset_search']:raise ValueError('trace protocol changed')
 for name,h in p['input_sha256'].items():
  f=(ROOT/name).resolve()
  if not f.is_relative_to(RESEARCH) or digest(f)!=h:raise ValueError('trace source changed: '+name)
 lock={'protocol_sha256':digest(ROOT/'protocol.json'),'code_sha256':{name:digest(ROOT/name) for name in ['trace_core.py','generate_trace.py']}}
 target=ROOT/'analysis_lock.json'
 if target.exists() and read(target)!=lock:raise ValueError('trace implementation changed after lock')
 if create and not target.exists():write(target,lock)
 return p

def sources():
 old=RESEARCH/'vehicle_radar_multidrive';night=RESEARCH/'vehicle_night_training';reference=RESEARCH/'vehicle_reference_check';dayradar=[];daypayloads=[];probes=[];profiles={};windows=read(old/'inputs/fit/windows.json');windows=[{**w,'route':w['source_route']} for w in windows]
 with gzip.open(old/'inputs/fit/packets.jsonl.gz','rt',encoding='utf-8') as f:
  for line in f:
   packet=json.loads(line)
   if packet['file']=='radars.bag':
    b=base64.b64decode(packet['data'],validate=True);dayradar.append(io.decode_message('radar',b,packet['time_ns']));daypayloads.append(b)
 order=np.argsort([q['bag_time_ns'] for q in dayradar],kind='stable');dayradar=[dayradar[i] for i in order];daypayloads=[daypayloads[i] for i in order]
 for i in sorted({0,len(dayradar)//2,len(dayradar)-1}):probes.append({'source':'DAY pooled radar chunk extracts','message_kind':'radar','payload_base64':base64.b64encode(daypayloads[i]).decode(),'expected_fields':{k:dayradar[i][k] for k in ['speed_raw','sequence','header_time_ns']}})
 radars={};references={};publisher=read(night/'protocol.json')['publisher_sha256'];daycfg=read(reference/'protocol.json')['recordings']
 for route,role in zip(ROUTES[:3],DAY_ROLES):
  path=reference/'reference_bags'/(role+'.bag')
  if digest(path)!=daycfg[role]['reference_bag_publisher_sha256']:raise ValueError('day publisher reference hash mismatch')
  rows,profile,pp=decode_bag_blob(path.read_bytes(),'reference');radars[route]=dayradar;references[route]=rows;profiles[route]={'radar_scope':'pooled fixed-window chunk extracts; no complete radar bag claim','reference':profile};probes.extend({'source':str(path.relative_to(RESEARCH)),**v} for v in pp)
 for route in ROUTES[3:]:
  group=route.split('/')[0];profiles[route]={'radar_scope':'complete captured fitting-recording radar bag'}
  for name,kind in [('radars.bag','radar'),('reference.bag','reference')]:
   path=night/'source'/group/name
   if digest(path)!=publisher[route][name]:raise ValueError('night publisher source hash mismatch')
   rows,profile,pp=decode_bag_blob(path.read_bytes(),kind);(radars if kind=='radar' else references)[route]=rows;profiles[route][kind]=profile;probes.extend({'source':str(path.relative_to(RESEARCH)),**v} for v in pp)
 windows.extend(read(night/'data_report.json')['windows'])
 if len(windows)!=15 or any(w['role']!='fit' or w['route'] not in ROUTES for w in windows):raise ValueError('fixed fitting windows changed')
 return radars,references,{w['id']:w for w in windows},profiles,probes

def compute():
 freeze();old=RESEARCH/'vehicle_radar_multidrive';night=RESEARCH/'vehicle_night_training';members=read(RESEARCH/'vehicle_route_validation/membership.json');dayindex=read(old/'prepared/fit/index.json');nightindex=read(night/'night_index.json');labels={(q['window_id'],q['time_ns']):q for q in read(old/'inputs/fit/labels.json')};radar,reference,windows,profiles,probes=sources()
 with np.load(old/'prepared/fit/data.npz',allow_pickle=False) as z:day={k:z[k].copy() for k in ['X','y','median_feature']}
 with np.load(night/'night_fit.npz',allow_pickle=False) as z:dark={k:z[k].copy() for k in day}
 data={k:np.concatenate([day[k],dark[k]]) for k in day}
 with np.load(RESEARCH/'vehicle_matched_baselines/oof_predictions.npz',allow_pickle=False) as z:
  if not np.array_equal(data['y'],z['target_mps']):raise ValueError('prepared/frozen target population differs')
  prediction=z['context_boosting__prediction_mps'].copy()
 if len(members)!=27178 or len(data['y'])!=27178:raise ValueError('supported population changed')
 indexed={}
 for rows,offset in [(dayindex,0),(nightindex,len(day['y']))]:
  for q in rows:
   if q['array_index'] is not None:indexed[offset+q['array_index']]=q
 for i,m in enumerate(members):
  q=indexed[i]
  if m['id']!=i or q['role']!='fit' or q['time_ns']!=m['time_ns'] or q['window_id']!=m['window_id'] or windows[m['window_id']]['route']!=m['route']:raise ValueError('source index/member identity differs')
 traces={};assignment=np.zeros(len(members),dtype=int);window_labels=np.array([m['window_id'] for m in members]);y=data['y'];raw_counts={};source_checks=0
 for name,w in windows.items():
  ids=np.flatnonzero(window_labels==name);times=np.array([members[i]['time_ns'] for i in ids],dtype=np.int64);rows=radar[w['route']];ref=match_reference(reference[w['route']],times)
  if not np.all(ref['present']) or not np.array_equal(ref['reference_mps'],y[ids]):raise ValueError('raw reference label reconstruction differs')
  for local,i in enumerate(ids):
   q=indexed[i];stored_header=q.get('reference_header_ns') if i>=len(day['y']) else labels[(name,int(times[local]))]['reference_header_time_ns']
   if stored_header!=int(ref['reference_header_ns'][local]):raise ValueError('stored reference timestamp differs')
  x,h=replay_grid(rows,times,w['start_ns'])
  if not np.array_equal(x,data['X'][ids]) or not np.array_equal(h['abs_signed_median_raw'],data['median_feature'][ids]):raise ValueError('raw radar tensor/anchor reconstruction differs')
  for key,value in {**{k:v for k,v in ref.items() if k not in ['present','reference_mps']},**h}.items():
   if key not in traces:traces[key]=np.empty(len(members),dtype=value.dtype)
   traces[key][ids]=value
  assignment[ids]+=1;source_checks+=int(np.prod(x.shape));raw_counts[name]=sum(w['start_ns']<=q['bag_time_ns']<=w['end_ns'] for q in rows)
 if not np.all(assignment==1):raise ValueError('trace rows omitted or repeated')
 parent=read(RESEARCH/'vehicle_failure_diagnosis/results.json')
 if sum(raw_counts.values())!=sum(v['raw_endpoints_in_fixed_windows'] for v in parent['raw_support_costs'].values()):raise ValueError('raw endpoint denominator differs')
 traces['row_id']=np.arange(len(members),dtype=np.int64);traces['reference_age_bin']=age_bins(traces['reference_age_ns']);result=summarize(members,y,prediction,traces);result.update({'raw_window_endpoint_counts':raw_counts,'source_profiles':profiles,'raw_tensor_values_reproduced':source_checks,'raw_reference_labels_reproduced':len(y),'source_decode_probe_count':len(probes),'raw_support_costs':parent['raw_support_costs']})
 return result,traces,probes

def finite_stats(x):
 x=np.asarray(x,dtype=float);good=x[np.isfinite(x)]
 return {'total':len(x),'finite':len(good),'missing':int(len(x)-len(good)),'min':float(good.min()) if len(good) else None,'median':float(np.median(good)) if len(good) else None,'p95':float(np.quantile(good,.95)) if len(good) else None,'max':float(good.max()) if len(good) else None}

def summarize(members,y,pred,traces):
 speed=d.speed_bins(y);ages=traces['reference_age_bin'];mixed=traces['mixed_sign'];window=np.array([m['window_id'] for m in members]);allrows=np.ones(len(y),dtype=bool)
 def cell(mask):
  return {'count':int(mask.sum()),'hgb':d.metrics(y[mask],pred[mask]),'reference_age_ns':finite_stats(traces['reference_age_ns'][mask]),'local_age_proxy_mps':finite_stats(traces['local_age_proxy_mps'][mask]),'radar_header_age_ns':finite_stats(traces['radar_header_age_ns'][mask]),'sign_cancellation_raw':finite_stats(traces['sign_cancellation_raw'][mask]),'mixed_sign_rows':int(np.sum(mixed[mask])),'positive_cancellation_rows_above_numeric_tolerance':int(np.sum(traces['sign_cancellation_raw'][mask]>1e-6))}
 return {'scope':'Raw fitting-source replay and retrospective timing/radar associations; no established cause','supported_rows':len(y),'pooled':cell(allrows),'by_reference_speed':{name:cell(speed==i) for i,name in enumerate(d.SPEED_NAMES)},'by_reference_age':{str(k):cell(ages==k) for k in range(3)},'by_window':{name:cell(window==name) for name in sorted(set(window.tolist()))},'by_speed_and_sign':{name:{'mixed':cell((speed==i)&mixed),'other':cell((speed==i)&~mixed)} for i,name in enumerate(d.SPEED_NAMES)},'by_speed_and_reference_age':{name:{str(k):cell((speed==i)&(ages==k)) for k in range(3)} for i,name in enumerate(d.SPEED_NAMES)},'all_supported_rows_retained':True,'reference_timestamp_offset_searched':False,'reference_correction_applied':False,'new_model_fitted':False,'root_cause_established':False,'physical_bound_established':False,'reference_independence_established':False,'parent_development_gate_still_failed':True,'promotion_allowed':False,'reserved_final_opened':False,'witness_enabled':False,'decision':'TRACE_AUDIT_COMPLETE_PARENT_GATE_STILL_FAILED'}
