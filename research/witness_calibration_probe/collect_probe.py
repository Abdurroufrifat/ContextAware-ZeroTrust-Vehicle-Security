"""Acquire fixed development byte ranges. No final-evaluation routes are accessed."""
from pathlib import Path
import concurrent.futures,csv,hashlib,json,struct,urllib.request
import probe_core as c
HERE=Path(__file__).resolve().parent
BASE=HERE/'source_index'
OUT=HERE/'evidence';OUT.mkdir(exist_ok=True)
protocol=json.loads((HERE/'protocol.json').read_text())
root='https://www.frdr-dfdr.ca/repo/files/1/published/publication_1084/submitted_data/'+protocol['route']+'/'
metadata={name:json.loads((BASE/(name+'.json')).read_text()) for name in ['radars.bag','imus.bag']}
chunks={name:c.chunk_infos((BASE/(name+'.index')).read_bytes()) for name in metadata}
tmin=min(q['start_ns'] for q in chunks['radars.bag']);tmax=max(q['end_ns'] for q in chunks['radars.bag'])
windows=[];required={name:set() for name in metadata}
for role,fracs in protocol['window_fractions'].items():
 for fraction in fracs:
  target=tmin+int((tmax-tmin)*fraction)
  ci=next((i for i,q in enumerate(chunks['radars.bag']) if q['start_ns']<=target<=q['end_ns']),None)
  if ci is None:raise ValueError('target falls in unrecorded gap; stop rather than reselect')
  chunk=chunks['radars.bag'][ci]
  ii=[i for i,q in enumerate(chunks['imus.bag']) if q['start_ns']<=chunk['end_ns'] and q['end_ns']>=chunk['start_ns']-250000000]
  if not ii:raise ValueError('missing IMU/OBD chunk window')
  windows.append({'id':f'{role}_{fraction:.2f}','role':role,'fraction':fraction,'radar_chunk':ci,'imu_chunks':ii,'start_ns':chunk['start_ns'],'end_ns':chunk['end_ns']})
  required['radars.bag'].add(ci);required['imus.bag'].update(ii)
assert len({w['radar_chunk'] for w in windows})==len(windows)
plan={'protocol_sha256':hashlib.sha256((HERE/'protocol.json').read_bytes()).hexdigest(),'windows':windows,'ranges':[]}
for name,ids in required.items():
 for i in sorted(ids):
  q=chunks[name][i];start=q['position'];end=(chunks[name][i+1]['position'] if i+1<len(chunks[name]) else metadata[name]['index_pos'])-1
  plan['ranges'].append({'file':name,'chunk_index':i,'start':start,'end':end,'source_size':metadata[name]['size'],'relative_cache':f'{name}.{i:04d}.bin','url':root+name})
planpath=HERE/'acquisition_plan.json'
if planpath.exists() and json.loads(planpath.read_text())!=plan:raise ValueError('frozen acquisition plan changed')
planpath.write_text(json.dumps(plan,indent=2)+'\n')
print('FIXED PLAN',len(windows),'windows',len(plan['ranges']),'ranges',sum(q['end']-q['start']+1 for q in plan['ranges']),'bytes',flush=True)
cache=OUT/'chunks';cache.mkdir(exist_ok=True)
old=json.loads((OUT/'downloads.json').read_text()) if (OUT/'downloads.json').exists() else {}
def download(q):
 dest=cache/q['relative_cache'];key=q['relative_cache']
 if dest.exists():
  data=dest.read_bytes()
  if key not in old or len(data)!=q['end']-q['start']+1 or hashlib.sha256(data).hexdigest()!=old[key]['sha256']:raise ValueError('unverified existing cache')
  return key,old[key]
 err=None
 for attempt in range(2):
  try:
   req=urllib.request.Request(q['url'],headers={'Range':f"bytes={q['start']}-{q['end']}",'Accept-Encoding':'identity','User-Agent':'CO-ACT-CPS-research/1.0'})
   with urllib.request.urlopen(req,timeout=30) as r:
    expect=f"bytes {q['start']}-{q['end']}/{q['source_size']}"
    c.validate_range_response(r.status,r.headers.get('Content-Range'),r.headers.get('Content-Length'),q['start'],q['end'],q['source_size'])
    actual_range=r.headers['Content-Range']
    data=r.read(q['end']-q['start']+2)
    if len(data)!=q['end']-q['start']+1:raise ValueError('range length mismatch')
   dest.write_bytes(data)
   info={'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'content_range':actual_range}
   print('DOWNLOADED',key,len(data),flush=True);return key,info
  except Exception as e:
   err=e;print('TRANSFER RETRY',key,type(e).__name__,str(e)[:120],flush=True)
 raise RuntimeError(f"range acquisition failed for {key}: {err}")
completed=dict(old)
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
 jobs={pool.submit(download,q):q for q in plan['ranges']}
 for job in concurrent.futures.as_completed(jobs):
  key,info=job.result();completed[key]=info;(OUT/'downloads.json').write_text(json.dumps(completed,indent=2)+'\n')

def extract(name,i,topic,kind):
 data=(cache/f'{name}.{i:04d}.bin').read_bytes();outer=list(c.records(data))
 if outer[0][0].get('op')!=b'\x05' or outer[0][0].get('compression')!=b'none':raise ValueError('unsupported chunk layout/compression')
 body=outer[0][1]
 if c.u32(outer[0][0]['size'])!=len(body):raise ValueError('uncompressed chunk size mismatch')
 conn=next(q['conn'] for q in metadata[name]['connections'] if q['topic']==topic)
 actual={};rows=[]
 for h,d in c.records(body):
  if h['op']!=b'\x02':continue
  cid=c.u32(h['conn']);actual[cid]=actual.get(cid,0)+1
  if cid==conn:rows.append(c.decode_message(kind,d,c.ros_time(h['time'])))
 if actual!=chunks[name][i]['counts']:raise ValueError('decoded chunk counts disagree with source bag index')
 return rows

raw_radar=[];raw_obd=[];allpairs=[];windows_summary=[]
for w in windows:
 radar=extract('radars.bag',w['radar_chunk'],'/omnipresense/radar/front_bumper','radar')
 obd=sorted([r for i in w['imu_chunks'] for r in extract('imus.bag',i,'/obd2/speed','obd')],key=lambda r:r['bag_time_ns'])
 pairs,stats=c.causal_pairs(radar,obd,w['start_ns'])
 for rows,dest in [(radar,raw_radar),(obd,raw_obd),(pairs,allpairs)]:
  for r in rows:dest.append({'window_id':w['id'],'role':w['role'],**r})
 windows_summary.append({**w,**stats,'raw_radar_records':len(radar),'raw_obd_records':len(obd),
  'obd_unit_labels':sorted({r['unit'] for r in obd}),
  'radar_header_future':sum(r['header_time_ns']>r['bag_time_ns'] for r in radar),
  'radar_speed_min':min((r['speed_raw'] for r in radar),default=None),'radar_speed_max':max((r['speed_raw'] for r in radar),default=None),
  'obd_kph_min':min((r['speed_raw'] for r in obd),default=None),'obd_kph_max':max((r['speed_raw'] for r in obd),default=None)})
for name,rows in [('radar_raw',raw_radar),('obd_raw',raw_obd),('causal_pairs',allpairs)]:
 with (HERE/(name+'.csv')).open('w',newline='') as f:
  if rows:
   writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
(HERE/'window_summary.json').write_text(json.dumps(windows_summary,indent=2)+'\n')
print('EXTRACTED',len(raw_radar),'radar',len(raw_obd),'OBD',len(allpairs),'causal pairs',flush=True)
