"""Acquire fixed continuous blocks, coalescing adjacent chunks into <=8 MiB HTTP ranges."""
from pathlib import Path
import base64,concurrent.futures,gzip,hashlib,json,sys,urllib.request
import core_io as c
import final_core as f
from data_stream import build_stream
ROOT=Path(__file__).resolve().parent;role=sys.argv[1]
protocol=json.loads((ROOT/'protocol.json').read_text());cfg=protocol['recordings'][role]
if role.startswith('evaluation') and not (ROOT/'policy_lock.json').exists():raise SystemExit('Freeze the calibrated policy before evaluation acquisition')
meta={n:json.loads((ROOT/'metadata'/role/(n+'.json')).read_text()) for n in ('radars.bag','imus.bag','gnss.bag')}
indices={n:c.chunk_infos((ROOT/'metadata'/role/(n+'.index')).read_bytes()) for n in meta}
first=min(q['start_ns'] for q in indices['radars.bag']);last=max(q['end_ns'] for q in indices['radars.bag']);windows=[];needed={n:set() for n in meta}
for wi,fraction in enumerate(cfg['centers']):
 center=first+int((last-first)*fraction);start=center-protocol['block_duration_ns']//2;end=start+protocol['block_duration_ns']
 if start-1000000000<first or end>last:raise ValueError('fixed block outside recording; no replacement permitted')
 selection={n:[i for i,q in enumerate(chunks) if q['start_ns']<=end and q['end_ns']>=start-1000000000] for n,chunks in indices.items()}
 if not all(selection.values()):raise ValueError('missing channel')
 for n,ids in selection.items():needed[n].update(ids)
 windows.append({'id':f'{role}_{wi}','role':role,'fraction':fraction,'start_ns':start,'end_ns':end,'chunks':selection})
plan={'role':role,'protocol_sha256':hashlib.sha256((ROOT/'protocol.json').read_bytes()).hexdigest(),'windows':windows,'ranges':[],'groups':[]}
for n,ids in needed.items():
 child=[]
 for i in sorted(ids):
  start=indices[n][i]['position'];end=(indices[n][i+1]['position'] if i+1<len(indices[n]) else meta[n]['index_pos'])-1
  child.append({'file':n,'chunk':i,'start':start,'end':end,'key':f'{n}.{i:04d}.bin'})
 plan['ranges'].extend(child);group=[]
 for q in child:
  if group and (q['start']!=group[-1]['end']+1 or q['end']-group[0]['start']+1>8388608):plan['groups'].append({'file':n,'children':group});group=[]
  group.append(q)
 if group:plan['groups'].append({'file':n,'children':group})
folder=ROOT/'roles'/role;folder.mkdir(parents=True,exist_ok=True);p=folder/'plan.json'
if p.exists() and json.loads(p.read_text())!=plan:raise ValueError('frozen selection changed')
p.write_text(json.dumps(plan,indent=2)+'\n');cache=ROOT/'source_cache'/role;cache.mkdir(parents=True,exist_ok=True)
records=json.loads((folder/'downloads.json').read_text()) if (folder/'downloads.json').exists() else {}
print('CONTINUOUS',role,len(windows),'blocks',len(plan['ranges']),'chunks',len(plan['groups']),'HTTP groups',sum(q['end']-q['start']+1 for q in plan['ranges']),'bytes',flush=True)
def download(item):
 number,group=item;n=group['file'];children=group['children'];start=children[0]['start'];end=children[-1]['end'];key=str(number)
 if key in records and all((cache/q['key']).is_file() and hashlib.sha256((cache/q['key']).read_bytes()).hexdigest()==records[key]['children'][q['key']] for q in children):return key,records[key]
 error=None
 for attempt in range(2):
  try:
   request=urllib.request.Request(meta[n]['url'],headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity','User-Agent':'CO-ACT-CPS-research/1.0'})
   with urllib.request.urlopen(request,timeout=55) as response:
    c.validate_range_response(response.status,response.headers.get('Content-Range'),response.headers.get('Content-Length'),start,end,meta[n]['size']);cr=response.headers['Content-Range'];blob=response.read(end-start+2)
   if len(blob)!=end-start+1:raise ValueError('short grouped response')
   hashes={}
   for q in children:
    data=blob[q['start']-start:q['end']-start+1];(cache/q['key']).write_bytes(data);hashes[q['key']]=hashlib.sha256(data).hexdigest()
   return key,{'sha256':hashlib.sha256(blob).hexdigest(),'bytes':len(blob),'content_range':cr,'children':hashes}
  except Exception as e:error=e
 raise RuntimeError(f'failed {role} group {key}: {error}')
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
 futures=[pool.submit(download,item) for item in enumerate(plan['groups'])];done=0
 for future in concurrent.futures.as_completed(futures):
  key,info=future.result();records[key]=info;(folder/'downloads.json').write_text(json.dumps(records,indent=2)+'\n');done+=1
  if done%5==0 or done==len(futures):print('TRANSFER',role,done,'/',len(futures),flush=True)
streams={};packets=0
with gzip.open(folder/'packets.jsonl.gz','wt',encoding='utf8') as output:
 for q in plan['ranges']:
  n=q['file'];i=q['chunk'];head,body=list(c.records((cache/q['key']).read_bytes()))[0]
  if head['op']!=b'\x05' or head['compression']!=b'none' or c.u32(head['size'])!=len(body):raise ValueError('unsupported source chunk')
  topic='/omnipresense/radar/front_bumper' if n=='radars.bag' else '/obd2/speed' if n=='imus.bag' else '/novatel/gps';conn=next(v['conn'] for v in meta[n]['connections'] if v['topic']==topic);counts={};rows=[]
  for h,b in c.records(body):
   if h['op']!=b'\x02':continue
   cid=c.u32(h['conn']);counts[cid]=counts.get(cid,0)+1
   if cid!=conn:continue
   t=c.ros_time(h['time']);output.write(json.dumps({'file':n,'chunk':i,'time_ns':t,'data':base64.b64encode(b).decode()},separators=(',',':'))+'\n');packets+=1
   rows.append(f.decode_gnss(b,t) if n=='gnss.bag' else c.decode_message('radar' if n=='radars.bag' else 'obd',b,t))
  if counts!=indices[n][i]['counts']:raise ValueError('source message counts changed')
  streams[n,i]=rows
model=json.loads((ROOT/'radar_model_lock.json').read_text())['models']['signed_median_balanced'];allrows=[];summary=[]
for w in windows:
 selected={n:sorted([r for i in ids for r in streams[n,i]],key=lambda r:r['bag_time_ns']) for n,ids in w['chunks'].items()}
 rows,stats=build_stream(selected['radars.bag'],selected['imus.bag'],selected['gnss.bag'],w['start_ns'],w['end_ns'],model)
 allrows.extend({'window_id':w['id'],'role':role,**r} for r in rows);summary.append({**w,**stats})
(folder/'stream.json').write_text(json.dumps(allrows,indent=2,allow_nan=False)+'\n');(folder/'windows.json').write_text(json.dumps(summary,indent=2)+'\n')
print('EXTRACTED',role,len(allrows),'scored endpoints;',sum(r['valid'] for r in allrows),'valid;',packets,'packets',flush=True)
