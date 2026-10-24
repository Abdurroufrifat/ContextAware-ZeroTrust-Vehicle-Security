"""Reclassify already opened recordings as development data, preserving origins."""
from pathlib import Path
import gzip, hashlib, io, json
ROOT=Path(__file__).resolve().parent
FIT_ROLES=('fit','evaluation_urban03','evaluation_urban04')
RESERVED='Urban02/2023-11-23-17-53-52-Night'

def validate_split(windows):
 if set(windows)!={'fit','calibration'}:raise ValueError('unexpected development split')
 seen=set();routes={role:set() for role in windows};intervals={}
 for role,rows in windows.items():
  if not rows:raise ValueError('empty split')
  for w in rows:
   source=w['source_role'];route=w['source_route']
   if route==RESERVED:raise ValueError('reserved test cannot enter development')
   if role=='fit' and source=='calibration':raise ValueError('calibration leaked into training')
   if source not in (FIT_ROLES if role=='fit' else ('calibration',)):raise ValueError('unapproved source role')
   if w['id'] in seen:raise ValueError('duplicate window')
   seen.add(w['id']);routes[role].add(route)
   if not isinstance(w['start_ns'],int) or not isinstance(w['end_ns'],int) or w['start_ns']>=w['end_ns']:raise ValueError('invalid interval')
   intervals.setdefault((role,route),[]).append((w['start_ns'],w['end_ns']))
 if routes['fit']&routes['calibration']:raise ValueError('same recording in both splits')
 for intervals_in_drive in intervals.values():
  intervals_in_drive.sort()
  if any(a[1]>=b[0] for a,b in zip(intervals_in_drive,intervals_in_drive[1:])):raise ValueError('overlapping windows')

def compose():
 parent=ROOT.parent;read=lambda p:json.loads(p.read_text(encoding='utf-8'))
 protocol=read(ROOT/'protocol.json');temporal=parent/'vehicle_temporal_policy';reference_path=parent/'vehicle_reference_check'/'matched_endpoints.json';reference=read(reference_path)
 routes=read(temporal/'protocol.json')['recordings'];windows={};outputs={};sources={'../vehicle_reference_check/matched_endpoints.json':hashlib.sha256(reference_path.read_bytes()).hexdigest()}
 for role,source_roles in protocol['development_source_roles'].items():
  if tuple(source_roles)!=(FIT_ROLES if role=='fit' else ('calibration',)):raise ValueError('fixed source roles changed')
  selected_windows=[];labels=[];packets=[]
  for source in source_roles:
   route=routes[source]['route']
   if protocol['source_recordings'][source]!=route:raise ValueError('source route changed')
   folder=temporal/'roles'/source
   for name in ('windows.json','packets.jsonl.gz'):
    path=folder/name;sources[str(path.relative_to(parent)).replace('\\','/')]=hashlib.sha256(path.read_bytes()).hexdigest()
   selected_windows.extend({**w,'role':role,'source_role':source,'source_route':route} for w in read(folder/'windows.json'))
   labels.extend({**r,'role':role,'source_role':source,'source_route':route} for r in reference if r['role']==source)
   packets.append(gzip.decompress((folder/'packets.jsonl.gz').read_bytes()))
  if len({(r['window_id'],r['time_ns']) for r in labels})!=len(labels):raise ValueError('duplicate endpoint')
  windows[role]=selected_windows;outputs[role]={'windows':selected_windows,'labels':labels,'packet_text':b''.join(packets)}
 validate_split(windows)
 return outputs,sources

def materialize():
 if (ROOT/'design_lock.json').exists():raise ValueError('inputs are already frozen')
 outputs,sources=compose()
 for role,data in outputs.items():
  folder=ROOT/'inputs'/role;folder.mkdir(parents=True,exist_ok=True)
  for name in ('labels','windows'):(folder/(name+'.json')).write_text(json.dumps(data[name],indent=2,allow_nan=False)+'\n',encoding='utf-8')
  with (folder/'packets.jsonl.gz').open('wb') as file:
   with gzip.GzipFile(filename='',mode='wb',fileobj=file,mtime=0,compresslevel=6) as compressed:compressed.write(data['packet_text'])
  print('POOLED',role,len(data['windows']),'blocks',len(data['labels']),'endpoints',flush=True)
 (ROOT/'source_lock.json').write_text(json.dumps(sources,indent=2)+'\n',encoding='utf-8')
if __name__=='__main__':materialize()
