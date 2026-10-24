"""Acquire exactly one allowed complete radar bag and verify publisher SHA256."""
import concurrent.futures,hashlib,os,urllib.request
from continuous_core import ROOT,read,write,freeze,check_route,ranges,digest
import core_io as c
def acquire():
 freeze();p=read(ROOT/'protocol.json');check_route(p['route'],p['url']);source=ROOT/'source';source.mkdir(exist_ok=True);target=source/'radars.bag'
 if target.exists():
  if target.stat().st_size!=p['radar_bytes'] or digest(target)!=p['radar_publisher_sha256']:raise ValueError('existing radar source is corrupt')
  print('FULL RADAR SOURCE ALREADY VERIFIED',flush=True);return
 cache=ROOT/'source_cache';cache.mkdir(exist_ok=True);plan=ranges(p['radar_bytes']);records={}
 def fetch(item):
  number,(start,end)=item;path=cache/f'{number:04d}.bin';error=None
  for attempt in range(2):
   try:
    req=urllib.request.Request(p['url'],headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity','User-Agent':'CO-ACT-CPS-research/1.0'})
    with urllib.request.urlopen(req,timeout=45) as response:
     c.validate_range_response(response.status,response.headers.get('Content-Range'),response.headers.get('Content-Length'),start,end,p['radar_bytes']);content_range=response.headers['Content-Range'];data=response.read(end-start+2)
    if len(data)!=end-start+1:raise ValueError('short source range')
    path.write_bytes(data);return number,{'start':start,'end':end,'bytes':len(data),'content_range':content_range,'sha256':hashlib.sha256(data).hexdigest()}
   except Exception as exc:error=exc
  raise RuntimeError('full source range failed: '+str(error))
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
  futures=[pool.submit(fetch,item) for item in enumerate(plan)]
  for future in concurrent.futures.as_completed(futures):
   number,record=future.result();records[str(number)]=record
   if len(records)%5==0 or len(records)==len(plan):print('FULL RADAR DOWNLOAD',len(records),'/',len(plan),'ranges',flush=True)
 temp=source/'radars.bag.tmp'
 with temp.open('wb') as out:
  for i in range(len(plan)):out.write((cache/f'{i:04d}.bin').read_bytes())
 if temp.stat().st_size!=p['radar_bytes'] or digest(temp)!=p['radar_publisher_sha256']:raise ValueError('publisher SHA256 mismatch; do not use source')
 os.replace(temp,target);write(ROOT/'download_manifest.json',{'ranges':records,'size':p['radar_bytes'],'publisher_sha256':p['radar_publisher_sha256'],'whole_file_verified':True});print('FULL RADAR PUBLISHER CHECKSUM: PASS',flush=True)
if __name__=='__main__':acquire()
