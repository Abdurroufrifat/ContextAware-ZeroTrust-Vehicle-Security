"""Acquire only predeclared fit radar/reference files; verify publisher hashes."""
import concurrent.futures,hashlib,json,os,urllib.request
from night_core import *
BASE='https://www.frdr-dfdr.ca'
def fetch(route,name):
 check_source(route,name);cfg=freeze();folder=ROOT/'metadata';folder.mkdir(exist_ok=True);listing_path=folder/(route.split('/')[0]+'.json')
 # Listings are acquired sequentially before independent source downloads.
 listing=read(listing_path);entry=next(x for x in listing['contents'] if x['name']==name)
 expected='/1/published/publication_1084/submitted_data/'+route+'/'+name
 if entry['path']!=expected or type(entry['size']) is not int or entry['size']<=0:raise ValueError('unapproved listing entry')
 path=source_path(route,name);sha=cfg['publisher_sha256'][route][name];size=entry['size']
 if path.exists():verified_bytes(path.read_bytes(),size,sha);return
 path.parent.mkdir(parents=True,exist_ok=True);part=path.with_suffix('.part');error=None
 for attempt in range(2):
  try:
   req=urllib.request.Request(BASE+'/repo/files'+expected,headers={'Range':f'bytes=0-{size-1}','Accept-Encoding':'identity','User-Agent':'CO-ACT-CPS-research/1.0'})
   with urllib.request.urlopen(req,timeout=55) as response:
    c.c.validate_range_response(response.status,response.headers.get('Content-Range'),response.headers.get('Content-Length'),0,size-1,size)
    blob=response.read(size+1)
   verified_bytes(blob,size,sha)
   with part.open('wb') as out:out.write(blob);out.flush();os.fsync(out.fileno())
   os.replace(part,path);print('PUBLISHER VERIFIED',route,name,size,'bytes',flush=True);return
  except Exception as exc:error=exc
 raise RuntimeError(f'{route}/{name}: {error}')
def acquire():
 freeze();(ROOT/'metadata').mkdir(exist_ok=True)
 for route in ROUTES:
  url=BASE+'/cache/1/publication_1084/file_sizes/file_sizes-'+hashlib.sha256(route.encode()).hexdigest()+'.json'
  with urllib.request.urlopen(url,timeout=40) as response:blob=response.read(1000000)
  listing=json.loads(blob)
  if listing['path']!='/1/published/publication_1084/submitted_data/'+route+'/':raise ValueError('listing recording differs')
  write(ROOT/'metadata'/(route.split('/')[0]+'.json'),listing)
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(lambda job:fetch(*job),[(route,name) for route in ROUTES for name in ['radars.bag','reference.bag']]))
 write(ROOT/'source_lock.json',{str(source_path(route,name).relative_to(ROOT)):digest(source_path(route,name)) for route in ROUTES for name in ['radars.bag','reference.bag']})
if __name__=='__main__':acquire()
