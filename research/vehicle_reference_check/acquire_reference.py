"""Download complete reference bags and verify captured publisher checksums."""
from pathlib import Path
import concurrent.futures,hashlib,json,urllib.request
import core_io as c
ROOT=Path(__file__).resolve().parent
protocol=json.loads((ROOT/'protocol.json').read_text());(ROOT/'reference_bags').mkdir(exist_ok=True)
def acquire(item):
 role,cfg=item;meta=json.loads((ROOT/'metadata'/role/'reference.bag.json').read_text());path=ROOT/'reference_bags'/(role+'.bag');expected=cfg['reference_bag_publisher_sha256'];size=meta['size']
 if path.exists() and len(path.read_bytes())==size and hashlib.sha256(path.read_bytes()).hexdigest()==expected:return role,{'bytes':size,'sha256':expected,'publisher_checksum_verified':True}
 error=None
 for attempt in range(2):
  try:
   req=urllib.request.Request(meta['url'],headers={'Range':f'bytes=0-{size-1}','Accept-Encoding':'identity','User-Agent':'CO-ACT-CPS-research/1.0'})
   with urllib.request.urlopen(req,timeout=55) as response:
    if response.status==206:c.validate_range_response(response.status,response.headers.get('Content-Range'),response.headers.get('Content-Length'),0,size-1,size)
    elif response.status!=200:raise ValueError('unexpected whole-file response')
    blob=response.read(size+1)
   actual=hashlib.sha256(blob).hexdigest()
   if len(blob)!=size or actual!=expected:raise ValueError('whole reference bag checksum mismatch')
   if blob[:4096]!=(ROOT/'metadata'/role/'reference.bag.header').read_bytes() or blob[meta['index_pos']:]!=(ROOT/'metadata'/role/'reference.bag.index').read_bytes():raise ValueError('cached metadata disagrees with publisher-verified source')
   temp=path.with_suffix('.part');temp.write_bytes(blob);temp.replace(path)
   print('REFERENCE SOURCE VERIFIED',role,size,'bytes',flush=True)
   return role,{'bytes':size,'sha256':actual,'publisher_checksum_verified':True}
  except Exception as e:error=e
 raise RuntimeError(f'{role}: {error}')
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:results=dict(pool.map(acquire,protocol['recordings'].items()))
(ROOT/'acquisition.json').write_text(json.dumps({'protocol_sha256':hashlib.sha256((ROOT/'protocol.json').read_bytes()).hexdigest(),'reference_files':results},indent=2)+'\n')
