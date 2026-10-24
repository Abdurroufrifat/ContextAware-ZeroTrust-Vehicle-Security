"""Read bag header and index only; do not open reference measurement payloads."""
from pathlib import Path
import concurrent.futures,json,struct,urllib.request
import core_io as c
ROOT=Path(__file__).resolve().parent
def get(url,start,end,size):
 req=urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity','User-Agent':'CO-ACT-CPS-research/1.0'})
 with urllib.request.urlopen(req,timeout=40) as r:
  c.validate_range_response(r.status,r.headers.get('Content-Range'),r.headers.get('Content-Length'),start,end,size);b=r.read(end-start+2)
 if len(b)!=end-start+1:raise ValueError('short metadata response')
 return b
def fetch(role):
 folder=ROOT/'metadata'/role;e=json.loads((folder/'source_listing.json').read_text());url='https://www.frdr-dfdr.ca/repo/files'+e['path'];size=e['size'];head=get(url,0,4095,size)
 if not head.startswith(b'#ROSBAG V2.0\n'):raise ValueError('unsupported bag format')
 h=c.fields(head[17:17+struct.unpack_from('<I',head,13)[0]]);idx=struct.unpack('<Q',h['index_pos'])[0];index=get(url,idx,size-1,size);connections=[]
 for rh,d in c.records(index):
  if rh['op']!=b'\x07':continue
  fields=c.fields(d);connections.append({'conn':c.u32(rh['conn']),'topic':rh['topic'].decode(),'type':fields['type'].decode(),'md5sum':fields['md5sum'].decode(),'message_definition':fields['message_definition'].decode()})
 (folder/'reference.bag.header').write_bytes(head);(folder/'reference.bag.index').write_bytes(index);meta={'file':'reference.bag','url':url,'size':size,'index_pos':idx,'connections':connections};(folder/'reference.bag.json').write_text(json.dumps(meta,indent=2)+'\n')
 print(role,[(q['topic'],q['type']) for q in connections],len(c.chunk_infos(index)),'chunks',flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(fetch,['fit','calibration','evaluation_urban03','evaluation_urban04']))
