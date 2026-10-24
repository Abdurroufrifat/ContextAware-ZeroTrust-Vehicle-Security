from pathlib import Path
import urllib.request,hashlib,json,concurrent.futures,struct,sys
ROOT=Path(__file__).resolve().parent;SRC=ROOT.parent
import core_io as c
p=json.loads((ROOT/'protocol.json').read_text())
( ROOT/'metadata').mkdir(exist_ok=True)
BASE='https://www.frdr-dfdr.ca'
def get(url,start=None,end=None,total=None):
 headers={'User-Agent':'CO-ACT-CPS-research/1.0'}
 if start is not None:headers['Range']=f'bytes={start}-{end}'
 with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=30) as r:
  if start is None:return r.read(2000000)
  match=__import__('re').fullmatch(r'bytes (\d+)-(\d+)/(\d+|\*)',r.headers.get('Content-Range',''))
  if r.status!=206 or not match or int(match[1])!=start or int(match[2])!=end:raise ValueError('range mismatch')
  if total is not None:c.validate_range_response(r.status,r.headers.get('Content-Range'),r.headers.get('Content-Length'),start,end,total)
  b=r.read(end-start+2)
  if len(b)!=end-start+1:raise ValueError('length mismatch')
  return b

def listing(item):
 role,cfg=item;folder=cfg['route'];dest=ROOT/'metadata'/role;dest.mkdir(exist_ok=True)
 url=BASE+'/cache/1/publication_1084/file_sizes/file_sizes-'+hashlib.sha256(folder.encode()).hexdigest()+'.json'
 b=get(url);(dest/'files.json').write_bytes(b);print('METADATA',role,len(b),flush=True)
 return role,json.loads(b)
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:lists=dict(pool.map(listing,p['recordings'].items()))

def bag(item):
 role,e=item;name=e['name'];dest=ROOT/'metadata'/role
 url=BASE+'/repo/files'+e['path']
 cached=SRC/'final_evaluation_work'/'metadata'/role
 if (cached/(name+'.header')).exists():head=(cached/(name+'.header')).read_bytes()
 else:head=get(url,0,65535,e['size'])
 if not head.startswith(b'#ROSBAG V2.0\n'):raise ValueError('not ROS bag')
 size=struct.unpack_from('<I',head,13)[0];h=c.fields(head[17:17+size]);idx=struct.unpack('<Q',h['index_pos'])[0]
 if (cached/(name+'.index')).exists():index=(cached/(name+'.index')).read_bytes()
 else:index=get(url,idx,e['size']-1,e['size'])
 (dest/(name+'.header')).write_bytes(head);(dest/(name+'.index')).write_bytes(index)
 con=[]
 for rh,d in c.records(index):
  if rh['op']!=b'\x07':continue
  data=c.fields(d);con.append({'conn':c.u32(rh['conn']),'topic':rh['topic'].decode(),'type':data['type'].decode(),'md5sum':data['md5sum'].decode(),'message_definition':data['message_definition'].decode()})
 meta={'file':name,'size':e['size'],'url':url,'index_pos':idx,'connections':con}
 (dest/(name+'.json')).write_text(json.dumps(meta,indent=2)+'\n')
 print('BAG INDEX',role,name,'connections',len(con),'chunks',len(c.chunk_infos(index)),flush=True)
 if name=='radars.bag' and not any(x['topic']=='/omnipresense/radar/front_bumper' for x in con):raise ValueError('Ground-speed radar missing; frozen recording cannot be replaced.')
 return meta
jobs=[(role,e) for role,x in lists.items() for e in x['contents'] if e['name'] in ('radars.bag','imus.bag','gnss.bag')]
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:list(pool.map(bag,jobs))
