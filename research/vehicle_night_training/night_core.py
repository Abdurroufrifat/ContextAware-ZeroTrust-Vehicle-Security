"""Fit-only night-source coverage experiment; no security permission is emitted."""
from pathlib import Path
import collections,hashlib,json,struct,sys
import numpy as np
ROOT=Path(__file__).resolve().parent
MASK=ROOT.parent/'vehicle_masked_history'
sys.path.insert(0,str(MASK))
import masked_core as m
from masked_core import e,c,read,write,digest
import reference_core as r
OLD=m.OLD
ROUTES=('Urban03/2023-11-23-19-12-16-Night','Urban04/2023-11-23-19-36-42-Night')
DEVELOPMENT='Urban01/2023-11-23-18-24-14-Night'
CONDITIONS=('day_only','day_plus_night')
KINDS=('causal_residual_tcn','masked_cnn_4ch')
SEEDS=(141,142,143)
def check_source(route,name):
 if route not in ROUTES or name not in ('radars.bag','reference.bag'):raise ValueError('only two fixed fit recordings and two bag types allowed')
 return True
def validate_routes(routes):
 if tuple(routes)!=ROUTES:raise ValueError('fit-only recordings changed or leaked')
def verified_bytes(blob,size,sha):
 if len(blob)!=size or hashlib.sha256(blob).hexdigest()!=sha:raise ValueError('publisher checksum or byte count mismatch')
 return blob
def windows(route,first,last):
 check_source(route,'radars.bag')
 if type(first) is not int or type(last) is not int or first>=last:raise ValueError('invalid recording interval')
 result=[]
 for i,fraction in enumerate((.2,.5,.8)):
  center=first+int((last-first)*fraction);start=center-45000000000;end=start+90000000000
  if start<first or end>last:raise ValueError('fixed window outside recording; no substitution')
  result.append({'id':route.split('/')[0]+'_night_'+str(i),'route':route,'role':'fit','fraction':fraction,'start_ns':start,'end_ns':end})
 if any(a['end_ns']>=b['start_ns'] for a,b in zip(result,result[1:])):raise ValueError('overlapping fixed windows')
 return result
def epoch_ids(size,steps,rng):
 if type(size) is not int or size<=0 or type(steps) is not int or steps<=0:raise ValueError('invalid update budget')
 needed=steps*128;parts=[]
 while needed:
  ids=rng.permutation(size)[:needed];parts.append(ids);needed-=len(ids)
 return np.concatenate(parts).reshape(steps,128)
def fixed_weights(day_y,y):
 count=np.bincount(np.digitize(day_y,[1,5,10]),minlength=4)
 if np.any(count==0):raise ValueError('day fitting bins must all be represented')
 return (len(day_y)/4)/count[np.digitize(y,[1,5,10])]
def decode(blob,name):
 if not blob.startswith(b'#ROSBAG V2.0\n'):raise ValueError('unsupported bag format')
 h=c.c.fields(blob[17:17+c.c.u32(blob,13)]);pos=struct.unpack('<Q',h['index_pos'])[0]
 if not 4096<=pos<len(blob):raise ValueError('invalid bag index')
 io=c.c;connections={}
 for rh,data in io.records(blob[pos:]):
  if rh['op']==b'\x07':
   info=io.fields(data);cid=io.u32(rh['conn'])
   if cid in connections:raise ValueError('duplicate connection')
   connections[cid]={'topic':rh['topic'].decode(),'type':info['type'].decode(),'md5sum':info['md5sum'].decode(),'message_definition':info['message_definition'].decode()}
 chunks=io.chunk_infos(blob[pos:]);rows=[];chunk=0;messages=0;wanted=0
 for head,body in io.records(blob[13:pos]):
  if head['op']!=b'\x05':continue
  if chunk>=len(chunks) or head['compression']!=b'none' or io.u32(head['size'])!=len(body):raise ValueError('unexpected bag chunk')
  counts=collections.Counter()
  for mh,data in io.records(body):
   if mh['op']!=b'\x02':continue
   cid=io.u32(mh['conn']);info=connections[cid];counts[cid]+=1;messages+=1;t=io.ros_time(mh['time'])
   if name=='radars.bag' and info['topic']=='/omnipresense/radar/front_bumper':rows.append(io.decode_message('radar',data,t));wanted+=1
   elif name=='reference.bag' and info['type'].split('/')[-1]=='INSPVA':rows.append(r.decode_reference('INSPVA',data,t));wanted+=1
  if dict(counts)!=chunks[chunk]['counts']:raise ValueError('chunk counts disagree with index')
  chunk+=1
 if chunk!=len(chunks) or not rows:raise ValueError('missing chunks or target channel')
 if name=='radars.bag' and any(a['bag_time_ns']>=b['bag_time_ns'] for a,b in zip(rows,rows[1:])):raise ValueError('ambiguous ground radar times')
 if name=='reference.bag':r.ReferenceIndex(rows)
 return rows,{'chunks':chunk,'all_messages':messages,'selected_messages':wanted,'connections':connections}
def freeze(create=False):
 protocol=read(ROOT/'protocol.json');validate_routes(protocol['fit_routes'])
 for route,files in protocol['publisher_sha256'].items():
  for name,h in files.items():check_source(route,name)
 for name,h in protocol['input_sha256'].items():
  p=(ROOT/name).resolve()
  if not p.is_relative_to(ROOT.parent) or digest(p)!=h:raise ValueError('prior input changed: '+name)
 lock={'protocol_sha256':digest(ROOT/'protocol.json'),'code_sha256':{n:digest(ROOT/n) for n in ['night_core.py','acquire_night.py','prepare_night.py','train_night.py','score_night.py']}}
 path=ROOT/'analysis_lock.json'
 if path.exists() and read(path)!=lock:raise ValueError('night analysis changed after freeze')
 if create and not path.exists():write(path,lock)
 return protocol
def source_path(route,name):
 check_source(route,name);return ROOT/'source'/route.split('/')[0]/name
def source_rows(route,name):
 cfg=read(ROOT/'protocol.json');listing=read(ROOT/'metadata'/(route.split('/')[0]+'.json'));entry=next(q for q in listing['contents'] if q['name']==name)
 expected='/1/published/publication_1084/submitted_data/'+route+'/'+name
 if entry['path']!=expected:raise ValueError('source listing path differs')
 blob=verified_bytes(source_path(route,name).read_bytes(),entry['size'],cfg['publisher_sha256'][route][name]);return decode(blob,name)
