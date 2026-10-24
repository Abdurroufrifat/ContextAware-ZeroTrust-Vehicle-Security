"""Full recording development replay; reference never supplies model features."""
from pathlib import Path
import collections,hashlib,importlib.util,json,math,struct
import numpy as np
import core_io as c
import estimator_core as e
import quality_core as q
ROOT=Path(__file__).resolve().parent
ROUTE='Urban01/2023-11-23-18-24-14-Night'
URL='https://www.frdr-dfdr.ca/repo/files/1/published/publication_1084/submitted_data/'+ROUTE+'/radars.bag'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,value):p.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def check_route(route,url):
 if route!=ROUTE or url!=URL:raise ValueError('only the fixed development recording may be acquired')
def ranges(size,block=8388608):
 if type(size) is not int or type(block) is not int or size<=0 or block<=0:raise ValueError('invalid byte range plan')
 return [(start,min(size-1,start+block-1)) for start in range(0,size,block)]
def segment(t,start):return (t-start)//300000000000
def freeze(create=False):
 protocol=read(ROOT/'protocol.json');check_route(protocol['route'],protocol['url'])
 for name,h in protocol['source_sha256'].items():
  p=(ROOT/name).resolve()
  if not p.is_relative_to(ROOT.parent) or digest(p)!=h:raise ValueError('continuous input changed: '+name)
 code=('core_io.py','reference_core.py','estimator_core.py','quality_core.py','continuous_core.py','acquire_full.py','run_continuous.py')
 lock={'protocol_sha256':digest(ROOT/'protocol.json'),'code_sha256':{n:digest(ROOT/n) for n in code},'models_refitted':False,'reserved_test_opened':False}
 path=ROOT/'analysis_lock.json'
 if path.exists() and read(path)!=lock:raise ValueError('continuous experiment implementation changed')
 if create and not path.exists():write(path,lock)
 return lock
def decode_radar():
 protocol=read(ROOT/'protocol.json');meta=read(ROOT.parent/'vehicle_temporal_policy/metadata/calibration/radars.bag.json');path=ROOT/'source/radars.bag';blob=path.read_bytes()
 if len(blob)!=protocol['radar_bytes'] or hashlib.sha256(blob).hexdigest()!=protocol['radar_publisher_sha256']:raise ValueError('full radar publisher checksum failed')
 original=ROOT.parent/'vehicle_temporal_policy/metadata/calibration'
 head=(original/'radars.bag.header').read_bytes()
 if blob[:len(head)]!=head or blob[meta['index_pos']:]!=(original/'radars.bag.index').read_bytes():raise ValueError('radar header/index changed')
 indices=c.chunk_infos(blob[meta['index_pos']:]);connection=next(v['conn'] for v in meta['connections'] if v['topic']=='/omnipresense/radar/front_bumper');rows=[];chunk=0;all_messages=0
 for h,body in c.records(blob[13:meta['index_pos']]):
  if h['op']!=b'\x05':continue
  if h['compression']!=b'none' or c.u32(h['size'])!=len(body):raise ValueError('unsupported radar chunk')
  counts=collections.Counter()
  for mh,data in c.records(body):
   if mh['op']!=b'\x02':continue
   conn=c.u32(mh['conn']);counts[conn]+=1;all_messages+=1
   if conn==connection:rows.append(c.decode_message('radar',data,c.ros_time(mh['time'])))
  if dict(counts)!=indices[chunk]['counts']:raise ValueError('full radar chunk counts disagree')
  chunk+=1
 if chunk!=len(indices) or not rows or any(a['bag_time_ns']>=b['bag_time_ns'] for a,b in zip(rows,rows[1:])):raise ValueError('missing chunks or ambiguous radar time')
 return rows,{'publisher_sha256':protocol['radar_publisher_sha256'],'bytes':len(blob),'chunks':chunk,'all_connection_messages':all_messages,'ground_radar_messages':len(rows),'first_ns':rows[0]['bag_time_ns'],'last_ns':rows[-1]['bag_time_ns']}
def prepare(rows):
 if not rows:raise ValueError('empty radar recording')
 grid=e.RadarGrid(rows);start=rows[0]['bag_time_ns'];xs=[];medians=[];index=[]
 for ordinal,row in enumerate(rows):
  t=row['bag_time_ns'];warm=t-start<6350000000;features=grid.at(t,start);lo=np.searchsorted(grid.times,t-1000000000,side='left');hi=np.searchsorted(grid.times,t,side='right');ready=bool(features is not None and hi-lo>=10)
  index.append({'ordinal':ordinal,'time_ns':t,'segment':int(segment(t,start)),'warmup':warm,'ready':ready,'array_index':len(xs) if ready else None})
  if ready:xs.append(features);medians.append(abs(float(np.median(grid.values[lo:hi]))))
 return {'X':np.asarray(xs,dtype=np.float32),'median_feature':np.asarray(medians,dtype=np.float64)},index
def reference_targets(index):
 path=ROOT.parent/'vehicle_reference_check/run_reference.py';spec=importlib.util.spec_from_file_location('_continuous_reference_reader',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);rows,_=module.decode_bag('calibration');import reference_core as ref
 lookup=ref.ReferenceIndex(rows['INSPVA']);targets=[]
 for point in index:
  p=lookup.latest(point['time_ns'],100000000);targets.append({'time_ns':point['time_ns'],'reference_mps':p['horizontal_mps'] if p else None,'reference_header_ns':p['header_time_ns'] if p else None,'reference_status':p['status'] if p else None})
 return targets
def account(y,p,keep,raw):
 y=np.asarray(y);p=np.asarray(p);keep=np.asarray(keep)
 if y.ndim!=1 or p.shape!=y.shape or keep.shape!=y.shape or keep.dtype.kind!='b' or raw<len(y) or not np.all(np.isfinite(p)):raise ValueError('unaligned continuous accounting')
 known=np.isfinite(y);accepted=keep&known;withheld=(~keep)&known
 return {'raw_endpoints':int(raw),'features_ready':len(y),'unready_raw':int(raw-len(y)),'ready_reference_missing':int((~known).sum()),'reference_coverage_on_ready':float(known.mean()) if len(y) else 0.,'usable_total':int(keep.sum()),'usable_reference_missing':int((keep&~known).sum()),'quality_withheld':int((~keep).sum()),'whole_acceptance_fraction':int(keep.sum())/raw if raw else 0.,'all_ready_reference_scored':q.error_metrics(y[known],p[known]),'accepted':q.error_metrics(y[accepted],p[accepted]),'withheld':q.error_metrics(y[withheld],p[withheld])}
