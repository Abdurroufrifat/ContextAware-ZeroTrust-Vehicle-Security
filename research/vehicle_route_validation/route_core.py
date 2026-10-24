"""Retrospective group splits; all fitted preprocessing stays inside each fold."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parent;CONTEXT=ROOT.parent/'vehicle_context_boosting'
sys.path.insert(0,str(CONTEXT))
import context_core as c
from train_context import export
read=c.read;write=c.write;digest=c.digest
GROUPS=('Urban01','Urban03','Urban04')
ROUTES=tuple(c.FIT_ROUTES)
def group_of(route):
 if route not in ROUTES:raise ValueError('only fixed fitting recordings may enter route validation')
 return route.split('/')[0]
def split(groups,held):
 groups=np.asarray(groups)
 if groups.ndim!=1 or set(groups.tolist())!=set(GROUPS) or held not in GROUPS:raise ValueError('unknown or missing route group')
 return np.flatnonzero(groups!=held),np.flatnonzero(groups==held)
def assign_membership(rows,size,offset):
 out=[None]*size;seen=set()
 for row in rows:
  if row['role']!='fit':raise ValueError('nonfit membership')
  route=row['route'];group=group_of(route);i=row['array_index']
  if i is None:continue
  if type(i) is not int or not 0<=i<size or out[i] is not None or (route,row['time_ns']) in seen:raise ValueError('ambiguous fitting membership')
  seen.add((route,row['time_ns']));out[i]={'id':offset+i,'route':route,'group':group,'day':route.endswith('-Day'),'window_id':row['window_id'],'time_ns':row['time_ns']}
 if any(x is None for x in out):raise ValueError('fitting membership missing')
 return out
def load_data():
 with np.load(c.OLD/'prepared/fit/data.npz',allow_pickle=False) as z:day={k:z[k].copy() for k in ['X','y','median_feature']}
 with np.load(c.NIGHT/'night_fit.npz',allow_pickle=False) as z:night={k:z[k].copy() for k in day}
 windows=read(c.OLD/'inputs/fit/windows.json');routes={w['id']:w['source_route'] for w in windows};old=read(c.OLD/'prepared/fit/index.json');oldrows=[{**v,'route':routes[v['window_id']]} for v in old];newrows=read(c.NIGHT/'night_index.json');membership=assign_membership(oldrows,len(day['y']),0)+assign_membership(newrows,len(night['y']),len(day['y']));data={k:np.concatenate([day[k],night[k]]) for k in day}
 if len(membership)!=27178 or not all(np.isfinite(a).all() for a in data.values()):raise ValueError('fixed supported fitting population differs')
 groups=np.array([r['group'] for r in membership]);dayflag=np.array([r['day'] for r in membership],dtype=bool);support={}
 for route in ROUTES:
  source=[r for r in oldrows+newrows if r['route']==route];raw=sum(w['raw_radar_in_block'] for w in windows if w['source_route']==route) if route.endswith('-Day') else len(source);supported=sum(r['array_index'] is not None for r in source);support[route]={'raw_endpoints_in_fixed_windows':raw,'indexed_endpoints':len(source),'upstream_prefix_not_indexed':raw-len(source),'unsupported_or_unmatched_indexed':len(source)-supported,'supported_rows_used':supported,'support_fraction_of_raw':supported/raw}
 return data,membership,groups,dayflag,support
def calibrate(data,groups,dayflag,held):
 train,test=split(groups,held);dayflag=np.asarray(dayflag)
 if dayflag.shape!=groups.shape or dayflag.dtype.kind!='b':raise ValueError('invalid day membership')
 ids=np.flatnonzero((groups!=held)&dayflag)
 if not len(ids):raise ValueError('no training-day calibration population')
 y=np.asarray(data['y']);median=np.asarray(data['median_feature']);counts=np.bincount(np.digitize(y[ids],[1,5,10]),minlength=4)
 if np.any(counts==0):raise ValueError('training-day speed bin unsupported; no fold replacement')
 anchor=c.e.fit_affine(median[ids],y[ids],'fit')
 return {'held_group':held,'anchor':anchor,'bin_counts':counts.tolist(),'calibration_ids':ids.tolist(),'train_ids':train.tolist(),'held_ids':test.tolist()}
def weights(y,calibration):
 counts=np.array(calibration['bin_counts'])
 if counts.shape!=(4,) or np.any(counts<=0) or counts.sum()!=len(calibration['calibration_ids']):raise ValueError('invalid training-day bin counts')
 return (len(calibration['calibration_ids'])/4)/counts[np.digitize(y,[1,5,10])]
def fold_inputs(data,cal):
 ids=np.array(cal['train_ids']);anchor=c.e.affine(cal['anchor'],data['median_feature'][ids]);x=data['X'][ids];aug=c.m.augment(x,np.random.default_rng(20261019));a=c.features(x,anchor,data['median_feature'][ids]);b=c.features(aug,anchor,data['median_feature'][ids]);target=np.tile(data['y'][ids]-anchor,2);w=np.tile(weights(data['y'][ids],cal)*.5,2)
 return np.concatenate([a,b]),target,w,np.tile(anchor,2)
def freeze(create=False):
 p=read(ROOT/'protocol.json')
 if tuple(p['routes'])!=ROUTES or tuple(p['groups'])!=GROUPS:raise ValueError('fixed route protocol changed')
 for name,h in p['input_sha256'].items():
  path=(ROOT/name).resolve()
  if not path.is_relative_to(ROOT.parent) or digest(path)!=h:raise ValueError('route input changed: '+name)
 lock={'protocol_sha256':digest(ROOT/'protocol.json'),'code_sha256':{name:digest(ROOT/name) for name in ['route_core.py','train_routes.py','score_routes.py']}}
 path=ROOT/'analysis_lock.json'
 if path.exists() and read(path)!=lock:raise ValueError('route implementation changed after freeze')
 if create and not path.exists():write(path,lock)
 return p
