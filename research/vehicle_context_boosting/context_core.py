"""Portable causal context summaries and numeric boosted trees, research use only."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parent;NIGHT=ROOT.parent/'vehicle_night_training';MASK=ROOT.parent/'vehicle_masked_history';OLD=ROOT.parent/'vehicle_radar_multidrive'
sys.path.insert(0,str(NIGHT))
import night_core as n
from night_core import read,write,digest,m,e,c
RESERVED='Urban02/2023-11-23-17-53-52-Night'
FIT_ROUTES=['Urban01/2023-11-23-15-38-43-Day','Urban03/2023-11-03-09-42-41-Day','Urban04/2023-11-01-12-16-16-Day',*n.ROUTES]
SPANS=(5,11,21,41,81,128)
STATS=('median','mean','trimmed_mean','q10','q90','mad','sd','range','ols_slope','half_median_slope','valid_fraction','mean_age')
FEATURE_NAMES=['current_signed_div10','anchor_mps','anchor_median_raw']+[f'w{w}_{s}' for w in SPANS for s in STATS]
def validate_fit_routes(routes):
 if list(routes)!=FIT_ROUTES:raise ValueError('fixed fit recordings changed or development/final leaked')
def features(x,anchor,median):
 x=np.asarray(x,dtype=float);anchor=np.asarray(anchor,dtype=float);median=np.asarray(median,dtype=float)
 if x.ndim!=3 or x.shape[1:]!=(4,128) or anchor.shape!=(len(x),) or median.shape!=(len(x),) or not np.isfinite(x).all() or not np.isfinite(anchor).all() or not np.isfinite(median).all() or np.any((x[:,3]!=0)&(x[:,3]!=1)):raise ValueError('invalid aligned context inputs')
 out=[x[:,0,-1],anchor,median]
 for width in SPANS:
  v=x[:,0,-width:]*10;ok=x[:,3,-width:]==1;count=ok.sum(1);safe=np.maximum(count,1);vv=np.where(ok,v,0.);mean=vv.sum(1)/safe;masked=np.where(ok,v,np.nan);masked[count==0]=0.
  q10,med,q90=np.nanquantile(masked,[.1,.5,.9],axis=1);mad=np.nanmedian(abs(masked-med[:,None]),axis=1);sd=np.sqrt((np.where(ok,(v-mean[:,None])**2,0.)).sum(1)/safe);ran=np.nanmax(masked,axis=1)-np.nanmin(masked,axis=1)
  sortedv=np.sort(np.where(ok,v,np.inf),axis=1);trim=(count*.1).astype(int);rank=np.arange(width)[None,:];take=(rank>=trim[:,None])&(rank<(count-trim)[:,None]);trimmed=np.where(take,sortedv,0.).sum(1)/np.maximum(take.sum(1),1)
  t=np.arange(-(width-1),1)*.05;tm=(ok*t).sum(1)/safe;center=t[None,:]-tm[:,None];den=np.where(ok,center**2,0.).sum(1);slope=np.divide(np.where(ok,center*(v-mean[:,None]),0.).sum(1),den,out=np.zeros(len(x)),where=den>0)
  half=width//2;parts=[];partgood=[]
  for values,valid in [(v[:,:half],ok[:,:half]),(v[:,half:],ok[:,half:])]:
   a=np.where(valid,values,np.nan);validrow=valid.any(1);a[~validrow]=0.;parts.append(np.nanmedian(a,axis=1));partgood.append(validrow)
  halfslope=np.where(partgood[0]&partgood[1],(parts[1]-parts[0])/(width*.025),0.);ages=np.where(ok,x[:,2,-width:],0.).sum(1)/safe;ages=np.where(count>0,ages,1.)
  out.extend([med,mean,trimmed,q10,q90,mad,sd,ran,slope,halfslope,count/width,ages])
 result=np.column_stack(out)
 if not np.isfinite(result).all():raise ValueError('nonfinite context summaries')
 return result

def predict(model,x,anchor):
 x=np.asarray(x,dtype=float);anchor=np.asarray(anchor,dtype=float)
 if x.ndim!=2 or x.shape[1]!=len(model['feature_names']) or anchor.shape!=(len(x),) or not np.isfinite(x).all() or not np.isfinite(anchor).all():raise ValueError('invalid boosted-tree inputs')
 result=np.full(len(x),float(model['baseline']))
 for nodes in model['trees']:
  leaf=np.array([v['leaf'] for v in nodes],dtype=bool);value=np.array([v.get('value',0.) for v in nodes]);feature=np.array([v.get('feature',0) for v in nodes],dtype=int);threshold=np.array([v.get('threshold',0.) for v in nodes]);left=np.array([v.get('left',0) for v in nodes],dtype=int);right=np.array([v.get('right',0) for v in nodes],dtype=int);at=np.zeros(len(x),dtype=int)
  for step in range(len(nodes)+1):
   if np.any(at<0) or np.any(at>=len(nodes)):raise ValueError('invalid tree child')
   active=~leaf[at]
   if not active.any():break
   rows=np.flatnonzero(active);j=at[rows]
   if np.any(feature[j]<0) or np.any(feature[j]>=x.shape[1]):raise ValueError('invalid tree feature')
   at[rows]=np.where(x[rows,feature[j]]<=threshold[j],left[j],right[j])
  else:raise ValueError('cyclic tree')
  result+=value[at]
 return np.maximum(0.,anchor+result)
def quality(prediction,peer,source_quality):
 p=np.asarray(prediction);q=np.asarray(peer);keep=np.asarray(source_quality)
 if p.ndim!=1 or q.shape!=p.shape or keep.shape!=p.shape or keep.dtype.kind!='b':raise ValueError('unaligned quality inputs')
 return keep&np.isfinite(p)&np.isfinite(q)&(p>=0)&(q>=0)&(abs(p-q)<=1.)
def fitting_data():
 with np.load(OLD/'prepared/fit/data.npz',allow_pickle=False) as z:day={k:z[k].copy() for k in ['X','y','median_feature']}
 with np.load(NIGHT/'night_fit.npz',allow_pickle=False) as z:night={k:z[k].copy() for k in day}
 return {k:np.concatenate([day[k],night[k]]) for k in day},day['y']
def freeze(create=False):
 p=read(ROOT/'protocol.json');validate_fit_routes(p['fit_routes'])
 for name,h in p['input_sha256'].items():
  path=(ROOT/name).resolve()
  if not path.is_relative_to(ROOT.parent) or digest(path)!=h:raise ValueError('context source changed: '+name)
 lock={'protocol_sha256':digest(ROOT/'protocol.json'),'code_sha256':{name:digest(ROOT/name) for name in ['context_core.py','train_context.py','score_context.py']}}
 path=ROOT/'analysis_lock.json'
 if path.exists() and read(path)!=lock:raise ValueError('context implementation changed after freeze')
 if create and not path.exists():write(path,lock)
 return p
