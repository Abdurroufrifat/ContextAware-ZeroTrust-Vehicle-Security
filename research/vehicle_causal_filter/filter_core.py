"""Resettable causal Kalman baseline; covariance is not a physical error bound."""
from pathlib import Path
import hashlib,json,sys,math
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'vehicle_masked_history'))
import masked_core as m
sys.path.insert(0,str(ROOT.parent/'vehicle_availability_audit'))
from availability_core import serve,request_times
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def fit_noise(y,p,role):
 if role!='fit':raise ValueError('only fitting data may estimate noise')
 y=np.asarray(y);p=np.asarray(p)
 if y.ndim!=1 or p.shape!=y.shape or len(y)<2 or not np.all(np.isfinite(y)) or not np.all(np.isfinite(p)):raise ValueError('invalid noise fitting arrays')
 error=p-y;center=float(np.median(error));sigma=float(1.4826*np.median(abs(error-center)))
 return {'fit_samples':len(y),'median_residual_mps':center,'robust_sigma_mps':sigma,'variance_mps2':max(.0001,sigma*sigma),'role':'fit','source':'Unchanged original TCN fit residuals; no development labels','bias_subtracted_from_outputs':False}
def filter_events(times,z,source_quality,r,q,reset_gap_ns,peer=None):
 times=np.asarray(times);z=np.asarray(z);source_quality=np.asarray(source_quality)
 if times.ndim!=1 or z.shape!=times.shape or source_quality.shape!=times.shape or source_quality.dtype.kind!='b' or times.dtype.kind not in 'iu' or np.any(np.diff(times)<=0):raise ValueError('unaligned or ambiguous filter events')
 if not all(isinstance(v,(float,int)) and math.isfinite(v) and v>0 for v in (r,q)) or type(reset_gap_ns) is not int or reset_gap_ns<=0:raise ValueError('invalid filter covariances or reset age')
 if np.any(~np.isfinite(z[source_quality])) or np.any(z[source_quality]<0):raise ValueError('invalid accepted source estimate')
 if peer is not None:
  peer=np.asarray(peer)
  if peer.shape!=z.shape or np.any(~np.isfinite(peer[source_quality])) or np.any(peer[source_quality]<0):raise ValueError('invalid peer estimates')
 n=len(times);est=np.full(n,np.nan);acc=np.full(n,np.nan);cov=np.full((n,2,2),np.nan);reset=np.zeros(n,dtype=bool);quality=np.zeros(n,dtype=bool);state=P=None;last=None
 for i,tick in enumerate(times):
  t=int(tick)
  if not source_quality[i]:state=P=last=None;continue
  if state is None or t-last>reset_gap_ns:
   state=np.array([float(z[i]),0.]);P=np.diag([r,1.]);reset[i]=True
  else:
   dt=(t-last)/1e9;F=np.array([[1.,dt],[0.,1.]]);Q=q*np.array([[dt**3/3,dt**2/2],[dt**2/2,dt]]);state=F@state;P=F@P@F.T+Q;K=P[:,0]/(P[0,0]+r);state=state+K*(z[i]-state[0]);A=np.eye(2)-np.outer(K,[1.,0.]);P=A@P@A.T+r*np.outer(K,K);P=(P+P.T)/2
  est[i]=max(0.,float(state[0]));acc[i]=state[1];cov[i]=P;quality[i]=peer is None or abs(est[i]-peer[i])<=1.;last=t
  if not quality[i]:state=P=last=None
 return {'estimate_mps':est,'acceleration_mps2':acc,'covariance':cov,'reset':reset,'quality':quality}
def freeze(create=False):
 p=read(ROOT/'protocol.json')
 for name,h in p['input_sha256'].items():
  f=(ROOT/name).resolve()
  if not f.is_relative_to(ROOT.parent) or digest(f)!=h:raise ValueError('filter input changed: '+name)
 lock={'protocol_sha256':digest(ROOT/'protocol.json'),'code_sha256':{n:digest(ROOT/n) for n in ['filter_core.py','run_filter.py']}}
 dest=ROOT/'analysis_lock.json'
 if dest.exists() and read(dest)!=lock:raise ValueError('filter analysis implementation changed')
 if create and not dest.exists():write(dest,lock)
 return p
