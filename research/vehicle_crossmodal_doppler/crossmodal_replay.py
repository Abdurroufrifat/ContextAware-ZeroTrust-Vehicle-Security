"""Causal development replay helpers; no reference access."""
import numpy as np
from crossmodal_core import agree
def latest_completed(completion,queries):
 c=np.asarray(completion,dtype=np.int64);q=np.asarray(queries,dtype=np.int64)
 if np.any(np.diff(c)<0):raise ValueError("completion must be ordered")
 return np.searchsorted(c,q,side="right")-1
def make_cases(queries,radar,optical,pair,radar_age,optical_age):
 q=np.asarray(queries,dtype=np.int64);r=np.asarray(radar,float);o=np.asarray(optical,float);pair=np.asarray(pair,bool)
 ra=np.asarray(radar_age,dtype=np.int64);oa=np.asarray(optical_age,dtype=np.int64)
 out={}
 def add(name,rv,ov,rage,oage,support,oq=True):
  accepted=np.array([bool(support[i]) and agree(rv[i],ov[i],rage[i],oage[i],True,oq) for i in range(len(q))])
  out[name]={"supported":np.asarray(support,bool),"accepted":accepted,"prediction":np.where(accepted,rv,np.nan)}
 add("nominal",r,o,ra,oa,pair)
 add("radar_plus3",r+3,o,ra,oa,pair)
 add("optical_plus3",r,o+3,ra,oa,pair)
 add("both_plus3",r+3,o+3,ra,oa,pair)
 j=np.searchsorted(q,q-1_000_000_000,side="right")-1;k=np.maximum(j,0)
 support=pair&(j>=0)&pair[k]
 lag=q-q[k]
 add("radar_replay_honest",r[k],o,ra[k]+lag,oa,support)
 add("radar_replay_forged",r[k],o,np.zeros(len(q),dtype=np.int64),oa,support)
 add("optical_replay_honest",r,o[k],ra,oa[k]+lag,support)
 add("optical_replay_forged",r,o[k],ra,np.zeros(len(q),dtype=np.int64),support)
 add("missing_optical",r,o,ra,oa,pair,False)
 add("stale_optical",r,o,ra,np.full(len(q),100_000_001,dtype=np.int64),pair)
 return out
def metrics(pred,reference):
 p=np.asarray(pred,float);y=np.asarray(reference,float);valid=np.isfinite(p)&np.isfinite(y);e=abs(p[valid]-y[valid])
 return {"predictions":int(np.isfinite(p).sum()),"scored":int(valid.sum()),"unknown_reference":int(np.sum(np.isfinite(p)&~np.isfinite(y))),"p95_mps":float(np.quantile(e,.95)) if len(e) else None,"max_mps":float(max(e)) if len(e) else None,"above2":int(sum(e>2))}
def interpolate(queries,times,values):
 q=np.asarray(queries,dtype=np.int64);t=np.asarray(times,dtype=np.int64);v=np.asarray(values,float)
 j=np.searchsorted(t,q);safe=np.minimum(j,len(t)-1);out=np.full(len(q),np.nan)
 exact=(j<len(t))&(q==t[safe]);out[exact]=v[safe[exact]]
 valid=(j>0)&(j<len(t))&~exact
 ix=np.flatnonzero(valid);lo=j[ix]-1;hi=j[ix]
 ok=(t[hi]-t[lo]<=200_000_000)&np.isfinite(v[lo])&np.isfinite(v[hi]);ix=ix[ok];lo=lo[ok];hi=hi[ok]
 out[ix]=v[lo]+(v[hi]-v[lo])*(q[ix]-t[lo])/(t[hi]-t[lo])
 return out
