import numpy as np
def metrics(pred,ref,admitted=None):
 p=np.asarray(pred,float);y=np.asarray(ref,float);ask=np.isfinite(p) if admitted is None else np.asarray(admitted,bool)
 if p.shape!=y.shape or ask.shape!=p.shape:raise ValueError("unmatched rows")
 keep=ask&np.isfinite(p);scored=keep&np.isfinite(y);e=abs(p[scored]-y[scored])
 return {"attempts":len(p),"admitted":int(keep.sum()),"invalid_prediction":int((ask&~np.isfinite(p)).sum()),"availability":float(keep.mean()) if len(p) else None,"scored":int(scored.sum()),"unknown_reference":int((keep&~np.isfinite(y)).sum()),"mae_mps":float(e.mean()) if len(e) else None,"p95_mps":float(np.quantile(e,.95)) if len(e) else None,"max_mps":float(max(e)) if len(e) else None,"above2":int(sum(e>2))}
def paired_mae_ci(a,b,block,draws,seed):
 a=np.asarray(a,float);b=np.asarray(b,float)
 if a.shape!=b.shape or a.ndim!=1 or not len(a) or not np.all(np.isfinite(a)&np.isfinite(b)) or block<1 or draws<1:raise ValueError("invalid paired bootstrap")
 delta=a-b;n=len(delta);rng=np.random.default_rng(seed);start=rng.integers(0,n,size=(draws,(n+block-1)//block))
 ix=((start[:,:,None]+np.arange(block))%n).reshape(draws,-1)[:,:n];means=delta[ix].mean(axis=1)
 low,high=np.quantile(means,[.025,.975])
 return {"difference":float(delta.mean()),"low":float(low),"high":float(high),"draws":draws,"block":block,"seed":seed,"common_rows":n}
def decisions(r,o,ra,oa,support,optical_present):
 r=np.asarray(r);o=np.asarray(o);support=np.asarray(support,bool)
 rok=support&np.isfinite(r)&(ra>=0)&(ra<=100_000_000)
 ook=support&optical_present&np.isfinite(o)&(oa>=0)&(oa<=100_000_000)
 return {"radar_only":rok,"optical_only":ook,"agreement_guard":rok&ook&(abs(r-o)<=1)}
