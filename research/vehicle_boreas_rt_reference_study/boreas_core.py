"""Local Doppler regressions. No reference input, security certificate or novelty claim."""
import hashlib
import numpy as np
DTYPE=np.dtype([("fields","<f4",(8,)),("flags","<u8")])
def decode(raw):
 if type(raw) is not bytes or not raw or len(raw)%40:raise ValueError("Empty/truncated Aeva record")
 records=np.frombuffer(raw,dtype=DTYPE);points=records["fields"].astype(np.float64)
 points[:,7]*=1e-9 # Explicit provisional unit inference documented in SCHEMA_ADDENDUM.md.
 return points,records["flags"].copy()
def fit(points,method):
 if method not in ("ols3","huber3","huber6time"):raise ValueError("Unknown method")
 p=np.asarray(points,dtype=np.float64)
 if p.ndim!=2 or p.shape[1]!=8:raise ValueError("Expected8 normalized fields")
 out={"status":"rejected","reason":"too_few_points","beta":None,"n_eligible":0,"n_used":0,"end_offset_s":None,"condition_number":None}
 finite=np.isfinite(p).all(axis=1)
 timed=np.isfinite(p[:,7])
 if not timed.any():return {**out,"reason":"unsupported_scan_timing"}
 end=float(np.max(p[timed,7]));start=float(np.min(p[timed,7]));out["end_offset_s"]=end
 if start<0 or end>.2:return {**out,"reason":"unsupported_scan_timing"}
 ranges=np.linalg.norm(p[:,:3],axis=1);eligible=np.flatnonzero(finite&(ranges>=3)&(ranges<=100));out["n_eligible"]=int(len(eligible))
 if len(eligible)<30:return out
 chosen=eligible[np.linspace(0,len(eligible)-1,min(10000,len(eligible)),dtype=np.int64)]
 rows=p[chosen];r=np.linalg.norm(rows[:,:3],axis=1);directions=rows[:,:3]/r[:,None]
 x=np.c_[directions,directions*((rows[:,7]-end)/.1)[:,None]] if method=="huber6time" else directions
 y=-rows[:,3];out["n_used"]=len(rows)
 singular=np.linalg.svd(x,compute_uv=False);rank=np.linalg.matrix_rank(x)
 condition=float(singular[0]/singular[-1]) if singular[-1]>0 else None;out["condition_number"]=condition
 if rank<x.shape[1] or (method=="huber6time" and (condition is None or condition>100000)):return {**out,"reason":"unobservable_design"}
 beta=np.linalg.lstsq(x,y,rcond=None)[0]
 if method!="ols3":
  for _ in range(20):
   residual=x@beta-y;weights=np.minimum(1.,.3/np.maximum(np.abs(residual),1e-12))
   root=np.sqrt(weights);beta=np.linalg.lstsq(x*root[:,None],y*root,rcond=None)[0]
 if not np.isfinite(beta).all():return {**out,"reason":"nonfinite_fit"}
 return {**out,"status":"estimated","reason":"fit","beta":beta.tolist()}
def predict(solution,age_us):
 if type(age_us) is not int or not 0<=age_us<=100000:raise ValueError("Unsupported query age")
 if solution["status"]!="estimated":return None
 beta=np.array(solution["beta"],dtype=np.float64)
 return beta[:3]+(beta[3:]*age_us/100000 if len(beta)==6 else 0)
def reference_speed(times_us,velocities,query_us):
 t=np.asarray(times_us);v=np.asarray(velocities)
 if type(query_us) is not int or t.ndim!=1 or v.shape!=(len(t),3) or not len(t) or np.any(np.diff(t)<=0):raise ValueError("Invalid reference/time")
 i=int(np.searchsorted(t,query_us))
 if i<len(t) and int(t[i])==query_us:return float(np.linalg.norm(v[i])) if np.isfinite(v[i]).all() else None
 if i==0 or i==len(t) or int(t[i])-int(t[i-1])>200000:return None
 alpha=(query_us-int(t[i-1]))/(int(t[i])-int(t[i-1]))
 vector=(1-alpha)*v[i-1]+alpha*v[i]
 return float(np.linalg.norm(vector)) if np.isfinite(vector).all() else None

def estimate_frame(raw,expected_sha256,method):
 rejected={"status":"rejected","reason":"missing_frame","beta":None,"n_eligible":0,"n_used":0,"end_offset_s":None,"condition_number":None}
 out={"raw_points":None,"flags_nonzero":None,"solution":rejected}
 if raw is None:return out
 if type(raw) is not bytes:raise ValueError("Expected immutable frame bytes")
 if expected_sha256 is None or hashlib.sha256(raw).hexdigest()!=expected_sha256:
  return {**out,"solution":{**rejected,"reason":"frame_hash_mismatch"}}
 try:points,flags=decode(raw)
 except ValueError:return {**out,"solution":{**rejected,"reason":"invalid_frame_schema"}}
 return {"raw_points":len(points),"flags_nonzero":int((flags!=0).sum()),"solution":fit(points,method)}
