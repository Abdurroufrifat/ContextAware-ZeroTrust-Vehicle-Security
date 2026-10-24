"""Causal differential-motion diagnostic; no reference-speed inputs."""
import numpy as np,time
from scipy.spatial.transform import Rotation
def rotations(q):
    q=np.asarray(q,float)
    if q.ndim!=2 or q.shape[1]!=4 or not np.isfinite(q).all() or np.any(abs(np.linalg.norm(q,axis=1)-1)>.02):raise ValueError("invalid xyzw quaternion")
    return Rotation.from_quat(q)
def acceleration_world_xy(q,acceleration):
    a=np.asarray(acceleration,float);r=rotations(q)
    if a.shape!=(len(np.asarray(q)),3) or not np.isfinite(a).all():raise ValueError("invalid acceleration")
    return r.apply(a)[:,:2]
def imu_origin_velocity(velocity_xy,q,gyro,lever):
    v=np.asarray(velocity_xy,float);g=np.asarray(gyro,float);l=np.asarray(lever,float);r=rotations(q)
    if v.shape!=(len(np.asarray(q)),2) or g.shape!=(len(v),3) or l.shape!=(3,) or not all(np.isfinite(x).all() for x in [v,g,l]):raise ValueError("invalid origin correction")
    return v-r.apply(np.cross(g,l))[:,:2]
def validate_imu(times,acceleration):
    t=np.asarray(times,np.int64);a=np.asarray(acceleration,float)
    if t.ndim!=1 or len(t)<1 or np.any(np.diff(t)<=0) or a.shape!=(len(t),2) or not np.isfinite(a).all():raise ValueError("invalid IMU series")
    return t,a
def _integrate(t,a,start,end,max_age):
    if end<=start:raise ValueError("interval must advance")
    first=int(np.searchsorted(t,start,side="right"))-1;last=int(np.searchsorted(t,end,side="right"))-1
    if first<0 or last<0:return {"supported":False,"delta":None,"reason":"missing_history"}
    knots=np.r_[np.int64(start),t[(t>start)&(t<end)],np.int64(end)]
    indices=np.searchsorted(t,knots[:-1],side="right")-1
    if end-int(t[last])>max_age or np.any(knots[1:]-t[indices]>max_age):
        return {"supported":False,"delta":None,"reason":"stale_imu"}
    delta=(a[indices]*(np.diff(knots)*1e-9)[:,None]).sum(axis=0)
    return {"supported":True,"delta":delta,"reason":"ready"}
def integrate_causal(times,acceleration,start,end,max_age=100_000_000):
    t,a=validate_imu(times,acceleration)
    return _integrate(t,a,int(start),int(end),int(max_age))
def compare_changes(velocity,past_index,delta_imu,supported):
    v=np.asarray(velocity,float);j=np.asarray(past_index,np.int64);d=np.asarray(delta_imu,float);ok=np.asarray(supported,bool)
    if v.ndim!=2 or v.shape[1]!=2 or d.shape!=v.shape or j.shape!=(len(v),) or ok.shape!=(len(v),):raise ValueError("unmatched intervals")
    ok=ok&(j>=0)&(j<len(v));k=np.clip(j,0,max(0,len(v)-1))
    ok=ok&np.isfinite(v).all(axis=1)&np.isfinite(v[k]).all(axis=1)&np.isfinite(d).all(axis=1)
    residual=np.full(len(v),np.nan);residual[ok]=np.linalg.norm(v[ok]-v[k[ok]]-d[ok],axis=1)
    return {"supported":ok,"residual":residual,"accepted":ok&(residual<=1.)}
def innovations(velocity,stamps,imu_times,acceleration,quality):
    v=np.asarray(velocity,float);t=np.asarray(stamps,np.int64);good=np.asarray(quality,bool);mt,a=validate_imu(imu_times,acceleration)
    if v.shape!=(len(t),2) or good.shape!=(len(t),) or np.any(np.diff(t)<=0):raise ValueError("invalid radar series")
    j=np.searchsorted(t,t-1_000_000_000,side="right")-1
    support=np.zeros(len(t),bool);delta=np.full((len(t),2),np.nan);elapsed=np.zeros(len(t),np.int64)
    for i,previous in enumerate(j):
        start=time.perf_counter_ns()
        if previous>=0 and good[i] and good[previous] and t[i]-t[previous]<=1_100_000_000:
            row=_integrate(mt,a,int(t[previous]),int(t[i]),100_000_000)
            if row["supported"]:support[i]=True;delta[i]=row["delta"]
        elapsed[i]=time.perf_counter_ns()-start
    out=compare_changes(v,j,delta,support)
    out.update(past_index=j,delta_imu=delta,check_ns=elapsed);return out
