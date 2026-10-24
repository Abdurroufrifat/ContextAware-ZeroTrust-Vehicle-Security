import numpy as np
def additive_forgery(nominal_delta,bias,past_index):
    d=np.asarray(nominal_delta,float);b=np.asarray(bias,float);j=np.asarray(past_index,np.int64)
    if d.shape!=b.shape or b.ndim!=2 or b.shape[1]!=2 or j.shape!=(len(b),):raise ValueError("unmatched forgery arrays")
    safe=np.clip(j,0,len(b)-1)
    return d+b-b[safe]
def explain_frames(stamps,past,quality,supported,accepted):
    t=np.asarray(stamps,np.int64);j=np.asarray(past,np.int64);q=np.asarray(quality,bool);s=np.asarray(supported,bool);a=np.asarray(accepted,bool)
    if any(x.shape!=t.shape for x in [j,q,s,a]) or np.any(a&~s):raise ValueError("inconsistent support masks")
    reasons=np.full(len(t),"accepted",dtype="U32")
    for i,previous in enumerate(j):
        if previous<0:reasons[i]="missing_history"
        elif previous>=len(t):raise ValueError("invalid past index")
        elif not q[i] or not q[previous]:reasons[i]="radar_quality"
        elif t[i]-t[previous]>1_100_000_000:reasons[i]="long_interval"
        elif not s[i]:reasons[i]="stale_imu"
        elif not a[i]:reasons[i]="innovation_exceeds_1"
    if np.any((reasons=="accepted")!=a):raise ValueError("inconsistent reason assignment")
    return reasons
