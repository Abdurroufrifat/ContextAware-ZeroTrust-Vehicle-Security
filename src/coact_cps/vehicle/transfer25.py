"""Frozen preprocessing order and descriptive presumed-normal alarm metrics."""

import numpy as np


def scaled_windows(features, time, start, window, scaler):
    f,t,s = map(np.asarray,(features,time,start))
    mean,scale = np.asarray(scaler['mean']),np.asarray(scaler['scale'])
    if (f.ndim != 2 or f.shape[1] != 3 or not np.isfinite(f).all()
            or t.shape != (len(f),) or s.shape != t.shape
            or t.dtype.kind not in 'iu' or s.dtype.kind not in 'iu'
            or type(window) is not int or not 2 <= window <= 256
            or mean.shape != (3,) or scale.shape != (3,)
            or not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale<=0)
            or np.any(t<=s) or np.any(t-s>2000) or np.any(np.diff(t)<=0)):
        raise ValueError('finite aligned features, timestamps and positive frozen scaler required')
    positions=[];xs=[];begin=0
    for i in range(len(f)):
        if i and s[i]!=t[i-1]:begin=i
        if i-begin+1>=window:
            positions.append(i);xs.append(f[i-window+1:i+1])
    # Match original vehicle_ai_data13.windows -> Scaling.transform ordering.
    x=np.asarray(xs,dtype=np.float32).reshape(-1,window,3)
    return ((x.astype(float)-mean)/scale).astype(np.float32),np.asarray(positions,dtype=np.int64)


def normal_metrics(nodes,calibration,threshold,positions,time,start):
    nodes,p,t,s=map(np.asarray,(nodes,positions,time,start))
    med,scale=np.asarray(calibration['median']),np.asarray(calibration['scale'])
    if (nodes.shape!=(len(p),3) or not len(p) or not np.isfinite(nodes).all()
            or p.ndim!=1 or p.dtype.kind not in 'iu' or np.any(p<0) or np.any(p>=len(t))
            or np.any(np.diff(p.astype(np.int64))<=0) or t.shape!=s.shape or t.ndim!=1
            or med.shape!=(3,) or scale.shape!=(3,) or np.any(scale<=0)
            or not np.isfinite(med).all() or not np.isfinite(scale).all()
            or not np.isfinite(threshold)):
        raise ValueError('finite scored nodes, frozen normalization and ordered positions required')
    z=(nodes-med)/scale;score=z.max(axis=1);alarm=score>threshold
    longest=current=0
    for j,i in enumerate(p):
        continuous=j>0 and i==p[j-1]+1 and s[i]==t[p[j-1]]
        current=(current+1 if continuous else 1) if alarm[j] else 0
        longest=max(longest,current)
    return dict(epochs=len(t),scored_endpoints=len(p),unscored_epochs=len(t)-len(p),
                alarm_endpoints=int(alarm.sum()),presumed_normal_alarm_fraction=float(alarm.mean()),
                channel_alarm_fractions=(z>threshold).mean(axis=0).tolist(),
                longest_alarm_run_epochs=longest),score,alarm
