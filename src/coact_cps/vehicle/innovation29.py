"""Causal signed sensor innovations and fixed normal-only score geometry."""
import numpy as np

HISTORY=5
MAX_INTERVAL_MS=500
ARMS=('innovation_max','innovation_covariance')


def innovations(features,time_ms,start_ms):
    x=np.asarray(features,dtype=float);t,s=np.asarray(time_ms),np.asarray(start_ms)
    if (x.ndim!=2 or x.shape[1]!=3 or len(x)==0 or not np.isfinite(x).all()
        or t.shape!=(len(x),) or s.shape!=t.shape or t.dtype.kind not in 'iu' or s.dtype.kind not in 'iu'):
        raise ValueError('finite three-channel features and integer time vectors required')
    t,s=t.astype(np.int64),s.astype(np.int64);dt=t-s
    if np.any(dt<=0) or np.any(dt>MAX_INTERVAL_MS) or np.any(np.diff(t)<=0) or np.any(s[1:]<t[:-1]):
        raise ValueError('ordered nonoverlapping intervals of at most 500ms required')
    result=np.full(x.shape,np.nan)
    for i in range(HISTORY,len(x)):
        first=i-HISTORY
        if not np.array_equal(s[first+1:i+1],t[first:i]):continue
        duration=(int(t[i])-int(t[first]))/1000
        integral=np.sum(x[first+1:i+1,2]*dt[first+1:i+1]/1000)
        result[i]=[x[i,0]-x[i,1],(x[i,0]-x[first,0]-integral)/duration,
                   (x[i,1]-x[first,1]-integral)/duration]
    return result


def fit_geometry(values,labels):
    x,y=np.asarray(values,dtype=float),np.asarray(labels)
    if (x.ndim!=2 or x.shape[1]!=3 or len(x)<2 or not np.isfinite(x).all()
        or y.shape!=(len(x),) or y.dtype.kind!='b' or y.any()):
        raise ValueError('finite exclusively presumed-normal innovation rows required')
    median=np.median(x,axis=0)
    scale=np.maximum(np.quantile(x,.75,axis=0)-np.quantile(x,.25,axis=0),.01)
    z=(x-median)/scale;covariance=.9*(z.T@z/len(z))+.1*np.eye(3)
    return dict(median=median.tolist(),scale=scale.tolist(),covariance=covariance.tolist())


def score(values,profile,arm):
    x=np.asarray(values,dtype=float)
    median,scale,covariance=[np.asarray(profile[k],dtype=float) for k in ('median','scale','covariance')]
    if (arm not in ARMS or x.ndim!=2 or x.shape[1]!=3 or not np.isfinite(x).all()
        or median.shape!=(3,) or scale.shape!=(3,) or covariance.shape!=(3,3)
        or not all(np.isfinite(v).all() for v in (median,scale,covariance))
        or np.any(scale<=0) or not np.allclose(covariance,covariance.T,rtol=0,atol=1e-12)):
        raise ValueError('finite available innovations and valid fixed score geometry required')
    try:chol=np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:raise ValueError('positive definite score geometry required') from exc
    z=(x-median)/scale
    if arm=='innovation_max':return np.abs(z).max(axis=1)
    whitened=np.linalg.solve(chol,z.T).T
    return np.sqrt(np.sum(whitened**2,axis=1))
