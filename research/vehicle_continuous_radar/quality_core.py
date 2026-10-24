"""Research quality flag only. Model agreement cannot authorize a request."""
import hashlib,math,numbers
import numpy as np
def usable(tcn,cnn,ready):
 valid=lambda x:isinstance(x,numbers.Real) and not isinstance(x,(bool,np.bool_)) and math.isfinite(x) and x>=0
 return ready is True and valid(tcn) and valid(cnn) and abs(tcn-cnn)<=1.
def matched_random(keep,seed):
 keep=np.asarray(keep)
 if keep.ndim!=1 or keep.dtype.kind!='b':raise ValueError('expected a boolean vector')
 order=sorted(range(len(keep)),key=lambda i:hashlib.sha256(f'{seed}:{i}'.encode('ascii')).digest());result=np.zeros(len(keep),dtype=bool);result[order[:int(keep.sum())]]=True
 return result
def error_metrics(y,p):
 if not len(y):return {'count':0,'mae_mps':None,'rmse_mps':None,'p95_mps':None,'maximum_mps':None,'above_2_mps':0}
 e=abs(p-y);return {'count':len(e),'mae_mps':float(e.mean()),'rmse_mps':float(np.sqrt(np.mean(e**2))),'p95_mps':float(np.quantile(e,.95)),'maximum_mps':float(e.max()),'above_2_mps':int((e>2).sum())}
def metrics(y,p,keep):
 y=np.asarray(y);p=np.asarray(p);keep=np.asarray(keep)
 if y.ndim!=1 or p.shape!=y.shape or keep.shape!=y.shape or keep.dtype.kind!='b' or not np.all(np.isfinite(y)) or not np.all(np.isfinite(p)):raise ValueError('unaligned/nonfinite quality inputs')
 return {'all_supported':error_metrics(y,p),'accepted':error_metrics(y[keep],p[keep]),'withheld':error_metrics(y[~keep],p[~keep]),'supported_acceptance_fraction':float(keep.mean()) if len(keep) else 0.}
