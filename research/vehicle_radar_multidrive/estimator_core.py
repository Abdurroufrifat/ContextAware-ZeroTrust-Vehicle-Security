"""Radar-only causal features and portable NumPy inference; no permission output."""
import math
import numpy as np
class RadarGrid:
 def __init__(self,rows):
  times=[r['bag_time_ns'] for r in rows]
  if any(type(t) is not int for t in times) or any(a>=b for a,b in zip(times,times[1:])):raise ValueError('ambiguous radar clock')
  good=[r for r in rows if math.isfinite(r['speed_raw']) and 0<=r['bag_time_ns']-r['header_time_ns']<=250000000]
  self.times=np.asarray([r['bag_time_ns'] for r in good],dtype=np.int64);self.values=np.asarray([r['speed_raw'] for r in good],dtype=np.float64)
 def at(self,t,start):
  if type(t) is not int or type(start) is not int or t-start<6350000000 or not len(self.times):return None
  ticks=np.int64(t)-np.arange(127,-1,-1,dtype=np.int64)*50000000;j=np.searchsorted(self.times,ticks,side='right')-1
  if np.any(j<0):return None
  age=ticks-self.times[j]
  if self.times[j[-1]]!=t or np.any(self.times[j]<start) or np.any(age>300000000):return None
  values=self.values[j];return np.stack((values/10,np.abs(values)/10,age/300000000,np.ones(128))).astype(np.float32)
def grid_features(rows,t,start):return RadarGrid(rows).at(t,start)
def sg_coefficients():
 t=np.arange(-50,1,dtype=np.float64)/50;design=np.stack((np.ones(51),t,t*t),axis=1);return np.linalg.pinv(design)[0]
def balanced_weights(y):
 bins=np.digitize(y,[1,5,10]);counts=np.bincount(bins,minlength=4);weights=1/counts[bins];return weights/weights.mean()
def fit_affine(x,y,role):
 if role!='fit':raise ValueError('only fitting labels may calibrate an estimator')
 if len(x)<2 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)) or np.ptp(y)<3:raise ValueError('insufficient fitting support')
 w=balanced_weights(y);mx=np.average(x,weights=w);my=np.average(y,weights=w);den=np.sum(w*(x-mx)**2)
 if den<=1e-12:raise ValueError('degenerate fitting feature')
 slope=float(np.sum(w*(x-mx)*(y-my))/den)
 if slope<=0:raise ValueError('nonphysical fitted slope')
 return {'slope':slope,'intercept':float(my-slope*mx)}
def affine(model,x):return np.maximum(0,model['slope']*x+model['intercept'])
def conv(x,weight,bias,left,right=0,dilation=1):
 x=np.pad(x,((0,0),(0,0),(left,right)));width=(weight.shape[2]-1)*dilation+1;windows=np.lib.stride_tricks.sliding_window_view(x,width,axis=2)[...,::dilation]
 return np.einsum('bctk,ock->bot',windows,weight,optimize=True)+bias[None,:,None]
def neural(model,x,anchor):
 s={k:np.asarray(v,dtype=np.float64) for k,v in model['state'].items()};x=np.asarray(x,dtype=np.float64)
 if model['kind']=='causal_residual_tcn':
  h=np.maximum(0,conv(x,s['stem.weight'],s['stem.bias'],0))
  for i,d in enumerate((1,2,4,8,16)):
   z=np.maximum(0,conv(h,s[f'layers.{i}.weight'],s[f'layers.{i}.bias'],4*d,dilation=d));h=(h+z)/math.sqrt(2)
  out=anchor+h[:,:,-1]@s['head.weight'][0]+s['head.bias'][0]
 elif model['kind']=='paper_inspired_cnn':
  h=x[:,:1,:]
  for i in range(6):
   h=conv(h,s[f'layers.{i}.weight'],s[f'layers.{i}.bias'],3,3);prefix=f'norms.{i}.'
   h=(h-s[prefix+'running_mean'][None,:,None])/np.sqrt(s[prefix+'running_var'][None,:,None]+1e-5);h=h*s[prefix+'weight'][None,:,None]+s[prefix+'bias'][None,:,None]
   h=.5*h*(1+np.tanh(math.sqrt(2/math.pi)*(h+.044715*h*h*h)))
  out=10*h[:,0,-1]
 else:raise ValueError('unknown model kind')
 return np.maximum(0,out)
def ensemble(models,x,anchor,batch=128):
 predictions=[]
 for model in models:
  predictions.append(np.concatenate([neural(model,x[i:i+batch],anchor[i:i+batch]) for i in range(0,len(x),batch)]))
 return np.mean(predictions,axis=0),predictions
