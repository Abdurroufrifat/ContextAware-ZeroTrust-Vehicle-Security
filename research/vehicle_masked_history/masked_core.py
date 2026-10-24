"""Causal explicit missing-history features and portable model inference."""
from pathlib import Path
import hashlib,json,sys,math
import numpy as np
ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'vehicle_radar_multidrive'
PREVIOUS=ROOT.parent/'vehicle_continuous_radar'
sys.path.insert(0,str(PREVIOUS))
import continuous_core as c
import estimator_core as e
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,value):p.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
class MaskedGrid(e.RadarGrid):
 def at_masked(self,t,start):
  if type(t) is not int or type(start) is not int or t<start or not len(self.times):return None
  ticks=np.int64(t)-np.arange(127,-1,-1,dtype=np.int64)*50000000;j=np.searchsorted(self.times,ticks,side='right')-1;safe=np.maximum(j,0);ages=ticks-self.times[safe]
  valid=(j>=0)&(ticks>=start)&(self.times[safe]>=start)&(ages>=0)&(ages<=300000000)
  if not valid[-1] or self.times[safe[-1]]!=t or int(valid.sum())<16:return None
  lo=np.searchsorted(self.times,max(start,t-1000000000),side='left');hi=np.searchsorted(self.times,t,side='right')
  if hi-lo<10:return None
  values=np.where(valid,self.values[safe],0.);age=np.where(valid,ages/300000000,1.)
  x=np.stack((values/10,np.abs(values)/10,age,valid)).astype(np.float32)
  return x,abs(float(np.median(self.values[lo:hi])))
def features(rows,t,start):return MaskedGrid(rows).at_masked(t,start)
def augment(x,generator):
 out=np.array(x,copy=True)
 for i in range(len(out)):
  mode=int(generator.integers(0,4))
  if mode==0:continue
  if mode==1:a,b=0,int(generator.integers(1,113))
  else:
   length=int(generator.choice([4,8,16,32]));a=int(generator.integers(0,128-length));b=a+length
  out[i,:2,a:b]=0;out[i,2,a:b]=1;out[i,3,a:b]=0
 return out
def neural(model,x,anchor):
 if model['kind']=='causal_residual_tcn':return e.neural(model,x,anchor)
 if model['kind']!='masked_cnn_4ch':raise ValueError('unknown masked model kind')
 s={k:np.asarray(v,dtype=np.float64) for k,v in model['state'].items()};h=np.asarray(x,dtype=np.float64)
 for i in range(6):
  h=e.conv(h,s[f'layers.{i}.weight'],s[f'layers.{i}.bias'],3,3);p=f'norms.{i}.';h=(h-s[p+'running_mean'][None,:,None])/np.sqrt(s[p+'running_var'][None,:,None]+1e-5);h=h*s[p+'weight'][None,:,None]+s[p+'bias'][None,:,None];h=.5*h*(1+np.tanh(math.sqrt(2/math.pi)*(h+.044715*h*h*h)))
 return np.maximum(0,10*h[:,0,-1])
def ensemble(models,x,anchor):
 vectors=[np.concatenate([neural(model,x[i:i+128],anchor[i:i+128]) for i in range(0,len(x),128)]) for model in models]
 return np.mean(vectors,axis=0),vectors
def freeze(create=False):
 protocol=read(ROOT/'protocol.json')
 for name,h in protocol['input_sha256'].items():
  p=(ROOT/name).resolve()
  if not p.is_relative_to(ROOT.parent) or digest(p)!=h:raise ValueError('masked experiment source changed: '+name)
 lock={'protocol_sha256':digest(ROOT/'protocol.json'),'code_sha256':{n:digest(ROOT/n) for n in ['masked_core.py','models_torch.py','train_masked.py','score_masked.py']}}
 path=ROOT/'analysis_lock.json'
 if path.exists() and read(path)!=lock:raise ValueError('masked experiment implementation changed')
 if create and not path.exists():write(path,lock)
 return protocol
