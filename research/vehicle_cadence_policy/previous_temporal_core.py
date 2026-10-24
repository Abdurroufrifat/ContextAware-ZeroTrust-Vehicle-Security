"""Experimental temporal comparator. Not a production certificate verifier."""
import collections,math,statistics
from final_core import speed_bin

class TemporalPolicy:
 def __init__(self,params):
  self._centers=tuple(float(x) for x in params['centers']);self._k=float(params['allowance_mps']);self._h=float(params['cusum_limit']);self._hard=float(params['hard_limit'])
  if len(self._centers)!=4 or not all(math.isfinite(x) for x in self._centers+(self._k,self._h,self._hard)) or self._k<0 or self._h<=0 or self._hard<=1:raise ValueError('invalid temporal parameters')
  self.latched=False;self.admitted=False;self.confirmed_now=False;self._first=None;self._last=None;self._count=0;self._pending=False;self._reset_epoch=True;self._plus=0.;self._minus=0.
 def _quarantine(self):
  self.admitted=False;self._first=None;self._count=0;self._pending=False;self._reset_epoch=True
 def step(self,t,gnss,wheel,pred,valid=True):
  self.confirmed_now=False
  if not isinstance(t,int) or isinstance(t,bool) or (self._last is not None and t<=self._last):self._quarantine();return False
  dt=0 if self._last is None else (t-self._last)/1e9;self._last=t
  if not valid or not all(isinstance(v,(int,float)) and math.isfinite(v) for v in (gnss,wheel,pred)):
   self._quarantine();self._plus=self._minus=0.;return False
  if dt>.1+1e-12:self._quarantine();self._plus=self._minus=0.;dt=0
  base=abs(gnss-wheel)<=1;discrepancy=max(abs(gnss-pred),abs(wheel-pred));clean=base and discrepancy<=1
  if not self.admitted and self._reset_epoch and clean:
   self._plus=self._minus=0.;self._reset_epoch=False;dt=0
  residual=(gnss+wheel)/2-pred-self._centers[speed_bin(pred)]
  self._plus=max(0.,self._plus+(residual-self._k)*dt);self._minus=max(0.,self._minus+(-residual-self._k)*dt)
  self.confirmed_now=not base or discrepancy>self._hard or max(self._plus,self._minus)>self._h
  if self.confirmed_now:self.latched=True;self._quarantine();return False
  if not clean:
   if not self.admitted:self._quarantine()
   return False
  if self.admitted:return True
  if self._pending:
   self.admitted=True;self.latched=False;self._pending=False;return True
  if self._first is None:self._first=t
  self._count+=1
  if self._count>=5 and t-self._first>=1000000000:self._pending=True
  return False

def calibrate(fitrows,calrows):
 if any(r.get('role')!='fit' for r in fitrows) or any(r.get('role')!='calibration' for r in calrows):raise ValueError('fitting/calibration role violation')
 fit=[r for r in fitrows if r['valid']]
 if not fit:raise ValueError('no valid fitting data')
 global_center=statistics.median((r['gnss_mps']+r['label_mps'])/2-r['prediction_mps'] for r in fit)
 centers=[]
 for b in range(4):
  values=[(r['gnss_mps']+r['label_mps'])/2-r['prediction_mps'] for r in fit if speed_bin(r['prediction_mps'])==b]
  centers.append(statistics.median(values) if values else global_center)
 byblock=collections.defaultdict(list)
 for r in calrows:byblock[r['window_id']].append(r)
 maximum=0.;hard=0.;valid_count=0
 for rows in byblock.values():
  plus=minus=0.;last=None
  for r in rows:
   t=r['time_ns']
   if not r['valid']:plus=minus=0.;last=None;continue
   dt=0 if last is None else (t-last)/1e9;last=t
   if not 0<=dt<=.1+1e-12:plus=minus=0.;dt=0
   residual=(r['gnss_mps']+r['label_mps'])/2-r['prediction_mps']-centers[speed_bin(r['prediction_mps'])]
   plus=max(0.,plus+(residual-.1)*dt);minus=max(0.,minus+(-residual-.1)*dt);maximum=max(maximum,plus,minus)
   hard=max(hard,abs(r['gnss_mps']-r['prediction_mps']),abs(r['label_mps']-r['prediction_mps']));valid_count+=1
 if not valid_count:raise ValueError('no valid calibration data')
 return {'centers':centers,'allowance_mps':.1,'cusum_limit':1.10*maximum+1e-6,'hard_limit':max(2.5,1.10*hard),'calibration_cusum_max':maximum,'calibration_valid_rows':valid_count}
