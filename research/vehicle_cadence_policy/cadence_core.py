"""Experimental cadence correction, with separate per-endpoint freshness checks upstream."""
import collections,math
from previous_temporal_core import TemporalPolicy,calibrate
from final_core import speed_bin
class CadencePolicy(TemporalPolicy):
 def __init__(self,params):
  super().__init__(params)
  gap=params['continuity_gap_ns']
  if type(gap) is not int or not 100000000<=gap<=300000000:raise ValueError('invalid continuity deadline')
  self._gap_ns=gap
 def step(self,t,gnss,wheel,pred,valid=True):
  self.confirmed_now=False
  if type(t) is not int or (self._last is not None and t<=self._last):self._quarantine();return False
  delta=0 if self._last is None else t-self._last;self._last=t;dt=delta/1e9
  if not valid or not all(type(v) in (int,float) and math.isfinite(v) for v in (gnss,wheel,pred)):
   self._quarantine();self._plus=self._minus=0.;return False
  if delta>self._gap_ns:self._quarantine();self._plus=self._minus=0.;dt=0
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
  if self._pending:self.admitted=True;self.latched=False;self._pending=False;return True
  if self._first is None:self._first=t
  self._count+=1
  if self._count>=5 and t-self._first>=1000000000:self._pending=True
  return False
def calibrate_cadence(fitrows,calrows):
 # The previous calibration validates roles and supplies the identical centers
 # and hard threshold. Only time continuity changes its CUSUM maximum.
 original=calibrate(fitrows,calrows);gaps=[]
 for rows in (fitrows,calrows):
  last=None;block=None
  for r in rows:
   t=r['time_ns']
   if r['window_id']==block:
    if t<=last:raise ValueError('non-increasing development clock')
    gaps.append(t-last)
   last=t;block=r['window_id']
 if not gaps:raise ValueError('no development cadence')
 maximum_gap=max(gaps);deadline=max(100000000,min(300000000,math.ceil(1.10*maximum_gap/10000000)*10000000))
 blocks=collections.defaultdict(list)
 for r in calrows:blocks[r['window_id']].append(r)
 maximum=0.
 for rows in blocks.values():
  plus=minus=0.;last=None
  for r in rows:
   t=r['time_ns']
   if not r['valid']:plus=minus=0.;last=None;continue
   delta=0 if last is None else t-last;last=t;dt=delta/1e9
   if not 0<=delta<=deadline:plus=minus=0.;dt=0
   residual=(r['gnss_mps']+r['label_mps'])/2-r['prediction_mps']-original['centers'][speed_bin(r['prediction_mps'])]
   plus=max(0.,plus+(residual-original['allowance_mps'])*dt);minus=max(0.,minus+(-residual-original['allowance_mps'])*dt);maximum=max(maximum,plus,minus)
 return {**original,'cusum_limit':1.10*maximum+1e-6,'calibration_cusum_max':maximum,'continuity_gap_ns':deadline,'development_max_gap_ns':maximum_gap,'deadline_cap_ns':300000000}
