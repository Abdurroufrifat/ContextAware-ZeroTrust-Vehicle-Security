"""Research consistency veto; trusted initialization and motion bounds are premises."""
from fractions import Fraction as F

class TemporalGuard:
 def __init__(self,initial_now_us=1010000,initial_low_mmps=9750,initial_high_mmps=10250):
  if any(type(x) is not int or x<0 for x in (initial_now_us,initial_low_mmps,initial_high_mmps)) or initial_low_mmps>initial_high_mmps:
   raise ValueError("invalid initial contract")
  self.now=initial_now_us;self.low=F(initial_low_mmps);self.high=F(initial_high_mmps);self.latched=False
 def evaluate(self,base_decision,now_us):
  base=dict(base_decision);used=False
  reason=None
  if type(now_us) is not int or now_us<=self.now:
   self.latched=True;reason="temporal_invalid_time"
  elif not self.latched:
   dt=F(now_us-self.now,1000000)
   self.low=max(F(0),self.low-4000*dt);self.high=max(F(0),self.high+2000*dt)
   numerical=base.get("reason") in ("candidate_inside_reachable_budget","propagated_anchor_inside_budget","budget_exhausted")
   if numerical and base.get("anchor_status")==base.get("command_status")=="authenticated":
    try:
     fields=[base[k] for k in ("low_n","low_d","high_n","high_d")]
     if any(type(x) is not int for x in fields) or fields[1]<=0 or fields[3]<=0:raise ValueError("invalid rational")
     lower,upper=F(fields[0],fields[1]),F(fields[2],fields[3])
     if lower<0 or lower>upper:raise ValueError("invalid interval")
    except (KeyError,ValueError,ZeroDivisionError):
     self.latched=True;reason="temporal_invalid_interval"
    else:
     used=True;self.low=max(self.low,lower);self.high=min(self.high,upper)
     if self.low>self.high:self.latched=True;reason="temporal_inconsistency_latched"
  if type(now_us) is int and now_us>self.now:self.now=now_us
  out={**base,"base_decision":base,"temporal_latched":self.latched,"temporal_used_interval":used,
   "temporal_low_n":None if self.latched else self.low.numerator,
   "temporal_low_d":None if self.latched else self.low.denominator,
   "temporal_high_n":None if self.latched else self.high.numerator,
   "temporal_high_d":None if self.latched else self.high.denominator,
   "temporal_trusted_initialization_assumed":True,"temporal_motion_bound_assumed":True,
   "physical_validation_established":False,"promotion_allowed":False}
  if self.latched and base.get("reason")!="lockdown_context":
   out.update(selection="abstain",reason=reason or "temporal_inconsistency_latched",
    research_speed_mmps=None,research_speed_mps=None,conditional_budget_satisfied=False)
  return out
