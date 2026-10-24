"""Research-only scalar reachable interval; caller must justify all premises."""
from fractions import Fraction as F

def uint(x):return type(x) is int and 0<=x<=2**63-1

def admit(*,candidate_mmps,anchor_mmps,uncertainty_mmps,measured_us,now_us,
          clock_uncertainty_us,execution_error_mmps2,commands,assumptions):
 result={"selection":"abstain","value_mmps":None,"reason":"assumptions_unknown",
 "low_n":None,"low_d":None,"high_n":None,"high_d":None,
 "conditional_budget_satisfied":False,"physical_validation_established":False,"promotion_allowed":False}
 if assumptions is not True:return result
 if not all(uint(v) for v in (anchor_mmps,uncertainty_mmps,measured_us,now_us,clock_uncertainty_us,execution_error_mmps2)):
  return {**result,"reason":"invalid_or_missing_bound"}
 age=now_us-measured_us;lower_time=measured_us-clock_uncertainty_us;upper_time=measured_us+clock_uncertainty_us
 if age-clock_uncertainty_us<0 or age+clock_uncertainty_us>100000:
  return {**result,"reason":"uncertain_age_outside_window"}
 if lower_time<0 or type(commands) not in (list,tuple) or len(commands)>16:
  return {**result,"reason":"invalid_or_missing_history"}
 records=[]
 for record in commands:
  if type(record) not in (list,tuple) or len(record)!=3:
   return {**result,"reason":"invalid_history"}
  start,end,command=record
  if not uint(start) or not uint(end) or start>=end or end>now_us or type(command) not in (int,F) or not -4000<=command<=2000:
   return {**result,"reason":"invalid_history"}
  if records and start!=records[-1][1]:return {**result,"reason":"history_gap_or_overlap"}
  records.append((start,end,F(command)))
 if lower_time<now_us and (not records or records[0][0]>lower_time or records[-1][1]!=now_us):
  return {**result,"reason":"incomplete_history"}
 cuts={lower_time,upper_time}
 for start,end,_ in records:
  if lower_time<start<upper_time:cuts.add(start)
  if lower_time<end<upper_time:cuts.add(end)
 lows=[];highs=[]
 for sampled in sorted(cuts):
  lo=max(F(0),F(anchor_mmps-uncertainty_mmps));hi=F(anchor_mmps+uncertainty_mmps)
  for start,end,command in records:
   dt=F(max(0,end-max(start,sampled)),1000000)
   lo=max(F(0),lo+max(F(-4000),command-execution_error_mmps2)*dt)
   hi=max(F(0),hi+min(F(2000),command+execution_error_mmps2)*dt)
  lows.append(lo);highs.append(hi)
 low=min(lows);high=max(highs)
 result.update(low_n=low.numerator,low_d=low.denominator,high_n=high.numerator,high_d=high.denominator)
 if uint(candidate_mmps) and max(abs(candidate_mmps-low),abs(candidate_mmps-high))<=500:
  return {**result,"selection":"candidate","value_mmps":candidate_mmps,"reason":"candidate_inside_reachable_budget","conditional_budget_satisfied":True}
 center=round((low+high)/2)
 if max(abs(center-low),abs(center-high))<=500:
  return {**result,"selection":"propagated_anchor","value_mmps":center,"reason":"propagated_anchor_inside_budget","conditional_budget_satisfied":True}
 return {**result,"reason":"budget_exhausted"}
