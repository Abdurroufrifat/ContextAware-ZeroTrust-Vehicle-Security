"""Fixed model-selection ablations behind the unchanged authenticated gateway."""
from pathlib import Path
import sys,math,json
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/"vehicle_command_gateway"))
import command_gateway as cg
sys.path.insert(0,str(ROOT/"vehicle_clock_bound_admission"))
from clock_core import select as clock_select
def history_window(records,lower,now):
 chosen=[tuple(row) for row in records if row[1]>lower and row[0]<now]
 if not chosen or len(chosen)>16 or chosen[0][0]>lower or chosen[-1][1]!=now:raise ValueError("history not covered")
 if any(a[1]!=b[0] for a,b in zip(chosen,chosen[1:])):raise ValueError("history gap")
 return chosen
def numeric(*,policy,candidate,anchor,stamp,now,clock,history,risk):
 if policy not in ("static","clock","tube"):raise ValueError("unknown policy")
 if risk not in ("normal","elevated","lockdown"):raise ValueError("unsupported context")
 if risk=="lockdown":return {"selection":"abstain","value_mmps":None,"reason":"lockdown_context"}
 c=candidate if risk=="normal" else None
 if policy=="tube":
  return cg.admit(candidate_mmps=c,anchor_mmps=anchor,uncertainty_mmps=250,measured_us=stamp,now_us=now,
   clock_uncertainty_us=clock,execution_error_mmps2=1000,commands=history,assumptions=True)
 if policy=="clock":
  return clock_select(candidate_mmps=c,anchor_mmps=anchor,uncertainty_mmps=250,measured_us=stamp,now_us=now,
   clock_uncertainty_us=clock,acceleration_mmps2=4000,assumptions=True)
 if type(anchor) is not int or anchor<0 or not 0<=now-stamp<=100000:return {"selection":"abstain","value_mmps":None,"reason":"static_invalid_anchor"}
 use=type(c) is int and c>=0 and abs(c-anchor)+250<=500
 return {"selection":"candidate" if use else "anchor","value_mmps":c if use else anchor,"reason":"age_agnostic_component_ablation","conditional_budget_satisfied":False}
def adapt_verified(policy,decision,candidate_body,anchor_body,now,clock):
 if policy not in ("static","clock","tube"):raise ValueError("unknown policy")
 if policy=="tube":return decision
 numerical={"candidate_inside_reachable_budget","propagated_anchor_inside_budget","budget_exhausted","uncertain_age_outside_window"}
 if decision["reason"] not in numerical or decision["anchor_status"]!="authenticated" or decision["command_status"]!="authenticated":return decision
 c=candidate_body["speed_mmps"] if candidate_body is not None and decision["candidate_status"]=="authenticated" else None
 out=numeric(policy=policy,candidate=c,anchor=anchor_body["speed_mmps"],stamp=anchor_body["measured_us"],now=now,
  clock=clock,history=[],risk=decision["risk_context"])
 value=out["value_mmps"]
 return {**decision,"selection":out["selection"],"reason":out["reason"],"research_speed_mmps":value,
  "research_speed_mps":value/1000 if value is not None else None,"low_n":None,"low_d":None,"high_n":None,"high_d":None,"conditional_budget_satisfied":out.get("conditional_budget_satisfied",False)}
def idm_command(estimate,lead_speed,gap):
 if estimate is None:return -4000
 if not all(type(x) in (int,float) and math.isfinite(x) for x in (estimate,lead_speed,gap)) or estimate<0 or lead_speed<0:raise ValueError("invalid controller input")
 desired=2+max(0.,estimate*1.5+estimate*(estimate-lead_speed)/(2*math.sqrt(2*4)))
 acceleration=2*(1-(estimate/18.)**4-(desired/max(gap,.001))**2)
 return max(-4000,min(2000,round(acceleration*1000)))
