"""Opt-in research integration. Authentication does not establish measurement truth."""
from pathlib import Path
from fractions import Fraction as F
import sqlite3,sys
import command_envelope as ce
p=ce.p
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/"vehicle_command_tube"))
from tube_core import admit

class CommandGateway:
 def __init__(self,guard,policies,command_rule,*,clock_uncertainty_us,execution_error_mmps2,research_execution_assumption=False):
  if not p.auth.unsigned(clock_uncertainty_us) or not p.auth.unsigned(execution_error_mmps2) or type(research_execution_assumption) is not bool:raise ValueError("Invalid protected execution/clock policy")
  self.sources=p.Gateway(guard,policies);self.commands=ce.CommandGuard(guard,command_rule)
  if command_rule.domain_id in {v.domain_id for v in self.sources._roles.values()}:raise ValueError("Command issuer must have a distinct declared domain")
  self.clock=clock_uncertainty_us;self.bound=execution_error_mmps2;self.execution_assumed=research_execution_assumption
 def evaluate(self,candidate_packet,anchor_packet,command_packet,now_us,risk="normal",operation="research_speed"):
  cs=as_=hs="not_checked"
  def result(selection,reason,value=None,detail=None):
   out=p.decision(selection,reason,cs,as_,value,risk)
   out.update(command_status=hs,conditional_budget_satisfied=False,anchor_clock_uncertainty_us=self.clock,
    execution_error_mmps2=self.bound,physical_validation_established=False)
   if detail is not None:
    out.update({k:detail[k] for k in ("conditional_budget_satisfied","low_n","low_d","high_n","high_d")})
   return out
  if operation!="research_speed":return result("abstain","physical_operation_unavailable")
  if type(risk) is not str or risk not in ("normal","elevated","lockdown"):return result("abstain","unsupported_risk_context")
  if risk=="lockdown":return result("abstain","lockdown_context")
  if not p.auth.unsigned(now_us):return result("abstain","invalid_receiver_time")
  if not self.execution_assumed:return result("abstain","execution_assumption_not_supplied")
  try:
   history,hs=self.commands.verify(command_packet,now_us)
   if history is None:return result("abstain","command_"+hs)
   candidate,cs=self.sources._authenticate(candidate_packet,"candidate",now_us)
   anchor,as_=self.sources._authenticate(anchor_packet,"anchor",now_us)
  except (ValueError,OSError,sqlite3.Error):return result("abstain","state_unavailable")
  if anchor is None:return result("abstain","anchor_"+as_)
  protected=self.sources._roles["anchor"]
  if not protected.research_assumption:return result("abstain","anchor_assumption_not_supplied")
  if not protected.valid_from_us<=anchor["measured_us"]-self.clock<=now_us<=protected.valid_until_us:return result("abstain","anchor_calibration_outside_interval")
  if candidate is not None and not p.in_interval(self.sources._roles["candidate"],candidate,now_us):candidate=None;cs="policy_interval_expired"
  selected_candidate=candidate["speed_mmps"] if candidate is not None and risk=="normal" else None
  records=[(s,e,F(n,d)) for s,e,n,d in history["commands"]]
  admitted=admit(candidate_mmps=selected_candidate,anchor_mmps=anchor["speed_mmps"],
   uncertainty_mmps=protected.uncertainty_mmps,measured_us=anchor["measured_us"],now_us=now_us,
   clock_uncertainty_us=self.clock,execution_error_mmps2=self.bound,commands=records,assumptions=True)
  return result(admitted["selection"],admitted["reason"],admitted["value_mmps"],admitted)
