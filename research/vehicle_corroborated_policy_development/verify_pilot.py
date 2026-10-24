"""Pilot audit: frozen-reference baselines plus integrated guard mathematics and packet replay."""
from dev_common import *
import json,gzip,math,importlib.util,sys,time,copy
from fractions import Fraction as Q
from verification_contracts import verify_checkpoint,verify_command_history,accepted_history
from verification_math import reference_context,close
from dev_policy import CorroboratedPolicy
def loadmodule(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def main():
 result=json.loads((R/"pilot_results.json").read_text());trials=result["trials"];cfg=json.loads((R/"PILOT_PROTOCOL.json").read_text())
 assert len(trials)==52 and result["pilot"]
 assert {(t["case"],t["method"],t["seed"],t["profile"]) for t in trials}=={(c,m,2069,"fresh_merge") for c in cfg["primary_cases"]+cfg["stress_cases"] for m in cfg["methods"]}
 assert result["lock_digest"]==sha(R/"development_lock.json")
 assert all(sha(P/n)==h for n,h in json.loads((R/"development_lock.json").read_text()).items())
 for t in trials:verify_checkpoint(R,t,result["lock_digest"])
 ref=loadmodule("pilot_frozen_independent_reference",B/"verify_study.py")
 ref.B=R;ref.CFG=cfg
 subset={**result,"trials":[t for t in trials if t["method"]!="corroborated"]}
 dump(R/"runtime/reference_subset.json",subset)
 argv=sys.argv[:];sys.argv=["verify","--pilot",str(R/"runtime/reference_subset.json")]
 try:ref.main()
 finally:sys.argv=argv
 counts={"guard_trials":0,"guard_queries":0,"guard_abstentions":0,"packet_replayed_queries":0,"raw_vehicle_transitions":0,"qualified_selected":0,"qualified_selected_errors":0}
 for t in trials:
  if t["method"]!="corroborated":continue
  with gzip.open(R/t["trace"],"rt") as f:rows=[json.loads(x) for x in f]
  with gzip.open(R/t["trajectory"],"rt") as f:states=[json.loads(x) for x in f]
  assert len(rows)==200 and len(states)==2000
  c=cfg["profiles"][t["profile"]]
  for a,b in zip(states,states[1:]):
   assert b["time_us"]==a["time_us"]+10000
   qi=a["time_us"]//100000-1
   if qi>=0:close(b["ego_accel"],rows[qi]["next_applied_mps2"])
   close(b["lead_accel"],c["accelerations"][min(4,a["time_us"]//4000000)])
   for role in ("ego","lead"):
    v=max(0,a[role+"_v"]+.01*b[role+"_accel"])
    close(b[role+"_v"],v);close(b[role+"_x"],a[role+"_x"]+.01*v);counts["raw_vehicle_transitions"]+=1
  state_dir=P/"research/project_release/runtime"/("guard-audit-"+str(t["trial_id"])+"-"+str(time.time_ns()))
  policy=CorroboratedPolicy(state_dir,rows[0]["context"],KEYS);counters={};ledger=[];errors=[];accepted_previous=0
  for i,row in enumerate(rows):
   out=row["output"];source=out["source_output"];ed=source["estimator_decision"];base=ed.get("base_decision",{});record=row["input"];now=record["now_us"];ctx=row["context"]
   assert set(record)=={"now_us","risk","packets","scene_packet","radar_packet","advisory"}
   assert row["step"]==i and row["trial_id"]==t["trial_id"]
   counts["guard_queries"]+=1
   ledger.append((1010000 if i==0 else now-100000,now,accepted_previous))
   anchor=row["truth"]["anchor_time_us"]
   commands=accepted_history(ledger,None if anchor is None else anchor+1000000-5000,now)
   verify_command_history(record["packets"]["command"],KEYS["command_logger"],ctx,i+1,now,commands)
   ref.speed_body(record["packets"]["candidate"],"candidate",now,ctx,i+1);ref.speed_body(record["packets"]["anchor"],"anchor",now,ctx,i+1)
   scene,ss=ref.REF.verify_packet("scene",record["scene_packet"],now,ctx,counters)
   radar,rs=ref.REF.verify_packet("radar",record["radar_packet"],now,ctx,counters)
   assert out["scene_status"]==ss and out["radar_status"]==rs
   val=out["estimate_mmps"];status=out["guard"]["status"]
   if val is None:
    assert out["requested_mmps2"]==out["desired_mmps2"]==-4000
    assert out["controller_input"] is out["fused_context"] is out["projected_margin_m"] is None
    assert all(out["estimator_decision"].get(k) is False for k in ("conditional_budget_satisfied","refined_budget_satisfied","scene_budget_satisfied"))
    counts["guard_abstentions"]+=1
   else:
    assert status=="corroborated" and val==source["estimate_mmps"]
    lo=Q(base["low_n"],base["low_d"]);hi=Q(base["high_n"],base["high_d"])
    assert base["conditional_budget_satisfied"] is True and max(abs(val-lo),abs(val-hi))<=500
    truth=row["truth"]["ego_v"]*1000
    if float(lo)-1e-7<=truth<=float(hi)+1e-7:
     counts["qualified_selected"]+=1;counts["qualified_selected_errors"]+=abs(val-truth)>500.00001
    expected=None if scene is None or radar is None else reference_context(max(0,val/1000-.5),val/1000+.5,scene,radar)
    close(out["fused_context"],expected)
    if expected is not None:
     desired=ref.REF.nominal(val/1000,expected["lead_mps"],expected["gap_m"]);assert out["desired_mmps2"]==desired
     cmd=out["requested_mmps2"];margin=ref.REF.full_path_margin(cmd,expected["ego_high"],expected["lead_low"],expected["gap_low"])
     close(margin,out["projected_margin_m"])
     assert -4000<=cmd<=desired
     if margin<0:assert cmd==-4000
     elif cmd<desired:assert ref.REF.full_path_margin(cmd+1,expected["ego_high"],expected["lead_low"],expected["gap_low"])<1e-7
   packet,actor_status=ref.REF.verify_packet("actuator",row["delivered_command"],now,ctx,counters)
   accepted=packet["accel_mmps2"] if packet is not None else -4000
   assert accepted==row["accepted_mmps2"] and actor_status==row["actuator_status"]
   replay=json.loads(json.dumps(policy.process(record)));assert replay==out
   issued=policy.command_packet(out["requested_mmps2"],now,i+1);assert issued==row["issued_command"]
   av,ast=policy.actuate(row["delivered_command"],now);assert (av,ast)==(accepted,actor_status)
   card=policy.card(out,av,ast);assert card==row["operator_card"] and card["selection_guard"]==out["guard"]
   assert not({"truth","attack_case","label"}&set(card))
   counts["packet_replayed_queries"]+=1;accepted_previous=accepted
   err=None if val is None else abs(val/1000-row["truth"]["ego_v"])
   close(err,row["selected_error_mps"])
   if err is not None:errors.append(err)
  assert t["selected"]==len(errors);close(t["mae_mps"],sum(errors)/len(errors) if errors else None)
  gaps=[x["lead_x"]-x["ego_x"]-5 for x in states];assert t["collision_trial"]==(min(gaps)<=0)
  close(t["minimum_gap_m"],min(gaps));close(t["progress_m"],states[-1]["ego_x"]-60)
  counts["guard_trials"]+=1
 assert counts["guard_trials"]==13 and counts["guard_queries"]==counts["packet_replayed_queries"]==2600 and counts["qualified_selected_errors"]==0
 dump(R/"pilot_verification.json",{"passed":True,"scope":"Development audit;39 reference trials independently audited;13 integrated guard trials checked and replayed. No heldout novelty or global safety claim.","counts":counts,"results_sha256":sha(R/"pilot_results.json"),"development_lock_digest":sha(R/"development_lock.json")})
 print("DEVELOPMENT_INTEGRATION_AUDIT_PASS",json.dumps(counts),flush=True)
if __name__=="__main__":main()
