"""Frozen offline integration matrix. No vehicle data or physical operations."""
from common import *
install_guard()
import gzip,itertools,shutil,tempfile
from fractions import Fraction as F
import command_gateway as cg
ce=cg.ce;p=cg.p
config=json.loads((R/"protocol.json").read_text())
assert not (R/"results.json").exists(),"Frozen output exists; run verify_matrix.py instead"
paths=[R/name for name in ("DESIGN_PLAN.md","protocol.json","TIMING_SCOPE.md","common.py","command_envelope.py","command_gateway.py","test_integration.py","run_matrix.py")]
paths += [R.parent/name for name in ("vehicle_policy_gateway/policy_gateway.py","vehicle_evidence_auth/auth_guard.py","vehicle_command_tube/tube_core.py","vehicle_command_tube/file_manifest.json")]
lock={str(path.relative_to(P)):sha(path) for path in paths};write("input_lock.json",lock)
keys={name:__import__("hashlib").sha256(("COACTCPS-PUBLIC-TEST-KEY-"+name).encode()).digest() for name in ("candidate","anchor","command_logger")}
write("public_test_keys.json",{k:v.hex() for k,v in keys.items()})
runtime=R/"runtime";runtime.mkdir(exist_ok=True)
totals={};cells=[];rows=0
def packet(role,speed,seq,stamp,now,context):
 body=dict(schema=1,source_id=role,key_id="c-key" if role=="candidate" else "a-key",
  vehicle_id=context.vehicle_id,session_id=context.session_id,capture_id=context.capture_id,
  channel="wheel_speed" if role=="candidate" else "protected_speed",seq=seq,measured_us=stamp,speed_mmps=speed)
 return p.Packet(json.dumps(p.auth.sign(body,keys[role])),now)
def wire(row):return None if row is None else {"wire":row.wire,"received_us":row.received_us}
def alter(packet,field,value):
 obj=json.loads(packet.wire);obj["body"][field]=value
 return p.Packet(json.dumps(obj),packet.received_us)
with gzip.open(R/"decisions.jsonl.gz","wt",encoding="utf-8",compresslevel=3) as f:
 for cid,(case,risk,age,clock) in enumerate(itertools.product(config["cases"],config["contexts"],config["nominal_age_us"],config["anchor_clock_uncertainty_us"])):
  with tempfile.TemporaryDirectory(dir=runtime) as folder:
   root=Path(folder);state=root/"state.sqlite";backup=root/"old.sqlite"
   context=p.auth.Context("vehicle-A","session-A",f"capture-{cid}")
   p.auth.create_state(state,context);shutil.copyfile(state,backup)
   rules=[p.auth.KeyRule("candidate","c-key","wheel_speed",keys["candidate"]),p.auth.KeyRule("anchor","a-key","protected_speed",keys["anchor"])]
   policies=[p.SourcePolicy("candidate","wheel_speed","candidate","drive",None,0,2000000),p.SourcePolicy("anchor","protected_speed","anchor","instrument",250,0,2000000,True)]
   selections={};bad500=bad2000=selected=conditional_bad=replayed=0;last=None
   for step in range(config["queries_per_condition"]):
    active=step>=8;now=1000000+step*1000;seq=step+1;actual_age=100000 if active and case=="anchor_clock_window" else age
    assumption=not(active and case=="unknown_execution_bound");enabled=not(active and case=="command_revoked_key")
    command_rule=ce.CommandRule("command_logger","h-key",keys["command_logger"],"control",enabled)
    guard=p.auth.Guard(state,context,rules)
    gateway=cg.CommandGateway(guard,policies,command_rule,clock_uncertainty_us=clock,execution_error_mmps2=1000,research_execution_assumption=assumption)
    c=packet("candidate",10500,seq,now,now,context);a=packet("anchor",10000,seq,now-actual_age,now,context)
    records=[[now-actual_age-clock,now,0,1]]
    body=dict(schema=1,source_id="command_logger",key_id="h-key",vehicle_id=context.vehicle_id,
     session_id=context.session_id,capture_id=context.capture_id,channel="command_history",seq=seq,issued_us=now,commands=records)
    truth=F(10000);source_holds=integrity=execution=True;snapshot=False;replay_event=False
    if active:
     if case=="candidate_plus3":c=packet("candidate",13000,seq,now,now,context)
     if case=="candidate_missing":c=None
     if case=="candidate_bad_tag":c=alter(c,"speed_mmps",13000)
     if case=="anchor_bad_tag":a=alter(a,"speed_mmps",13000)
     if case=="anchor_stale":a=packet("anchor",10000,seq,now-100001,now,context)
     if case=="command_unknown_key":body["key_id"]="unknown"
     if case=="command_context_mismatch":body["session_id"]="wrong"
     if case=="command_old_issue":
      body["issued_us"]=now-100001;body["commands"]=[[now-actual_age-clock-100001,now-100001,0,1]]
     if case=="signed_joint_source_bias":
      c=packet("candidate",13000,seq,now,now,context);a=packet("anchor",13000,seq,now-actual_age,now,context);source_holds=False
     if case in ("signed_command_history_bias","execution_bound_violated"):
      c=None;body["commands"]=[[now-actual_age-clock,now,2000,1]]
      truth=F(10000)-F(4000*(actual_age+clock),1000000)
      integrity=case!="signed_command_history_bias";execution=case!="execution_bound_violated"
    h=p.Packet(json.dumps(ce.sign(body,keys["command_logger"])),now)
    if active:
     if case=="command_bad_tag":h=alter(h,"seq",seq+1000)
     if case=="command_gap":
      h=alter(h,"commands",[[now-actual_age-clock,now-10000,0,1],[now-9000,now,0,1]])
     if case=="command_missing":h=None
     if case in ("command_replay","restart_replay"):h=last[2];replay_event=True
     if case=="report_replay":c,a=last[:2];replay_event=True
     if case=="database_snapshot_rollback":
      shutil.copyfile(backup,state);c,a,h,now=last;replay_event=True;snapshot=True
    if not active:last=(c,a,h,now)
    source_holds=a is not None and abs(json.loads(a.wire)["body"]["speed_mmps"]-10000)<=250
    result=gateway.evaluate(c,a,h,now,risk=risk)
    estimate=result["research_speed_mmps"];error=None if estimate is None else abs(F(estimate)-truth)
    holds=source_holds and integrity and execution
    selections[result["selection"]]=selections.get(result["selection"],0)+1
    if estimate is not None:
     selected+=1;bad500+=error>500;bad2000+=error>2000;conditional_bad+=error>500 and holds and assumption;replayed+=replay_event
    event=dict(condition=cid,case=case,risk=risk,nominal_age_us=age,clock_us=clock,step=step,now_us=now,
      context=dict(vehicle_id=context.vehicle_id,session_id=context.session_id,capture_id=context.capture_id),
      command_key_enabled=enabled,execution_assumption=assumption,packets={"candidate":wire(c),"anchor":wire(a),"command":wire(h)},
      snapshot_restored=snapshot,replayed_evidence_event=replay_event,evaluation={"true_speed_n":truth.numerator,"true_speed_d":truth.denominator,
       "anchor_bound_holds":source_holds,"command_history_intact":integrity,"execution_bound_holds":execution,"model_holds":holds,
       "selected_error_n":error.numerator if error is not None else None,"selected_error_d":error.denominator if error is not None else None},decision=result)
    f.write(json.dumps(event,separators=(",",":"),allow_nan=False)+"\n");rows+=1
   assert conditional_bad==0
   cells.append(dict(condition=cid,case=case,risk=risk,nominal_age_us=age,clock_us=clock,queries=16,selected=selected,
    selections=selections,error_above500=bad500,error_above2000=bad2000,conditional_error_above500=conditional_bad,replayed_evidence_outputs=replayed))
  if (cid+1)%48==0:print("completed",cid+1,"conditions",flush=True)
assert len(cells)==264 and rows==4224
assert all(sha(P/name)==digest for name,digest in lock.items())
write("results.json",{"conditions":cells,"condition_count":264,"decisions":rows,"physical_validation_established":False,
 "specific_novelty_established":False,"promotion_allowed":False,"scope":"Offline logical-time authenticated research integration; not closed-loop or physical"})
print(json.dumps({"conditions":len(cells),"decisions":rows,"selected":sum(c["selected"] for c in cells),"error_above500":sum(c["error_above500"] for c in cells),
"replayed_evidence_outputs":sum(c["replayed_evidence_outputs"] for c in cells)},indent=2),flush=True)
