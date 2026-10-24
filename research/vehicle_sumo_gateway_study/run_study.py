"""Frozen SUMO software-in-the-loop integration. No real vehicle/data access."""
from sg_common import *
install_guard(local_sumo=True)
import traci,contextlib,random,gzip,itertools,time,math,xml.etree.ElementTree as ET
import sumo_bridge as bridge
from sg_evolution import step_bound
import bisect
cg=bridge.cg;ce=cg.ce;p=cg.p
config=json.loads((R/"PROTOCOL.json").read_text())
KEYS={name:hashlib.sha256(("COACT-SUMO-PUBLIC-TEST-"+name).encode()).digest() for name in ("candidate","anchor","command_logger")}
EPOCH=1000000
def packet(role,speed,seq,stamp,now,ctx):
 body=dict(schema=1,source_id=role,key_id="c-key" if role=="candidate" else "a-key",vehicle_id=ctx.vehicle_id,
 session_id=ctx.session_id,capture_id=ctx.capture_id,channel="wheel_speed" if role=="candidate" else "protected_speed",
 seq=seq,measured_us=stamp,speed_mmps=speed)
 return p.Packet(json.dumps(p.auth.sign(body,KEYS[role])),now)
def wire(obj):return None if obj is None else {"wire":obj.wire,"received_us":obj.received_us}
def percentile(a,q=.95):
 a=sorted(a)
 if not a:return None
 z=(len(a)-1)*q;i=int(z);return a[i]+(a[min(i+1,len(a)-1)]-a[i])*(z-i)
def lead_input(t):
 return 1. if t<4 else 0. if t<8 else -2. if t<12 else 1. if t<16 else 0.
def run_trial(trial_id,case,age,seed,policy,reference=False,pilot=False):
 root=(R/"runtime"/("reference_pilot" if reference else "trial_pilot") if pilot else R/"simulator_outputs"/f"trial_{trial_id:03d}")
 root.mkdir(parents=True,exist_ok=False)
 rng=random.Random(seed);sensor=[(rng.randint(-100,100),rng.randint(-100,100)) for _ in range(201)]
 rng=random.Random(seed+10000);disturbance=[rng.randint(-500,500) for _ in range(201)]
 ctx=p.auth.Context("sumo-ego","sumo-session",f"trial-{trial_id}")
 state=root/"replay.sqlite";p.auth.create_state(state,ctx)
 rules=[p.auth.KeyRule("candidate","c-key","wheel_speed",KEYS["candidate"]),p.auth.KeyRule("anchor","a-key","protected_speed",KEYS["anchor"])]
 policies=[p.SourcePolicy("candidate","wheel_speed","candidate","drive",None,0,100000000),p.SourcePolicy("anchor","protected_speed","anchor","instrument",250,0,100000000,True)]
 gateway=cg.CommandGateway(p.auth.Guard(state,ctx,rules),policies,ce.CommandRule("command_logger","h-key",KEYS["command_logger"],"control"),
 clock_uncertainty_us=5000,execution_error_mmps2=1000,research_execution_assumption=True)
 cmd=[str(BINARY),"-n",str(R/"road.net.xml"),"-r",str(R/("reference.rou.xml" if reference else "controlled.rou.xml")),
 "--step-length","0.01","--no-step-log","true","--precision","6","--seed",str(seed),"--collision.action","warn",
 "--collision.mingap-factor","0","--time-to-teleport","-1","--collision-output",str(root/"collisions.xml.gz"),
 "--fcd-output",str(root/"fcd.xml.gz"),"--fcd-output.acceleration","true"]
 rows=[];ledger=[];execution=[];anchor=None;last_control=10000;requested=0;applied=disturbance[0]
 phase=100000-age
 captures={phase+k*100000 for k in range(200) if phase+k*100000>=10000}
 queries={k*100000 for k in range(1,201)}
 events=sorted(captures|queries)
 with (root/"console.log").open("w",encoding="utf-8") as console,contextlib.redirect_stdout(console):
  traci.start(cmd,stdout=console)
  try:
   traci.simulationStep(.01)
   traci.vehicle.setSpeedMode("lead",0);traci.vehicle.setSpeedFactor("lead",1.)
   traci.vehicle.setLaneChangeMode("lead",512);traci.vehicle.setLaneChangeMode("ego",512)
   if not reference:
    traci.vehicle.setSpeedMode("ego",0);traci.vehicle.setAcceleration("ego",applied/1000,.1)
   traci.vehicle.setAcceleration("lead",lead_input(.01),1.)
   if 10000 in captures:anchor=(EPOCH+10000,traci.vehicle.getSpeed("ego")*1000)
   for stamp in events:
    if stamp==10000:continue
    traci.simulationStep(stamp/1000000)
    ego=traci.vehicle.getSpeed("ego");lead=traci.vehicle.getSpeed("lead")
    pos=traci.vehicle.getLanePosition("ego");leadpos=traci.vehicle.getLanePosition("lead");gap=leadpos-pos-5
    traci.vehicle.setAcceleration("lead",lead_input(stamp/1000000),1.)
    if stamp in captures:anchor=(EPOCH+stamp,ego*1000)
    if stamp not in queries:continue
    if reference:
     rows.append(dict(trial_id=trial_id,case=case,age_us=None,clock_us=None,seed=seed,policy="native_idm",native_idm_reference=True,
      step=stamp//100000-1,now_us=EPOCH+stamp,time_s=stamp/1000000,ego_speed_mps=ego,lead_speed_mps=lead,
      ego_position_m=pos,lead_position_m=leadpos,gap_m=gap,packets=None,decision=None,selected_disagreement_mps=None,
      requested_mmps2=None,applied_mmps2=None,model_premises_hold=False,decision_wall_seconds=0.))
     continue
    now=EPOCH+stamp;step=stamp//100000-1;seq=step+1;active=8000000<=stamp<14000000
    ledger.append((EPOCH+last_control,now,requested));execution.append((EPOCH+last_control,now,requested,applied))
    risk="elevated" if case=="context_change" and 8000000<=stamp<11000000 else "lockdown" if case=="context_change" and 11000000<=stamp<14000000 else "normal"
    c_speed=max(0,round(ego*1000)+sensor[step][0]);a_speed=None if anchor is None else max(0,round(anchor[1])+sensor[step][1])
    if active and case in ("candidate_minus3","joint_minus3"):c_speed=max(0,c_speed-3000)
    if active and case=="joint_minus3" and a_speed is not None:a_speed=max(0,a_speed-3000)
    candidate=packet("candidate",c_speed,seq,now,now,ctx)
    anchor_packet=packet("anchor",a_speed,seq,anchor[0],now,ctx) if a_speed is not None else None
    if active and case=="candidate_bad_tag":
     altered=json.loads(candidate.wire);altered["body"]["speed_mmps"]+=3000;candidate=p.Packet(json.dumps(altered),now)
    if active and case=="anchor_dropout":anchor_packet=None
    lower=(anchor[0]-5000) if anchor else now
    try:genuine=bridge.history_window(ledger,lower,now)
    except ValueError:genuine=[]
    supplied=[tuple(row) for row in genuine]
    if active and case=="forged_history":supplied=[(a,b,min(2000,c+2000)) for a,b,c in genuine]
    hbody=dict(schema=1,source_id="command_logger",key_id="h-key",vehicle_id=ctx.vehicle_id,session_id=ctx.session_id,
     capture_id=ctx.capture_id,channel="command_history",seq=seq,issued_us=now,commands=[[a,b,c,1] for a,b,c in supplied])
    history=p.Packet(json.dumps(ce.sign(hbody,KEYS["command_logger"])),now)
    begin=time.perf_counter()
    full=gateway.evaluate(candidate,anchor_packet,history,now,risk=risk)
    cb=json.loads(candidate.wire)["body"];ab=json.loads(anchor_packet.wire)["body"] if anchor_packet is not None else None
    decision=bridge.adapt_verified(policy,full,cb,ab,now,5000)
    elapsed=time.perf_counter()-begin
    value=decision["research_speed_mmps"]
    output=None if value is None else value/1000
    requested_next=bridge.idm_command(output,lead,gap)
    violation=active and case=="execution_break"
    actual_next=max(-4000,min(2000,requested_next+(2500 if violation else disturbance[step+1])))
    source_ok=anchor_packet is not None and abs(a_speed-anchor[1])<=250
    command_ok=bool(genuine) and genuine==supplied
    execution_ok=bool(anchor) and all(abs(a-u)<=1000 for s,e,u,a in execution if e>lower and s<now)
    error=None if value is None else abs(value-ego*1000)/1000
    rows.append(dict(trial_id=trial_id,case=case,age_us=age,clock_us=5000,seed=seed,policy=policy,native_idm_reference=reference,
     step=step,now_us=now,time_s=stamp/1000000,context=dict(vehicle_id=ctx.vehicle_id,session_id=ctx.session_id,capture_id=ctx.capture_id),
     risk=risk,active=active,ego_speed_mps=ego,lead_speed_mps=lead,ego_position_m=pos,lead_position_m=leadpos,gap_m=gap,
     anchor_actual_time_us=anchor[0] if anchor else None,anchor_actual_speed_mmps=anchor[1] if anchor else None,
     packets={"candidate":wire(candidate),"anchor":wire(anchor_packet),"command":wire(history)},
     genuine_history=genuine,declared_input_error_bound_mmps2=1000,anchor_bound_holds=source_ok,command_history_intact=command_ok,
     execution_bound_holds=execution_ok,model_premises_hold=source_ok and command_ok and execution_ok,
     decision=decision,selected_disagreement_mps=error,requested_mmps2=None if reference else requested_next,
     applied_mmps2=None if reference else actual_next,decision_wall_seconds=elapsed))
    if not reference:traci.vehicle.setAcceleration("ego",actual_next/1000,.1)
    requested,applied,last_control=requested_next,actual_next,stamp
  finally:traci.close()
 assert len(rows)==200
 if not reference:
  with gzip.open(root/"fcd.xml.gz","rt",encoding="utf-8") as f:trajectory=ET.parse(f).getroot()
  states=[]
  for tick in trajectory.findall("timestep"):
   for vehicle in tick.findall("vehicle"):
    if vehicle.attrib["id"]=="ego":states.append((EPOCH+round(float(tick.attrib["time"])*1000000)+10000,float(vehicle.attrib["speed"])))
  starts=[x["now_us"] for x in rows];commands=[x["requested_mmps2"] for x in rows];transitions=[]
  for (start,before),(end,after) in zip(states,states[1:]):
   index=bisect.bisect_right(starts,start)-1;u=commands[index] if index>=0 else 0
   transitions.append((start,end,step_bound(before,after,u,end-start)))
  for row in rows:
   lower=row["anchor_actual_time_us"]-5000;now=row["now_us"]
   sampled=states[0][0]<=lower and states[-1][0]>=now and all(ok for start,end,ok in transitions if end>lower and start<now)
   row["commanded_input_bound_holds"]=row["execution_bound_holds"]
   row["sampled_evolution_bound_holds"]=sampled
   row["execution_bound_holds"]=row["execution_bound_holds"] and sampled
   row["model_premises_hold"]=row["anchor_bound_holds"] and row["command_history_intact"] and row["execution_bound_holds"]
  checked(root/"evolution_summary.json").write_text(json.dumps({"sampled_transitions":len(transitions),"outside_declared_transition_bounds":sum(not x[2] for x in transitions),"fcd_precision_mps":0.000001,"numerical_tolerance_mps":0.000002},indent=2)+"\n",encoding="utf-8")
 with gzip.open(root/"trace.jsonl.gz","wt",encoding="utf-8",compresslevel=3) as f:
  for row in rows:f.write(json.dumps(row,separators=(",",":"),allow_nan=False)+"\n")
 with gzip.open(root/"collisions.xml.gz","rt",encoding="utf-8") as f:collisions=list(ET.parse(f).getroot())
 errors=[x["selected_disagreement_mps"] for x in rows if x["selected_disagreement_mps"] is not None]
 summary=dict(trial_id=trial_id,case=case,age_us=age,clock_us=5000,seed=seed,policy="native_idm" if reference else policy,native_idm_reference=reference,
  queries=200,selected=len(errors),availability=None if reference else len(errors)/200,selected_error_p95_mps=percentile(errors),
  selected_error_gt05=sum(x>.5 for x in errors),selected_error_gt2=sum(x>2 for x in errors),
  conditional_error_gt05=sum(x["selected_disagreement_mps"] is not None and x["selected_disagreement_mps"]>.5 and x["model_premises_hold"] for x in rows),
  model_premise_violation_queries=sum(not x["model_premises_hold"] for x in rows),
  speed_target_mae_mps=math.fsum(abs(x["ego_speed_mps"]-18) for x in rows)/200,
  progress_m=rows[-1]["ego_position_m"]-60,minimum_sampled_gap_m=min(x["gap_m"] for x in rows),collision_events=len(collisions),
  decision_wall_p95_s=percentile([x["decision_wall_seconds"] for x in rows]),trace=str((root/"trace.jsonl.gz").relative_to(R)))
 checked(root/"summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
 return summary
def main():
 if "--pilot" in sys.argv:
  result=run_trial(999,"nominal",90000,17,"tube",pilot=True);write("runner_pilot.json",result);print(json.dumps(result,indent=2));return
 assert not (R/"results.json").exists(),"Results already frozen"
 assert (R/"input_lock.json").exists(),"Freeze inputs first"
 lock=json.loads((R/"input_lock.json").read_text());assert all(sha(P/n)==h for n,h in lock.items())
 trials=[]
 for tid,(case,age,seed,policy) in enumerate(itertools.product(config["cases"],config["ages_us"],config["seeds"],config["policies"])):
  trials.append(run_trial(tid,case,age,seed,policy))
  if (tid+1)%6==0:print("Completed",tid+1,"of144 controlled trials",flush=True)
 for index,seed in enumerate(config["seeds"]):trials.append(run_trial(144+index,"native_idm_reference",90000,seed,"tube",reference=True))
 assert len(trials)==147 and sum(t["queries"] for t in trials if not t["native_idm_reference"])==28800
 assert all(sha(P/n)==h for n,h in lock.items())
 write("results.json",{"trials":trials,"controlled_trials":144,"controlled_queries":28800,"native_reference_trials":3,
 "scope":"SUMO software-in-the-loop; generated reports and command execution assumptions","physical_vehicle_validation_established":False,
 "global_algorithmic_novelty_established":False,"human_effectiveness_established":False,"production_promotion_allowed":False})
 print("SUMO_MATRIX_COMPLETE",flush=True)
if __name__=="__main__":main()
