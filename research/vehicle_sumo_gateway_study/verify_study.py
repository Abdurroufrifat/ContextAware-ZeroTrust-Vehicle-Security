"""Independent packet/model/FCD replay; imports no gateway, controller or bridge."""
from sg_common import *
install_guard()
import gzip,xml.etree.ElementTree as ET,math,hmac,random,itertools,bisect
from fractions import Fraction as F
cfg=json.loads((R/"PROTOCOL.json").read_text());result_path=checked(Path(sys.argv[sys.argv.index("--results")+1])) if "--results" in sys.argv else R/"results.json"
assert result_path.resolve().is_relative_to(R.resolve())
result=json.loads(result_path.read_text())
lock=json.loads((R/"input_lock.json").read_text());assert all(sha(P/n)==v for n,v in lock.items())
env=json.loads((R/"environment.json").read_text());assert hashlib.sha256(BINARY.read_bytes()).hexdigest()==env["sumo_sha256"]
keys={n:hashlib.sha256(("COACT-SUMO-PUBLIC-TEST-"+n).encode()).digest() for n in ("candidate","anchor","command_logger")}
report_fields=("schema","source_id","key_id","vehicle_id","session_id","capture_id","channel","seq","measured_us","speed_mmps")
command_fields=("schema","source_id","key_id","vehicle_id","session_id","capture_id","channel","seq","issued_us","commands")
def close(a,b,tol=1e-8):
 if a is None or b is None:assert a is b,(a,b)
 else:assert math.isclose(a,b,rel_tol=1e-9,abs_tol=tol),(a,b)
def quantile(values):
 if not values:return None
 v=sorted(values);x=(len(v)-1)*.95;lo=int(x);return v[lo]+(v[min(lo+1,len(v)-1)]-v[lo])*(x-lo)
def inspect_packet(obj,role,context,now,counters):
 if obj is None:return None,"missing"
 message=json.loads(obj["wire"]);assert set(message)=={"body","tag"};body=message["body"]
 command=role=="command_logger";fields=command_fields if command else report_fields
 assert set(body)==set(fields) and body["schema"]==1
 assert [body[k] for k in ("vehicle_id","session_id","capture_id")]==[context[k] for k in ("vehicle_id","session_id","capture_id")]
 assert body["source_id"]==role and body["key_id"]==("h-key" if command else "c-key" if role=="candidate" else "a-key")
 assert body["channel"]==("command_history" if command else "wheel_speed" if role=="candidate" else "protected_speed")
 assert all(type(body[k]) is int and 0<=body[k]<2**63 for k in ("seq","issued_us") if command) if command else all(type(body[k]) is int and 0<=body[k]<2**63 for k in ("seq","measured_us","speed_mmps"))
 if command:
  assert len(body["commands"])<=16
  for i,(a,b,n,d) in enumerate(body["commands"]):
   assert all(type(x) is int for x in (a,b,n,d)) and 0<=a<b<=body["issued_us"] and 1<=d<=10**9 and -4000*d<=n<=2000*d
   if i:assert a==body["commands"][i-1][1]
 domain=b"COACTCPS-COMMAND-HMAC-SHA256-v1\x00" if command else b"COACTCPS-EVIDENCE-HMAC-SHA256-v1\x00"
 data=domain+json.dumps([body[k] for k in fields],ensure_ascii=True,separators=(",",":")).encode("ascii")
 expected=hmac.new(keys[role],data,hashlib.sha256).hexdigest()
 if not hmac.compare_digest(expected,message["tag"]):return None,"bad_tag"
 stamp=body["issued_us" if command else "measured_us"]
 assert stamp<=obj["received_us"]<=now and now-stamp<=100000 and now-obj["received_us"]<=100000
 assert body["seq"]>counters.get(role,-1);counters[role]=body["seq"]
 return body,"authenticated"
def propagate(initial,start,history,upper):
 increments=[]
 for a,b,n,d in history:
  elapsed=F(max(0,b-max(a,start)),1000000)
  u=F(n,d);extreme=min(F(2000),u+1000) if upper else max(F(-4000),u-1000)
  increments.append(extreme*elapsed)
 suffix=F(0);values=[F(0)]
 for amount in reversed(increments):suffix+=amount;values.append(suffix)
 values.append(initial+suffix)
 return max(values)
def tube(c,a,now,history,risk):
 low=high=None
 if a is None:return "abstain","anchor_missing",None,False,low,high
 begin=a["measured_us"]-5000;end=a["measured_us"]+5000
 if now-begin>100000 or now-end<0:return "abstain","uncertain_age_outside_window",None,False,low,high
 if not history or history[0][0]>begin or history[-1][1]!=now:return "abstain","incomplete_history",None,False,low,high
 starts={begin,end}
 for s,e,_,_ in history:
  if begin<s<end:starts.add(s)
  if begin<e<end:starts.add(e)
 lo=max(F(0),F(a["speed_mmps"]-250));hi=F(a["speed_mmps"]+250)
 low=min(propagate(lo,s,history,False) for s in starts);high=max(propagate(hi,s,history,True) for s in starts)
 candidate=c["speed_mmps"] if c is not None and risk=="normal" else None
 if candidate is not None and max(abs(candidate-low),abs(candidate-high))<=500:return "candidate","candidate_inside_reachable_budget",candidate,True,low,high
 center=round((low+high)/2)
 if max(abs(center-low),abs(center-high))<=500:return "propagated_anchor","propagated_anchor_inside_budget",center,True,low,high
 return "abstain","budget_exhausted",None,False,low,high
def policy(c,a,h,now,risk,kind):
 chosen,why,value,budget,low,high=tube(c,a,now,h,risk)
 if kind=="tube" or why not in {"candidate_inside_reachable_budget","propagated_anchor_inside_budget","budget_exhausted","uncertain_age_outside_window"}:
  return chosen,why,value,budget,low,high
 candidate=c["speed_mmps"] if c is not None and risk=="normal" else None
 if kind=="static":
  use=candidate is not None and abs(candidate-a["speed_mmps"])+250<=500
  return ("candidate" if use else "anchor"),"age_agnostic_component_ablation",candidate if use else a["speed_mmps"],False,None,None
 radius=F(250)+F(4000*(now-a["measured_us"]+5000),1000000)
 if now-a["measured_us"]+5000>100000:return "abstain","uncertain_age_outside_window",None,False,None,None
 if radius>500:return "abstain","budget_exhausted",None,False,None,None
 use=candidate is not None and abs(candidate-a["speed_mmps"])+radius<=500
 return ("candidate" if use else "anchor"),("candidate_within_conditional_budget" if use else "anchor_within_conditional_budget"),candidate if use else a["speed_mmps"],True,None,None
trials=result["trials"];expected=list(itertools.product(cfg["cases"],cfg["ages_us"],cfg["seeds"],cfg["policies"]))
assert len(trials)==147 and result["controlled_trials"]==144 and result["controlled_queries"]==28800 and result["native_reference_trials"]==3
assert {t["trial_id"] for t in trials}==set(range(147))
assert all(t["native_idm_reference"] is (t["trial_id"]>=144) for t in trials)
assert all(result[k] is False for k in ("physical_vehicle_validation_established","global_algorithmic_novelty_established","human_effectiveness_established","production_promotion_allowed"))
source_errors=0;verified_rows=0;fcd_matches=0;transition_count=0;assumption_failures=0
for index,summary in enumerate(trials):
 assert summary["trial_id"]==index
 reference=index>=144
 if reference:
  assert summary["native_idm_reference"] is True and summary["case"]=="native_idm_reference" and summary["seed"]==cfg["seeds"][index-144] and summary["policy"]=="native_idm"
 else:
  case,age,seed,kind=expected[index]
  assert (summary["case"],summary["age_us"],summary["seed"],summary["policy"])==(case,age,seed,kind)
 root=checked(R/"simulator_outputs"/f"trial_{index:03d}")
 assert checked(R/summary["trace"])==root/"trace.jsonl.gz"
 with gzip.open(root/"trace.jsonl.gz","rt",encoding="utf-8") as f:rows=[json.loads(line) for line in f]
 assert len(rows)==200 and [x["step"] for x in rows]==list(range(200))
 with gzip.open(root/"fcd.xml.gz","rt",encoding="utf-8") as f:xml=ET.parse(f).getroot()
 fcd={};ego_states=[]
 for tick in xml.findall("timestep"):
  tick_us=1000000+round(float(tick.attrib["time"])*1000000)+10000
  for vehicle in tick.findall("vehicle"):
   fields=vehicle.attrib;fcd[(tick_us,fields["id"])]=fields
   if fields["id"]=="ego":ego_states.append((tick_us,float(fields["speed"])))
 transitions=[];query_times=[x["now_us"] for x in rows]
 if not reference:
  for (start,v0),(end,v1) in zip(ego_states,ego_states[1:]):
   j=bisect.bisect_right(query_times,start)-1;u=rows[j]["requested_mmps2"] if j>=0 else 0
   assert end-start==10000
   low=max(0,v0+max(-4000,u-1000)*(end-start)/1e9);high=max(0,v0+min(2000,u+1000)*(end-start)/1e9)
   transitions.append((start,end,low-.000002<=v1<=high+.000002))
  transition_count+=len(transitions)
  check=json.loads((root/"evolution_summary.json").read_text())
  assert check["sampled_transitions"]==len(transitions) and check["outside_declared_transition_bounds"]==sum(not x[2] for x in transitions)
 rng=random.Random(summary["seed"]);noise=[(rng.randint(-100,100),rng.randint(-100,100)) for _ in range(201)]
 rng=random.Random(summary["seed"]+10000);actuation=[rng.randint(-500,500) for _ in range(201)]
 counters={};ledger=[];execution=[];prior_time=1010000;prior_u=0;prior_applied=actuation[0]
 for step,row in enumerate(rows):
  now=1100000+100000*step
  assert row["trial_id"]==index and row["now_us"]==now and row["native_idm_reference"] is reference
  assert row["case"]==summary["case"] and row["policy"]==summary["policy"] and row["seed"]==summary["seed"]
  close(row["time_s"],(now-1000000)/1e6)
  ego=fcd[(now,"ego")];lead=fcd[(now,"lead")]
  close(row["ego_speed_mps"],float(ego["speed"]),.000001);close(row["lead_speed_mps"],float(lead["speed"]),.000001)
  close(row["ego_position_m"],float(ego["pos"]),.000001);close(row["lead_position_m"],float(lead["pos"]),.000001)
  close(row["gap_m"],row["lead_position_m"]-row["ego_position_m"]-5)
  fcd_matches+=2
  if reference:
   assert row["packets"] is row["decision"] is row["selected_disagreement_mps"] is row["requested_mmps2"] is row["applied_mmps2"] is None
   continue
  assert row["age_us"]==age and row["clock_us"]==5000
  active=8000000<=now-1000000<14000000
  risk="elevated" if case=="context_change" and 8000000<=now-1000000<11000000 else "lockdown" if case=="context_change" and 11000000<=now-1000000<14000000 else "normal"
  assert row["active"] is active and row["risk"]==risk
  assert row["context"]=={"vehicle_id":"sumo-ego","session_id":"sumo-session","capture_id":f"trial-{index}"}
  stamp=now-age;assert row["anchor_actual_time_us"]==stamp
  recorded_anchor=float(fcd[(stamp,"ego")]["speed"])*1000
  close(row["anchor_actual_speed_mmps"],recorded_anchor,.001)
  anchor_truth=row["anchor_actual_speed_mmps"]
  current=max(0,round(row["ego_speed_mps"]*1000)+noise[step][0]);anchor_speed=max(0,round(anchor_truth)+noise[step][1])
  if active and case in ("candidate_minus3","joint_minus3"):current=max(0,current-3000)
  if active and case=="joint_minus3":anchor_speed=max(0,anchor_speed-3000)
  c_raw=json.loads(row["packets"]["candidate"]["wire"])["body"]
  assert c_raw["seq"]==step+1 and c_raw["measured_us"]==now and c_raw["speed_mmps"]==current+(3000 if active and case=="candidate_bad_tag" else 0)
  missing=active and case=="anchor_dropout"
  assert (row["packets"]["anchor"] is None) is missing
  if not missing:
   a_raw=json.loads(row["packets"]["anchor"]["wire"])["body"]
   assert a_raw["seq"]==step+1 and a_raw["measured_us"]==stamp and a_raw["speed_mmps"]==anchor_speed
  ledger.append((prior_time,now,prior_u));execution.append((prior_time,now,prior_u,prior_applied));lower=stamp-5000
  genuine=[tuple(x) for x in ledger if x[1]>lower and x[0]<now]
  if not genuine or genuine[0][0]>lower:genuine=[]
  assert [list(x) for x in genuine]==row["genuine_history"]
  supplied=[(a,b,min(2000,u+2000) if active and case=="forged_history" else u,1) for a,b,u in genuine]
  h_raw=json.loads(row["packets"]["command"]["wire"])["body"]
  assert h_raw["seq"]==step+1 and h_raw["issued_us"]==now and h_raw["commands"]==[list(x) for x in supplied]
  d=row["decision"];c=a=h=None;cs=ats=hs="not_checked"
  if risk=="lockdown":expected_dec=("abstain","lockdown_context",None,False,None,None)
  else:
   h,hs=inspect_packet(row["packets"]["command"],"command_logger",row["context"],now,counters)
   c,cs=inspect_packet(row["packets"]["candidate"],"candidate",row["context"],now,counters)
   a,ats=inspect_packet(row["packets"]["anchor"],"anchor",row["context"],now,counters)
   if a is None:expected_dec=("abstain","anchor_"+ats,None,False,None,None)
   else:expected_dec=policy(c,a,h["commands"],now,risk,kind)
  choice,why,value,budget,lo,hi=expected_dec
  assert (d["selection"],d["reason"],d["research_speed_mmps"],d["conditional_budget_satisfied"])==(choice,why,value,budget),(index,step,d,expected_dec)
  assert (d["candidate_status"],d["anchor_status"],d["command_status"])==(cs,ats,hs)
  assert d["risk_context"]==risk
  close(d["research_speed_mps"],value/1000 if value is not None else None)
  for prefix,fraction in (("low",lo),("high",hi)):
   assert d.get(prefix+"_n")==(fraction.numerator if fraction is not None else None) and d.get(prefix+"_d")==(fraction.denominator if fraction is not None else None)
  assert all(d[k] is False for k in ("physical_speed_authorized","measurement_truth_established","independent_source_trust_established","physical_witness_enabled","production_policy_changed","promotion_allowed","physical_validation_established"))
  source_ok=not missing and abs(anchor_speed-anchor_truth)<=250
  log_ok=bool(genuine) and [list(x) for x in genuine]==[[a,b,n//den] for a,b,n,den in supplied]
  commanded=all(abs(v-u)<=1000 for start,end,u,v in execution if end>lower and start<now)
  sampled=ego_states[0][0]<=lower and ego_states[-1][0]>=now and all(ok for start,end,ok in transitions if end>lower and start<now)
  holds=source_ok and log_ok and commanded and sampled
  assert (row["anchor_bound_holds"],row["command_history_intact"],row["commanded_input_bound_holds"],row["sampled_evolution_bound_holds"],row["execution_bound_holds"],row["model_premises_hold"])==(source_ok,log_ok,commanded,sampled,commanded and sampled,holds)
  error=None if value is None else abs(value/1000-row["ego_speed_mps"]);close(row["selected_disagreement_mps"],error)
  if value is None:u=-4000
  else:
   v=value/1000;desired=2+max(0,1.5*v+v*(v-row["lead_speed_mps"])/(2*math.sqrt(8)))
   u=max(-4000,min(2000,round(2000*(1-(v/18)**4-(desired/max(.001,row["gap_m"]))**2))))
  actual=max(-4000,min(2000,u+(2500 if active and case=="execution_break" else actuation[step+1])))
  assert row["requested_mmps2"]==u and row["applied_mmps2"]==actual
  assert type(row["decision_wall_seconds"]) is float and row["decision_wall_seconds"]>=0
  prior_time,prior_u,prior_applied=now,u,actual;verified_rows+=1
  assumption_failures+=not holds
 errors=[x["selected_disagreement_mps"] for x in rows if x["selected_disagreement_mps"] is not None]
 assert summary["queries"]==200 and summary["selected"]==len(errors)
 close(summary["availability"],None if reference else len(errors)/200);close(summary["selected_error_p95_mps"],quantile(errors))
 assert summary["selected_error_gt05"]==sum(e>.5 for e in errors) and summary["selected_error_gt2"]==sum(e>2 for e in errors)
 assert summary["conditional_error_gt05"]==sum(r["selected_disagreement_mps"] is not None and r["selected_disagreement_mps"]>.5 and r["model_premises_hold"] for r in rows)
 assert summary["model_premise_violation_queries"]==sum(not r["model_premises_hold"] for r in rows)
 close(summary["speed_target_mae_mps"],math.fsum(abs(x["ego_speed_mps"]-18) for x in rows)/200)
 close(summary["progress_m"],rows[-1]["ego_position_m"]-60);close(summary["minimum_sampled_gap_m"],min(x["gap_m"] for x in rows))
 close(summary["decision_wall_p95_s"],quantile([x["decision_wall_seconds"] for x in rows]))
 with gzip.open(root/"collisions.xml.gz","rt",encoding="utf-8") as f:collisions=list(ET.parse(f).getroot())
 assert summary["collision_events"]==len(collisions)
 if (index+1)%24==0:print("Verified",index+1,"trials",flush=True)
assert verified_rows==28800 and fcd_matches==58800
assert all(result[k] is False for k in ("physical_vehicle_validation_established","global_algorithmic_novelty_established","human_effectiveness_established","production_promotion_allowed"))
write("verification.json",{"passed":True,"controlled_decisions_reconstructed":verified_rows,"native_reference_trials":3,"fcd_vehicle_states_matched":fcd_matches,
"realized_speed_transitions_checked":transition_count,"input_identities_checked":len(lock),"model_premise_violation_queries":assumption_failures,
"imports_production_decision_code":False,"physical_vehicle_validation_established":False,"global_algorithmic_novelty_established":False})
print("INDEPENDENT_SUMO_REPLAY_PASSED",flush=True)
