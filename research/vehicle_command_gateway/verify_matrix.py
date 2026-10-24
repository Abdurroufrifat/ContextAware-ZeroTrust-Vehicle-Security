"""Independent wire, replay and exact-interval oracle. No production module imports."""
from common import *
install_guard()
import gzip,hmac,re,itertools,collections
from fractions import Fraction as F
protocol=json.loads((R/"protocol.json").read_text());result=json.loads((R/"results.json").read_text())
design=list(itertools.product(protocol["cases"],protocol["contexts"],protocol["nominal_age_us"],protocol["anchor_clock_uncertainty_us"]))
assert len(design)==264 and protocol["queries_per_condition"]==16
assert result["condition_count"]==len(design) and result["decisions"]==len(design)*16
assert type(result["conditions"]) is list and len(result["conditions"])==len(design)
ids=[cell["condition"] for cell in result["conditions"]]
assert all(type(cid) is int for cid in ids) and set(ids)==set(range(len(design))) and len(set(ids))==len(ids)
for cell in result["conditions"]:
 case,risk,age,clock=design[cell["condition"]]
 assert (cell["case"],cell["risk"],cell["nominal_age_us"],cell["clock_us"])==(case,risk,age,clock)
 assert type(cell["nominal_age_us"]) is int and type(cell["clock_us"]) is int
 assert type(cell["queries"]) is int and cell["queries"]==16

lock=json.loads((R/"input_lock.json").read_text());assert all(sha(P/n)==h for n,h in lock.items())
keys={n:hashlib.sha256(("COACTCPS-PUBLIC-TEST-KEY-"+n).encode()).digest() for n in ("candidate","anchor","command_logger")}
RF=("schema","source_id","key_id","vehicle_id","session_id","capture_id","channel","seq","measured_us","speed_mmps")
CF=("schema","source_id","key_id","vehicle_id","session_id","capture_id","channel","seq","issued_us","commands")
def uint(v):return type(v) is int and 0<=v<2**63
def identity(v):return type(v) is str and re.fullmatch(r"[A-Za-z0-9_.:-]{1,64}",v) is not None
def unique(pairs):
 out={}
 for k,v in pairs:
  if k in out:raise ValueError("duplicate")
  out[k]=v
 return out
def parsed(packet,kind):
 raw=packet["wire"];fields=CF if kind=="command" else RF
 if type(raw) is not str or len(raw.encode())>(8192 if kind=="command" else 4096):raise ValueError("wire")
 obj=json.loads(raw,object_pairs_hook=unique)
 if type(obj) is not dict or set(obj)!={"body","tag"}:raise ValueError("envelope")
 b=obj["body"]
 if type(b) is not dict or set(b)!=set(fields) or type(b["schema"]) is not int or b["schema"]!=1:raise ValueError("schema")
 if not all(identity(b[k]) for k in ("source_id","key_id","vehicle_id","session_id","capture_id","channel")):raise ValueError("identifier")
 if not uint(b["seq"]):raise ValueError("sequence")
 if type(obj["tag"]) is not str or re.fullmatch("[0-9a-f]{64}",obj["tag"]) is None:raise ValueError("tag")
 if kind=="command":
  if not uint(b["issued_us"]) or type(b["commands"]) is not list or len(b["commands"])>16:raise ValueError("commands")
  previous=None
  for rec in b["commands"]:
   if type(rec) is not list or len(rec)!=4 or not all(type(v) is int for v in rec):raise ValueError("record")
   s,e,n,d=rec
   if not uint(s) or not uint(e) or not s<e<=b["issued_us"] or not 1<=d<=1000000000 or not -4000*d<=n<=2000*d or (previous is not None and previous!=s):raise ValueError("geometry")
   previous=e
 else:
  if not uint(b["measured_us"]) or not uint(b["speed_mmps"]):raise ValueError("measurement")
 return obj
def authenticate(packet,kind,row,state):
 if packet is None:return None,"missing"
 try:m=parsed(packet,kind)
 except (ValueError,TypeError,KeyError):return None,"malformed" if kind=="command" else "malformed_wire"
 b=m["body"];source="command_logger" if kind=="command" else kind;key_id={"command":"h-key","candidate":"c-key","anchor":"a-key"}[kind]
 channel={"command":"command_history","candidate":"wheel_speed","anchor":"protected_speed"}[kind]
 if kind!="command" and (b["source_id"]!=source or b["channel"]!=channel):return None,"role_mismatch"
 if (b["source_id"],b["key_id"])!=(source,key_id):return None,"unknown_key"
 if kind=="command" and not row["command_key_enabled"]:return None,"revoked_key"
 fields=CF if kind=="command" else RF
 prefix=b"COACTCPS-COMMAND-HMAC-SHA256-v1\x00" if kind=="command" else b"COACTCPS-EVIDENCE-HMAC-SHA256-v1\x00"
 data=prefix+json.dumps([b[k] for k in fields],ensure_ascii=True,separators=(",",":")).encode("ascii")
 if not hmac.compare_digest(hmac.new(keys[source],data,hashlib.sha256).hexdigest(),m["tag"]):return None,"bad_tag"
 if any(b[k]!=row["context"][k] for k in ("vehicle_id","session_id","capture_id")):return None,"context_mismatch"
 if b["channel"]!=channel:return None,"channel_denied"
 stamp=b["issued_us" if kind=="command" else "measured_us"];received=packet["received_us"];now=row["now_us"]
 if not uint(now) or not uint(received) or not stamp<=received<=now:return None,"invalid_clock_order"
 if now-received>100000:return None,"receipt_stale"
 if now-stamp>100000:return None,"measurement_stale"
 if now<state["last"]:return None,"clock_rollback"
 counter=(source,channel)
 if b["seq"]<=state["counters"].get(counter,-1):return None,"replay_or_reorder"
 state["counters"][counter]=b["seq"];state["last"]=now
 return b,"authenticated"
def reflected(v,increments):
 suffix=F(0);answer=F(0)
 for delta in reversed(increments):answer=max(answer,suffix);suffix+=delta
 return max(answer,v+suffix)
def interval_choice(candidate,anchor,commands,now,clock):
 stamp=anchor["measured_us"]
 if now-stamp-clock<0 or now-stamp+clock>100000:return None,"abstain","uncertain_age_outside_window",None,None
 early,late=stamp-clock,stamp+clock
 records=[(s,e,F(n,d)) for s,e,n,d in commands]
 if early<now and (not records or records[0][0]>early or records[-1][1]!=now):return None,"abstain","incomplete_history",None,None
 cuts={early,late}|{t for s,e,c in records for t in (s,e) if early<t<late}
 low=[];high=[]
 for t in cuts:
  future=[(F(max(0,e-max(s,t)),1000000),c) for s,e,c in records]
  low.append(reflected(F(max(0,anchor["speed_mmps"]-250)),[max(F(-4000),c-1000)*dt for dt,c in future]))
  high.append(reflected(F(anchor["speed_mmps"]+250),[min(F(2000),c+1000)*dt for dt,c in future]))
 l,h=min(low),max(high)
 center=round((l+h)/2);fits=lambda v:h-500<=v<=l+500
 if candidate is not None and fits(candidate):return candidate,"candidate","candidate_inside_reachable_budget",l,h
 if fits(center):return center,"propagated_anchor","propagated_anchor_inside_budget",l,h
 return None,"abstain","budget_exhausted",l,h
def expected(row,state):
 cs=as_=hs="not_checked";value=None;selection="abstain";detail=None;clock=row["clock_us"];risk=row["risk"]
 if risk=="lockdown":reason="lockdown_context"
 elif not row["execution_assumption"]:reason="execution_assumption_not_supplied"
 else:
  history,hs=authenticate(row["packets"]["command"],"command",row,state)
  if history is None:reason="command_"+hs
  else:
   candidate,cs=authenticate(row["packets"]["candidate"],"candidate",row,state)
   anchor,as_=authenticate(row["packets"]["anchor"],"anchor",row,state)
   if anchor is None:reason="anchor_"+as_
   elif not 0<=anchor["measured_us"]-clock<=row["now_us"]<=2000000:reason="anchor_calibration_outside_interval"
   else:
    v=candidate["speed_mmps"] if candidate is not None and risk=="normal" else None
    value,selection,reason,low,high=interval_choice(v,anchor,history["commands"],row["now_us"],clock)
    detail=dict(conditional_budget_satisfied=value is not None,low_n=low.numerator if low is not None else None,low_d=low.denominator if low is not None else None,
     high_n=high.numerator if high is not None else None,high_d=high.denominator if high is not None else None)
 out=dict(selection=selection,reason=reason,research_speed_mmps=value,research_speed_mps=value/1000 if value is not None else None,
  candidate_status=cs,anchor_status=as_,risk_context=risk,research_simulation_only=True,physical_speed_authorized=False,
  measurement_truth_established=False,independent_source_trust_established=False,physical_witness_enabled=False,production_policy_changed=False,promotion_allowed=False,
  command_status=hs,conditional_budget_satisfied=False,anchor_clock_uncertainty_us=clock,execution_error_mmps2=1000,physical_validation_established=False)
 if detail is not None:out.update(detail)
 return out
def rows(path):
 with gzip.open(path,"rt",encoding="utf-8") as f:
  for line in f:yield json.loads(line)

def check_construction(row,previous_packets):
 case=row["case"];step=row["step"];active=step>=8;clock=row["clock_us"];age=row["nominal_age_us"]
 now0=1000000+step*1000;actual_age=100000 if active and case=="anchor_clock_window" else age
 for kind,packet in row["packets"].items():
  absent=active and ((kind=="command" and case=="command_missing") or (kind=="candidate" and case in ("candidate_missing","signed_command_history_bias","execution_bound_violated")))
  assert (packet is None)==absent
  if absent:continue
  repeated=active and ((case=="database_snapshot_rollback") or (kind=="command" and case in ("command_replay","restart_replay")) or (kind!="command" and case=="report_replay"))
  if repeated:
   assert packet==previous_packets[kind];continue
  obj=json.loads(packet["wire"]);b=obj["body"];fields=CF if kind=="command" else RF
  assert packet["received_us"]==now0
  assert b["seq"]==step+1+(1000 if active and kind=="command" and case=="command_bad_tag" else 0)
  assert b["source_id"]==("command_logger" if kind=="command" else kind)
  for k,v in row["context"].items():assert b[k]==("wrong" if active and kind=="command" and case=="command_context_mismatch" and k=="session_id" else v)
  if kind=="command":
   assert b["key_id"]==("unknown" if active and case=="command_unknown_key" else "h-key")
   assert b["issued_us"]==now0-(100001 if active and case=="command_old_issue" else 0)
   n=2000 if active and case in ("signed_command_history_bias","execution_bound_violated") else 0
   start=now0-actual_age-clock-(100001 if active and case=="command_old_issue" else 0)
   end=b["issued_us"]
   wanted=[[start,now0-10000,0,1],[now0-9000,now0,0,1]] if active and case=="command_gap" else [[start,end,n,1]]
   assert b["commands"]==wanted
  else:
   wanted_speed=13000 if active and ((kind=="candidate" and case in ("candidate_plus3","candidate_bad_tag","signed_joint_source_bias")) or (kind=="anchor" and case in ("anchor_bad_tag","signed_joint_source_bias"))) else (10500 if kind=="candidate" else 10000)
   assert b["speed_mmps"]==wanted_speed
   assert b["measured_us"]==(now0 if kind=="candidate" else now0-(100001 if active and case=="anchor_stale" else actual_age))
  source="command_logger" if kind=="command" else kind
  prefix=("COACTCPS-COMMAND-HMAC-SHA256-v1" if kind=="command" else "COACTCPS-EVIDENCE-HMAC-SHA256-v1").encode("ascii")+bytes([0])
  payload=prefix+json.dumps([b[k] for k in fields],ensure_ascii=True,separators=(",",":")).encode("ascii")
  good=hmac.compare_digest(hmac.new(keys[source],payload,hashlib.sha256).hexdigest(),obj["tag"])
  bad=active and ((kind=="command" and case in ("command_bad_tag","command_gap")) or (kind=="candidate" and case=="candidate_bad_tag") or (kind=="anchor" and case=="anchor_bad_tag"))
  assert good is (not bad),(row["condition"],step,case,kind,good,bad,repr(prefix))

before=iter(rows(R/"pre_metadata_correction"/"decisions.jsonl.gz"))
counts=collections.defaultdict(collections.Counter);selections=collections.defaultdict(collections.Counter);changed_labels=checked=0
state=None;previous_packets=None
for row in rows(R/"decisions.jsonl.gz"):
 cid=row["condition"];case=row["case"];step=row["step"];active=step>=8
 check_construction(row,previous_packets)
 if step==7:previous_packets=row["packets"]
 design=list(itertools.product(protocol["cases"],protocol["contexts"],protocol["nominal_age_us"],protocol["anchor_clock_uncertainty_us"]))[cid]
 assert (case,row["risk"],row["nominal_age_us"],row["clock_us"])==design
 assert cid*16+step==checked and row["context"]==dict(vehicle_id="vehicle-A",session_id="session-A",capture_id=f"capture-{cid}")
 expected_now=1007000 if active and case=="database_snapshot_rollback" else 1000000+step*1000
 assert row["now_us"]==expected_now
 assert row["snapshot_restored"] is (active and case=="database_snapshot_rollback")
 assert row["replayed_evidence_event"] is (active and case in ("command_replay","report_replay","restart_replay","database_snapshot_rollback"))
 assert row["command_key_enabled"] is not(active and case=="command_revoked_key")
 assert row["execution_assumption"] is not(active and case=="unknown_execution_bound")
 if step==0 or row["snapshot_restored"]:state={"last":-1,"counters":{}}
 decision=expected(row,state);assert decision==row["decision"],(cid,step,decision,row["decision"])
 previous=next(before);assert previous["decision"]==decision and previous["packets"]==row["packets"]
 anchor_packet=row["packets"]["anchor"]
 anchor_ok=anchor_packet is not None and abs(json.loads(anchor_packet["wire"])["body"]["speed_mmps"]-10000)<=250
 integrity=not(active and case=="signed_command_history_bias");execution=not(active and case=="execution_bound_violated")
 truth=F(10000)-F(4000*(row["nominal_age_us"]+row["clock_us"]),1000000) if active and case in ("signed_command_history_bias","execution_bound_violated") else F(10000)
 ev=row["evaluation"];assert F(ev["true_speed_n"],ev["true_speed_d"])==truth
 for k,v in (("anchor_bound_holds",anchor_ok),("command_history_intact",integrity),("execution_bound_holds",execution),("model_holds",anchor_ok and integrity and execution)):assert ev[k] is v
 changed_labels+=previous["evaluation"]!=ev
 value=decision["research_speed_mmps"];error=None if value is None else abs(value-truth)
 assert ev["selected_error_n"]==(error.numerator if error is not None else None) and ev["selected_error_d"]==(error.denominator if error is not None else None)
 selections[cid][decision["selection"]]+=1;counts[cid]["queries"]+=1
 if error is not None:
  counts[cid]["selected"]+=1;counts[cid]["error_above500"]+=error>500;counts[cid]["error_above2000"]+=error>2000
  counts[cid]["conditional_error_above500"]+=error>500 and ev["model_holds"] and row["execution_assumption"]
  counts[cid]["replayed_evidence_outputs"]+=row["replayed_evidence_event"]
 checked+=1
assert next(before,None) is None and checked==4224 and changed_labels==96
for cell in result["conditions"]:
 cid=cell["condition"];assert cell["selections"]==dict(selections[cid])
 for k in ("queries","selected","error_above500","error_above2000","conditional_error_above500","replayed_evidence_outputs"):assert cell[k]==counts[cid][k],(cid,k)
assert all(c["conditional_error_above500"]==0 for c in result["conditions"])
assert all(sha(P/n)==h for n,h in lock.items())
assert all(result[k] is False for k in ("physical_validation_established","specific_novelty_established","promotion_allowed"))
report={"passed":True,"decisions_independently_verified":checked,"conditions":len(result["conditions"]),"source_identities_checked":len(lock),
 "prior_attempt_decisions_and_packets_unchanged":True,"evaluation_label_rows_corrected":changed_labels,
 "oracle":"Independent schema/HMAC/in-memory replay state and reverse-suffix scalar interval; no production imports",
 "physical_validation_established":False,"specific_novelty_established":False,"promotion_allowed":False}
write("verification.json",report);print(json.dumps(report,indent=2))
