"""Recorded-state development assessment, not revised closed-loop confirmation."""
from pathlib import Path
import json,gzip,hashlib,math,collections
from ambiguity_guard import corroborate
R=Path(__file__).resolve().parent;P=R.parent.parent;B=P/"research/vehicle_comparative_validation"
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
result=json.loads((B/"results.json").read_text());audit=json.loads((B/"verification.json").read_text())
assert audit["passed"] and audit["input_results_sha256"]==sha(B/"results.json")
groups={}
for t in result["trials"]:
 if t["method"]!="full":continue
 marker=json.loads(((B/t["trace"]).parent/"complete.json").read_text())
 assert marker["lock_digest"]==result["lock_digest"] and sha(B/t["trace"])==marker["files"]["trace.jsonl.gz"]
 key=(t["group"],t["case"]);a=groups.setdefault(key,{"trials":0,"queries":0,"original_errors":[],"guard_errors":[],"reasons":collections.Counter(),"original_bad_corrected":0,"original_bad_abstained":0,"original_good_made_bad":0})
 a["trials"]+=1
 with gzip.open(B/t["trace"],"rt") as f:
  for line in f:
   row=json.loads(line);o=row["output"];old=o["estimate_mmps"];base=o["estimator_decision"].get("base_decision",{})
   new,reason=corroborate(old,base);v=row["truth"]["ego_v"];a["queries"]+=1;a["reasons"][reason]+=1
   e=None if old is None else abs(old/1000-v);ne=None if new is None else abs(new/1000-v)
   if e is not None:a["original_errors"].append(e)
   if ne is not None:a["guard_errors"].append(ne)
   if e is not None and e>.5:
    if ne is None:a["original_bad_abstained"]+=1
    elif ne<=.5:a["original_bad_corrected"]+=1
   if e is not None and e<=.5 and ne is not None and ne>.5:a["original_good_made_bad"]+=1
rows=[]
for (group,case),a in sorted(groups.items()):
 item={"group":group,"case":case,**{k:v for k,v in a.items() if k not in ("original_errors","guard_errors")}}
 for prefix,field in [("original","original_errors"),("guard","guard_errors")]:
  es=a[field];item.update({prefix+"_selected":len(es),prefix+"_mae_mps":math.fsum(es)/len(es) if es else None,prefix+"_over_budget":sum(e>.5 for e in es),prefix+"_availability":len(es)/a["queries"]})
 rows.append(item)
out={"scope":"Offline development assessment on recorded original full-policy states. No revised control commands or trajectories were executed; contact improvement not measured.","results_sha256":sha(B/"results.json"),"guard_sha256":sha(R/"ambiguity_guard.py"),"full_policy_trials":sum(a["trials"] for a in rows),"queries":sum(a["queries"] for a in rows),"cases":rows,"limitations":["Uses already consumed confirmation data for development; no new-method heldout performance claim.","No evaluator truth is passed into the guard; truth is used only for scoring.","Base-certificate premises can fail under public sensor innovations and execution break; the guard does not prove source honesty.","Availability changes and remaining errors are retained.","Shared false range and closed-loop contact outcomes remain untested for this prototype."]}
(R/"ambiguity_offline_assessment.json").write_text(json.dumps(out,indent=2)+"\n")
print("OFFLINE_AMBIGUITY_ASSESSMENT",json.dumps(out),flush=True)
