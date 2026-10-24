"""Read-only diagnosis of frozen comparison traces; no policy or experiment changes."""
from pathlib import Path
from fractions import Fraction
import hashlib,json,gzip,math,collections
R=Path(__file__).resolve().parent;P=R.parent.parent;B=P/"research/vehicle_comparative_validation"
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def bounds(d,prefix):
 try:return tuple(Fraction(d[prefix+x+"_n"],d[prefix+x+"_d"]) for x in ("low","high"))
 except (KeyError,TypeError,ZeroDivisionError):return None
def stats(ts):
 n=sum(t["selected"] for t in ts)
 return {"trials":len(ts),"selected":n,"queries":sum(t["queries"] for t in ts),"selected_mae_mps":math.fsum(t["mae_mps"]*t["selected"] for t in ts if t["selected"])/n if n else None,"availability":n/sum(t["queries"] for t in ts),"over_budget":sum(t["over_0_5"] for t in ts),"contact_trials":sum(t["collision_trial"] for t in ts)}
def main():
 result=json.loads((B/"results.json").read_text());audit=json.loads((B/"verification.json").read_text())
 assert len(result["trials"])==1512 and audit["passed"] and audit["input_results_sha256"]==sha(B/"results.json")
 assert result["lock_digest"]==audit["lock_digest"]==sha(B/"experiment_lock.json")
 frozen=json.loads((B/"experiment_lock.json").read_text())
 assert all(sha(P/n)==h for n,h in frozen.items())
 full=[t for t in result["trials"] if t["group"]=="primary" and t["method"]=="full"];assert len(full)==144
 counts={};examples=[];trial_rows=[]
 for t in full:
  marker=json.loads(((B/t["trace"]).parent/"complete.json").read_text())
  assert all(sha((B/t["trace"]).parent/n)==h for n,h in marker["files"].items())
  with gzip.open(B/t["trace"],"rt",encoding="utf-8") as f:rows=[json.loads(x) for x in f]
  with gzip.open(B/t["trajectory"],"rt",encoding="utf-8") as f:trajectory=[json.loads(x) for x in f]
  contact=next((s["time_us"] for s in trajectory if s["lead_x"]-s["ego_x"]-5<=0),None)
  c=collections.Counter();errors=[];reasons=collections.Counter()
  for row in rows:
   c["queries"]+=1;o=row["output"];ed=o["estimator_decision"];base=ed.get("base_decision",{})
   v=o["estimate_mmps"];bv=base.get("research_speed_mmps");truth=row["truth"]["ego_v"]
   err=None if v is None else abs(v/1000-truth)
   assert err==row["selected_error_mps"]
   if err is None:continue
   errors.append(err);c["selected"]+=1
   pre=contact is None or round(row["time_s"]*1e6)<contact
   c["selected_pre_contact" if pre else "selected_after_contact"]+=1
   if err<=.5:continue
   c["bad"]+=1;c["bad_pre_contact" if pre else "bad_after_contact"]+=1;reasons[o["reason"]]+=1
   base_error=None if bv is None else abs(bv/1000-truth)
   if base_error is not None and base_error<=.5:c["bad_with_accurate_base"]+=1
   if base.get("conditional_budget_satisfied") is True:c["bad_with_base_certificate"]+=1
   bnd=bounds(base,"");ref=bounds(ed,"refined_")
   disjoint=bnd is not None and ref is not None and (bnd[1]<ref[0] or ref[1]<bnd[0])
   if disjoint:c["bad_with_disjoint_base_refinement"]+=1
   if ed.get("scene_decision",{}).get("reason")=="scene_witness_replacement":c["bad_scene_replacement"]+=1
   if o["controller_input"] is not None:c["bad_speed_admitted_to_controller"]+=1
   if pre and sum(e["case"]==t["case"] for e in examples)<2:
    examples.append({"case":t["case"],"trial_id":t["trial_id"],"step":row["step"],"time_s":row["time_s"],"truth_ego_mps":truth,"selected_mps":v/1000,"base_mps":None if bv is None else bv/1000,"base_error_mps":base_error,"base_certificate":base.get("conditional_budget_satisfied"),"base_interval_mmps":None if bnd is None else list(map(str,bnd)),"refined_interval_mmps":None if ref is None else list(map(str,ref)),"disjoint":disjoint,"controller_reason":o["reason"],"scene_authenticated":o["scene_status"]=="authenticated","trace":t["trace"],"trace_sha256":sha(B/t["trace"])})
  assert len(rows)==200 and c["selected"]==t["selected"] and c["bad"]==t["over_0_5"]
  assert math.isclose(math.fsum(errors)/len(errors),t["mae_mps"],abs_tol=1e-12)
  counts.setdefault(t["case"],collections.Counter()).update(c)
  trial_rows.append({"trial_id":t["trial_id"],"case":t["case"],"first_contact_time_us":contact,**c,"bad_controller_reasons":dict(reasons)})
 case_metrics={}
 for case in sorted(counts):
  case_metrics[case]={m:stats([t for t in result["trials"] if t["group"]=="primary" and t["case"]==case and t["method"]==m]) for m in ["full","no_learning","no_refinement","s3e_brake","ssf_zocbf"]}
 sources=["research/vehicle_kinematic_witness_study/scene_core.py","research/vehicle_command_scene_learning/command_scene_guard.py","research/vehicle_command_scene_learning/command_scene_core.py"]
 out={"scope":"Development diagnosis of existing confirmation evidence; no revised-method confirmation claim","results_sha256":sha(B/"results.json"),"audit_sha256":sha(B/"verification.json"),"experiment_lock_sha256":sha(B/"experiment_lock.json"),"full_primary_trials_checked":144,"query_count":sum(c["queries"] for c in counts.values()),"case_counts":counts,"case_metrics":case_metrics,"pre_contact_examples":examples,"per_trial":trial_rows,"source_bindings":{n:sha(P/n) for n in sources},"hypothesis":"Scene witness replacement can supersede a correct, conditionally qualified base estimate. Command refinement and learning then operate inside the scene-derived interval without corroborating the original anchor-command reachable interval.","limits":["Uses evaluator truth for diagnosis only; it cannot be supplied to a deployed guard.","Pre-contact and post-contact errors are separated; collision-clipped range can create large derived-speed artifacts.","The scene-source premise is false during these attacks; conditional mathematics is not disproved.","Shared-disinformation contact outcomes also affect the two published-method adaptations, so this analysis does not establish that estimator replacement caused those contacts.","These consumed results are development evidence for any future revision."]}
 (R/"DIAGNOSIS.json").write_text(json.dumps(out,indent=2)+"\n")
 text="# Frozen-study estimator failure diagnosis\n\n"
 text+="Checked 144 full-policy primary trials and 28,800 recorded queries against saved checkpoint hashes and the independent audit. No experiment was rerun or policy modified.\n\n"
 text+="The main error pattern is scene-derived witness replacement overriding a correct wheel/anchor base estimate. Refinement and the learned residual remain inside the replacement interval, so a small learned correction cannot repair an interval displaced by false scene evidence. Authentication establishes source identity, while scene measurement correctness remains an assumption.\n\n"
 text+="| Condition | Selected | Error >0.5 m/s | Before contact | After contact | Accurate base available |\n| --- | ---: | ---: | ---: | ---: | ---: |\n"
 for case,c in sorted(counts.items()):text+=f"| {case} | {c['selected']} | {c['bad']} | {c['bad_pre_contact']} | {c['bad_after_contact']} | {c['bad_with_accurate_base']} |\n"
 text+="\n## Concrete pre-contact evidence\n\n"
 for e in examples:text+=f"- {e['case']}, trial {e['trial_id']}, step {e['step']}: true speed {e['truth_ego_mps']:.5f} m/s; base {e['base_mps']:.3f}; final {e['selected_mps']:.3f}; base/refined intervals disjoint={e['disjoint']}; controller reason {e['controller_reason']}.\n"
 text+="\n## Interpretation\n\nThe full method has lower selected error than both adaptations in the nominal condition and several single-source conditions. Its aggregate disadvantage is concentrated in scene disinformation and shared disinformation. Report condition-specific performance with availability; do not remove these failed conditions to improve averages.\n\nAll 12 primary full-policy contact trials occur in shared disinformation. Both adaptations also have 12 primary contacts, so fixing speed estimation alone is not evidence that shared-gap collisions are solved. Post-contact range clipping must be distinguished from pre-contact estimator failures.\n\n"
 text+="## Next development decision\n\nTest a guard that checks whether scene-based selection contradicts the independently authenticated anchor/command reachable interval. Preserve useful learned estimates when corroborated; use a qualified base result or abstain when corroboration fails. Separately test whether implausible shared range changes can be detected from prior admitted context and motion bounds. Such checks require explicit initial-honesty and execution assumptions and cannot establish truth when every independent reference is coherently compromised.\n\nBefore integration, reproduce the saved counterexample in a regression test. Compare against midpoint, base-only and S3E choices, including genuine sensor disagreements and missing anchors. Any revised closed-loop method needs a new frozen protocol and fresh confirmation conditions. This study becomes development evidence for that revision. No human-effectiveness or new safety theorem claim follows from this report.\n"
 (R/"DIAGNOSIS.md").write_text(text)
 manifest={p.name:sha(p) for p in R.iterdir() if p.is_file() and p.name!="manifest.json"}
 (R/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
 print("FAILURE_DIAGNOSIS_COMPLETE",json.dumps(counts),flush=True)
if __name__=="__main__":main()
