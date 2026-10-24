"""Secondary read-only analysis. No experiment launch, policy change or manuscript."""
from pathlib import Path
import json,gzip,hashlib,csv,math,statistics,random
from assurance_core import *
S=Path(__file__).resolve().parent;P=S.parent.parent;R=P/"research/vehicle_evidence_lifecycle";B=R/"confirmation_v2"
load=lambda p:json.loads(p.read_text(encoding="utf-8"))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def dump(n,x):(S/n).write_text(json.dumps(x,indent=2),encoding="utf-8")
def main():
 assert not (S/"analysis.json").exists(),"Saved analysis exists; do not repeat."
 seal=load(R/"final_seal.json");assert seal["status"]=="COMPLETE_FOR_LIFECYCLE_SIMULATION_SCOPE"
 manifest=load(R/"package_manifest.json")["files"];inputs={}
 def bind(p):
  rel=str(p.relative_to(P));h=sha(p)
  if rel in manifest:assert manifest[rel]==h,rel
  inputs[rel]=h
 for p in [R/"final_seal.json",R/"package_manifest.json",R/"confirmation_v2_lock.json",B/"results.json",B/"verification.json",B/"comparison.json",B/"operator_replays.json",R/"PROTOCOL_V2.json",R/"harness_engine_v2.py"]:bind(p)
 data=load(B/"results.json");audit=load(B/"verification.json");cfg=load(R/"PROTOCOL_V2.json")
 assert audit["passed"] and seal["results_sha256"]==audit["results_sha256"]==sha(B/"results.json")
 rows=[];times={m:[] for m in cfg["methods"]};warm={m:[] for m in cfg["methods"]};cold={m:[] for m in cfg["methods"]}
 for t in data["trials"]:
  path=B/t["trace"];bind(path);r={**t,"point_certified":0,"command_accepted":0,"accepted_with_uncertified_point":0,"over_100ms":0};n=0
  with gzip.open(path,"rt") as f:
   for line in f:
    x=json.loads(line);c=x["operator_card"];v=facts(c);sec=x["decision_wall_seconds"];assert math.isfinite(sec) and sec>=0
    assert c["point_budget_satisfied"]==x["output"]["estimator_decision"]["point_budget_satisfied"]
    assert c["actuator_status"]==x["actuator_status"] and c["accepted_command_mmps2"]==x["accepted_mmps2"]
    r["point_certified"]+=v["speed_budget_certified"];r["command_accepted"]+=v["command_gate_accepted"]
    r["accepted_with_uncertified_point"]+=v["command_gate_accepted"] and not v["speed_budget_certified"]
    r["over_100ms"]+=sec>.1;times[t["method"]].append(sec);(cold if x["step"]==0 else warm)[t["method"]].append(sec);n+=1
  assert n==t["queries"]==200
  rows.append(r)
 assert len(rows)==1080
 aggregates=[]
 for scope in ["all","main","negative_controls"]:
  for profile in ["ALL",*cfg["profiles"]]:
   for method in cfg["methods"]:
    chosen=[t for t in rows if t["method"]==method and (profile=="ALL" or t["profile"]==profile) and (scope=="all" or (t["case"] not in NEGATIVE if scope=="main" else t["case"] in NEGATIVE))]
    aggregates.append({"scope":scope,"profile":profile,"method":method,**summary(chosen)})
 timing=[]
 for m in cfg["methods"]:
  for k,collection in [("all",times),("first_query",cold),("later_queries",warm)]:
   a=collection[m];timing.append({"method":m,"scope":k,"queries":len(a),"p50_ms":quantile(a,.5)*1000,"p95_ms":quantile(a,.95)*1000,"p99_ms":quantile(a,.99)*1000,"max_ms":max(a)*1000,"over_100ms":sum(x>.1 for x in a)})
 effects=[]
 for scope in ["all","main","negative_controls"]:
  for opponent in [m for m in cfg["methods"] if m!="joint"]:
   for metric in ["contact_fraction","mean_progress_m"]:
    effects_by_seed=[]
    for seed in cfg["seeds"]:
     chosen=[t for t in rows if t["seed"]==seed and (scope=="all" or (t["case"] not in NEGATIVE if scope=="main" else t["case"] in NEGATIVE))]
     a=summary([t for t in chosen if t["method"]=="joint"]);b=summary([t for t in chosen if t["method"]==opponent]);effects_by_seed.append(a[metric]-b[metric])
    rng=random.Random(8297);draws=sorted(statistics.mean(rng.choice(effects_by_seed) for _ in effects_by_seed) for _ in range(2000))
    effects.append({"scope":scope,"right":opponent,"metric":metric,"left":"joint","seed_clusters":6,"mean_left_minus_right":statistics.mean(effects_by_seed),"descriptive_95_interval":[draws[49],draws[1950]],"seed_effects":effects_by_seed})
 allrows=[a for a in aggregates if a["scope"]=="all" and a["profile"]=="ALL"]
 pareto=[a["method"] for a in allrows if not any(dominates(b,a) for b in allrows)]
 analysis={"scope":"Secondary analysis after confirmation; no tuning, trial exclusion or rerun. Main and negative controls use predeclared case labels. Six-seed intervals are descriptive, not query-independent inference.","timing_scope":"Recorded controller processing, command construction and actuator checking on a shared three-worker Windows run. Excludes subsequent trace writing and physics advancement. Offline simulated timestamps did not advance by wall time; no end-to-end deadline or real-time safety claim.","trials":len(rows),"queries":sum(t["queries"] for t in rows),"aggregates":aggregates,"timing":timing,"effects":effects,"pareto_methods":pareto}
 dump("analysis.json",analysis);dump("per_trial.json",rows);dump("input_manifest.json",inputs)
 for name,data_rows in [("coverage_and_tradeoffs.csv",aggregates),("timing.csv",timing)]:
  with (S/name).open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=list(data_rows[0]));w.writeheader();w.writerows(data_rows)
 replays=load(B/"operator_replays.json");payload=[]
 for r in replays:
  payload.append({**{k:v for k,v in r.items() if k!="cards"},"cards":[{"source_card":c,"explanation":facts(c)} for c in r["cards"]]})
 dump("operator_evidence.json",payload)
 html=(S/"viewer_template.html").read_text(encoding="utf-8").replace("__PAYLOAD__",json.dumps(payload,separators=(",",":")).replace("<","\\u003c"))
 (S/"operator_assurance_viewer.html").write_text(html,encoding="utf-8")
 text="# CO-ACT-CPS submission assurance supplement\n\nSecondary analysis of the frozen 1080-trial release. No experiments repeated, policy changed or manuscript written.\n\n"
 text+="| Method | Contacts /216 | Mean progress m | Certified point coverage | Observed loops >100ms |\n|---|---:|---:|---:|---:|\n"
 for a in allrows:text+=f"| {a['method']} | {a['contacts']} | {a['mean_progress_m']:.3f} | {a['point_budget_coverage']:.2%} | {a['over_100ms_fraction']:.2%} |\n"
 text+="\nA numeric estimate is not necessarily within its stated point budget. Command authorization uses the retained state set and is separate from point certification. The assurance viewer makes both statuses explicit and always discloses source truth and bootstrap limitations.\n\n"
 text+="The measured contact/progress tradeoff is not universal dominance. Always-brake has no contacts and little motion. Current-context control moves farther and has more contacts. Learning improves speed error but does not reduce the contact count relative to no-learning. These are closed-loop policy comparisons with common exogenous scenarios, not identical-state estimation experiments.\n\n"
 text+="Pareto methods for the two aggregate empirical objectives (fewer contacts, greater mean progress): "+", ".join(pareto)+". This is descriptive over the fixed evaluated policies; it is not an equal-progress comparison or an optimized Pareto frontier. No ratio of contacts to distance is used because it could reward unsafe progress.\n\n"
 text+="## Timing evidence\n\n"+analysis["timing_scope"]+"\n\n"
 text+="| Method | p50 ms | p95 ms | p99 ms | Maximum ms |\n|---|---:|---:|---:|---:|\n"
 for t in timing:
  if t["scope"]=="all":text+=f"| {t['method']} | {t['p50_ms']:.3f} | {t['p95_ms']:.3f} | {t['p99_ms']:.3f} | {t['max_ms']:.3f} |\n"
 text+="\nRuntime overruns are retained. These recordings do not prove a 100ms real-time deadline. A deployment claim would require an execution platform and latency-aware closed-loop qualification.\n\n## Fit with the requested call\n\n"
 text+="The publisher's indexed title confirms Human-Centered and Trustworthy AI for Cybersecurity in Cyber-Physical Systems; the full publisher page returned403 on2026-10-24. This supplement supports the user's supplied themes through attack-resilient closed-loop simulation, explicit uncertainty/assumptions and observation-derived operator explanations. Full call compliance is not certified. The operator component has technical fidelity and navigation checks; no participant study, trust improvement, workload reduction or human-effectiveness result is claimed.\n\n"
 text+="Primary framing source: He et al.,2021, The Challenges and Opportunities of Human-Centered AI for Trustworthy Robots and Autonomous Systems, https://arxiv.org/abs/2105.04408 . Its abstract discusses safety, security and usable interaction; it does not validate our viewer. Call: https://www.sciencedirect.com/special-issue/335890/human-centered-and-trustworthy-ai-for-cybersecurity-in-cyber-physical-systems .\n\n"
 text+="The candidate novelty remains the specific context-history/execution-envelope/command-lifetime contract and its failure analysis. This supplement strengthens inspectability and comparison, and introduces no first-ever or native-framework superiority claim.\n"
 (S/"ASSURANCE_REPORT.md").write_text(text,encoding="utf-8")
 import subprocess,sys
 subprocess.run([sys.executable,str(S/'make_figures.py')],check=True)
 print('ASSURANCE_BUILD_COMPLETE',len(rows),sum(t['queries'] for t in rows),flush=True)
if __name__=='__main__':main()
