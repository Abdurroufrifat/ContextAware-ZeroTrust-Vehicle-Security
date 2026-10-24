"""Build comparison artifacts only from verified saved confirmation results."""
from cv_common import B,P,OLD,sha,dump
import json,csv,math,gzip
from report_math import summarize,paired,cluster_effect,METRICS
from verification_contracts import verify_checkpoint
CFG=json.loads((B/"PROTOCOL.json").read_text())
PAIRS=[("full","no_learning","learning"),("no_learning","no_refinement","successful_refinement"),
 ("full","no_fusion","context_fusion"),("full","no_brake","braking_filter"),("full","no_actuator_auth","authenticated_actuation"),
 ("full","s3e_brake","published_S3E_adaptation"),("full","ssf_zocbf","published_SSF_adaptation")]
def table(rows,columns):
 return "| "+" | ".join(columns)+" |\n| "+" | ".join("---" for _ in columns)+" |\n"+"\n".join("| "+" | ".join(str(r.get(c,"")) for c in columns)+" |" for r in rows)
def fmt(v):
 if v is None:return "unavailable"
 return f"{v:.6f}" if isinstance(v,float) else str(v)
def main():
 result=json.loads((B/"results.json").read_text());trials=result["trials"]
 verification=json.loads((B/"verification.json").read_text());replay=json.loads((B/"policy_replay.json").read_text())
 assert not result["pilot"] and len(trials)==1512 and verification["passed"] and verification["counts"]["trials"]==1512
 assert replay["passed"] and replay["trials"]==1512 and replay["decisions"]==302400
 assert result["lock_digest"]==replay["lock_digest"]==sha(B/"experiment_lock.json")
 assert all(sha(P/n)==h for n,h in json.loads((B/"experiment_lock.json").read_text()).items())
 assert verification["lock_digest"]==result["lock_digest"]
 assert verification["input_results_sha256"]==replay["input_results_sha256"]==sha(B/"results.json")
 assert all(sha(B/n)==h for evidence in (verification,replay) for n,h in evidence["evidence_engine"].items())
 by_id={x["trial_id"]:x for x in replay["per_trial"]}
 assert len(by_id)==1512
 for t in trials:
  verify_checkpoint(B,t,result["lock_digest"])
  assert by_id[t["trial_id"]]["trace_sha256"]==sha(B/t["trace"])
  assert by_id[t["trial_id"]]["evidence_engine"]==replay["evidence_engine"]
  assert by_id[t["trial_id"]]["passed"] and by_id[t["trial_id"]]["queries"]==t["queries"]
 aggregates=[]
 for group,cases,methods in [("primary",CFG["primary_cases"],CFG["methods"]),("stress",CFG["stress_cases"],CFG["stress_methods"])]:
  for case in [None,*cases]:
   for method in methods:
    rows=[t for t in trials if t["group"]==group and t["method"]==method and (case is None or t["case"]==case)]
    aggregates.append({"group":group,"case":case or "ALL","method":method,**summarize(rows)})
 effects=[];raw_pairs=[]
 for group,methods in [("primary",CFG["methods"]),("stress",CFG["stress_methods"])]:
  for left,right,component in PAIRS:
   if left not in methods or right not in methods:continue
   rows=paired(trials,left,right,group)
   raw_pairs.append({"group":group,"left":left,"right":right,"component":component,"matched_trials":len(rows),"effects":rows})
   for metric in [*METRICS,"availability"]:
    effects.append({"group":group,"component":component,"left":left,"right":right,"metric":metric,"matched_trials":len(rows),**cluster_effect(rows,metric)})
 dump(B/"comparison.json",{"input_results_sha256":sha(B/"results.json"),"verification_sha256":sha(B/"verification.json"),"policy_replay_sha256":sha(B/"policy_replay.json"),"aggregates":aggregates,"paired_effects":effects,"scope":"New scalar longitudinal numerical closed-loop benchmark; common-task adaptations, no native full-framework superiority claim."})
 dump(B/"paired_baseline_results.json",{"pairs":raw_pairs})
 dump(B/"seed_cluster_summary.json",{"seed_clusters":CFG["seeds"],"effects":effects,"resampling_unit":"seed; all profiles and cases retained in each cluster","ci_scope":"Exploratory; six clusters; no multiplicity-adjusted confirmatory inference."})
 for name,rows in [("aggregate_results.csv",aggregates),("method_effects.csv",[{k:(json.dumps(v) if isinstance(v,(dict,list)) else v) for k,v in r.items()} for r in effects])]:
  with (B/name).open("w",newline="",encoding="utf-8") as f:
   writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 primary=[r for r in aggregates if r["group"]=="primary" and r["case"]=="ALL"]
 stress=[r for r in aggregates if r["group"]=="stress" and r["case"]=="ALL"]
 def display(rows):
  return [{"method":r["method"],"trials":r["trials"],"contacts":r["collision_trials"],"selected MAE (m/s)":fmt(r["selected_mae_mps"]),"availability":fmt(r["availability"]),"mean progress (m)":fmt(r["mean_progress_m"]),"fallbacks":r["fallback_queries"]} for r in rows]
 columns=["method","trials","contacts","selected MAE (m/s)","availability","mean progress (m)","fallbacks"]
 effectrows=[{"comparison":r["left"]+" - "+r["right"],"metric":r["metric"],"difference":fmt(r["mean_left_minus_right"]),"six-seed interval":str([round(x,6) for x in r["descriptive_cluster_bootstrap_95"]])} for r in effects if r["group"]=="primary" and r["metric"] in ("collision_trial","mae_mps","progress_m")]
 counts=verification["counts"]
 text="# CO-ACT-CPS comparative validation\n\n"
 text+="All 1,512 new confirmation trials are saved, independently checked and replayed from strict signed input packets. This adds 302,400 decisions to the preserved 540-trial predecessor. No predecessor trial was rerun and no model was trained on confirmation outcomes.\n\n"
 text+="## Primary common-task comparison\n\n"+table(display(primary),columns)+"\n\n"
 text+="Each method has 144 matched scenario-seed trials: three new lead profiles, eight conditions, six frozen seeds. All methods use the same plant, exogenous source-noise streams, actuator envelope and query schedule. Their closed-loop trajectories and later observations can differ because their actions differ. Selected MAE includes only available estimates; compare it together with availability, contact and progress.\n\n"
 text+="## Additional public-data-informed stresses\n\n"+table(display(stress),columns)+"\n\n"
 text+="The four stress methods have 90 trials each. Conditions are public command-receipt age, public motion innovation, burst source loss, reduced braking and execution break. These deliberately challenging conditions are not interchangeable with bounded nominal assumptions. Per-case CSV rows retain failures and missing estimates; unequal method populations are not combined.\n\n"
 text+="## Matched component and method effects\n\n"+table(effectrows,["comparison","metric","difference","six-seed interval"])+"\n\n"
 text+="Differences are left minus right on matched profile/condition/seed worlds. Lower contact rate and error are favorable; greater progress and availability are favorable. A favorable error difference alone is not an overall win. Intervals resample six seed clusters, retaining their cases and profiles. They are exploratory descriptive uncertainty, not independent-query evidence or proof of universal superiority.\n\n"
 text+="Refinement compares no_learning against no_refinement: successful refined-interval selection is replaced by a budget-projected midpoint of the original scene interval. The same fallback algorithm is retained when refinement is unavailable or conflicting. This is an eligibility-conditioned interval ablation, not deletion of the whole module. Eligible queries: "+str(counts["refinement_eligible_queries"])+". Learning compares full against no_learning; fusion, braking and actuation authentication compare full against their named ablations.\n\n"
 text+="## Saved-artifact verification\n\n"+"\n".join("    "+line for line in json.dumps(counts,indent=2).splitlines())+"\n\n"
 text+="The independent verifier checks every numerical transition, protocol identity, complete checkpoint hashes, prescribed source attacks, public heldout split, command-log MAC and accepted-command coverage, interval/filter arithmetic, manual forest outputs and aggregate metrics. A separate fresh-state replay compares every recorded policy decision, issued/accepted command and operator card to the strict packet-only implementation. Replay uses the same algorithm and is separate from independent mathematics.\n\n"
 text+="Conditional qualification uses actual sensor-set containment and stated execution bounds. All context violations, abstentions and contacts remain in overall results; qualified subsets do not erase failures outside those assumptions. Zero qualified violations is a finite benchmark result, not a universal safety theorem.\n\n"
 text+="## Literature and novelty boundary\n\n"
 text+="S3E and secure-safety-filter baselines are transparent common-task adaptations documented in BASELINE_CONTRACT.md. S3E retains disconnected agreement sets and temporal accepted-command prediction; SSF uses current plausible sets and a 10ms finite-grid, zero-order-held headway filter. These are not complete native reproductions of the papers' original systems. RAIL, GUARDIAN and software-defined-vehicle zero trust are conceptual neighbors with different tasks and assumptions. No ranking against their native systems is claimed.\n\n"
 text+="The contribution to assess is the combined signed-source/context-to-command pipeline, explicit assumption-aware refinement, bounded learned selection, independently audited closed-loop attacks and component evidence. Novelty depends on measured component effects and differences from cited methods. No single improved metric or implementation count establishes broad originality.\n\n"
 text+="## Data and simulation scope\n\n"
 text+="Public provenance: TU/e F1TENTH, DOI 10.5281/zenodo.12536536. Easy defines calibration summaries; medium and hard are heldout stress sequences. Command receipt age is a staleness proxy, not measured network delay. Motion second difference includes motion/reference artifacts, not calibrated sensor noise. Large tails and missing values are retained. No protected HeRCULES Library03, Urban02 NIGHT, RAV4 or vehicle_final_evaluation data were opened.\n\n"
 text+="New trials use a 10ms numerical longitudinal plant. Full 20s, all-profile parity against SUMO 1.27.1 matched speed and position exactly under the tested actuator contract. These are not 1,512 native SUMO socket runs. A contact is trajectory gap <=0, not a SUMO warning. Timing is observed wall time under concurrent CPU execution, not a hard real-time guarantee.\n\n"
 text+="The operator viewer replays recorded observation-derived cards with reason, source authentication, control margin and accepted command. It has no evaluator truth input and no human-subject evaluation. The study addresses trustworthy AI, cybersecurity, CPS and explanatory operator support within simulation. It does not establish human effectiveness or cover every call keyword. Journal acceptance and full special-issue compliance require editorial assessment.\n"
 (B/"RESULTS.md").write_text(text,encoding="utf-8")
 baseline="""# Published-method comparison scope

| Work | Primary source | Relation to this study |
| --- | --- | --- |
| S3E, Niazi et al., 2026 version | https://arxiv.org/html/2309.05075v4 | Scalar interval-union adaptation of prediction and p-q agreement; q=1, signed adapter, accepted-command temporal propagation; disconnected components retained. |
| Tan, Ong, Tabuada and Ames, CDC2025, Secure Safety Filter | https://arxiv.org/html/2505.06842v1 | Current plausible-state reconstruction plus least-change finite-grid headway ZO-CBF adaptation. No transfer of the nonlinear/unicycle theorem. DOI 10.1109/CDC57313.2025.11312949. |
| GUARDIAN, 2026 | https://arxiv.org/html/2602.06026v2 | Conceptual: neural verification and Hamilton-Jacobi safety differ from this frozen forest selector and longitudinal task. No native reproduction/ranking. |
| RAIL, VehicleSec2026 | https://www.usenix.org/conference/vehiclesec26/presentation/wasif-rail | Conceptual: interpretable risk, shield and human interaction in MetaDrive/CARLA. No native bandit/human study reproduction. |
| Khan et al., context-aware zero trust for software-defined vehicles, 2025 | https://doi.org/10.1016/j.adhoc.2024.103694 | Context-aware zero trust already exists; novelty cannot be the phrase alone. No unsupported native numerical comparison. |

Eight primary arms share plant, source adapter, uncertainty envelope, timing, actuator limits and exogenous random streams. S3E shares the proposed braking projection; SSF uses its declared filter. No hidden ground truth is supplied to policies. Radar is not counted as an independent ego-speed source because its conversion shares scene lead context. Initial speed set [0,30] is common prior plant information.

Performance, failures, availability and progress are in RESULTS.md and aggregate_results.csv. Effects are matched closed-loop tasks, not identical-state estimator errors. Shared disinformation and execution stresses test assumption failure. No universal robustness or complete native state-of-the-art reproduction is claimed.
"""
 (B/"BASELINE_COMPARISON.md").write_text(baseline,encoding="utf-8")
 import matplotlib
 matplotlib.use("Agg")
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,3,figsize=(15,5.4))
 for ax,metric,label in zip(axes,["collision_trials","selected_mae_mps","mean_progress_m"],["Contact trials / 144","Selected-estimate MAE (m/s)","Mean vehicle progress (m)"]):
  ax.barh([r["method"] for r in primary],[float("nan") if r[metric] is None else r[metric] for r in primary],color=["#126B78" if r["method"]=="full" else "#8DA7AF" for r in primary])
  ax.set_xlabel(label);ax.invert_yaxis();ax.grid(axis="x",alpha=.2);ax.set_axisbelow(True)
 fig.suptitle("Frozen common-task closed-loop comparison",fontsize=15)
 fig.text(.5,.015,"144 trials per arm; selected MAE must be interpreted with availability, contact and progress.",ha="center",fontsize=9)
 fig.tight_layout(rect=[0,.05,1,.95]);fig.savefig(B/"comparative_results.png",dpi=180);plt.close(fig)
 cards=[]
 for case in CFG["primary_cases"]+CFG["stress_cases"]:
  t=next(t for t in trials if t["method"]=="full" and t["profile"]=="urban_cutback" and t["seed"]==CFG["seeds"][0] and t["case"]==case)
  with gzip.open(B/t["trace"],"rt",encoding="utf-8") as f:rows=[json.loads(x) for x in f]
  cards.append({"case":case,"profile":t["profile"],"seed":t["seed"],"trial_id":t["trial_id"],"cards":[r["operator_card"] for r in rows]})
 dump(B/"operator_replays.json",{"scope":"Recorded full-policy cards; observation-derived; no human evaluation","replays":cards})
 payload=json.dumps(cards,separators=(",",":")).replace("<","\\u003c")
 (B/"operator_viewer.html").write_text((B/"operator_template.html").read_text(encoding="utf-8").replace("__PAYLOAD__",payload),encoding="utf-8")
 dump(B/"assembly_status.json",{"status":"AWAITING_REPORT_QA","trials":1512,"decisions":302400,"original_trials_preserved":540,"verification":True,"strict_policy_replay":True,"manuscript_written":False,"native_full_framework_superiority_claim":False,"human_evaluation":False,"results_sha256":sha(B/"results.json"),"experiment_lock_sha256":sha(B/"experiment_lock.json"),"limits":["scalar longitudinal numerical simulation","public-data proxies, not calibrated hard bounds","published-method adaptations","six-seed exploratory uncertainty","editorial novelty/acceptance not certified"]})
 print("COMPARATIVE_REPORT_BUILT",len(aggregates),len(effects),sum(len(c["cards"]) for c in cards),flush=True)
if __name__=="__main__":main()
