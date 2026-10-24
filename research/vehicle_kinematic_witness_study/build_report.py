from kw_common import *
install_guard()
import gzip,itertools,csv,math,os
os.environ["MPLCONFIGDIR"]=str(R/"runtime"/"mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
result=json.loads((R/"results.json").read_text());v=json.loads((R/"verification.json").read_text());assert v["passed"]
app={x["trial_id"]:x for x in json.loads((R/"scene_appraisal.json").read_text())}
enriched=[]
for t in result["trials"]:
 with gzip.open(R/t["trace"],"rt",encoding="utf-8") as f:rows=[json.loads(x) for x in f]
 a=[x for x in rows if x["active"]];chosen=[x for x in a if x["decision"]["research_speed_mmps"] is not None]
 e={**t,**{k:x for k,x in app[t["trial_id"]].items() if k not in ("trial_id","case","age_us","seed","policy","queries")}}
 e.update(active_queries=len(a),active_selected=len(chosen),active_error_gt05=sum(x["selected_disagreement_mps"]>.5 for x in chosen),
  active_error_gt2=sum(x["selected_disagreement_mps"]>2 for x in chosen),
  post_window_selected=sum(x["time_s"]>=14 and x["decision"]["research_speed_mmps"] is not None for x in rows),
  latched_queries=sum(x["decision"].get("temporal_latched",False) for x in rows))
 enriched.append(e)
totals=("queries","selected","selected_error_gt05","selected_error_gt2","conditional_error_gt05","model_premise_violation_queries","collision_events",
 "active_queries","active_selected","active_error_gt05","active_error_gt2","post_window_selected","latched_queries",
 "qualifying_scene_intervals","scene_contract_holds_queries","reference_outside_witness","reference_outside_witness_with_valid_scene",
 "selected_with_valid_scene","selected_error_gt05_with_valid_scene","selected_error_gt2_with_valid_scene","witness_replacements")
aggregate=[]
for key,group in itertools.groupby(sorted(enriched,key=lambda x:(x["case"],x["age_us"],x["policy"])),key=lambda x:(x["case"],x["age_us"],x["policy"])):
 group=list(group);x=dict(zip(("case","age_us","policy"),key));x["seeds"]=len(group)
 for name in totals:x[name]=sum(t[name] for t in group)
 x.update(availability=x["selected"]/x["queries"],mean_progress_m=math.fsum(t["progress_m"] for t in group)/3,
  mean_speed_target_mae_mps=math.fsum(t["speed_target_mae_mps"] for t in group)/3,
  minimum_sampled_gap_m=min(t["minimum_sampled_gap_m"] for t in group))
 aggregate.append(x)
cfg=json.loads((R/"PROTOCOL.json").read_text());pairs=[]
for case,age,seed in itertools.product(cfg["cases"],cfg["ages_us"],cfg["seeds"]):
 a,b=[next(t for t in enriched if (t["case"],t["age_us"],t["seed"],t["policy"])==(case,age,seed,policy)) for policy in ("temporal","kinematic")]
 pairs.append(dict(case=case,age_us=age,seed=seed,temporal_trial_id=a["trial_id"],kinematic_trial_id=b["trial_id"],
  selected_difference=b["selected"]-a["selected"],error_gt05_difference=b["selected_error_gt05"]-a["selected_error_gt05"],
  progress_difference_m=b["progress_m"]-a["progress_m"]))
write("enriched_trials.json",enriched);write("aggregate_results.json",aggregate);write("paired_results.json",pairs)
for name,data in (("aggregate_results.csv",aggregate),("paired_results.csv",pairs)):
 with (R/name).open("w",newline="",encoding="utf-8") as f:
  w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
prior=json.loads((TEMPORAL/"results.json").read_text())["trials"];reproduced=0
metrics=("queries","selected","selected_error_gt05","selected_error_gt2","conditional_error_gt05","model_premise_violation_queries",
 "availability","selected_error_p95_mps","speed_target_mae_mps","progress_m","minimum_sampled_gap_m","collision_events")
for t in result["trials"]:
 if t["policy"]!="temporal" or t["case"].startswith("scene_"):continue
 old=next(x for x in prior if (x["case"],x["age_us"],x["seed"],x["policy"])==(t["case"],t["age_us"],t["seed"],"temporal"))
 for k in metrics:
  a,b=t[k],old[k]
  assert a is b if a is None or b is None else math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-8),(k,a,b)
 reproduced+=1
assert reproduced==54
write("baseline_reproduction.json",dict(passed=True,reproduced_conditions=54,metrics_compared=list(metrics),wall_time_not_compared=True))
def cell(case,age,policy):return next(x for x in aggregate if (x["case"],x["age_us"],x["policy"])==(case,age,policy))
fig,axes=plt.subplots(2,4,figsize=(14,6.2),sharex=True,sharey=True)
cases=("nominal","joint_minus3","joint_slow_drift","scene_common_compromise")
for row,age in enumerate((50000,90000)):
 for col,case in enumerate(cases):
  ax=axes[row,col]
  for i,policy in enumerate(("temporal","kinematic")):
   x=cell(case,age,policy);bad=x["selected_error_gt05"];good=x["selected"]-bad;abstain=600-x["selected"]
   ax.barh(i,good,color="#287c5c",label="Selected error <=0.5 m/s" if (row,col,i)==(0,0,0) else None)
   ax.barh(i,bad,left=good,color="#b7463d",label="Selected error >0.5 m/s" if (row,col,i)==(0,0,0) else None)
   ax.barh(i,abstain,left=good+bad,color="#dedede",label="Abstained" if (row,col,i)==(0,0,0) else None)
   ax.text(605,i,str(x["selected"])+"/600",va="center",fontsize=8)
  if row==0:ax.set_title(("Nominal","Abrupt wheel+anchor attack","Gradual wheel+anchor drift","Scene also compromised")[col],fontsize=10)
  ax.set_xlim(0,790);ax.set_xticks([0,300,600]);ax.set_yticks([0,1],["Temporal guard","Scene witness"])
  ax.spines[["top","right"]].set_visible(False)
  if col==0:ax.set_ylabel(str(age//1000)+" ms anchor age")
  if row==1:ax.set_xlabel("Queries")
axes[0,0].invert_yaxis();fig.suptitle("Fresh SUMO comparison with an authenticated kinematic witness",fontsize=14)
h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc="lower center",bbox_to_anchor=(.5,.065),ncol=3,frameon=False)
fig.text(.5,.025,"3 fixed noise seeds; 600 queries/cell. Witness correctness requires independently trustworthy gap, absolute lead speed and aligned timestamps.",ha="center",fontsize=9)
fig.subplots_adjust(left=.11,right=.99,top=.88,bottom=.22,hspace=.36,wspace=.14)
fig.savefig(R/"kinematic_witness_comparison.png",dpi=180);plt.close(fig)
kin=[x for x in aggregate if x["policy"]=="kinematic"];nominal=[x for x in kin if x["case"]=="nominal"]
assessment=dict(fresh_sumo_validation_completed=True,fresh_trials=144,fresh_decisions=28800,baseline_conditions_reproduced=54,
 new_assumed_trust_domain="query-aligned scene range and independently supplied absolute lead speed",
 nominal_availability_at_least_95_percent=all(x["availability"]>=.95 for x in nominal),
 nominal_selected_error_gt05=sum(x["selected_error_gt05"] for x in nominal),
 joint_abrupt_selected_error_gt05=sum(x["selected_error_gt05"] for x in kin if x["case"]=="joint_minus3"),
 joint_drift_selected_error_gt05=sum(x["selected_error_gt05"] for x in kin if x["case"]=="joint_slow_drift"),
 scene_common_compromise_selected_error_gt05=sum(x["selected_error_gt05"] for x in kin if x["case"]=="scene_common_compromise"),
 selected_with_valid_scene=sum(x["selected_with_valid_scene"] for x in kin),
 selected_error_gt05_with_valid_scene=sum(x["selected_error_gt05_with_valid_scene"] for x in kin),
 global_security_guarantee_established=False,physical_scene_independence_established=False,new_algorithm_principle_established=False,
 physical_vehicle_validation_established=False,human_effectiveness_established=False,production_promotion_allowed=False)
write("contribution_assessment.json",assessment)
tests=json.loads((R/"test_validation.json").read_text())
lines=["# Authenticated kinematic witness: completed simulation study","23 October 2026.","",
"Implemented a bounded scene witness using the gap and lead-speed inputs already supplied to the controller. Both old and new policies were run in a frozen 144-trial grid: 12 cases, two delays, three fixed noise seeds, 28,800 decisions. No ego-speed reference is supplied to the new estimator.","",
"This adds a separately authenticated scene domain and assumes its measurement bounds, independence from the tested ego-speed estimates and query-time alignment. It is not physical independent evidence. In a real vehicle, lead speed derived from the compromised ego speed could make the check circular. The gap/lead pair is one common trust domain. See CONTRACT.md.","",
"## Results","",
"Each row represents three fixed noise seeds and 600 queries. The two policies create different trajectories under otherwise matched scenarios.","",
"| Case | Age ms | Policy | Selected / 600 | Error >0.5 m/s | Error >2 m/s | Selected with valid scene | Errors with valid scene >0.5 | Mean progress m |",
"|---|---:|---|---:|---:|---:|---:|---:|---:|"]
for case in ("nominal","joint_minus3","joint_slow_drift","scene_bad_tag","scene_dropout","scene_common_compromise"):
 for age in (50000,90000):
  for policy in ("temporal","kinematic"):
   x=cell(case,age,policy);lines.append(f'| {case} | {age//1000} | {policy} | {x["selected"]} | {x["selected_error_gt05"]} | {x["selected_error_gt2"]} | {x["selected_with_valid_scene"]} | {x["selected_error_gt05_with_valid_scene"]} | {x["mean_progress_m"]:.3f} |')
lines+=["","For the old policy, valid scene data merely describe information it did not use; its errors in that column are not a violation of the new witness algorithm's contract. The new policy's selection can keep the original point or replace it with a scene-bounded midpoint. Original cryptographic/context/age abstentions remain abstentions. A bad/missing scene report cannot fall back to the two compromised speed channels.","",
"scene_common_compromise deliberately falsifies the new scene source together with the two speed sources. It is a retained counterexample to any universal-defense claim. Authentication does not establish sensor truth. The common-compromise gap offset persists after the active attack window; a constant offset can cancel in successive gap differences without satisfying the original absolute range-error assumption.","",
"All cases, active-window denominators, witness replacements, qualifying-input counts, sampled gaps, collisions, post-window selection and per-seed paired progress differences are in aggregate_results.csv and paired_results.csv. Zero accepted error with zero selections is abstention, not accurate control.","",
"## Verification","",
f'Independent replay reconstructs {v["controlled_decisions_reconstructed"]:,} decisions, {v["fcd_vehicle_states_matched"]:,} query vehicle states and {v["realized_speed_transitions_checked"]:,} ego execution transitions. An additional {v["scene_motion_and_geometry_transition_appraisals"]:,} two-vehicle transitions are appraised for scene motion/position assumptions.',
"All 54 repeated temporal baseline conditions reproduce prior numerical results. The verifier checks original HMAC/counter decisions, new scene HMAC/context/sequence/time, fixed sensor noise and common-compromise transformations, witness intervals, controller commands, raw FCD, collisions and summaries. It imports no production decision code.",
f'Tests: {tests["focused"]["counts"]["passed"]} focused passed; {tests["core"]["counts"]["passed"]} core passed; {tests["core"]["counts"].get("skipped",0)} core skipped. The previous legacy calibration warning remains.',
"",
"## Contribution and limits","",
"The demonstrated addition is an authenticated, error-bounded use of existing controller context for speed admission, with a directly tested trust-boundary counterexample. The kinematic identity, bounded propagation and HMAC are established ideas; a new estimation principle or learned AI model is not claimed. The previous scalar published-method comparison remains in vehicle_secure_set_comparison with its own assumptions and cohorts.",
"",
"Results are software-in-the-loop evidence. Scene association, independent absolute lead speed, physical range/clock/error bounds and physical command timing remain unvalidated. Machine-readable reasons are not evidence of human-centered effectiveness. No protected holdout, manuscript, production policy or physical vehicle operation is changed.",""]
(R/"README.md").write_text("\n".join(lines),encoding="utf-8")
print(json.dumps(assessment,indent=2))
