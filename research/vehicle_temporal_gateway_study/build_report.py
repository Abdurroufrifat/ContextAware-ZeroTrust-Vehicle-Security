from tg_common import *
install_guard()
import gzip,itertools,csv,math,os
os.environ["MPLCONFIGDIR"]=str(R/"runtime"/"mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
results=json.loads((R/"results.json").read_text());verification=json.loads((R/"verification.json").read_text())
assert verification["passed"]
trials=results["trials"];enriched=[]
for t in trials:
 with gzip.open(R/t["trace"],"rt",encoding="utf-8") as f:rows=[json.loads(x) for x in f]
 out=dict(t);active=[x for x in rows if x["active"]]
 selected=[x for x in active if x["decision"]["research_speed_mmps"] is not None]
 latched=[x for x in rows if x["decision"].get("temporal_latched",False)]
 out.update(active_queries=len(active),active_selected=len(selected),
  active_error_gt05=sum(x["selected_disagreement_mps"]>.5 for x in selected),
  active_error_gt2=sum(x["selected_disagreement_mps"]>2 for x in selected),
  latched_queries=len(latched),first_latch_time_s=latched[0]["time_s"] if latched else None,
  post_window_queries=sum(x["time_s"]>=14 for x in rows),
  post_window_selected=sum(x["time_s"]>=14 and x["decision"]["research_speed_mmps"] is not None for x in rows))
 enriched.append(out)
aggregate=[]
for key,group in itertools.groupby(sorted(enriched,key=lambda x:(x["case"],x["age_us"],x["policy"])),key=lambda x:(x["case"],x["age_us"],x["policy"])):
 group=list(group);out=dict(zip(("case","age_us","policy"),key));out["seeds"]=len(group)
 for name in ("queries","selected","selected_error_gt05","selected_error_gt2","conditional_error_gt05","model_premise_violation_queries",
  "collision_events","active_queries","active_selected","active_error_gt05","active_error_gt2","latched_queries","post_window_queries","post_window_selected"):
  out[name]=sum(x[name] for x in group)
 out.update(availability=out["selected"]/out["queries"],mean_progress_m=math.fsum(x["progress_m"] for x in group)/len(group),
  mean_speed_target_mae_mps=math.fsum(x["speed_target_mae_mps"] for x in group)/len(group),
  minimum_sampled_gap_m=min(x["minimum_sampled_gap_m"] for x in group))
 aggregate.append(out)
pairs=[]
for case,age,seed in itertools.product(json.loads((R/"PROTOCOL.json").read_text())["cases"],[50000,90000],[17,29,43]):
 a,b=[next(x for x in enriched if (x["case"],x["age_us"],x["seed"],x["policy"])==(case,age,seed,policy)) for policy in ("tube","temporal")]
 pairs.append(dict(case=case,age_us=age,seed=seed,selected_difference=b["selected"]-a["selected"],
  error_gt05_difference=b["selected_error_gt05"]-a["selected_error_gt05"],
  progress_difference_m=b["progress_m"]-a["progress_m"],temporal_first_latch_time_s=b["first_latch_time_s"],
  tube_trial_id=a["trial_id"],temporal_trial_id=b["trial_id"]))
write("aggregate_results.json",aggregate);write("paired_results.json",pairs);write("enriched_trials.json",enriched)
for name,data in (("aggregate_results.csv",aggregate),("paired_results.csv",pairs)):
 with (R/name).open("w",newline="",encoding="utf-8") as f:
  w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
# Reproduce prior baseline outcomes for the 48 original case/age/seed conditions.
old=json.loads((OLD/"results.json").read_text())["trials"];reproduced=0
metrics=("queries","selected","selected_error_gt05","selected_error_gt2","conditional_error_gt05","model_premise_violation_queries",
 "availability","selected_error_p95_mps","speed_target_mae_mps","progress_m","minimum_sampled_gap_m","collision_events")
for t in trials:
 if t["policy"]!="tube" or t["case"]=="joint_slow_drift":continue
 previous=next(x for x in old if not x["native_idm_reference"] and (x["case"],x["age_us"],x["seed"],x["policy"])==(t["case"],t["age_us"],t["seed"],"tube"))
 for field in metrics:
  a,b=t[field],previous[field]
  assert a is b if a is None or b is None else math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-8),(field,a,b)
 reproduced+=1
assert reproduced==48
write("baseline_reproduction.json",dict(passed=True,reproduced_conditions=reproduced,metrics_compared=list(metrics),wall_time_not_compared=True))
def cell(case,age,policy):return next(x for x in aggregate if (x["case"],x["age_us"],x["policy"])==(case,age,policy))
fig,axes=plt.subplots(2,3,figsize=(12,6.1),sharex=True,sharey=True)
for row,age in enumerate((50000,90000)):
 for col,case in enumerate(("nominal","joint_minus3","joint_slow_drift")):
  ax=axes[row,col]
  for i,policy in enumerate(("tube","temporal")):
   x=cell(case,age,policy);bad=x["selected_error_gt05"];good=x["selected"]-bad;abstain=x["queries"]-x["selected"]
   ax.barh(i,good,color="#287c5c",label="Selected error <=0.5 m/s" if (row,col,i)==(0,0,0) else None)
   ax.barh(i,bad,left=good,color="#b7463d",label="Selected error >0.5 m/s" if (row,col,i)==(0,0,0) else None)
   ax.barh(i,abstain,left=good+bad,color="#dedede",label="Abstained" if (row,col,i)==(0,0,0) else None)
   ax.text(605,i,f'{x["selected"]}/600',va="center",fontsize=9)
  if row==0:ax.set_title(dict(nominal="Nominal",joint_minus3="Abrupt coordinated -3 m/s",joint_slow_drift="Gradual coordinated drift")[case],fontsize=11)
  ax.set_xlim(0,760);ax.set_xticks([0,200,400,600]);ax.set_yticks([0,1],["Existing gateway","Temporal veto"])
  ax.spines[["top","right"]].set_visible(False)
  if col==0:ax.set_ylabel(str(age//1000)+" ms anchor age",fontsize=10)
  if row==1:ax.set_xlabel("Queries")
axes[0,0].invert_yaxis()
fig.suptitle("Fresh paired SUMO trials: selected errors and abstention",fontsize=14)
h,l=axes[0,0].get_legend_handles_labels()
fig.legend(h,l,loc="lower center",bbox_to_anchor=(.5,.055),ncol=3,frameon=False,fontsize=10)
fig.text(.5,.025,"3 fixed noise seeds; 600 queries per cell; 5 ms clock bound. Different policy trajectories remain separately scored.",ha="center",fontsize=9)
fig.subplots_adjust(left=.14,right=.99,top=.88,bottom=.21,hspace=.37,wspace=.12)
fig.savefig(R/"temporal_gateway_comparison.png",dpi=180);plt.close(fig)
nominal=[cell("nominal",age,"temporal") for age in (50000,90000)]
abrupt=[cell("joint_minus3",age,"temporal") for age in (50000,90000)]
drift=[cell("joint_slow_drift",age,"temporal") for age in (50000,90000)]
assessment=dict(fresh_sumo_validation_completed=True,fresh_trials=108,fresh_decisions=21600,baseline_conditions_reproduced=48,
 nominal_availability_at_least_95_percent=all(x["availability"]>=.95 for x in nominal),
 nominal_selected_error_gt05=sum(x["selected_error_gt05"] for x in nominal),
 abrupt_joint_selected_error_gt05=sum(x["selected_error_gt05"] for x in abrupt),
 gradual_joint_selected_error_gt05=sum(x["selected_error_gt05"] for x in drift),
 universal_coordinated_compromise_defense_established=False,new_algorithm_principle_established=False,
 physical_vehicle_validation_established=False,human_effectiveness_established=False,production_promotion_allowed=False)
write("contribution_assessment.json",assessment)
lines=["# Temporal consistency gateway: fresh SUMO results","23 October 2026.","",
"Implemented a persistent interval-consistency veto around the existing signed-command gateway, then ran the full predeclared 108-trial grid once. The grid includes both sudden coordinated false readings and gradual coordinated drift. No threshold or failure case was removed after the run.","",
"The guard preserves the original authentication/context/age/error rules and can only suppress an existing selection. It requires an additional trusted initial interval and declared global motion bounds. A contradiction latches abstention; recovery requires a separately justified trusted reset. See CONTRACT.md and the frozen DESIGN_PLAN.md.","",
"## Results","",
"Each row has three fixed noise seeds and 600 queries. Different policies drive their own trajectories; these are paired scenarios, not identical motion traces.","",
"| Case | Age (ms) | Policy | Selected / 600 | Error >0.5 m/s | Error >2 m/s | Mean progress (m) | Collisions |",
"|---|---:|---|---:|---:|---:|---:|---:|"]
for case in ("nominal","joint_minus3","joint_slow_drift","execution_break","anchor_dropout","context_change"):
 for age in (50000,90000):
  for policy in ("tube","temporal"):
   x=cell(case,age,policy)
   lines.append(f'| {case} | {age//1000} | {policy} | {x["selected"]} | {x["selected_error_gt05"]} | {x["selected_error_gt2"]} | {x["mean_progress_m"]:.3f} | {x["collision_events"]} |')
lines+=["","All cases, active-window scores, latch/recovery consequences, progress, sampled gap and per-seed paired differences are in aggregate_results.csv and paired_results.csv. Active-window zero-error counts must be read with active_selected; abstention is not accurate control.","",
"## Verification and limits","",
"An independent packet/model/FCD replay reconstructs all 21,600 decisions and 43,200 query vehicle states, including HMAC checks, counters, controller commands, source transformations and the temporal interval recurrence. It imports no production decision code. All 48 original baseline case/age/seed conditions reproduce their prior numerical outcomes; pipeline wall time is excluded from that comparison.","",
"Source/clock/command-error and initialization bounds are simulation premises. Gradual false data may remain dynamically plausible, so a temporal consistency check alone does not authenticate measurement truth. Latching can force prolonged fallback braking and reduce progress. Read the gradual-drift and post-window outcomes before interpreting a reduction in selected errors.","",
"This is an implemented engineering extension and a reproducible conditional study. Set propagation/intersection and temporal residual checks are established mathematical ideas; this study does not certify a new algorithm principle or universal security superiority. Physical vehicle control, physical latency and human-centered effectiveness remain unvalidated. Raw failures and protected holdout separation are preserved.",""]
(R/"README.md").write_text("\n".join(lines),encoding="utf-8")
print(json.dumps(assessment,indent=2))
