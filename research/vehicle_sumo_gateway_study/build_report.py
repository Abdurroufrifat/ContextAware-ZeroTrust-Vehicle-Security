from sg_common import *
install_guard()
import csv,math,collections
os.environ["MPLCONFIGDIR"]=str(R/"runtime"/"MPLCONFIGDIR")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
r=json.loads((R/"results.json").read_text());v=json.loads((R/"verification.json").read_text());t=json.loads((R/"test_validation.json").read_text())
assert v["passed"] and t["focused"]["exit_code"]==t["core"]["exit_code"]==0
groups=collections.defaultdict(list)
for row in r["trials"]:
 if not row["native_idm_reference"]:groups[(row["case"],row["age_us"],row["policy"])].append(row)
cells=[]
for (case,age,policy),rows in groups.items():
 assert len(rows)==3
 cells.append(dict(case=case,age_us=age,clock_us=5000,policy=policy,seeds=3,queries=sum(x["queries"] for x in rows),
  selected=sum(x["selected"] for x in rows),availability=sum(x["selected"] for x in rows)/600,
  selected_error_gt05=sum(x["selected_error_gt05"] for x in rows),selected_error_gt2=sum(x["selected_error_gt2"] for x in rows),
  conditional_error_gt05=sum(x["conditional_error_gt05"] for x in rows),
  mean_speed_target_mae_mps=math.fsum(x["speed_target_mae_mps"] for x in rows)/3,
  mean_progress_m=math.fsum(x["progress_m"] for x in rows)/3,
  minimum_sampled_gap_m=min(x["minimum_sampled_gap_m"] for x in rows),collision_events=sum(x["collision_events"] for x in rows)))
paired=[]
for case in json.loads((R/"PROTOCOL.json").read_text())["cases"]:
 for age in (50000,90000):
  for comparator in ("static","clock"):
   differences=[]
   for seed in (17,29,43):
    a=next(x for x in r["trials"] if (x["case"],x["age_us"],x["seed"],x["policy"])==(case,age,seed,"tube"))
    b=next(x for x in r["trials"] if (x["case"],x["age_us"],x["seed"],x["policy"])==(case,age,seed,comparator))
    differences.append({"seed":seed,"progress_difference_m":a["progress_m"]-b["progress_m"],"tracking_mae_difference_mps":a["speed_target_mae_mps"]-b["speed_target_mae_mps"],"selected_difference":a["selected"]-b["selected"]})
   paired.append({"case":case,"age_us":age,"comparator":comparator,"paired_seeds":differences,
    "mean_progress_difference_m":math.fsum(x["progress_difference_m"] for x in differences)/3,
    "mean_tracking_mae_difference_mps":math.fsum(x["tracking_mae_difference_mps"] for x in differences)/3})
for name,rows in (("all_trials.csv",r["trials"]),("aggregate_results.csv",cells)):
 with (R/name).open("w",encoding="utf-8",newline="") as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
write("aggregate_results.json",{"cells":cells,"paired":paired})
normal=[x for x in cells if x["case"]=="nominal" and x["age_us"]==90000];by={x["policy"]:x for x in normal}
benefit=by["tube"]["availability"]>=.95 and by["tube"]["mean_progress_m"]>by["clock"]["mean_progress_m"] and by["tube"]["selected_error_gt05"]==0
assessment={"pc_simulation_evaluation_completed":True,"conditional_nonzero_clock_advantage_demonstrated":benefit,
 "evidence_condition":"nominal90ms declared anchor age,5ms declared clock bound; same generated-source/error/command assumptions,3 fixed noise seeds",
 "tube":by["tube"],"clock_component_baseline":by["clock"],"static_component_baseline":by["static"],
 "new_algorithm_principle_claim_supported":False,"global_first_claim_supported":False,"novel_ai_model_validated_by_this_study":False,
 "scope":"Original implementation/integration and cross-simulator application evidence; not native reproduction of a complete published security framework",
 "physical_vehicle_validation_established":False,"human_effectiveness_established":False,"journal_acceptance_guaranteed":False}
write("contribution_assessment.json",assessment)
table=["| Case | Age (ms) | Policy | Selected /600 | Mean progress (m) | Errors >0.5m/s | Errors >2m/s | Collisions |","|---|---:|---|---:|---:|---:|---:|---:|"]
for case in ("nominal","candidate_minus3","joint_minus3","execution_break"):
 for cell in [x for x in cells if x["case"]==case and x["age_us"]==90000]:
  table.append(f'| {case} |90| {cell["policy"]} |{cell["selected"]}|{cell["mean_progress_m"]:.3f}|{cell["selected_error_gt05"]}|{cell["selected_error_gt2"]}|{cell["collision_events"]}|')
text="# Authenticated gateway in a SUMO vehicle-following loop\n\n"
text+="Completed a fixed CPU-only software-in-the-loop application experiment: 144 controlled runs, 28,800 signed-policy queries, plus 3 separately reported native SUMO IDM reference runs. The same existing signed gateway, source roles, replay state and exact command-history selector are reused; no older module or result was altered.\n\n"
text+=f'At 90 ms nominal anchor age with 5 ms declared clock uncertainty, command-history selection is {by["tube"]["availability"]:.2%} versus {by["clock"]["availability"]:.2%} for the clock-aware component ablation. Mean progress is {by["tube"]["mean_progress_m"]:.3f}m versus {by["clock"]["mean_progress_m"]:.3f}m; selected errors above 0.5 m/s are {by["tube"]["selected_error_gt05"]}. These are paired generated-noise simulation results, not physical vehicle measurements.\n\n'
text+="The static-source ablation may also retain availability; it does not provide the same current-query age/clock motion bound. Do not claim uniform superiority. Full data preserve coordinated signed-source compromise, signed false history, input-bound violations, missing sources, bad tags and contextual denial. Zero selections is abstention, not a successful detection rate.\n\n"
text+="## Selected results\n\n"+"\n".join(table)+"\n\n"
text+="## What the evidence supports\n\n"
text+="The added result is a reproducible integration and conditional application demonstration at a nonzero declared clock bound in a simulator separate from the original scalar test plant. The mathematical interval propagation, HMACs, IDM and general zero-trust/context principles are established techniques. No global-first or new-AI-model claim is established. Recent full AV security frameworks were not natively reproduced by these component ablations; see published_comparison_scope.json.\n\n"
text+="The upstream recorded-data studies and their failures remain separate. In particular the 96-frame Boreas comparison does not supply this SUMO study's source assumptions, and no new physical dataset score is substituted for it. No new learned model was trained in this study.\n\n"
text+="## Simulation and trust limits\n\n"
text+="The road is one straight 5000 m lane; two vehicles, 20 s runs, 10 ms simulator steps, 100 ms policy queries. Radar range and lead speed are uncorrupted simulator observations. Candidate/anchor reports and declared uncertainty are generated from simulated ego speed; they are not independent physical sensors. There is no planner, lateral maneuver, pedestrian or road-generalization claim.\n\n"
text+="Controlled ego uses speedMode0 so built-in safety checks do not secretly repair its commands. Realized FCD speed transitions are appraised against the reflected/clipped command/error model, not just commanded values. FCD precision tolerance is 2e-6 m/s for numerical replay only. Native IDM reference runs use privileged current simulator state and built-in safety and therefore are not ranked as attacked-policy competitors.\n\n"
text+="The gateway consumes offline logical query time. CPU/HMAC/SQLite work occurs while SUMO logical time is paused; measured pipeline runtimes do not establish a physical 100 ms deadline. Authentication does not prove physical measurement truth or protect the PC replay database against rollback. All production/physical authority flags remain false.\n\n"
text+="## Verification\n\n"
text+=f'Focused suite: {t["focused"]["counts"]["passed"]} passed. Core suite: {t["core"]["counts"]["passed"]} passed, {t["core"]["counts"]["skipped"]} skipped; the existing legacy checkpoint calibration warning remains. Independent replay reconstructs all 28,800 decisions and summaries, verifies HMAC/context/counter sequences, exact interval outputs using reverse suffix sums, controller/input transformations, and 58,800 query vehicle states against raw SUMO FCD. It checks realized discrete-speed evolution and preserves every premise failure.\n\n'
text+="One source review found the missing realized-motion appraisal; five tests first failed then passed after the correction, before the final grid. 23 new regression tests pass. Four mutation probes reject omitted/duplicate trials, relabeled reference cohorts and false physical claims.\n\n"
text+="## Files and entry\n\n"
text+="PROTOCOL.json and DESIGN_PLAN.md define the fixed grid; input_lock.json locks 19 experiment inputs. simulator_outputs/trial_NNN contains every packet/decision trace, raw FCD, collisions and final replay state. all_trials.csv and aggregate_results.csv report separate denominators. verification.json and test_validation.json contain check results. run_all.py is the combined offline verification entry; --tests reruns tests into runtime/recheck. No manuscript, commit, push or physical vehicle action.\n"
(R/"README.md").write_text(text,encoding="utf-8")
fig,axes=plt.subplots(1,2,figsize=(10,3.8),constrained_layout=True)
colors={"static":"#777777","clock":"#cf7c1c","tube":"#26749c"}
for policy in ("static","clock","tube"):
 chosen=sorted([x for x in cells if x["case"]=="nominal" and x["policy"]==policy],key=lambda x:x["age_us"])
 axes[0].plot([x["age_us"]/1000 for x in chosen],[100*x["availability"] for x in chosen],("s--" if policy=="static" else "o-"),label=policy,color=colors[policy],markersize=9 if policy=="static" else 5,markerfacecolor="none" if policy=="static" else colors[policy])
 axes[1].plot([x["age_us"]/1000 for x in chosen],[x["mean_progress_m"] for x in chosen],("s--" if policy=="static" else "o-"),label=policy,color=colors[policy],markersize=9 if policy=="static" else 5,markerfacecolor="none" if policy=="static" else colors[policy])
for ax in axes:
 ax.set_xlabel("Nominal anchor age (ms)");ax.set_xticks([50,90]);ax.grid(alpha=.2)
axes[0].set_ylabel("Selected queries (%)");axes[0].set_ylim(-3,103);axes[0].legend()
axes[1].set_ylabel("Mean distance travelled (m)")
fig.suptitle("SUMO nominal runs; declared clock bound 5 ms; 3 fixed noise seeds\nGenerated reports and declared bounds; no physical vehicle validation",fontsize=10)
fig.savefig(R/"nominal_comparison.png",dpi=180);plt.close(fig)
print(json.dumps(assessment,indent=2))
