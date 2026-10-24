from common import *
import csv,gzip,math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
v=json.loads((R/"independent_verification.log").read_text(encoding="utf-8"));assert v["passed"]
tests=json.loads((R/"test_validation.json").read_text())
assert all(tests[name]["exit_code"]==0 for name in ("focused","core"))
assert tests["focused"]["counts"]["passed"]==239 and tests["core"]["counts"]["passed"]==731
write("independent_verification.json",v)
results=json.loads((R/"results.json").read_text());trials=results["trials"]
groups={}
for row in trials:groups.setdefault((row["case"],row["nominal_age_us"],row["clock_us"],row["policy"]),[]).append(row)
aggregates=[]
for (case,age,clock,policy),rows in groups.items():
 assert {x["seed"] for x in rows}=={17,29,43} and len(rows)==3
 maes=[x["tracking_mae_mps"] for x in rows]
 aggregates.append(dict(case=case,nominal_age_us=age,clock_us=clock,policy=policy,noise_seeds=3,queries=1800,
  selected_queries=sum(x["selected_queries"] for x in rows),fallback_steps=sum(x["fallback_steps"] for x in rows),
  tracking_mae_mean_mps=math.fsum(maes)/3,tracking_mae_min_mps=min(maes),tracking_mae_max_mps=max(maes),
  selected_error_above2000_count=sum(x["selected_error_above2000_count"] for x in rows),
  mean_final_position_m=math.fsum(x["final_position_m"] for x in rows)/3))
assert len(aggregates)==96
with (R/"all_conditions.csv").open("w",newline="",encoding="utf-8") as f:
 writer=csv.DictWriter(f,fieldnames=list(aggregates[0]));writer.writeheader();writer.writerows(aggregates)
with (R/"all_conditions.csv").open(newline="",encoding="utf-8") as f:
 reread=list(csv.DictReader(f))
assert len(reread)==96
for original,row in zip(aggregates,reread):
 for key,value in original.items():
  assert str(value)==row[key]
# Plot every normal-case timing condition with all three fixed seeds, no selection.
curves={}
with gzip.open(R/"trace.csv.gz","rt",newline="",encoding="utf-8") as f:
 for row in csv.DictReader(f):
  if row["case"]!="normal":continue
  key=(int(row["nominal_age_us"]),int(row["clock_us"]),row["policy"])
  if key not in curves:curves[key]=np.zeros(600)
  curves[key][int(row["step"])]+=int(row["v_n"])/int(row["v_d"])/3000
colors={"unguarded_candidate":"#666666","static_native":"#138c90","clock_aware":"#8b42b6"}
names={"unguarded_candidate":"Candidate only","static_native":"Static selector","clock_aware":"Clock-aware"}
fig,axes=plt.subplots(2,2,figsize=(11,6.5),sharex=True,sharey=True)
time=np.arange(600)*.05;goal=np.where(time<8,15,np.where(time<18,22,10))
for ax,(age,clock) in zip(axes.flat,((50000,0),(50000,1000),(100000,0),(100000,1000))):
 ax.step(time,goal,where="post",color="black",linestyle="--",linewidth=1.2,label="Desired speed")
 for policy in colors:
  ax.plot(time,curves[(age,clock,policy)],color=colors[policy],linewidth=1.6,label=names[policy])
 ax.set_title(f"Anchor age {age/1000:g} ms; clock bound {clock/1000:g} ms",fontsize=10)
 ax.set_ylim(-.5,24);ax.set_xlim(0,30);ax.grid(alpha=.2)
for ax in axes[:,0]:ax.set_ylabel("Simulated speed (m/s)")
for ax in axes[-1,:]:ax.set_xlabel("Model time (s)")
handles,labels=axes[0,0].get_legend_handles_labels()
fig.legend(handles,labels,loc="lower center",ncol=4,frameon=False,fontsize=9)
fig.suptitle("Normal-case feedback: mean of three fixed noise seeds",fontsize=12)
fig.tight_layout(rect=(0,.055,1,.95));fig.savefig(R/"normal_speed_tracking.png",dpi=160);plt.close(fig)
lines=["# Closed-loop longitudinal component study","",
"Completed 22 October 2026. All 288 frozen trials and 172,800 feedback steps are retained. This study evaluates numerical access-policy components in a simple plant model; it is not a manuscript or a physical vehicle experiment.","",
"## New evidence","",
"Selected estimates now feed the next acceleration command. This exposes tracking and fallback effects that a fixed-output replay cannot measure. All policies use the same public speed commands and fixed noise schedules. Their trajectories differ through feedback.","",
"At 50 ms nominal age plus a 1 ms clock bound, a candidate -3 m/s bias raises mean tracking error from 1.436624 to 2.253985 m/s for the unguarded control. The clock-aware control has mean tracking error 1.452861 m/s in that attack condition and no selected errors above 2 m/s. These values average all three fixed noise seeds in the stated condition; all other cells remain in all_conditions.csv.","",
"The joint +3 m/s case remains a failure: at that same timing condition, the clock-aware control selects 600 outputs above 2 m/s error across 1,800 queries. The forged anchor violates the stipulated source-error bound. Sensor agreement and this numerical guard do not resolve coordinated source compromise.","",
"The 100 ms conditions reveal a different limitation. The clock-aware policy selects no outputs in all 48 such trials. Automatic fallback braking stops the model after 25 m; mean speed-tracking error is 14.491667 m/s. That loss of operation is not counted as security success. The 0.25 m/s measurement bound plus nominal age growth already exceeds the 0.5 m/s admission budget, even before added clock uncertainty.","",
"![All normal-case timing conditions, mean of three noise seeds](normal_speed_tracking.png)","",
"## All normal-case timing conditions","",
"MAE is tracking disagreement with the public commanded speed, not sensor accuracy. Selected counts are over 1,800 simulated queries per row.","",
"| Age ms | Clock bound ms | Control | Selected/queries | Mean tracking MAE m/s | Mean final position m |",
"| ---: | ---: | --- | ---: | ---: | ---: |"]
for row in aggregates:
 if row["case"]=="normal":
  lines.append(f'| {row["nominal_age_us"]/1000:g} | {row["clock_us"]/1000:g} | {row["policy"]} | {row["selected_queries"]}/1800 | {row["tracking_mae_mean_mps"]:.6f} | {row["mean_final_position_m"]:.3f} |')
lines+=["",
"all_conditions.csv includes all 96 case/timing/control cells, each with all three seed results represented by mean/min/max tracking MAE and full query/selection denominators. results.json retains all 288 individual trials; trace.csv.gz retains every exact rational state and command. No seed or trial was removed.",
"",
"## Limits that remain",
"",
"The static selector is evaluated under this experiment's measurement-time interpretation; its original protected-copy/request-time contract differs. The clock-aware selector also depends on true bounds, not merely claimed ones. Joint bias and clock-bound violation are intentionally outside those assumptions.",
"",
"Only noise realizations vary across the three seeds. This is one stipulated plant, not independent-route replication. The model has no lateral dynamics, road friction, objects, collisions, TTC or validated actuator behavior. Automatic braking is an experimental fallback choice; no real-world safety claim follows.",
"",
"No packet signing, key/replay database, network timing or complete published security framework is reproduced here. This study is not pooled with earlier gateway or vehicle-data results. No physical reference, source protection or calibration was established.",
"",
"The numerical guard and controller use established arithmetic and feedback ideas. This study tests a project-specific question about consequences and limitations; specific algorithmic novelty and EAAI readiness remain unestablished.",
"",
"## Verification and status",
"",
"- Twenty-one new behavioral tests failed before implementation and now pass.",
"- Independent recurrence, source-generation and selection replay checked all 172,800 steps without importing the plant or selector implementations.",
"- All 288 trial summaries and 96 exported aggregate rows were checked. Model-valid, assumptions-supplied selections of the clock-aware method have no error-budget violations; its all-abstention and outside-model failures remain visible.",
"- Cumulative tests: 239 focused and 731 configured core tests pass; four core tests skipped. Existing legacy residual-calibration warning retained.",
"- Scoped automated review found no important actionable correctness, leakage or fairness issue.",
"",
"Read-only reproduction: verify_loop.py. run_simulation.py refuses to overwrite results.json. Parameters and full membership are in PROTOCOL.md and configuration.json.",
"",
"Previous failed development intervals, unknown references, thresholds, reserved payload restrictions and production state are unchanged. No manuscript, installation, message, commit, push or vehicle action was performed."]
(R/"README.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
# Check highlighted report values against the saved aggregate, rather than only prose.
def cell(case,age,clock,policy):return next(r for r in aggregates if (r["case"],r["nominal_age_us"],r["clock_us"],r["policy"])==(case,age,clock,policy))
assert round(cell("candidate_minus3",50000,1000,"unguarded_candidate")["tracking_mae_mean_mps"],6)==2.253985
assert round(cell("candidate_minus3",50000,1000,"clock_aware")["tracking_mae_mean_mps"],6)==1.452861
assert cell("both_plus3",50000,1000,"clock_aware")["selected_error_above2000_count"]==600
assert all(t["selected_queries"]==0 and t["final_position_m"]==25 for t in trials if t["policy"]=="clock_aware" and t["nominal_age_us"]==100000)
assert all(sha(P/name)==digest for name,digest in json.loads((R/"input_lock.json").read_text()).items())
write("completion.json",{"closed_loop_model_study_complete":True,"trials":288,"feedback_steps_verified":172800,
"aggregate_cells":96,"focused_passed":239,"core_passed":731,"core_skipped":4,
"physical_validation_established":False,"specific_novelty_established":False,"journal_ready":False,"production_changed":False})
manifest={path.name:sha(path) for path in sorted(R.iterdir()) if path.is_file() and path.name!="file_manifest.json"}
write("file_manifest.json",manifest);assert all(sha(R/name)==digest for name,digest in manifest.items())
print(json.dumps({"completed":True,"package_files":len(manifest),"aggregate_cells":96,"figure":"normal_speed_tracking.png"},indent=2))
