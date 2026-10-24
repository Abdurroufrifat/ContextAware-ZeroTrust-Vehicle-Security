"""Build inspectable development results; no new experiments or threshold changes."""
from common import *
install_guard()
import csv,gzip,collections,math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
result=json.loads((R/"results.json").read_text());trials=result["trials"]
assert json.loads((R/"replay_validation.json").read_text())["passed"]
tests=json.loads((R/"test_validation.json").read_text())
assert all(t["exit_code"]==0 for t in tests.values())
assert json.loads((R/"limitation_witness.json").read_text())["passed"]
def csvfile(name,rows):
 with (R/name).open("w",newline="",encoding="utf-8") as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
flat=[]
for t in trials:
 flat.append({**{k:v for k,v in t.items() if k!="selections"},**{"selection_"+k:v for k,v in t["selections"].items()}})
# Uniform fields across controls.
fields=list(dict.fromkeys(k for row in flat for k in row))
for row in flat:
 for k in fields:row.setdefault(k,0 if k.startswith("selection_") else None)
csvfile("all_trials.csv",flat)
groups=collections.defaultdict(list)
for t in trials:groups[tuple(t[k] for k in ("case","nominal_age_us","clock_us","actuation","policy"))].append(t)
aggregates=[]
for keys,ts in groups.items():
 assert len(ts)==3 and {t["seed"] for t in ts}=={17,29,43}
 row=dict(zip(("case","nominal_age_us","clock_us","actuation","policy"),keys))
 row.update(seed_count=3,queries=sum(t["queries"] for t in ts),selected_queries=sum(t["selected_queries"] for t in ts),
  selected_fraction=sum(t["selected_queries"] for t in ts)/1800,
  tracking_mae_mean_over_seeds_mps=math.fsum(t["tracking_mae_mps"] for t in ts)/3,
  tracking_p95_mean_over_seeds_mps=math.fsum(t["tracking_p95_mps"] for t in ts)/3,
  selected_error_above500_count=sum(t["selected_error_above500_count"] for t in ts),
  selected_error_above2000_count=sum(t["selected_error_above2000_count"] for t in ts),
  selected_error_above500_with_model_and_assumptions=sum(t["selected_error_above500_with_model_and_assumptions"] for t in ts),
  selected_outside_model_or_assumptions=sum(t["selected_outside_model_or_assumptions"] for t in ts))
 aggregates.append(row)
assert len(aggregates)==352
csvfile("aggregate_results.csv",aggregates)
write("aggregate_results.json",{"cells":aggregates,"aggregation":"3 fixed seeds per cell; counts summed, per-trial tracking metrics averaged; no pooled P95","cell_count":352})
def cell(case,age,clock,policy):
 return next(a for a in aggregates if (a["case"],a["nominal_age_us"],a["clock_us"],a["actuation"],a["policy"])==(case,age,clock,"bounded_noise",policy))
lines=["| Case | Age / clock bound (ms) | Control | Selected / 1800 | Tracking MAE (m/s) | Selected error >0.5 m/s | Selected error >2 m/s |",
 "|---|---:|---|---:|---:|---:|---:|"]
for case in ["normal","candidate_minus3","both_plus3","command_history_missing","command_history_tamper","actuation_bound_violated"]:
 for age,clock in [(50000,1000),(100000,0),(100000,1000)]:
  for policy in ["clock_aware","command_tube"]:
   a=cell(case,age,clock,policy)
   lines.append(f'| {case} | {age/1000:g} / {clock/1000:g} | {policy} | {a["selected_queries"]} | {a["tracking_mae_mean_over_seeds_mps"]:.6f} | {a["selected_error_above500_count"]} | {a["selected_error_above2000_count"]} |')
table="\n".join(lines)
totals={}
for policy in ["clock_aware","command_tube"]:
 ts=[t for t in trials if t["policy"]==policy]
 totals[policy]={"trials":len(ts),"selected":sum(t["selected_queries"] for t in ts),
 "error_above500_count":sum(t["selected_error_above500_count"] for t in ts),
 "error_above500_with_model_and_assumptions":sum(t["selected_error_above500_with_model_and_assumptions"] for t in ts)}
write("control_totals.json",totals)
readme="""# Command-history reachable-interval development package

## Result and scope
The delayed-evidence work package is complete at the conditional scalar-simulator level. It is not a physical self-driving vehicle validation or an established novel security algorithm.

The new research-only command_tube control propagates an older bounded anchor using recorded requested control commands. It unions the possible current-speed intervals over timestamp endpoints and intervening command boundaries. It then checks the same 0.5 m/s output budget using exact rational arithmetic. All possible ages must remain within 0..100 ms inclusive. See CONTRACT_CLARIFICATION.md for the acceleration-input premise at zero speed.

## Frozen study
PROTOCOL.md and input_lock.json record the pre-run design and 15 source identities. The grid is 11 cases × 2 ages × 2 clock bounds × 3 seeds × 2 actuation variants × 4 controls = 1056 trials, 633600 queries. All 288 earlier exact-actuation trials of the original three controls reproduce unchanged.

The nominal ages are 50 and 100 ms; clock bounds 0 and 1 ms. Requested acceleration is bounded to [-4,+2] m/s². Command execution input error is declared bounded by 1 m/s²; bounded_noise uses a separate fixed-seed ±0.5 m/s² disturbance. This input-error premise, command-log protection, timing and the anchor uncertainty are not established by physical measurements.

## Main result
In the normal 100 ms / zero-clock-uncertainty bounded-noise cell, command_tube selects 1794/1800 queries (startup accounts for six missing anchors) with zero selected errors over 0.5 m/s. Mean tracking MAE across three fixed seeds is 1.533196 m/s versus 14.490948 m/s for clock_aware, which selects no outputs and uses fallback throughout. The target-speed changes and actuator limits create tracking transients even in normal runs.

At 50 ms / 1 ms both controls select 1794/1800. Tracking MAE is slightly worse for command_tube (1.531989 versus 1.516332 m/s); this is not uniform superiority. At 100 ms / 1 ms both reject all outputs because the possible age exceeds 100 ms. The zero clock bound in the recovered 100 ms cell is a declared model condition, not a realistic-clock qualification.

## Failures and limits
Across the entire stress grid, command_tube has 3600 selected errors above 0.5 m/s among 107670 selected queries, versus 2406 among 74202 for clock_aware. All of these errors occur outside the evaluated assumptions/premises. Greater admission can increase exposure when those premises fail; this is not an overall security-superiority result. The new control consumes additional command-history information and assumptions, so this is not an equal-information architecture ranking.

- both_plus3 still produces 600 selected errors over 2 m/s per three-seed eligible cell because both source reports are forged. Those rows violate the anchor premise. They are preserved, not counted as a passed security defense.
- Missing command history causes abstention during the affected window. Availability and tracking suffer.
- The standalone log-tamper and execution-bound violations in the grid retain an honest current candidate. Their lack of selected errors in particular cells cannot establish security against those violations.
- limitation_witness.py adds explicitly post-study, unpooled fixed counterexamples: identical honest/forged observations produce a 3 m/s error in the forged world; forged command history and a broken execution bound can each produce a 0.55 m/s error from an otherwise valid anchor.
- Zero selected outputs is reported as abstention/fallback, never as successful attack mitigation.
- This is a longitudinal point-mass component model. It contains no objects, collision avoidance, roads, planner validation, vehicle network transport, physical command integrity, or full authenticated-gateway execution.

## Verification
24 new unit tests passed after observed preimplementation failures. The cumulative focused suite passes 263 tests; the core suite passes 731 with four skips and the existing legacy checkpoint residual-calibration warning.
verify_tube.py independently reconstructs all 633600 plant, source, selection and command rows and all trial summaries without importing the production plant, controller or admission modules. Its interval oracle uses a reverse-suffix closed form for reflected affine maps rather than the production forward recurrence. All 15 locked inputs match and all 288 previous trials reproduce. No selected error exceeds 0.5 m/s when the evaluated model premises and declared assumptions both hold.
A fresh read-only code review found a wording issue about acceleration input at rest; CONTRACT_CLARIFICATION.md resolves it without changing the frozen protocol, algorithms, thresholds or numeric results.

## Files
- all_trials.csv: all 1056 trial summaries.
- aggregate_results.csv / .json: 352 condition/control cells, each three fixed seeds. Counts sum; tracking metrics average per-trial values. Averaged P95 is not pooled P95 and no population inference is claimed.
- trace.csv.gz: all simulator queries, including unselected and violated-premise rows.
- tracking_comparison.png: normal and candidate-underreporting cases at 100 ms / zero clock bound, bounded noise, seed 17. Simulator actual speed and desired target, not physical reference data.
- replay_validation.json, test_validation.json, limitation_witness.json: executable verification evidence.
- FULL_TASK_STATUS.md: remaining full-project dependencies.
- file_manifest.json and verify_package.py: final package integrity and stored-evidence checks.

## Recheck
Use the existing project Python environment to run verify_package.py, verify_tube.py, limitation_witness.py or run_checks.py. run_study.py refuses to overwrite frozen results. No manuscript was produced. Protected final-evaluation payloads remain unopened.

## Selected cells
All rows below use bounded_noise and summarize the same three fixed seeds. Tracking MAE is averaged over seeds; counts sum. The full 352-cell export includes every case, both actuation variants and all controls.

"""+table+"\n"
(R/"README.md").write_text(readme,encoding="utf-8")
status="""# Full project status after the command-history study

1. Delayed-evidence engineering: completed at the declared simulator/component scope. The new control recovers eligible 100 ms evidence with command-history assumptions, preserves the 100 ms and 0.5 m/s limits, and includes negative cases plus executable limits. It does not solve stale timestamps or coordinated compromise.
2. Specific publishable novelty: not established. The existing published-method scope audit and equal-input studies remain valid. Interval propagation and indistinguishability are standard reasoning tools; this package does not establish that their combination is a new contribution. Additional numbered tests do not establish novelty.
3. Physical validation: blocked by missing independent reference uncertainty/calibration, clock and source-integrity evidence. A PC and existing unqualified recordings do not establish those properties. An instrumented vehicle is not automatically required, but suitable independently qualified data or measurements would be required for the present physical-error claim.
4. Final project evaluation and release: this development package is frozen and checked; the physical final-evaluation claim remains ineligible. Library03, Urban02 NIGHT, RAV4 and vehicle_final_evaluation payloads remain unopened. The older canonical development release remains unchanged.

There is no honest fixed step count that turns missing measurement evidence into a completed physical validation. The current result can support a limited simulator/conditional engineering study. Completing the originally intended physical-vehicle security claim requires the missing evidence, or an explicit change of research scope followed by a justified novelty and evaluation plan.

No manuscript, submission, commit, push or physical-validation promotion has been performed.
"""
(R/"FULL_TASK_STATUS.md").write_text(status,encoding="utf-8")
# Scientific figure: use only this simulator's traces.
data=collections.defaultdict(lambda:{"t":[],"v":[],"target":[]})
with gzip.open(R/"trace.csv.gz","rt",newline="",encoding="utf-8") as f:
 for row in csv.DictReader(f):
  if row["case"] in ("normal","candidate_minus3") and row["nominal_age_us"]=="100000" and row["clock_us"]=="0" and row["seed"]=="17" and row["actuation"]=="bounded_noise":
   a=data[(row["case"],row["policy"])];a["t"].append(int(row["time_us"])/1e6);a["v"].append(int(row["v_n"])/int(row["v_d"])/1000);a["target"].append(int(row["target_mmps"])/1000)
fig,axes=plt.subplots(2,1,figsize=(9,7),sharex=True,constrained_layout=True)
labels={"unguarded_candidate":"Unguarded candidate","static_native":"Static native","clock_aware":"Clock-aware","command_tube":"Command-history interval"}
for ax,case,title in zip(axes,["normal","candidate_minus3"],["Normal sources","Candidate underreports by 3 m/s during 10–20 s"]):
 for policy in labels:
  a=data[(case,policy)];assert len(a["t"])==600
  ax.plot(a["t"],a["v"],label=labels[policy],linewidth=1.5)
 ax.plot(a["t"],a["target"],"k--",label="Desired target",linewidth=1)
 ax.set_title(title);ax.set_ylabel("Simulated speed (m/s)");ax.grid(alpha=.2)
axes[0].legend(ncol=3,fontsize=8);axes[-1].set_xlabel("Time (s)")
fig.suptitle("Conditional component model: 100 ms age, zero clock uncertainty\nBounded actuation noise; seed 17; no physical validation",fontsize=11)
fig.savefig(R/"tracking_comparison.png",dpi=170);plt.close(fig)
print(json.dumps({"trial_rows":len(flat),"aggregate_cells":len(aggregates),"totals":totals},indent=2))
