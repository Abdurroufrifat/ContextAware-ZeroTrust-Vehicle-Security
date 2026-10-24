"""Summarize only this frozen synthetic experiment; no population inference."""
from learned_common import *
install_guard()
import gzip,math,csv,itertools,statistics
cfg=json.loads((R/"PROTOCOL.json").read_text())
result=json.loads((R/"results.json").read_text())
assert json.loads((R/"verification.json").read_text())["passed"]
appraisals={x["trial_id"]:x for x in json.loads((R/"scene_appraisal.json").read_text())}
summaries=[]
for t in result["trials"]:
 with gzip.open(checked(R/t["trace"]),"rt",encoding="utf-8") as f:rows=[json.loads(x) for x in f]
 errors=[x["selected_disagreement_mps"] for x in rows if x["selected_disagreement_mps"] is not None]
 active=[x for x in rows if x["active"]];active_errors=[x["selected_disagreement_mps"] for x in active if x["selected_disagreement_mps"] is not None]
 copied={**t,"selected_mae_mps":math.fsum(errors)/len(errors) if errors else None,
  "active_queries":len(active),"active_selected":len(active_errors),"active_error_gt05":sum(e>.5 for e in active_errors),
  "active_error_gt2":sum(e>2 for e in active_errors),
  "old_speed_source_premise_conditioned_error_gt05":t["conditional_error_gt05"],
  "scene_premise_conditioned_error_gt05":appraisals[t["trial_id"]]["selected_error_gt05_with_valid_scene"],
  "selected_with_valid_scene":appraisals[t["trial_id"]]["selected_with_valid_scene"],
  "learned_projection_changes":sum(d["decision"].get("learned_status")=="projected" and d["decision"]["research_speed_mmps"]!=round(d["decision"]["learned_prediction_mmps"]) for d in rows)}
 summaries.append(copied)
write("enriched_trials.json",summaries)
def aggregate(group):
 selected=sum(x["selected"] for x in group);queries=sum(x["queries"] for x in group)
 return dict(trials=len(group),queries=queries,selected=selected,availability=selected/queries,
  errors_gt05=sum(x["selected_error_gt05"] for x in group),errors_gt2=sum(x["selected_error_gt2"] for x in group),
  selected_mae_mps=sum(x["selected_mae_mps"]*x["selected"] for x in group if x["selected"])/selected if selected else None,
  selected_with_valid_scene=sum(x["selected_with_valid_scene"] for x in group),
  errors_gt05_with_valid_scene=sum(x["scene_premise_conditioned_error_gt05"] for x in group),
  collision_events=sum(x["collision_events"] for x in group),
  mean_progress_m=statistics.mean(x["progress_m"] for x in group),
  max_trial_decision_p95_ms=max(x["decision_wall_p95_s"] for x in group)*1000)
aggregates=[]
for profile,case,policy in itertools.product(cfg["profiles"],cfg["cases"],cfg["policies"]):
 group=[x for x in summaries if (x["profile"],x["case"],x["policy"])==(profile,case,policy)]
 assert len(group)==2
 aggregates.append(dict(profile=profile,case=case,policy=policy,**aggregate(group)))
write("aggregate_results.json",aggregates)
for name,rows in (("aggregate_results.csv",aggregates),("all_trials.csv",summaries)):
 with checked(R/name).open("w",encoding="utf-8",newline="") as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
pairs=[]
for profile,case,seed in itertools.product(cfg["profiles"],cfg["cases"],cfg["seeds"]):
 group={x["policy"]:x for x in summaries if (x["profile"],x["case"],x["seed"])==(profile,case,seed)}
 base=group["kinematic"];raw=group["learned_unshielded"];safe=group["learned_shielded"]
 pairs.append(dict(profile=profile,case=case,seed=seed,
  shield_minus_raw_gt05=safe["selected_error_gt05"]-raw["selected_error_gt05"],
  shield_minus_raw_valid_scene_gt05=safe["scene_premise_conditioned_error_gt05"]-raw["scene_premise_conditioned_error_gt05"],
  shield_minus_kinematic_mae_mps=safe["selected_mae_mps"]-base["selected_mae_mps"],
  shield_minus_kinematic_progress_m=safe["progress_m"]-base["progress_m"],
  shield_minus_kinematic_selected=safe["selected"]-base["selected"],
  shield_minus_raw_progress_m=safe["progress_m"]-raw["progress_m"]))
write("paired_results.json",pairs)
seed_sensitivity=[]
for seed in cfg["seeds"]:
 block=[x for x in pairs if x["seed"]==seed]
 seed_sensitivity.append(dict(seed=seed,paired_scenarios=len(block),
  shield_minus_raw_gt05=sum(x["shield_minus_raw_gt05"] for x in block),
  shield_minus_raw_valid_scene_gt05=sum(x["shield_minus_raw_valid_scene_gt05"] for x in block),
  mean_shield_minus_kinematic_mae_mps=statistics.mean(x["shield_minus_kinematic_mae_mps"] for x in block)))
write("seed_sensitivity.json",dict(blocks=seed_sensitivity,independent_rng_seeds=2,shared_rng_across_cases_and_profiles=True,
 inference="Descriptive finite-grid evidence. No independent-cell bootstrap, significance test or population confidence interval."))
envelope=[]
for error_mm in (1.5,3,5,6.25,7.5,10,20,50,100):
 E=error_mm/1000;best=min((2*E/h+.075+3*h,h) for h in [k/1000 for k in range(10,101)])
 envelope.append(dict(per_gap_bound_mm=error_mm,radius_at100ms_mps=2*E/.1+.075+.3,
  best_grid_radius_mps=best[0],best_spacing_ms=round(best[1]*1000),symmetric_interval_budget_feasible=best[0]<=.5))
write("scene_error_sensitivity.json",dict(rows=envelope,scope="Analytic bound away from zero; not observed sensor accuracy."))
totals={policy:aggregate([x for x in summaries if x["policy"]==policy]) for policy in cfg["policies"]}
nominal={policy:aggregate([x for x in summaries if x["policy"]==policy and x["case"]=="nominal"]) for policy in cfg["policies"]}
common={policy:aggregate([x for x in summaries if x["policy"]==policy and x["case"]=="scene_common_compromise"]) for policy in cfg["policies"]}
paired_better=sum(x["shield_minus_kinematic_mae_mps"]<0 for x in pairs)
assessment=dict(passed_reconstruction=True,totals=totals,nominal=nominal,common_scene_compromise=common,
 shield_lower_mae_than_original_scenarios=paired_better,paired_scenarios=len(pairs),
 conditional_bound_violations=totals["learned_shielded"]["errors_gt05_with_valid_scene"],
 learning_improves_over_original_overall=totals["learned_shielded"]["selected_mae_mps"]<totals["kinematic"]["selected_mae_mps"],
 physical_control_validation=False,new_algorithmic_principle=False,
 supported_contribution="Audited provenance-aware admission of learned speed estimates; exact conditional interval constraint and fresh closed-loop domain-shift stress evidence.",
 native_published_framework_superiority_established=False,production_promotion=False)
write("contribution_assessment.json",assessment)
selection=json.loads((R/"model_selection.json").read_text())
lines=["# Learned vehicle-security experiment","",
 "This is a research result record, not a manuscript.","",
 f"Five residual models trained on {selection['train_rows']:,} development rows and selected using {selection['validation_rows']:,} separate-seed validation rows. Selected model: {selection['selected']}.",
 "Models were frozen before three new driving profiles and two new seeds were run. Each of nine cases was run with the original scene-witness policy, unshielded learning and shielded learning: 162 fresh closed-loop trials, 32,400 decisions.","",
 "| Policy | Selected / queries | Errors >0.5 m/s | Valid-scene selected | Errors >0.5 with valid scene | Selected MAE (m/s) |",
 "|---|---:|---:|---:|---:|---:|"]
for policy,v in totals.items():
 lines.append(f"| {policy} | {v['selected']:,} / {v['queries']:,} | {v['errors_gt05']:,} | {v['selected_with_valid_scene']:,} | {v['errors_gt05_with_valid_scene']:,} | {v['selected_mae_mps']:.4f} |")
lines+=["","All-case totals include deliberate compromise of the scene trust domain. Success under a valid scene contract does not establish physical sensor accuracy.",
 "",f"The shield had lower selected MAE than the original policy in {paired_better} of {len(pairs)} matched scenario/seed conditions. Overall learning benefit over the original policy: {assessment['learning_improves_over_original_overall']}. Adding learning is not claimed to improve the original estimator.",
 "","## What this establishes","",
 "The integer projection constrains a learned speed estimate to at most 0.5 m/s discrepancy from every point in the authenticated scene interval, while preserving original admission abstentions. Independent replay checks source authentication, time/counters/context, exact scene bounds, every forest tree prediction, feasible projection, controller commands and raw SUMO speed/position transitions.",
 "","## Evidence boundaries","",
 "- Longitudinal software-in-the-loop evidence; no real vehicle operated or validated.",
 "- The 1.5 mm scene gap-error premise and independent absolute lead-speed premise are not physically qualified. Analytic sensitivity makes these narrow measurement requirements explicit.",
 "- Two new RNG seeds shared across profiles/cases. Seed sensitivity is descriptive; no population confidence or independent-cell significance claim.",
 "- AVMON residual learning, multi-source/context-aware zero trust and safety filtering precede this study. No new mathematical principle or native framework superiority claimed.",
 "- Recorded HeRCULES and Boreas reference results remain separate and unchanged, including failed estimates.",
 "- No human effectiveness, durable anti-rollback security, production deployment or journal acceptance established.",
 "","## Reproduce verification","",
 "Run the project Python interpreter with research/vehicle_learned_shield_study/run_all.py.",
 "Verification checks frozen results; it does not retrain, tune, overwrite or rerun the driving grid.",
 "","Per-profile/case results: aggregate_results.csv. Exact trial outcomes: all_trials.csv. Premise-scoped counts: scene_appraisal.json. Split/selection: model_selection.json. Prior-method distinctions: literature_scope.json."]
checked(R/"README.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
checked(P/"LEARNED_VEHICLE_RESULTS.md").write_text("\n".join(lines[:15])+"\n\nFull scope and verification: research/vehicle_learned_shield_study/README.md.\n",encoding="utf-8")
print("REPORT_COMPLETE",json.dumps(assessment,allow_nan=False),flush=True)
