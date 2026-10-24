"""Independent summary and paired-cell validation; imports no report builder."""
from learned_common import *
install_guard()
import gzip,math,itertools,statistics
cfg=json.loads((R/"PROTOCOL.json").read_text());source=json.loads((R/"results.json").read_text())["trials"]
enriched=json.loads((R/"enriched_trials.json").read_text());app={x["trial_id"]:x for x in json.loads((R/"scene_appraisal.json").read_text())}
assert len(source)==len(enriched)==162
for original,t in zip(source,enriched):
 assert all(t[k]==v for k,v in original.items())
 with gzip.open(checked(R/t["trace"]),"rt",encoding="utf-8") as f:rows=[json.loads(x) for x in f]
 errors=[x["selected_disagreement_mps"] for x in rows if x["selected_disagreement_mps"] is not None]
 assert math.isclose(t["selected_mae_mps"],sum(errors)/len(errors),abs_tol=1e-12)
 assert t["scene_premise_conditioned_error_gt05"]==app[t["trial_id"]]["selected_error_gt05_with_valid_scene"]
 assert t["selected_with_valid_scene"]==app[t["trial_id"]]["selected_with_valid_scene"]
 assert t["old_speed_source_premise_conditioned_error_gt05"]==original["conditional_error_gt05"]
aggregates=json.loads((R/"aggregate_results.json").read_text());assert len(aggregates)==81
for a in aggregates:
 group=[t for t in enriched if all(t[k]==a[k] for k in ("profile","case","policy"))];assert len(group)==2
 for dst,src in (("queries","queries"),("selected","selected"),("errors_gt05","selected_error_gt05"),("errors_gt2","selected_error_gt2"),("selected_with_valid_scene","selected_with_valid_scene"),("errors_gt05_with_valid_scene","scene_premise_conditioned_error_gt05"),("collision_events","collision_events")):
  assert a[dst]==sum(t[src] for t in group),(a,dst)
 assert math.isclose(a["availability"],a["selected"]/a["queries"],abs_tol=1e-12)
 assert math.isclose(a["mean_progress_m"],sum(t["progress_m"] for t in group)/2,abs_tol=1e-12)
pairs=json.loads((R/"paired_results.json").read_text());assert len(pairs)==54
assert {(x["profile"],x["case"],x["seed"]) for x in pairs}==set(itertools.product(cfg["profiles"],cfg["cases"],cfg["seeds"]))
for pair in pairs:
 group={t["policy"]:t for t in enriched if all(t[k]==pair[k] for k in ("profile","case","seed"))}
 safe,raw,base=[group[k] for k in ("learned_shielded","learned_unshielded","kinematic")]
 expectations={
 "shield_minus_raw_gt05":safe["selected_error_gt05"]-raw["selected_error_gt05"],
 "shield_minus_raw_valid_scene_gt05":safe["scene_premise_conditioned_error_gt05"]-raw["scene_premise_conditioned_error_gt05"],
 "shield_minus_kinematic_mae_mps":safe["selected_mae_mps"]-base["selected_mae_mps"],
 "shield_minus_kinematic_progress_m":safe["progress_m"]-base["progress_m"],
 "shield_minus_kinematic_selected":safe["selected"]-base["selected"],
 "shield_minus_raw_progress_m":safe["progress_m"]-raw["progress_m"]}
 assert all(pair[k]==v for k,v in expectations.items())
sensitivity=json.loads((R/"seed_sensitivity.json").read_text());assert sensitivity["independent_rng_seeds"]==2 and sensitivity["shared_rng_across_cases_and_profiles"]
for block in sensitivity["blocks"]:
 group=[p for p in pairs if p["seed"]==block["seed"]];assert len(group)==27
 assert block["shield_minus_raw_valid_scene_gt05"]==sum(p["shield_minus_raw_valid_scene_gt05"] for p in group)
assessment=json.loads((R/"contribution_assessment.json").read_text())
for policy,total in assessment["totals"].items():
 group=[t for t in enriched if t["policy"]==policy]
 assert total["queries"]==10800 and total["selected"]==sum(t["selected"] for t in group)
 assert total["errors_gt05"]==sum(t["selected_error_gt05"] for t in group)
 assert total["errors_gt05_with_valid_scene"]==sum(t["scene_premise_conditioned_error_gt05"] for t in group)
assert assessment["conditional_bound_violations"]==0
assert all(assessment[k] is False for k in ("physical_control_validation","new_algorithmic_principle","native_published_framework_superiority_established","production_promotion"))
write("report_verification.json",dict(passed=True,trials=162,aggregate_cells=81,paired_cells=54,seed_clusters=2,population_inference_claimed=False))
print("INDEPENDENT_REPORT_VERIFIED",flush=True)
