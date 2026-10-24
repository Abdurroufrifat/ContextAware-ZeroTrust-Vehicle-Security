from kw_common import *
install_guard()
import gzip,math,csv,ast
trials=json.loads((R/"results.json").read_text())["trials"];enriched=json.loads((R/"enriched_trials.json").read_text())
a=json.loads((R/"aggregate_results.json").read_text());pairs=json.loads((R/"paired_results.json").read_text())
app={x["trial_id"]:x for x in json.loads((R/"scene_appraisal.json").read_text())}
assert len(trials)==len(enriched)==144 and len(a)==48 and len(pairs)==72
for t,e in zip(trials,enriched):
 assert all(e[k]==v for k,v in t.items())
 assert all(e[k]==v for k,v in app[t["trial_id"]].items())
 with gzip.open(R/t["trace"],"rt",encoding="utf-8") as f:rows=[json.loads(x) for x in f]
 active=[x for x in rows if x["active"]];chosen=[x for x in active if x["decision"]["research_speed_mmps"] is not None]
 assert e["active_queries"]==len(active)==60 and e["active_selected"]==len(chosen)
 assert e["active_error_gt05"]==sum(x["selected_disagreement_mps"]>.5 for x in chosen)
 assert e["active_error_gt2"]==sum(x["selected_disagreement_mps"]>2 for x in chosen)
 assert e["post_window_selected"]==sum(x["time_s"]>=14 and x["decision"]["research_speed_mmps"] is not None for x in rows)
 assert e["latched_queries"]==sum(x["decision"].get("temporal_latched",False) for x in rows)
sum_fields=("queries","selected","selected_error_gt05","selected_error_gt2","conditional_error_gt05","model_premise_violation_queries","collision_events",
 "active_queries","active_selected","active_error_gt05","active_error_gt2","post_window_selected","latched_queries","qualifying_scene_intervals",
 "scene_contract_holds_queries","reference_outside_witness","reference_outside_witness_with_valid_scene","selected_with_valid_scene",
 "selected_error_gt05_with_valid_scene","selected_error_gt2_with_valid_scene","witness_replacements")
for x in a:
 group=[t for t in enriched if (t["case"],t["age_us"],t["policy"])==(x["case"],x["age_us"],x["policy"])]
 assert len(group)==x["seeds"]==3 and {t["seed"] for t in group}=={17,29,43}
 for k in sum_fields:assert x[k]==sum(t[k] for t in group)
 assert x["availability"]==x["selected"]/600
 assert math.isclose(x["mean_progress_m"],math.fsum(t["progress_m"] for t in group)/3)
 assert math.isclose(x["mean_speed_target_mae_mps"],math.fsum(t["speed_target_mae_mps"] for t in group)/3)
 assert x["minimum_sampled_gap_m"]==min(t["minimum_sampled_gap_m"] for t in group)
for x in pairs:
 old,new=[next(t for t in enriched if (t["case"],t["age_us"],t["seed"],t["policy"])==(x["case"],x["age_us"],x["seed"],p)) for p in ("temporal","kinematic")]
 assert x["temporal_trial_id"]==old["trial_id"] and x["kinematic_trial_id"]==new["trial_id"]
 assert x["selected_difference"]==new["selected"]-old["selected"]
 assert x["error_gt05_difference"]==new["selected_error_gt05"]-old["selected_error_gt05"]
 assert math.isclose(x["progress_difference_m"],new["progress_m"]-old["progress_m"])
for name,data in (("aggregate_results.csv",a),("paired_results.csv",pairs)):
 with (R/name).open(newline="",encoding="utf-8") as f:rows=list(csv.DictReader(f))
 assert len(rows)==len(data)
 for row,item in zip(rows,data):assert row=={k:("" if v is None else str(v)) for k,v in item.items()}
assessment=json.loads((R/"contribution_assessment.json").read_text());kin=[x for x in a if x["policy"]=="kinematic"]
assert assessment["nominal_availability_at_least_95_percent"] is all(x["availability"]>=.95 for x in kin if x["case"]=="nominal")
for case,key in (("nominal","nominal_selected_error_gt05"),("joint_minus3","joint_abrupt_selected_error_gt05"),("joint_slow_drift","joint_drift_selected_error_gt05"),("scene_common_compromise","scene_common_compromise_selected_error_gt05")):
 assert assessment[key]==sum(x["selected_error_gt05"] for x in kin if x["case"]==case)
assert assessment["selected_with_valid_scene"]==sum(x["selected_with_valid_scene"] for x in kin)
assert assessment["selected_error_gt05_with_valid_scene"]==sum(x["selected_error_gt05_with_valid_scene"] for x in kin)==0
assert assessment["selected_with_valid_scene"]>0
assert all(assessment[k] is False for k in ("global_security_guarantee_established","physical_scene_independence_established","new_algorithm_principle_established","physical_vehicle_validation_established","human_effectiveness_established","production_promotion_allowed"))
tree=ast.parse((R/"scene_core.py").read_text())
assert {n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)}=={"fractions"}
assert {n.name for t in ast.walk(tree) if isinstance(t,ast.Import) for n in t.names}=={"json","hmac","hashlib"}
fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=="evaluate")
assert [x.arg for x in fn.args.args]==["self","packet","now","base_decision"]
tests=json.loads((R/"test_validation.json").read_text())
assert tests["focused"]["exit_code"]==tests["core"]["exit_code"]==0
assert tests["focused"]["counts"]["passed"]==495 and tests["core"]["counts"]["passed"]==731 and tests["core"]["counts"]["skipped"]==4
write("report_verification.json",dict(passed=True,trial_summaries=144,aggregate_cells=48,paired_conditions=72,csv_matches=True,
 no_reference_source_import_in_estimator=True,focused_tests_passed=495,core_tests_passed=731,core_skipped=4))
print("KINEMATIC_REPORT_VERIFIED",flush=True)
