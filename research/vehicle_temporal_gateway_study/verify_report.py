from tg_common import *
install_guard()
import gzip,math,csv,itertools,ast
result=json.loads((R/"results.json").read_text());trials=result["trials"]
aggregate=json.loads((R/"aggregate_results.json").read_text());pairs=json.loads((R/"paired_results.json").read_text())
enriched=json.loads((R/"enriched_trials.json").read_text())
assert len(trials)==len(enriched)==108 and len(aggregate)==36 and len(pairs)==54
for t,e in zip(trials,enriched):
 assert all(e[k]==v for k,v in t.items())
 with gzip.open(R/t["trace"],"rt",encoding="utf-8") as f:rows=[json.loads(x) for x in f]
 selected=[x for x in rows if x["active"] and x["decision"]["research_speed_mmps"] is not None]
 latched=[x for x in rows if x["decision"].get("temporal_latched",False)]
 assert e["active_queries"]==sum(x["active"] for x in rows)==60
 assert e["active_selected"]==len(selected)
 assert e["active_error_gt05"]==sum(abs(x["decision"]["research_speed_mmps"]-x["ego_speed_mps"]*1000)>500 for x in selected)
 assert e["active_error_gt2"]==sum(abs(x["decision"]["research_speed_mmps"]-x["ego_speed_mps"]*1000)>2000 for x in selected)
 assert e["latched_queries"]==len(latched)
 assert e["first_latch_time_s"]==(latched[0]["time_s"] if latched else None)
 assert e["post_window_queries"]==sum(x["time_s"]>=14 for x in rows)
 assert e["post_window_selected"]==sum(x["time_s"]>=14 and x["decision"]["research_speed_mmps"] is not None for x in rows)
for x in aggregate:
 group=[t for t in enriched if (t["case"],t["age_us"],t["policy"])==(x["case"],x["age_us"],x["policy"])]
 assert len(group)==x["seeds"]==3 and {t["seed"] for t in group}=={17,29,43}
 for name in ("queries","selected","selected_error_gt05","selected_error_gt2","conditional_error_gt05","model_premise_violation_queries","collision_events",
  "active_queries","active_selected","active_error_gt05","active_error_gt2","latched_queries","post_window_queries","post_window_selected"):
  assert x[name]==sum(t[name] for t in group)
 assert x["availability"]==x["selected"]/600
 assert math.isclose(x["mean_progress_m"],math.fsum(t["progress_m"] for t in group)/3)
 assert math.isclose(x["mean_speed_target_mae_mps"],math.fsum(t["speed_target_mae_mps"] for t in group)/3)
 assert x["minimum_sampled_gap_m"]==min(t["minimum_sampled_gap_m"] for t in group)
for x in pairs:
 a,b=[next(t for t in enriched if (t["case"],t["age_us"],t["seed"],t["policy"])==(x["case"],x["age_us"],x["seed"],policy)) for policy in ("tube","temporal")]
 assert x["selected_difference"]==b["selected"]-a["selected"]
 assert x["error_gt05_difference"]==b["selected_error_gt05"]-a["selected_error_gt05"]
 assert math.isclose(x["progress_difference_m"],b["progress_m"]-a["progress_m"])
 assert x["temporal_first_latch_time_s"]==b["first_latch_time_s"]
 assert x["tube_trial_id"]==a["trial_id"] and x["temporal_trial_id"]==b["trial_id"]
for name,data in (("aggregate_results.csv",aggregate),("paired_results.csv",pairs)):
 with (R/name).open(newline="",encoding="utf-8") as f:csvrows=list(csv.DictReader(f))
 assert len(csvrows)==len(data)
 for csvrow,row in zip(csvrows,data):
  assert set(csvrow)==set(row)
  assert all(csvrow[k]==("" if v is None else str(v)) for k,v in row.items())
assessment=json.loads((R/"contribution_assessment.json").read_text())
nominal=[x for x in aggregate if x["case"]=="nominal" and x["policy"]=="temporal"]
assert assessment["nominal_availability_at_least_95_percent"] is all(x["availability"]>=.95 for x in nominal)
for case,key in (("nominal","nominal_selected_error_gt05"),("joint_minus3","abrupt_joint_selected_error_gt05"),("joint_slow_drift","gradual_joint_selected_error_gt05")):
 assert assessment[key]==sum(x["selected_error_gt05"] for x in aggregate if x["case"]==case and x["policy"]=="temporal")
assert all(assessment[k] is False for k in ("universal_coordinated_compromise_defense_established","new_algorithm_principle_established","physical_vehicle_validation_established","human_effectiveness_established","production_promotion_allowed"))
# The guard has no filesystem/network/truth-source imports and exposes only base decision and query time.
tree=ast.parse((R/"temporal_guard.py").read_text())
assert [n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]==["fractions"]
assert not any(isinstance(n,ast.Import) for n in ast.walk(tree))
fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=="evaluate")
assert [a.arg for a in fn.args.args]==["self","base_decision","now_us"]
tests=json.loads((R/"test_validation.json").read_text())
assert tests["focused"]["exit_code"]==tests["core"]["exit_code"]==0
assert tests["focused"]["counts"]["passed"]==477 and tests["core"]["counts"]["passed"]==731 and tests["core"]["counts"]["skipped"]==4
write("report_verification.json",dict(passed=True,trial_summaries=108,aggregate_cells=36,paired_seed_conditions=54,
 csv_matches=True,no_truth_source_import_in_guard=True,focused_tests_passed=477,core_tests_passed=731,core_tests_skipped=4))
print("TEMPORAL_REPORT_VERIFIED",flush=True)
