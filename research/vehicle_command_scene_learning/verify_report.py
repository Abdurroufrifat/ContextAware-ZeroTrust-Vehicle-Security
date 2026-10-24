from cs_common import *
install_guard()
import gzip,itertools,math
source=json.loads((R/"results.json").read_text())["trials"];enriched=json.loads((R/"enriched_trials.json").read_text());assert len(source)==len(enriched)==243
app={x["trial_id"]:x for x in json.loads((R/"scene_appraisal.json").read_text())}
for s,t in zip(source,enriched):
 assert all(t[k]==v for k,v in s.items())
 with gzip.open(checked(R/t["trace"]),"rt",encoding="utf-8") as f:rows=[json.loads(line) for line in f]
 es=[x["selected_disagreement_mps"] for x in rows if x["selected_disagreement_mps"] is not None]
 assert math.isclose(t["selected_mae_mps"],sum(es)/len(es),abs_tol=1e-12)
 assert t["valid_certificate_selected"]==app[t["trial_id"]]["selected_with_valid_joint_certificate"]
 assert t["valid_certificate_gt05"]==app[t["trial_id"]]["joint_certificate_errors_gt05"]
aggregates=json.loads((R/"aggregate_results.json").read_text());assert len(aggregates)==81
for a in aggregates:
 group=[t for t in enriched if all(t[k]==a[k] for k in ("profile","case","policy"))];assert len(group)==3
 for dest,src in (("queries","queries"),("selected","selected"),("errors_gt05","selected_error_gt05"),("errors_gt2","selected_error_gt2"),("valid_certificate_selected","valid_certificate_selected"),("valid_certificate_gt05","valid_certificate_gt05"),("collision_events","collision_events")):
  assert a[dest]==sum(t[src] for t in group)
 assert math.isclose(a["selected_mae_mps"],sum(t["selected_mae_mps"]*t["selected"] for t in group)/a["selected"],abs_tol=1e-12)
cfg=json.loads((R/"PROTOCOL.json").read_text());pairs=json.loads((R/"paired_results.json").read_text());assert len(pairs)==81
assert {(x["profile"],x["case"],x["seed"]) for x in pairs}==set(itertools.product(cfg["profiles"],cfg["cases"],cfg["seeds"]))
for pair in pairs:
 group={x["policy"]:x for x in enriched if all(x[k]==pair[k] for k in ("profile","case","seed"))}
 old,mid,learner=[group[p] for p in cfg["policies"]]
 for dest,other in (("learned_minus_original_mae",old),("learned_minus_midpoint_mae",mid)):
  assert pair[dest]==learner["selected_mae_mps"]-other["selected_mae_mps"]
assessment=json.loads((R/"contribution_assessment.json").read_text())
for scope,case in (("totals",None),("nominal","nominal"),("common_scene_compromise","scene_common_compromise")):
 for policy,a in assessment[scope].items():
  group=[t for t in enriched if t["policy"]==policy and (case is None or t["case"]==case)]
  assert a["selected"]==sum(t["selected"] for t in group) and a["queries"]==sum(t["queries"] for t in group)
  assert a["valid_certificate_gt05"]==sum(t["valid_certificate_gt05"] for t in group)==0
for flag,scope,other in (("learned_better_original_nominal","nominal","kinematic"),("learned_better_midpoint_nominal","nominal","refined_midpoint"),("learned_better_original_all_cases","totals","kinematic"),("learned_better_midpoint_all_cases","totals","refined_midpoint")):
 assert assessment[flag] is (assessment[scope]["refined_learned"]["selected_mae_mps"]<assessment[scope][other]["selected_mae_mps"])
assert all(assessment[k] is False for k in ("global_first_ever_claimed","native_recent_framework_superiority_claimed","physical_vehicle_control_validated","complete_special_issue_compliance_certified","production_promotion"))
write("report_verification.json",dict(passed=True,trials=243,aggregate_cells=81,paired_cells=81,seed_clusters=3))
print("INDEPENDENT_COMMAND_SCENE_REPORT_PASSED",flush=True)
