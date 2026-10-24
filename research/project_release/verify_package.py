from release_common import *
install_guard()
import importlib.metadata as md
m=json.loads((R/"file_manifest.json").read_text());files=m["files"]
assert files and len({x["path"] for x in files})==len(files)
for item in files:
 path=checked(P/item["path"])
 assert path.stat().st_size==item["bytes"] and sha(path)==item["sha256"],item["path"]
index=json.loads((R/"evidence_index.json").read_text())
assert index["performance_metrics_pooled"] is False and len(index["entries"])==9
for item in index["source_bindings"]:
 path=checked(P/item["path"]);assert path.stat().st_size==item["bytes"] and sha(path)==item["sha256"],item["path"]
app=json.loads((R/"application_replay.json").read_text());tests=json.loads((R/"test_validation.json").read_text());s=json.loads((R/"completion_status.json").read_text())
assert app["passed"] and app["trajectories"]==34 and app["decisions"]==6800 and len(app["comparisons"])==34
assert len({x["trial_id"] for x in app["comparisons"]})==34 and all(x["decisions"]==200 for x in app["comparisons"])
assert app["new_driving_evidence"] is app["physical_authorization"] is False
assert tests["focused"]["exit_code"]==tests["core"]["exit_code"]==0
assert tests["focused"]["counts"]["passed"]==s["focused_tests_passed"]==510
assert tests["core"]["counts"]["passed"]==s["core_tests_passed"]==731 and tests["core"]["counts"]["skipped"]==s["core_tests_skipped"]==4
assert s["application_implementation_complete"] and s["frozen_pc_evidence_scope_verified"] and s["specific_conditional_integration_result_supported"]
for name in ("full_original_research_objectives_completed","independent_physical_reference_error_bound_qualified","physical_vehicle_control_validation_complete",
 "physical_scene_independence_established","complete_recent_av_security_framework_natively_reproduced","new_algorithm_principle_or_new_learned_AV_model_established",
 "human_effectiveness_established","complete_special_issue_compliance_certified","journal_readiness_certified","production_promotion_allowed"):
 assert s[name] is False,name
claims=json.loads((R/"claim_matrix.json").read_text())
assert len(claims)==7 and sum(x["supported"] for x in claims)==3
assert all(e in {x["id"] for x in index["entries"]} for claim in claims for e in claim["evidence"])
scene=json.loads((KW/"contribution_assessment.json").read_text())
assert scene["selected_with_valid_scene"]==12330 and scene["selected_error_gt05_with_valid_scene"]==0 and scene["scene_common_compromise_selected_error_gt05"]==314
ref=json.loads((P/"research/vehicle_boreas_rt_reference_study/reference_qualification.json").read_text())
assert ref["hard_reference_error_bound_established"] is False and ref["physical_control_validation_established"] is False
env=json.loads((R/"environment.json").read_text())
for name,version in env["packages"].items():
 if version is not None:assert md.version(name)==version,(name,"environment changed")
print(json.dumps(dict(release_integrity_passed=True,release_files=len(files),source_bindings=len(index["source_bindings"]),
 application_integration_verified=True,application_replay_decisions=6800,focused_tests=510,core_tests=731,skipped=4,
 full_original_research_objectives_completed=False,physical_validation=False,journal_readiness_certified=False),indent=2))
