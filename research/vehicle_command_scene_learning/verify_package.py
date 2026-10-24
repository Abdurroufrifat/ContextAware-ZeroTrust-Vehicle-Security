from cs_common import *
install_guard()
import sys,importlib.metadata
manifest=json.loads((R/"file_manifest.json").read_text())
for name,h in manifest.items():assert sha(P/name)==h,("package",name)
for name in ("model_lock.json","experiment_lock.json"):
 for path,h in json.loads((R/name).read_text()).items():assert sha(P/path)==h,(name,path)
selection=json.loads((R/"model_selection.json").read_text())
for path,h in selection["input_bindings"].items():assert sha(P/path)==h
assert set(selection["trial_ids"]["train"]).isdisjoint(selection["trial_ids"]["validation"])
assert selection["test_results_accessed"] is False and selection["absolute_speed_gap_profile_seed_attack_label_features"] is False
env=json.loads((R/"environment.json").read_text());assert env["python"]==sys.version
assert hashlib.sha256(BINARY.read_bytes()).hexdigest()==env["sumo_sha256"]
for name,version in env["packages"].items():assert importlib.metadata.version(name)==version
tests=json.loads((R/"test_validation.json").read_text());assert all(v["exit_code"]==0 for v in tests.values())
assert tests["focused"]["counts"]["passed"]==572 and tests["core"]["counts"]["passed"]==731 and tests["core"]["counts"]["skipped"]==4
application=json.loads((R/"application_replay.json").read_text());assert application["passed"] and application["trials"]==54 and application["decisions"]==10800
for name,h in application["source_bindings"].items():assert sha(P/name)==h
assert application["truth_fields_passed_to_model"] is application["new_driving_evidence"] is application["physical_vehicle_operation"] is False
assert json.loads((R/"verification.json").read_text())["controlled_decisions_reconstructed"]==48600
assert json.loads((R/"report_verification.json").read_text())["passed"]
assessment=json.loads((R/"contribution_assessment.json").read_text())
assert all(assessment[k] for k in ("learned_better_original_nominal","learned_better_midpoint_nominal","learned_better_original_all_cases","learned_better_midpoint_all_cases"))
assert assessment["totals"]["refined_learned"]["valid_certificate_gt05"]==0
assert assessment["physical_vehicle_control_validated"] is assessment["complete_special_issue_compliance_certified"] is False
print("COMMAND_SCENE_PACKAGE_VERIFIED",len(manifest),"files",flush=True)
