from kw_common import *
install_guard()
m=json.loads((R/"file_manifest.json").read_text());files=m["files"]
assert len(files)>1000 and len({x["path"] for x in files})==len(files)
for item in files:
 path=checked(P/item["path"]);assert path.stat().st_size==item["bytes"] and sha(path)==item["sha256"],item["path"]
lock=json.loads((R/"input_lock.json").read_text());assert len(lock)==43 and all(sha(P/n)==h for n,h in lock.items())
assert {x.name for x in (R/"simulator_outputs").iterdir() if x.is_dir()}=={f"trial_{i:03d}" for i in range(144)}
v=json.loads((R/"verification.json").read_text());q=json.loads((R/"report_verification.json").read_text());a=json.loads((R/"contribution_assessment.json").read_text())
assert v["passed"] and v["controlled_decisions_reconstructed"]==28800 and v["fcd_vehicle_states_matched"]==57600
assert v["realized_speed_transitions_checked"]==287856 and v["scene_motion_and_geometry_transition_appraisals"]==575712
assert q["passed"] and q["focused_tests_passed"]==495 and q["core_tests_passed"]==731 and q["core_skipped"]==4
assert a["fresh_sumo_validation_completed"] and a["fresh_trials"]==144 and a["fresh_decisions"]==28800 and a["baseline_conditions_reproduced"]==54
assert a["nominal_availability_at_least_95_percent"] and a["nominal_selected_error_gt05"]==a["joint_abrupt_selected_error_gt05"]==a["joint_drift_selected_error_gt05"]==0
assert a["scene_common_compromise_selected_error_gt05"]==314
assert a["selected_with_valid_scene"]==12330 and a["selected_error_gt05_with_valid_scene"]==0
assert all(a[k] is False for k in ("global_security_guarantee_established","physical_scene_independence_established","new_algorithm_principle_established","physical_vehicle_validation_established","human_effectiveness_established","production_promotion_allowed"))
print(json.dumps(dict(package_integrity_passed=True,manifest_files=len(files),fresh_trials=144,fresh_decisions=28800,
 selected_under_valid_scene=12330,selected_errors_gt05_under_valid_scene=0,common_scene_compromise_errors_gt05=314,
 global_security_guarantee=False,physical_vehicle_validation=False),indent=2))
