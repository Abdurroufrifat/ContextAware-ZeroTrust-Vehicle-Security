from tg_common import *
install_guard()
m=json.loads((R/"file_manifest.json").read_text());files=m["files"]
assert len(files)>700 and len({x["path"] for x in files})==len(files)
for item in files:
 path=checked(P/item["path"]);assert path.stat().st_size==item["bytes"] and sha(path)==item["sha256"],item["path"]
lock=json.loads((R/"input_lock.json").read_text());assert len(lock)==31 and all(sha(P/n)==h for n,h in lock.items())
assert {x.name for x in (R/"simulator_outputs").iterdir() if x.is_dir()}=={f"trial_{i:03d}" for i in range(108)}
v=json.loads((R/"verification.json").read_text());q=json.loads((R/"report_verification.json").read_text());a=json.loads((R/"contribution_assessment.json").read_text())
assert v["passed"] and v["controlled_decisions_reconstructed"]==21600 and v["fcd_vehicle_states_matched"]==43200 and v["realized_speed_transitions_checked"]==215892
assert q["passed"] and q["focused_tests_passed"]==477 and q["core_tests_passed"]==731 and q["core_tests_skipped"]==4
assert a["fresh_sumo_validation_completed"] and a["fresh_trials"]==108 and a["fresh_decisions"]==21600 and a["baseline_conditions_reproduced"]==48
assert a["nominal_availability_at_least_95_percent"] and a["nominal_selected_error_gt05"]==0 and a["abrupt_joint_selected_error_gt05"]==0
assert a["gradual_joint_selected_error_gt05"]==306
assert all(a[k] is False for k in ("universal_coordinated_compromise_defense_established","new_algorithm_principle_established","physical_vehicle_validation_established","human_effectiveness_established","production_promotion_allowed"))
print(json.dumps(dict(package_integrity_passed=True,manifest_files=len(files),fresh_trials=108,fresh_decisions=21600,
 nominal_availability_retained=True,abrupt_joint_selected_errors=0,gradual_joint_selected_errors_gt05=306,
 universal_security_claim=False,physical_vehicle_validation=False),indent=2))
