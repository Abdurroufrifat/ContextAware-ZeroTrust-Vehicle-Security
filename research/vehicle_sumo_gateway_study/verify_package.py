"""Verify frozen SUMO research artifacts and prevent inflated completion claims."""
from sg_common import *
install_guard()
m=json.loads((R/"file_manifest.json").read_text());files=m["files"]
assert files and len({x["path"] for x in files})==len(files)
for item in files:
 path=checked(P/item["path"])
 assert path.stat().st_size==item["bytes"] and sha(path)==item["sha256"],item["path"]
assert {p.name for p in (R/"simulator_outputs").iterdir() if p.is_dir()}=={f"trial_{i:03d}" for i in range(147)}
r=json.loads((R/"results.json").read_text());v=json.loads((R/"verification.json").read_text());t=json.loads((R/"test_validation.json").read_text())
assert r["controlled_trials"]==144 and r["controlled_queries"]==28800 and r["native_reference_trials"]==3
assert v["passed"] and v["controlled_decisions_reconstructed"]==28800 and v["fcd_vehicle_states_matched"]==58800
assert v["realized_speed_transitions_checked"]==287856
assert t["focused"]["exit_code"]==t["core"]["exit_code"]==0 and t["focused"]["counts"]["passed"]==445 and t["core"]["counts"]["passed"]==731 and t["core"]["counts"]["skipped"]==4
probes=json.loads((R/"verifier_probes.json").read_text());assert probes["all_rejected"] and len(probes["probes"])==4 and all(x["rejected"] for x in probes["probes"])
a=json.loads((R/"contribution_assessment.json").read_text());assert a["pc_simulation_evaluation_completed"] and a["conditional_nonzero_clock_advantage_demonstrated"]
assert a["tube"]["selected"]==597 and a["tube"]["queries"]==600 and a["tube"]["selected_error_gt05"]==0 and a["clock_component_baseline"]["selected"]==0
assert all(a[k] is False for k in ("new_algorithm_principle_claim_supported","global_first_claim_supported","novel_ai_model_validated_by_this_study","physical_vehicle_validation_established","human_effectiveness_established","journal_acceptance_guaranteed"))
assert all(x["conditional_error_gt05"]==0 for x in r["trials"] if not x["native_idm_reference"] and x["policy"] in ("tube","clock"))
print(json.dumps({"package_integrity_passed":True,"manifest_files":len(files),"simulation_scope_completed":True,"controlled_trials":144,"controlled_queries":28800,"focused_tests_passed":445,"core_tests_passed":731,"core_skipped":4,"physical_vehicle_validation":False,"new_algorithm_or_ai_model_claim":False,"journal_acceptance_guaranteed":False},indent=2))
