from ssc_common import *
m=json.loads((R/"file_manifest.json").read_text());paths=[x["path"] for x in m["files"]]
assert len(paths)==len(set(paths)) and len(paths)>150
for item in m["files"]:
 path=checked(P/item["path"])
 assert path.stat().st_size==item["bytes"] and sha(path)==item["sha256"],item["path"]
lock=json.loads((R/"input_lock.json").read_text());assert len(lock)==298 and all(sha(P/n)==h for n,h in lock.items())
v=json.loads((R/"verification.json").read_text());assert v["passed"] and v["estimator_decisions_reconstructed"]==57600 and v["summaries_reconstructed"]==432
t=json.loads((R/"test_validation.json").read_text());assert t["focused"]["exit_code"]==t["core"]["exit_code"]==0
assert t["focused"]["counts"]["passed"]==461 and t["core"]["counts"]["passed"]==731 and t["core"]["counts"]["skipped"]==4
a=json.loads((R/"contribution_assessment.json").read_text())
assert a["numerical_primary_method_comparison_completed"] and a["scalar_reproduction_only"]
assert all(a[k] is False for k in ("stateful_candidate_ready_to_replace_current_policy","new_algorithm_principle_established","fresh_closed_loop_validation","physical_vehicle_validation","human_effectiveness_established","journal_acceptance_guaranteed"))
assert {x.name for x in (R/"decisions").iterdir()}=={f"trial_{i:03d}.jsonl.gz" for i in range(144)}
print(json.dumps(dict(package_integrity_passed=True,manifest_files=len(paths),frozen_inputs=len(lock),new_decisions=57600,
 focused_tests=461,core_tests=731,skipped=4,new_algorithm_claim=False,physical_vehicle_validation=False),indent=2))
