"""Check frozen package identities and stored evidence; no experiment generation."""
from common import *
install_guard()
manifest=json.loads((R/"file_manifest.json").read_text())
assert manifest["schema"]==1 and manifest["physical_validation_established"] is False
for name,digest in manifest["files"].items():assert sha(R/name)==digest,name
lock=json.loads((R/"input_lock.json").read_text())
for name,digest in lock.items():assert sha(P/name)==digest,name
result=json.loads((R/"results.json").read_text())
replay=json.loads((R/"replay_validation.json").read_text())
tests=json.loads((R/"test_validation.json").read_text())
witness=json.loads((R/"limitation_witness.json").read_text())
assert result["trial_count"]==1056 and result["query_rows"]==633600 and result["original_trials_reproduced"]==288
assert replay["passed"] and replay["feedback_steps_replayed"]==633600 and replay["original_trials_reproduced"]==288
assert replay["conditional_error_above500_count"]==0 and replay["input_identities_checked"]==15
assert tests["focused"]["exit_code"]==tests["core"]["exit_code"]==0
assert tests["focused"]["counts"]["passed"]==263 and tests["core"]["counts"]["passed"]==731 and tests["core"]["counts"]["skipped"]==4
assert witness["passed"] and witness["post_study_not_pooled"]
assert [w["selected_error_mmps"] for w in witness["worlds"]]==[3000,0]
assert [w["selected_error_mmps"] for w in witness["additional_broken_premise_witnesses"]]==[550,550]
assert all(result[k] is False for k in ("physical_validation_established","specific_novelty_established","promotion_allowed"))
import csv
for name,count in (("all_trials.csv",1056),("aggregate_results.csv",352)):
 with (R/name).open(newline="",encoding="utf-8") as f:assert sum(1 for _ in csv.DictReader(f))==count
print(json.dumps({"passed":True,"package_files_verified":len(manifest["files"]),"pre_run_inputs_verified":len(lock),
"trials":1056,"verified_trace_rows":633600,"aggregate_cells":352,"prior_trials_reproduced":288,
"physical_validation_established":False,"specific_novelty_established":False,"promotion_allowed":False},indent=2))
