"""Read-only package identity and evidence inventory checks."""
from common import *
install_guard()
manifest=json.loads((R/"file_manifest.json").read_text())
for name,digest in manifest["files"].items():assert sha(R/name)==digest,name
lock=json.loads((R/"input_lock.json").read_text())
for name,digest in lock.items():assert sha(P/name)==digest,name
v=json.loads((R/"verification.json").read_text());r=json.loads((R/"results.json").read_text())
t=json.loads((R/"test_validation.json").read_text());s=json.loads((R/"summary.json").read_text())
assert v["passed"] and v["decisions_independently_verified"]==4224 and v["conditions"]==264 and v["source_identities_checked"]==12
assert r["condition_count"]==264 and r["decisions"]==4224 and len(r["conditions"])==264
assert t["focused"]["exit_code"]==t["core"]["exit_code"]==0
assert t["focused"]["counts"]["passed"]==390 and t["core"]["counts"]["passed"]==731 and t["core"]["counts"]["skipped"]==4
assert s["totals"]==dict(queries=4224,selected=1472,error_above500=80,error_above2000=48,conditional_error_above500=0,replayed_evidence_outputs=48)
for obj in (manifest,r,s,v):
 for field in ("physical_validation_established","specific_novelty_established","promotion_allowed"):assert obj[field] is False
print(json.dumps({"passed":True,"package_files_checked":len(manifest["files"]),"locked_inputs_checked":len(lock),
"decisions":4224,"conditions":264,"focused_tests_passed":390,"core_tests_passed":731,"core_tests_skipped":4,
"physical_validation_established":False,"specific_novelty_established":False,"promotion_allowed":False},indent=2))
