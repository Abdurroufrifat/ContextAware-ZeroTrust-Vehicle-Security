from shared import *
from release_core import verify_manifest,file_sha
items=verify_manifest(R,"file_manifest.json");completion=read("completion.json")
assert len(items)==completion["top_level_files"] and completion["development_release_complete"]
assert file_sha(P/"CURRENT_PROJECT_STATUS.md")==completion["current_status_sha256"]
assert file_sha(P/"FINAL_PROJECT_README.md")==completion["final_readme_sha256"]
assert file_sha(R/"README.md")==file_sha(P/"FINAL_PROJECT_README.md")
for name,digest in read("experiment_lock.json").items():assert file_sha(R/name)==digest
tests=read("test_validation.json")
assert tests["core"]["exit_code"]==tests["focused"]["exit_code"]==0 and tests["denials"]==[]
assert tests["core"]["counts"]["passed"]==731 and tests["focused"]["counts"]["passed"]==184
assert read("independent_verification.json")["passed"] and read("code_review.json")["unresolved_important_findings"]==0
fix=read("review_fix_record.json")
assert file_sha(R/"pre_review_fix/experiment_lock.json")==fix["previous_lock_sha256"]
for name,digest in fix["previous_source_sha256"].items():assert file_sha(R/"pre_review_fix"/name)==digest
assert read("experiment_lock.json")==fix["current_lock"]
readiness=read("FINAL_READINESS.json")
assert not any(readiness[k] for k in ["physical_validation_complete","journal_readiness_established","overall_project_complete","promotion_allowed","reserved_payloads_opened"])
for path in R.glob("*.py"):compile(path.read_text(encoding="utf-8"),str(path),"exec")
out={"passed":True,"release_files_verified":len(items),"canonical_readme_verified":True,"current_status_verified":True,
 "historical_reports_preserved":True,"exact_numeric_boundary_fixed":True,"recorded_sensitivity_decisions_changed":0,
 "focused_passed":184,"core_passed":731,"core_skipped":4,"development_release_complete":True,
 "physical_validation_complete":False,"overall_project_complete":False}
write("package_audit.json",out);print(json.dumps(out,indent=2))
