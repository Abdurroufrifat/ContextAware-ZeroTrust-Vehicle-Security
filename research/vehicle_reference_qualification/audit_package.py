from shared import *
manifest=read("file_manifest.json");completion=read("completion.json")
for name,digest in manifest.items():
    assert Path(name).name==name and "/" not in name and "\\" not in name
    assert sha(R/name)==digest,name
assert len(manifest)==completion["top_level_files"] and completion["batch_complete"] and not completion["overall_project_complete"]
assert sha(P/"CURRENT_PROJECT_STATUS.md")==completion["current_status_sha256"]
for path,digest in read("input_lock.json").items():
    source=R/"PROJECT_STATUS_BEFORE_REFERENCE_AUDIT.md" if Path(path)==P/"CURRENT_PROJECT_STATUS.md" else Path(path)
    assert sha(source)==digest,path
prior=json.loads((R.parent/"vehicle_inertial_consistency/completion.json").read_text())
assert sha(R/"PROJECT_STATUS_BEFORE_REFERENCE_AUDIT.md")==prior["current_status_sha256"]
verify_lock()
assert read("independent_verification.json")["passed"] and read("code_review.json")["unresolved_important_findings"]==0
assert read("summary.json")["verified_reference_error_bound_mps"] is None
assert not read("RESEARCH_DECISION.json")["qualified_independent_absolute_reference_found"]
tests=read("test_validation.json")
assert tests["core"]["exit_code"]==tests["focused"]["exit_code"]==0 and tests["denials"]==[]
assert tests["focused"]["counts"]["passed"]==150 and tests["core"]["counts"]["passed"]==731
for p in R.glob("*.py"):compile(p.read_text(encoding="utf-8"),str(p),"exec")
out={"passed":True,"top_level_files_verified":len(manifest),"old_results_and_status_predecessor_verified":True,
 "current_status_verified":True,"python_compilation_passed":True,"focused_passed":150,"core_passed":731,"core_skipped":4,
 "actual_absolute_reference_qualified":False,"overall_project_complete":False}
write("package_audit.json",out);print(json.dumps(out,indent=2))
