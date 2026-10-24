from shared import *
import numpy as np
manifest=read("file_manifest.json");completion=read("completion.json")
for name,digest in manifest.items():
    assert Path(name).name==name and "/" not in name and "\\" not in name
    assert sha(R/name)==digest,name
assert len(manifest)==completion["top_level_files"] and completion["batch_complete"] and not completion["overall_project_complete"]
assert sha(P/"CURRENT_PROJECT_STATUS.md")==completion["current_status_sha256"]
for path,digest in read("input_lock.json").items():
    source=R/"PROJECT_STATUS_BEFORE_INERTIAL.md" if Path(path)==P/"CURRENT_PROJECT_STATUS.md" else Path(path)
    assert sha(source)==digest,path
previous=json.loads((R.parent/"vehicle_neural_adaptation/completion.json").read_text())
assert sha(R/"PROJECT_STATUS_BEFORE_INERTIAL.md")==previous["current_status_sha256"]
verify_lock()
for name,digest in read("detail_experiment_lock.json").items():assert sha(R/name)==digest
details=read("details_summary.json")
for name,digest in details["initial_outcomes_preserved"].items():assert sha(R/name)==digest
assert read("independent_verification.json")["passed"] and read("code_review.json")["unresolved_important_findings"]==0
tests=read("test_validation.json")
assert tests["core"]["exit_code"]==tests["focused"]["exit_code"]==0 and tests["denials"]==[]
assert tests["focused"]["counts"]["passed"]==134 and tests["core"]["counts"]["passed"]==731
total=0
for seq in ["Library01","Library02"]:
    d=np.load(R/(seq+"_inertial.npz"),allow_pickle=False);extra=np.load(R/(seq+"_details.npz"),allow_pickle=False)
    np.testing.assert_array_equal(extra["new_query_reasons"]=="accepted",d["new_accepted"])
    np.testing.assert_array_equal(extra["additive_forgery_accepted"],d["accepted"])
    np.testing.assert_allclose(extra["additive_forgery_residual"],d["residual"],atol=1e-12,rtol=0,equal_nan=True)
    assert sum(details["sequences"][seq]["new_query_reason_counts"].values())==len(d["queries"])
    total+=len(d["queries"])
for path in R.glob("*.py"):compile(path.read_text(encoding="utf-8"),str(path),"exec")
out={"passed":True,"top_level_files_verified":len(manifest),"full_queries":total,
 "old_results_and_status_predecessor_verified":True,"initial_outcomes_preserved":True,
 "separate_review_corrections_verified":True,"current_status_verified":True,"python_compilation_passed":True,
 "focused_passed":134,"core_passed":731,"core_skipped":4,"overall_project_complete":False}
write("package_audit.json",out);print(json.dumps(out,indent=2))
