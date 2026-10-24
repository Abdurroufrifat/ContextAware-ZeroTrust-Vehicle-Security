from shared import *
import numpy as np
manifest=read("file_manifest.json");completion=read("completion.json")
for name,digest in manifest.items():
    assert Path(name).name==name and "/" not in name and "\\" not in name
    assert sha(R/name)==digest,name
assert len(manifest)==completion["top_level_files"] and completion["batch_complete"]
assert not completion["overall_project_complete"] and sha(P/"CURRENT_PROJECT_STATUS.md")==completion["current_status_sha256"]
data=read("data_audit.json")
for path,digest in data["old_hashes"].items():
    expected=R/"PROJECT_STATUS_BEFORE_ADAPTATION.md" if Path(path)==P/"CURRENT_PROJECT_STATUS.md" else Path(path)
    assert sha(expected)==digest,path
predecessor=json.loads((R.parent/"vehicle_neural_baseline/completion.json").read_text())
assert sha(R/"PROJECT_STATUS_BEFORE_ADAPTATION.md")==predecessor["current_status_sha256"]
verify_lock()
assert read("independent_verification.json")["passed"] and read("raw_feature_verification.json")["passed"]
assert read("code_review.json")["unresolved_important_findings"]==0
tests=read("test_validation.json")
assert tests["focused"]["exit_code"]==tests["core"]["exit_code"]==0 and tests["denials"]==[]
assert tests["focused"]["counts"]["passed"]==110 and tests["core"]["counts"]["passed"]==731
for path in R.glob("*.py"):compile(path.read_text(encoding="utf-8"),str(path),"exec")
p=np.load(R/"predictions.npz",allow_pickle=False);assert len(p["stamps"])==564
assert sum(p["roles"]=="temporal_test")==61 and sum(p["roles"]=="transfer")==248
out={"passed":True,"top_level_files_verified":len(manifest),"attempts":564,"neural_seeds":3,
 "prior_results_and_predecessor_status_verified":True,"current_status_verified":True,
 "top_level_python_compilation_passed":True,"focused_passed":110,"core_passed":731,"core_skipped":4,
 "overall_project_complete":False}
write("package_audit.json",out);print(json.dumps(out,indent=2))
