from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parent
P=R.parent.parent
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
manifest=json.loads((R/"file_manifest.json").read_text())
for name,digest in manifest.items():
    assert Path(name).name==name and "/" not in name and "\\" not in name,name
    assert sha(R/name)==digest,name
completion=json.loads((R/"completion.json").read_text())
assert completion["batch_complete"] and not completion["overall_project_complete"]
assert completion["native_models_validated"]==3 and not completion["valid_target_neural_comparison"]
assert len(manifest)==completion["top_level_files"]
assert sha(P/"CURRENT_PROJECT_STATUS.md")==completion["current_status_sha256"]
previous=json.loads((R.parent/"vehicle_matched_evidence/completion.json").read_text())
assert sha(R/"PROJECT_STATUS_BEFORE_NEURAL_ADMISSION.md")==previous["current_status_sha256"]
inventory=json.loads((R/"source_inventory.json").read_text())
assert sha(R/"author_repository.zip")==inventory["zip_sha256"]
for item in inventory["extracted_source"]:
    assert sha(R/"author_source"/item["path"])==item["sha256"],item["path"]
models=json.loads((R/"model_admission_audit.json").read_text())
for item in models["models"]:
    assert sha(R/item["weight_file"])==item["weight_sha256"]
    assert not item["target_admission"]["admitted"] and not item["dataset_inference_executed"]
for path,digest in models["unchanged_prior_hashes"].items():assert sha(Path(path))==digest
for path in R.glob("*.py"):compile(path.read_text(encoding="utf-8"),str(path),"exec")
verification=json.loads((R/"independent_verification.json").read_text())
assert verification["passed"] and verification["native_architecture_weight_pairs_reverified"]==3
assert verification["real_dataset_neural_forward_count"]==0
out={"passed":True,"top_level_files_verified":len(manifest),"source_archive_verified":True,
     "extracted_source_files_verified":len(inventory["extracted_source"]),"weight_files_verified":3,
     "prior_results_unchanged":True,"predecessor_status_verified":True,"current_status_verified":True,
     "top_level_python_compilation_passed":True,"valid_target_neural_comparison":False,
     "overall_project_complete":False}
(R/"package_audit.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
print(json.dumps(out,indent=2))
