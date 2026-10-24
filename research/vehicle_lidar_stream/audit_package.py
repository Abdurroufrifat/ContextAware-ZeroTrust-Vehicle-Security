import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
checks={}
for folder,filename in [(ROOT,"file_manifest.json"),(ROOT.parent/"vehicle_radar_context","source_timing_file_manifest.json"),
    (ROOT.parent/"vehicle_radar_context","balanced_file_manifest.json")]:
    manifest=json.loads((folder/filename).read_text())
    for name,sha in manifest.items():
        p=folder/name
        assert p.parent==folder and hashlib.sha256(p.read_bytes()).hexdigest()==sha,(filename,name)
    checks[filename]=len(manifest)
completion=json.loads((ROOT/"completion.json").read_text())
summary=json.loads((ROOT/"stream_benchmark_summary.json").read_text())
verify=json.loads((ROOT/"independent_verification.json").read_text())
assert verify["all_passed"] and completion["tests_passed"]==100
assert summary["metrics"]["geometry_accepted"]["discrepancy_above_2_mps"]==4
assert summary["metrics"]["deadline_eligible"]["accepted"]==0
assert not completion["method_promoted"] and not completion["full_project_complete"]
result=dict(package_hashes=checks,prior_result_packages_unchanged=True,failed_candidate_excluded=True,full_project_complete=False)
(ROOT/"package_audit.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))
