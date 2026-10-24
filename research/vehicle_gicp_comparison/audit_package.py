import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
packages={}
for folder,manifest in [(ROOT,"file_manifest.json"),(ROOT.parent/"vehicle_lidar_stream","file_manifest.json"),
    (ROOT.parent/"vehicle_radar_context","source_timing_file_manifest.json"),
    (ROOT.parent/"vehicle_radar_context","balanced_file_manifest.json")]:
    expected=json.loads((folder/manifest).read_text())
    for name,sha in expected.items():
        p=folder/name
        assert p.parent==folder and hashlib.sha256(p.read_bytes()).hexdigest()==sha,(folder.name,name)
    packages[folder.name+"/"+manifest]=len(expected)
s=json.loads((ROOT/"comparison_summary.json").read_text())
v=json.loads((ROOT/"independent_verification.json").read_text())
assert v["all_passed"] and v["variant_outputs_checked"]==1132
assert not s["full_raw_replay_trigger_passed"] and not s["promotion_allowed"]
assert all(s["metrics"][method]["accepted"]==0 for method in ("ICP","PLANE_ICP","GICP"))
result=dict(all_passed=True,package_files=packages,prior_packages_unchanged=True,
    full_raw_replay_correctly_skipped=True,full_project_complete=False)
(ROOT/"package_audit.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))
