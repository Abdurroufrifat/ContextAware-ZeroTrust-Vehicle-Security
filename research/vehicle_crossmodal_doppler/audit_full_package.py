from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parent
for n,h in json.loads((R/"full_file_manifest.json").read_text()).items():
 assert Path(n).name==n
 assert hashlib.sha256((R/n).read_bytes()).hexdigest()==h,n
c=json.loads((R/"full_completion.json").read_text());assert c["full_development_study_complete"] and not c["overall_project_complete"]
assert hashlib.sha256((R.parent.parent/"CURRENT_PROJECT_STATUS.md").read_bytes()).hexdigest()==c["current_status_sha256"]
v=json.loads((R/"full_independent_verification.json").read_text());assert v["passed"] and c["optical_frames"]==v["optical_frames"] and c["queries"]==v["fixed_interval_requests"]
for p in R.glob("*.py"):compile(p.read_text(encoding="utf-8"),str(p),"exec")
out={"passed":True,"small_artifacts_verified":c["module_small_artifacts"],"full_queries_verified":c["queries"],"full_optical_frames":c["optical_frames"],"current_status_verified":True,"raw_source_hashes_verified_by_full_independent_audit":True,"overall_project_complete":False}
(R/"full_package_audit.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
print(json.dumps(out,indent=2))
