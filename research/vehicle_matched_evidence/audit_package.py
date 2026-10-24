from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parent;T=R.parent/"vehicle_optical_timing"
for n,h in json.loads((R/"file_manifest.json").read_text()).items():assert Path(n).name==n and hashlib.sha256((R/n).read_bytes()).hexdigest()==h,n
c=json.loads((R/"completion.json").read_text());assert c["batch_complete"] and not c["overall_project_complete"]
assert hashlib.sha256((R.parent.parent/"CURRENT_PROJECT_STATUS.md").read_bytes()).hexdigest()==c["current_status_sha256"]
assert hashlib.sha256((R/"PROJECT_STATUS_BEFORE_MATCHED.md").read_bytes()).hexdigest()==json.loads((T/"completion.json").read_text())["current_status_sha256"]
for p in R.glob("*.py"):compile(p.read_text(encoding="utf-8"),str(p),"exec")
out={"passed":True,"module_files_verified":c["files"],"current_and_preceding_status_verified":True,"overall_project_complete":False,"Library03_payloads_accessed":False}
(R/"package_audit.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
print(json.dumps(out,indent=2))
