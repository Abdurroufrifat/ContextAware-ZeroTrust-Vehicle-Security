from pathlib import Path
import json,hashlib
R=Path(__file__).resolve().parent;C=R.parent/"vehicle_crossmodal_doppler"
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(8*1024**2),b""):h.update(b)
 return h.hexdigest()
for n,h in json.loads((R/"file_manifest.json").read_text()).items():assert Path(n).name==n and sha(R/n)==h,n
c=json.loads((R/"completion.json").read_text());assert not c["overall_project_complete"] and c["batch_complete"]
assert sha(R.parent.parent/"CURRENT_PROJECT_STATUS.md")==c["current_status_sha256"]
assert sha(R/"PROJECT_STATUS_BEFORE_TIMING.md")==json.loads((C/"full_completion.json").read_text())["current_status_sha256"]
for p in R.glob("*.py"):compile(p.read_text(encoding="utf-8"),str(p),"exec")
out={"passed":True,"module_files":c["files"],"current_status_verified":True,"preceding_status_preserved_exactly":True,"overall_project_complete":False,"physical_witness_admitted":False}
(R/"package_audit.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
print(json.dumps(out,indent=2))
