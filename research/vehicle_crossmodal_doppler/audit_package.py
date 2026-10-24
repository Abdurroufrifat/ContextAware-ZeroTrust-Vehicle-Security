from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parent
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(8*1024**2),b""):h.update(b)
 return h.hexdigest()
for name,digest in json.loads((R/"file_manifest.json").read_text()).items():
 assert Path(name).name==name and sha(R/name)==digest,name
c=json.loads((R/"completion.json").read_text());assert sha(R.parent.parent/"CURRENT_PROJECT_STATUS.md")==c["current_status_sha256"]
assert not c["overall_project_complete"] and not c["full_sequence_study_complete"]
for p in R.glob("*.py"):compile(p.read_text(encoding="utf-8"),str(p),"exec")
out={"passed":True,"module_files_verified":c["module_files"],"current_status_verified":True,"syntax_checked_all_module_scripts":True,"full_sequence_study_complete":False,"physical_witness_admitted":False}
(R/"package_audit.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
print(json.dumps(out,indent=2))
