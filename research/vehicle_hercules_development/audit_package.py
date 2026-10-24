from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parent
files=json.loads((R/"file_manifest.json").read_text())
for name,digest in files.items():
 assert Path(name).name==name
 assert hashlib.sha256((R/name).read_bytes()).hexdigest()==digest,name
c=json.loads((R/"completion.json").read_text())
assert hashlib.sha256((R.parent.parent/"CURRENT_PROJECT_STATUS.md").read_bytes()).hexdigest()==c["current_status_sha256"]
assert c["overall_project_complete"] is False
out={"passed":True,"module_files_verified":len(files),"development_inputs_previously_verified":14,"current_status_verified":True,"scope":"explicit module-only artifact audit"}
(R/"package_audit.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
print(json.dumps(out,indent=2))
