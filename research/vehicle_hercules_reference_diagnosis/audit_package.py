from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parent
for name,digest in json.loads((R/"file_manifest.json").read_text()).items():
 assert Path(name).name==name
 assert hashlib.sha256((R/name).read_bytes()).hexdigest()==digest,name
completion=json.loads((R/"completion.json").read_text())
assert hashlib.sha256((R.parent.parent/"CURRENT_PROJECT_STATUS.md").read_bytes()).hexdigest()==completion["current_status_sha256"]
assert not completion["overall_project_complete"]
result={"passed":True,"module_files_verified":completion["module_files"],"current_status_verified":True,"scope":"explicit module artifacts; source/model identities verified separately"}
(R/"package_audit.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
print(json.dumps(result,indent=2))
