import json,re
from pathlib import Path
R=Path(__file__).resolve().parent;P=R.parent.parent
files=sorted((P/"tests").glob("test_*.py"));hits=[]
for f in files:
 for i,line in enumerate(f.read_text(encoding="utf-8",errors="replace").splitlines(),1):
  if re.search(r"urban02|rav4|vehicle_final_evaluation|urlopen|requests\.|download|https?://",line,re.I):hits.append(dict(file=f.name,line=i,text=line.strip()[:180]))
prior=json.loads((P/"research"/"vehicle_signed_axis_correction"/"prior_snapshot_manifest.json").read_text())
protected=[n for n in prior if "vehicle_final_evaluation" in n.lower() or "rav4" in n.lower() or ("urban02" in n.lower() and "night" in n.lower())]
result=dict(core_test_files=len(files),hits=hits,axis_verifier_prior_protected_paths=protected)
(R/"core_preflight.json").write_text(json.dumps(result,indent=2))
print(json.dumps(dict(core_test_files=len(files),hits=hits[:35],total_hits=len(hits),axis_verifier_prior_protected_count=len(protected)),indent=2))
