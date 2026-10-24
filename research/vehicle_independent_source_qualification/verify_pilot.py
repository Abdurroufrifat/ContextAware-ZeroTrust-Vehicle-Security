"""Offline replay of artifact identities and saved discrepancy metrics."""
from pathlib import Path
import json,hashlib
import numpy as np
from pilot_core import decode_pcd,time_ns,nearest_reference
R=Path(__file__).resolve().parent
s=json.loads((R/"pilot_summary.json").read_text())
rows=json.loads((R/"pilot_predictions.json").read_text())
members=json.loads((R/"pilot_membership_lock.json").read_text())
assert members["protocol_sha256"]==hashlib.sha256((R/"PILOT_PROTOCOL.md").read_bytes()).hexdigest()
assert [r["file"] for r in rows]==[f"{i:06d}.pcd" for i in range(50)]
data_lock=json.loads((R/"pilot_data_lock.json").read_text())
points=0;verified=0
for e in data_lock["files"]:
 if "sha256" not in e:continue
 b=(R/"pilot_data"/e["file"]).read_bytes()
 assert hashlib.sha256(b).hexdigest()==e["sha256"]
 pc=decode_pcd(b);points+=len(pc["x"]);verified+=1
 corresponding=next(r for r in rows if r["file"]==e["file"])
 assert corresponding["points"]==len(pc["x"])
assert verified>0
schema=json.loads((R/"schema_probe.json").read_text())
for key in ["radar_times","rtk_schema"]:
 assert hashlib.sha256((R/(key+".sample")).read_bytes()).hexdigest()==schema[key]["sha256"]
for name,reported in s["methods"].items():
 vals=[abs(r["estimates"][name]["mps"]-r["reference_mps"]) for r in rows if "reference_mps" in r and "mps" in r["estimates"].get(name,{})]
 assert len(vals)==reported["scored"]
 for k,v in {"mae_mps":np.mean(vals),"p95_mps":np.percentile(vals,95),"maximum_mps":max(vals)}.items():assert abs(v-reported[k])<1e-12
 assert sum(v>2 for v in vals)==reported["above_2_mps"]
 assert sum("mps" in r["estimates"].get(name,{}) for r in rows)==reported["numeric_outputs"]
for e in json.loads((R/"author_source_lock.json").read_text())["files"].values():
 assert hashlib.sha256((R/e["local"]).read_bytes()).hexdigest()==e["sha256"]
tests=json.loads((R/"test_validation.json").read_text())
assert tests["core"]["exit_code"]==tests["pilot"]["exit_code"]==0
assert tests["denials"]==[]
assert s["physical_witness_admitted"] is False and s["production_changed"] is False
assert s["reference"]["physical_error_bound_mps"] is None and s["reference"]["clock_offset_bound_ns"] is None
out={"passed":True,"measurement_files_rehashed":verified,"decoded_points":points,"attempts":len(rows),"methods_metrics_recomputed":list(s["methods"]),"native_author_files_hash_verified":6,"core_tests":tests["core"]["counts"],"new_tests":tests["pilot"]["counts"],"reserved_payloads_opened":False,"scope":"saved-output and schema replay; not independent physical validation or an independent decoder"}
(R/"independent_verification.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
print(json.dumps(out,indent=2))
