"""Freeze reference-free physical-scan predictions before any reference scoring."""
from common import *
install_guard()
import numpy as np
from boreas_core import estimate_frame,predict
assert not (R/"predictions.json").exists(),"Predictions already frozen"
membership=json.loads((R/"membership.json").read_text());data=json.loads((R/"data_lock.json").read_text())
acquired={e["key"]:e for e in data["objects"]}
assert set(acquired)=={e["key"] for e in membership["objects"]}
names=("PROTOCOL.md","SCHEMA_ADDENDUM.md","membership.json","data_lock.json","common.py","boreas_core.py","test_boreas_core.py","predict_scans.py")
lock={str((R/n).relative_to(P)):sha(R/n) for n in names};write("input_lock.json",lock)
rows=[]
frames=[e for e in membership["objects"] if e["kind"]=="frame"]
assert len(frames)==96
for index,e in enumerate(frames):
 record=acquired[e["key"]]
 try:raw=checked(R/"data"/e["key"]).read_bytes() if record["status"]=="downloaded" else None
 except OSError:raw=None
 for method in ("ols3","huber3","huber6time"):
  prepared=estimate_frame(raw,record.get("sha256"),method);solution=prepared["solution"]
  rows.append({"frame_id":index,"key":e["key"],"method":method,"raw_points":prepared["raw_points"],"flags_nonzero":prepared["flags_nonzero"],
   "solution":solution,"query_vectors":{str(age):(predict(solution,age).tolist() if solution["status"]=="estimated" else None) for age in (0,50000,100000)}})
 if (index+1)%24==0:print("predicted",index+1,"/",96,flush=True)
assert len(rows)==288 and all(sha(P/n)==digest for n,digest in lock.items())
write("predictions.json",{"rows":rows,"methods":3,"frames":96,"reference_used_for_fit":False,"physical_error_certificate_established":False,"novelty_established":False})
print("FROZEN 288 reference-free fits; reference values not parsed or used for fitting",flush=True)
