from pathlib import Path
import json,hashlib
R=Path(__file__).resolve().parent
p=R/"run_development.py";s=p.read_text()
s=s.replace('tic=time.perf_counter_ns()\n   try:', 'tic=time.perf_counter_ns()\n   dirs=np.empty((0,3));vr=np.empty(0)\n   try:')
p.write_text(s,encoding="utf-8")
files=["EXPERIMENT_PROTOCOL.md","hercules_core.py","run_development.py","test_hercules_core.py","download_lock.json"]
assert not (R/"experiment_lock.json").exists()
(R/"experiment_lock.json").write_text(json.dumps({n:hashlib.sha256((R/n).read_bytes()).hexdigest() for n in files},indent=2),encoding="utf-8")
old=R.parent/"vehicle_independent_source_qualification"
s=(old/"run_checks.py").read_text()
s=s.replace('[str(R/"test_pilot_core.py"),str(R/"test_native_compat.py")]', '[str(R/"test_hercules_core.py"),str(R.parent/"vehicle_independent_source_qualification"/"test_pilot_core.py"),str(R.parent/"vehicle_independent_source_qualification"/"test_native_compat.py")]')
s=s.replace('if "vehicle_final_evaluation" in p or', 'if "/data/library03/" in p or "vehicle_final_evaluation" in p or')
(R/"run_checks.py").write_text(s,encoding="utf-8")
print("Frozen experiment identities and prepared isolated checks.")
