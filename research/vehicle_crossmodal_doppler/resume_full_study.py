"""Explicit future resumption; no timer, quota bypass or production promotion."""
from pathlib import Path
import subprocess,sys,json,hashlib
R=Path(__file__).resolve().parent
for n,h in json.loads((R/"prefix_experiment_lock.json").read_text()).items():
 assert hashlib.sha256((R/n).read_bytes()).hexdigest()==h,n
tests=json.loads((R/"test_validation.json").read_text())
assert tests["focused"]["exit_code"]==tests["core"]["exit_code"]==0
for name in ["acquire_full_optical.py","run_full_stream.py","verify_full_study.py"]:
 subprocess.run([sys.executable,str(R/name)],cwd=R,check=True)
print("Full development replay and arithmetic audit complete. Read full_summary.json; numerical gates may fail. Overall project and physical certification remain incomplete.")
