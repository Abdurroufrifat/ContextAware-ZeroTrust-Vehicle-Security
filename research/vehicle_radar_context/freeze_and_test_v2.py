import hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
p=ROOT/"protocol_v2_freeze.json"
h=hashlib.sha256((ROOT/"protocol_v2.json").read_bytes()).hexdigest()
if p.exists(): assert json.loads(p.read_text())["sha256"]==h
else: p.write_text(json.dumps(dict(sha256=h,frozen_before_v2_data_evaluation=True),indent=2)+"\n")
# Prove the revised exact fixture also fails the original algorithm.
test=(ROOT/"test_balanced.py").read_text()
ns={"__name__":"regression_probe"}
exec(test,ns)
ns["fit_velocity"]=__import__("radar_motion").fit_velocity
a,b,s,true=ns["BalancedTests"]().scene()
r=ns["fit_velocity"](a,b,s)
assert r["accepted"] and abs(r["state"][0]-8)>2
print("RED confirmed: original point-count estimator returns",r["state"][0])
p=subprocess.run([sys.executable,"-m","unittest","discover","-s",str(ROOT),"-p","test_*.py","-v"],capture_output=True,text=True)
(ROOT/"tests_v2.log").write_text(p.stdout+p.stderr)
print(p.stdout+p.stderr);sys.exit(p.returncode)
