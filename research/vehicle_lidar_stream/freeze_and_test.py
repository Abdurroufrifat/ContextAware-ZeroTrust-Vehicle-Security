import hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
h=hashlib.sha256((ROOT/"protocol_v1.json").read_bytes()).hexdigest()
p=ROOT/"protocol_freeze.json"
if p.exists(): assert json.loads(p.read_text())["sha256"]==h
else: p.write_text(json.dumps(dict(sha256=h,frozen_before_motion_outcomes=True),indent=2)+"\n")
r=subprocess.run([sys.executable,"-m","unittest","discover","-s",str(ROOT),"-p","test_*.py","-v"],capture_output=True,text=True)
(ROOT/"tests.log").write_text(r.stdout+r.stderr)
print(r.stdout+r.stderr);sys.exit(r.returncode)
