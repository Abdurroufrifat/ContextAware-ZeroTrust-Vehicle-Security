"""Keep this batch's validation logs separate from earlier manifests."""
import json,re,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
results={}
for name,folder in [("stream",ROOT),("radar_timing",ROOT.parent/"vehicle_radar_context"),
    ("CAN_context",ROOT.parent/"vehicle_context_policy"),("authentication",ROOT.parent/"vehicle_evidence_auth")]:
    p=subprocess.run([sys.executable,"-m","unittest","discover","-s",str(folder),"-p","test_*.py","-v"],capture_output=True,text=True)
    log=p.stdout+p.stderr;(ROOT/("tests_"+name+".log")).write_text(log)
    n=re.search(r"Ran (\d+) tests",log)
    results[name]=dict(exit_code=p.returncode,count=int(n.group(1)) if n else None,tail=log.splitlines()[-4:])
    print(name,json.dumps(results[name]),flush=True)
(ROOT/"test_validation.json").write_text(json.dumps(results,indent=2)+"\n")
sys.exit(1 if any(v["exit_code"] for v in results.values()) else 0)
