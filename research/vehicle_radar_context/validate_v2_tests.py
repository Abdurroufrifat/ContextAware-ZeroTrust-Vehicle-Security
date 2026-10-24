import json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
results={}
for name,folder in [("radar",ROOT),("CAN_context",ROOT.parent/"vehicle_context_policy"),("authentication",ROOT.parent/"vehicle_evidence_auth")]:
    p=subprocess.run([sys.executable,"-m","unittest","discover","-s",str(folder),"-p","test_*.py","-v"],capture_output=True,text=True)
    (ROOT/("tests_v2_"+name+".log")).write_text(p.stdout+p.stderr)
    results[name]=dict(exit_code=p.returncode,tail=(p.stdout+p.stderr).splitlines()[-4:])
    print(name,json.dumps(results[name]),flush=True)
(ROOT/"test_validation_v2.json").write_text(json.dumps(results,indent=2)+"\n")
sys.exit(1 if any(r["exit_code"] for r in results.values()) else 0)
