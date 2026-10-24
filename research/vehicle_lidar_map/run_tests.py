import json,re,subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parent
result={}
for name,folder in [("map",R),("compiled_backend",R.parent/"vehicle_gicp_comparison"),("stream",R.parent/"vehicle_lidar_stream"),("radar_timing",R.parent/"vehicle_radar_context"),("CAN_context",R.parent/"vehicle_context_policy"),("authentication",R.parent/"vehicle_evidence_auth")]:
 p=subprocess.run([sys.executable,"-m","unittest","discover","-s",str(folder),"-p","test_*.py","-v"],capture_output=True,text=True)
 log=p.stdout+p.stderr
 (R/("tests_"+name+".log")).write_text(log)
 count=re.search(r"Ran (\d+) tests?",log)
 result[name]=dict(exit_code=p.returncode,count=int(count.group(1)) if count else None,tail=log.splitlines()[-5:])
 print(name,json.dumps(result[name]),flush=True)
(R/"test_validation.json").write_text(json.dumps(result,indent=2)+"\n")
sys.exit(int(any(v["exit_code"] for v in result.values())))
