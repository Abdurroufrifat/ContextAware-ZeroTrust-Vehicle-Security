import json,re,subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parent
results={}
for name,folder in [("diagnosis",R),("packet",R.parent/"vehicle_packet_map"),("full_map",R.parent/"vehicle_lidar_map"),("compiled_backend",R.parent/"vehicle_gicp_comparison"),("stream",R.parent/"vehicle_lidar_stream"),("radar_timing",R.parent/"vehicle_radar_context"),("CAN_context",R.parent/"vehicle_context_policy"),("authentication",R.parent/"vehicle_evidence_auth")]:
 p=subprocess.run([sys.executable,"-m","unittest","discover","-s",str(folder),"-p","test_*.py","-v"],capture_output=True,text=True)
 log=p.stdout+p.stderr;(R/("tests_"+name+".log")).write_text(log)
 m=re.search(r"Ran (\d+) tests?",log);results[name]=dict(exit_code=p.returncode,count=int(m.group(1)) if m else None,tail=log.splitlines()[-5:])
 print(name,json.dumps(results[name]),flush=True)
(R/"test_validation.json").write_text(json.dumps(results,indent=2)+"\n")
sys.exit(int(any(x["exit_code"] for x in results.values())))
