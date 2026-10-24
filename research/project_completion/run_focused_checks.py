import json,re,subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parent;B=R.parent;result={}
for name,folder in [("completion",R),("quality",B/"vehicle_quality_calibration"),("diagnosis",B/"vehicle_motion_diagnosis"),("packet",B/"vehicle_packet_map"),("full_map",B/"vehicle_lidar_map"),("compiled_backend",B/"vehicle_gicp_comparison"),("stream",B/"vehicle_lidar_stream"),("radar_timing",B/"vehicle_radar_context"),("CAN_context",B/"vehicle_context_policy"),("authentication",B/"vehicle_evidence_auth")]:
 p=subprocess.run([sys.executable,"-m","unittest","discover","-s",str(folder),"-p","test_*.py","-v"],capture_output=True,text=True)
 log=p.stdout+p.stderr;(R/("tests_"+name+".log")).write_text(log)
 m=re.search(r"Ran (\d+) tests?",log);result[name]=dict(exit_code=p.returncode,count=int(m.group(1)) if m else None)
 print(name,json.dumps(result[name]),flush=True)
(R/"focused_test_validation.json").write_text(json.dumps(result,indent=2)+"\n")
sys.exit(int(any(v["exit_code"] for v in result.values())))
