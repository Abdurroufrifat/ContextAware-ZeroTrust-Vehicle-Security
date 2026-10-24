from pathlib import Path
import argparse,sys,json,subprocess,os,hashlib,datetime
P=Path(__file__).resolve().parent;R=P/"research/vehicle_evidence_lifecycle"
def main():
 p=argparse.ArgumentParser(description="CO-ACT-CPS evidence lifecycle simulation");p.add_argument("command",choices=("status","resume","report"),nargs="?",default="status");a=p.parse_args()
 if a.command=="status":
  print("Saved confirmation markers:",len(list((R/"confirmation_v2/confirmation").glob("trial_*/attempt_*/complete.json"))),"/1080 (qualification requires saved audit)")
  for name in ("runtime/pipeline_state.json","final_seal.json"):
   f=R/name
   if f.exists():
    value=json.loads(f.read_text())
    if name=="final_seal.json":assert value["package_manifest_sha256"]==hashlib.sha256((R/"package_manifest.json").read_bytes()).hexdigest()
    print(json.dumps(value,indent=2))
  return 0
 if a.command=="report":
  for name in ("RESULTS.md","lifecycle_results.png","operator_viewer.html"):print(R/"confirmation_v2"/name)
  return 0
 lock=R/"runtime/pipeline.lock"
 if lock.exists():
  owner=json.loads(lock.read_text());q='Get-CimInstance Win32_Process -Filter "ProcessId='+str(owner["pid"])+'" | Select-Object -ExpandProperty CommandLine'
  r=subprocess.run(["powershell","-NoProfile","-Command",q],capture_output=True,text=True);assert r.returncode==0,r.stderr
  if "finish.py" in r.stdout and "vehicle_evidence_lifecycle" in r.stdout:print("Lifecycle pipeline is active; use status.");return 2
  lock.rename(lock.with_name("pipeline_stale_"+datetime.datetime.now().strftime("%Y%m%dT%H%M%S")+".json"))
 env=os.environ.copy();env["PYTHONPATH"]=os.pathsep.join([str(P/"src"),str(P/"scripts"),str(R),env.get("PYTHONPATH","")])
 return subprocess.call([sys.executable,str(R/"finish.py")],cwd=R,env=env)
if __name__=="__main__":raise SystemExit(main())
