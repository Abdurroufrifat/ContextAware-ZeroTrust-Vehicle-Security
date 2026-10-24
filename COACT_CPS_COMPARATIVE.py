"""User entry point for the comparative closed-loop simulation extension."""
from pathlib import Path
import argparse,json,sys,subprocess,os,hashlib,datetime
P=Path(__file__).resolve().parent;B=P/"research"/"vehicle_comparative_validation"
def main():
 parser=argparse.ArgumentParser(description="CO-ACT-CPS frozen comparative simulation")
 parser.add_argument("command",nargs="?",choices=("status","verify","report","operator","resume"),default="status")
 parser.add_argument("--scenario",default="scene_disinformation");parser.add_argument("--step",type=int,default=80)
 args=parser.parse_args()
 env=os.environ.copy();env["PYTHONPATH"]=os.pathsep.join([str(P/"src"),str(P/"scripts"),str(B),env.get("PYTHONPATH","")])
 if args.command=="status":
  markers=list((B/"confirmation").glob("trial_*/attempt_*/complete.json"))
  print("Saved completion markers:",len(markers),"/1512 (integrity qualification requires verify)")
  for n in ("runtime/pipeline_state.json","final_seal.json"):
   p=B/n
   if p.exists():
    value=json.loads(p.read_text())
    if n=="final_seal.json":
     assert value["package_manifest_sha256"]==hashlib.sha256((B/"package_manifest.json").read_bytes()).hexdigest()
     assert value["development_manifest_sha256"]==hashlib.sha256((B/"development_manifest.json").read_bytes()).hexdigest()
    print(json.dumps(value,indent=2))
  return 0
 if args.command=="report":
  for n in ("RESULTS.md","BASELINE_COMPARISON.md","comparative_results.png","operator_viewer.html"):
   p=B/n;print(str(p) if p.exists() else "Pending: "+str(p))
  return 0
 if args.command=="operator":
  p=B/"operator_replays.json"
  if not p.exists():print("Recorded operator artifacts await verified completion.");return 2
  data=json.loads(p.read_text())["replays"];r=next((r for r in data if r["case"]==args.scenario),None)
  if r is None or not 0<=args.step<200:parser.error("unknown condition or decision index")
  print(json.dumps(r["cards"][args.step],indent=2));print("Viewer:",B/"operator_viewer.html");return 0
 if args.command=="resume":
  lock=B/"runtime/pipeline.lock"
  if lock.exists():
   owner=json.loads(lock.read_text());pid=owner["pid"]
   query="Get-CimInstance Win32_Process -Filter \"ProcessId="+str(pid)+"\" | Select-Object -ExpandProperty CommandLine"
   active=subprocess.run(["powershell","-NoProfile","-Command",query],capture_output=True,text=True)
   assert active.returncode==0,active.stderr
   if "pipeline.py" in active.stdout and "vehicle_comparative_validation" in active.stdout:print("Completion pipeline is active and monitors runner liveness. Use status.");return 2
   lock.rename(lock.with_name("pipeline_stale_"+datetime.datetime.now().strftime("%Y%m%dT%H%M%S")+".json"))
  return subprocess.call([sys.executable,str(B/"pipeline.py")],cwd=B,env=env)
 for n in ("verify_study.py","replay_policy.py","verify_report.py"):
  rc=subprocess.call([sys.executable,str(B/n)],cwd=B,env=env)
  if rc:return rc
 for name in ("experiment_lock.json","verification_lock.json","package_manifest.json","development_manifest.json"):
  contents=json.loads((B/name).read_text());bindings=contents["files"] if "files" in contents else contents
  assert all(hashlib.sha256((P/n).read_bytes()).hexdigest()==h for n,h in bindings.items()),name
 seal=json.loads((B/"final_seal.json").read_text())
 assert seal["status"]=="COMPLETE_FOR_COMPARATIVE_SIMULATION_SCOPE"
 assert seal["package_manifest_sha256"]==hashlib.sha256((B/"package_manifest.json").read_bytes()).hexdigest()
 assert seal["development_manifest_sha256"]==hashlib.sha256((B/"development_manifest.json").read_bytes()).hexdigest()
 print("COMPARATIVE_SAVED_ARTIFACT_VERIFICATION_PASS");return 0
if __name__=="__main__":raise SystemExit(main())
