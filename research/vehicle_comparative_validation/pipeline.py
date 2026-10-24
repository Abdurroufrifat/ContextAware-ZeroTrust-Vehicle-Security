"""Finish the saved study automatically; fail closed at every evidence gate."""
from cv_common import B,P,sha,dump
import os,sys,json,subprocess,time,datetime
def state(stage,status,detail=None):
 dump(B/"runtime/pipeline_state.json",{"stage":stage,"status":status,"detail":detail,"utc":datetime.datetime.now(datetime.timezone.utc).isoformat()})
def verify_locks():
 for name in ("experiment_lock.json","verification_lock.json"):
  assert all(sha(P/n)==h for n,h in json.loads((B/name).read_text()).items()),name
def run(name,args,cwd=None):
 state(name,"RUNNING");print("START",name,flush=True)
 with (B/"runtime"/(name+".log")).open("w",encoding="utf-8") as f:
  rc=subprocess.call([sys.executable,*args],cwd=cwd or B,env=ENV,stdout=f,stderr=subprocess.STDOUT)
 if rc:
  state(name,"FAILED",{"returncode":rc,"log":str(B/"runtime"/(name+".log"))})
  print((B/"runtime"/(name+".log")).read_text(encoding="utf-8")[-5000:],flush=True)
  raise RuntimeError(name+" exit "+str(rc))
 print("PASS",name,flush=True);return {"name":name,"returncode":rc}
ENV=os.environ.copy();ENV["PYTHONPATH"]=os.pathsep.join([str(P/"src"),str(P/"scripts"),str(B),ENV.get("PYTHONPATH","")])
def main():
 lock=B/"runtime/pipeline.lock";B.joinpath("runtime").mkdir(exist_ok=True)
 fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,json.dumps({"pid":os.getpid(),"script":str(B/"pipeline.py")}).encode());os.close(fd)
 try:
  verify_locks()
  checks=[]
  checks.append(run("comparative_tests",["-m","pytest","-q",str(B)]))
  checks.append(run("predecessor_component_tests",["-m","pytest","-q",*[str(P/"research/vehicle_closed_loop_cps"/n) for n in ("test_control_core.py","test_cps_session.py","test_simulator_insertion.py","test_report_core.py")]],P))
  checks.append(run("project_tests",["-m","pytest"],P))
  dump(B/"runtime/preflight_tests.json",{"passed":True,"checks":checks,"verification_lock_digest":sha(B/"verification_lock.json")})
  state("confirmation","WAITING_FOR_SAVED_RESULTS")
  while not (B/"results.json").exists():
   time.sleep(30)
   text=(B/"runtime/confirmation_console.log").read_text()
   if "Traceback" in text:
    state("confirmation","FAILED","See preserved confirmation_console.log; resume only unfinished IDs.")
    raise RuntimeError("confirmation process reported exception")
   if not (B/"results.json").exists():
    query="Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python' -and $_.CommandLine -like '*vehicle_comparative_validation*runner.py*' } | Select-Object -ExpandProperty ProcessId"
    active=subprocess.run(["powershell","-NoProfile","-Command",query],capture_output=True,text=True)
    assert active.returncode==0,active.stderr
    if not active.stdout.strip():run("resumed_confirmation",[str(B/"runner.py")])
  verify_locks()
  run("independent_confirmation",[str(B/"verify_study.py")])
  run("strict_packet_replay",[str(B/"replay_policy.py")])
  run("comparison_report",[str(B/"build_report.py")])
  run("report_viewer_verification",[str(B/"verify_report.py")])
  verify_locks()
  run("seal_release",[str(B/"seal_release.py")])
  state("complete","COMPLETE_FOR_COMPARATIVE_SIMULATION_SCOPE",{"trials":1512,"decisions":302400,"original_trials_preserved":540})
  print("ALL_COMPARATIVE_SIMULATION_GATES_PASSED",flush=True)
 except Exception as e:
  if not (B/"runtime/pipeline_state.json").exists() or json.loads((B/"runtime/pipeline_state.json").read_text())["status"]!="FAILED":state("pipeline","FAILED",repr(e))
  raise
 finally:lock.unlink(missing_ok=True)
if __name__=="__main__":main()
