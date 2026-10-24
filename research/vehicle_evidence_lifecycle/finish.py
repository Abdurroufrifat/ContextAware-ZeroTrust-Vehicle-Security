"""Single-owner, resumable finish pipeline; no completed experiment is repeated."""
from lc_common import *
import os,sys,json,datetime,subprocess
def state(stage,status,detail=None):
 dump(R/"runtime/pipeline_state.json",{"stage":stage,"status":status,"detail":detail,"utc":datetime.datetime.now(datetime.timezone.utc).isoformat()})
def main():
 path=R/"runtime/pipeline.lock";fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,json.dumps({"pid":os.getpid(),"script":str(R/"finish.py")}).encode());os.close(fd)
 env=os.environ.copy();env["PYTHONPATH"]=os.pathsep.join([str(P/"src"),str(P/"scripts"),str(R),env.get("PYTHONPATH","")])
 def run(stage,args,cwd=None):
  state(stage,"RUNNING");print("START",stage,flush=True)
  f=R/"runtime"/(stage+"-"+datetime.datetime.now().strftime("%Y%m%dT%H%M%S")+".log")
  with f.open("w",encoding="utf-8") as out:rc=subprocess.call([sys.executable,*args],cwd=cwd or R,env=env,stdout=out,stderr=subprocess.STDOUT)
  if rc:state(stage,"FAILED",{"log":str(f),"returncode":rc});print(f.read_text()[-5000:],flush=True);raise RuntimeError(stage)
  print("PASS",stage,flush=True);return {"stage":stage,"returncode":0,"log":str(f)}
 try:
  assert all(sha(P/n)==h for n,h in json.loads((R/"confirmation_v2_lock.json").read_text()).items())
  if (R/"final_seal.json").exists():print("LIFECYCLE_ALREADY_SEALED",flush=True);return
  if not (R/"confirmation_v2/results.json").exists():run("confirmation",[str(R/"study_runner_v2.py"),"--phase","confirmation"])
  else:print("SAVED_CONFIRMATION_EXISTS_NO_EXPERIMENT_RERUN",flush=True)
  run("confirmation_audit",[str(R/"verify_v3.py"),"--phase","confirmation"])
  run("report",[str(R/"report.py")]);run("figures",[str(R/"make_figures.py")]);run("report_viewer_audit",[str(R/"verify_report.py")])
  checks=[run("component_tests",["-m","pytest","-q",str(R),"--basetemp",str(P/"research/project_release/runtime"/("lifecycle-final-"+datetime.datetime.now().strftime("%Y%m%dT%H%M%S")))]),run("project_tests",["-m","pytest"],P)]
  dump(R/"runtime/final_tests.json",{"passed":True,"checks":checks,"confirmation_lock_digest":sha(R/"confirmation_v2_lock.json")})
  run("seal",[str(R/"seal.py")]);state("complete","COMPLETE_FOR_LIFECYCLE_SIMULATION_SCOPE",{"trials":1080,"decisions":216000})
  print("ALL_LIFECYCLE_GATES_PASSED",flush=True)
 except Exception as e:
  current=json.loads((R/"runtime/pipeline_state.json").read_text()) if (R/"runtime/pipeline_state.json").exists() else {}
  if current.get("status")!="FAILED":state("pipeline","FAILED",repr(e))
  raise
 finally:path.unlink(missing_ok=True)
if __name__=="__main__":main()
