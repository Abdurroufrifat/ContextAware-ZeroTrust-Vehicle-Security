"""Combined final verification entry; no new experiments or physical commands."""
from sg_common import *
install_guard()
import subprocess
checks=[R/"verify_package.py",R/"verify_study.py",R.parent/"vehicle_boreas_rt_reference_study"/"run_all.py"]
if "--tests" in sys.argv:checks.append(R/"run_checks.py")
checks.append(R/"verify_package.py")
runs=[]
for script in checks:
 args=[sys.executable,str(checked(script))]
 if script.name=="run_checks.py":args.append("--recheck")
 run=subprocess.run(args,cwd=P,capture_output=True,text=True)
 runs.append({"script":str(script.relative_to(P)),"exit_code":run.returncode,"stdout":run.stdout,"stderr":run.stderr})
 print(script.parent.name+"/"+script.name+": "+("PASS" if run.returncode==0 else "FAIL"),flush=True)
 if run.returncode:
  print(run.stdout+run.stderr);break
target=R/"runtime"/"last_run.json";target.parent.mkdir(exist_ok=True)
checked(target).write_text(json.dumps(runs,indent=2)+"\n",encoding="utf-8")
if any(x["exit_code"] for x in runs):raise SystemExit(1)
print("PC_RESEARCH_EVALUATION_VERIFIED; simulation is not physical vehicle validation; novelty/acceptance are not certified.",flush=True)
