"""Verify external-recording comparison and inherited frozen research packages."""
from common import *
install_guard()
import subprocess
checks=[R/"verify_package.py",R/"verify_study.py",R.parent/"vehicle_command_gateway"/"run_all.py"]
if "--tests" in sys.argv:checks.append(R/"run_checks.py")
checks.append(R/"verify_package.py")
outputs=[]
for script in checks:
    args=[sys.executable,str(checked(script))]
    if script.name=="run_checks.py":args.append("--recheck")
    run=subprocess.run(args,capture_output=True,text=True,cwd=P)
    outputs.append({"script":str(script.relative_to(P)),"exit_code":run.returncode,"stdout":run.stdout,"stderr":run.stderr})
    print(script.parent.name+"/"+script.name+": "+("PASS" if run.returncode==0 else "FAIL"),flush=True)
    if run.returncode:
        print(run.stdout+run.stderr);break
target=checked(R/"runtime"/"last_run.json");target.parent.mkdir(exist_ok=True)
target.write_text(json.dumps(outputs,indent=2)+"\n",encoding="utf-8")
if any(row["exit_code"] for row in outputs):raise SystemExit(1)
print("RECORDED_REFERENCE_STUDY_VERIFIED; candidate novelty rejected; no physical-control promotion.",flush=True)
