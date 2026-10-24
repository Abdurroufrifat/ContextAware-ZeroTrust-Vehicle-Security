from tg_common import *
install_guard()
import subprocess
runtime=R/"runtime";runtime.mkdir(exist_ok=True);checks=[]
for script in (R/"verify_package.py",R/"verify_temporal_study.py",R/"verify_report.py",R.parent/"vehicle_secure_set_comparison"/"run_all.py",R/"verify_package.py"):
 z=subprocess.run([sys.executable,str(script)],capture_output=True,text=True)
 (runtime/(script.parent.name+"_"+script.name+".log")).write_text(z.stdout+z.stderr,encoding="utf-8")
 checks.append(dict(script=str(script.relative_to(P)),exit_code=z.returncode))
 print(script.parent.name+"/"+script.name,":","PASS" if z.returncode==0 else "FAIL",flush=True)
 if z.returncode:
  print(z.stdout+z.stderr);write("runtime/latest_verification.json",dict(passed=False,checks=checks));raise SystemExit(z.returncode)
write("runtime/latest_verification.json",dict(passed=True,checks=checks))
print("TEMPORAL_GATEWAY_STUDY_VERIFIED; gradual-drift failure retained; physical validation not established.",flush=True)
