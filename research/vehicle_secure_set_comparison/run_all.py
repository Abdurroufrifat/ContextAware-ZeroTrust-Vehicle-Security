from ssc_common import *
import subprocess,time
runtime=R/"runtime";runtime.mkdir(exist_ok=True);out=[]
for script in (R/"verify_package.py",R/"verify_comparison.py",OLD/"run_all.py",R/"verify_package.py"):
 z=subprocess.run([sys.executable,str(script)],capture_output=True,text=True)
 out.append(dict(script=str(script.relative_to(P)),exit_code=z.returncode))
 (runtime/(script.parent.name+"_"+script.name+".log")).write_text(z.stdout+z.stderr,encoding="utf-8")
 print(script.parent.name+"/"+script.name,":","PASS" if z.returncode==0 else "FAIL",flush=True)
 if z.returncode:
  print(z.stdout+z.stderr);dump(runtime/"latest_verification.json",dict(passed=False,checks=out));raise SystemExit(z.returncode)
dump(runtime/"latest_verification.json",dict(passed=True,checks=out))
print("PUBLISHED_METHOD_SCALAR_COMPARISON_VERIFIED; no new physical or new-algorithm claim.",flush=True)
