from release_common import *
install_guard()
import subprocess
runtime=R/"runtime";runtime.mkdir(exist_ok=True);checks=[]
tasks=[("release integrity",[sys.executable,str(R/"verify_package.py")]),
 ("unified signed vehicle demo",[sys.executable,str(P/"COACT_CPS_RESEARCH.py"),"demo"]),
 ("original core CLI",[sys.executable,str(P/"COACT_CPS_RESEARCH.py"),"core","--help"]),
 ("complete frozen evidence chain",[sys.executable,str(KW/"run_all.py")]),
 ("final release integrity",[sys.executable,str(R/"verify_package.py")])]
for label,command in tasks:
 z=subprocess.run(command,capture_output=True,text=True,cwd=P)
 (runtime/(label.replace(" ","_")+".log")).write_text(z.stdout+z.stderr,encoding="utf-8")
 ok=z.returncode==0
 if ok and label=="unified signed vehicle demo":
  data=json.loads(z.stdout);ok=data["queries"]==200 and data["selected"]==199 and data["selected_reference_error_gt05"]==0 and data["physical_operation"] is False
 if ok and label=="original core CLI":ok="coact-cps" in z.stdout and "inspect-csv" in z.stdout
 checks.append(dict(check=label,passed=ok,exit_code=z.returncode))
 print(label+": "+("PASS" if ok else "FAIL"),flush=True)
 if not ok:
  print(z.stdout+z.stderr);dump(runtime/"latest_verification.json",dict(passed=False,checks=checks));raise SystemExit(1)
dump(runtime/"latest_verification.json",dict(passed=True,checks=checks))
print("COACT_CPS_RESEARCH_RELEASE_VERIFIED. PC application/evidence scope complete; original physical/novelty/CFP research claims remain unestablished.",flush=True)
