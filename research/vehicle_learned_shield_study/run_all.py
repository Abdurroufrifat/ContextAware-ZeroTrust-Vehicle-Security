"""Read-only verification of the frozen learned-estimator research study."""
from learned_common import *
import subprocess,time
runtime=R/"runtime";runtime.mkdir(exist_ok=True);steps=[]
commands=[
 ("package integrity",[sys.executable,str(R/"verify_package.py")]),
 ("independent 32400-decision replay",[sys.executable,str(R/"verify_learned_study.py")]),
 ("independent report replay",[sys.executable,str(R/"verify_report.py")]),
 ("frozen predecessor evidence chain",[sys.executable,str(P/"COACT_CPS_RESEARCH.py"),"verify"]),
 ("final package integrity",[sys.executable,str(R/"verify_package.py")])]
for i,(name,command) in enumerate(commands):
 started=time.perf_counter()
 x=subprocess.run(command,cwd=P,text=True,capture_output=True)
 checked(runtime/f"verification_{i}.log").write_text(x.stdout+x.stderr,encoding="utf-8")
 steps.append(dict(name=name,exit_code=x.returncode,seconds=time.perf_counter()-started))
 print(name+": "+("PASS" if x.returncode==0 else "FAIL"),flush=True)
 if x.returncode:
  print((x.stdout+x.stderr)[-4000:]);break
checked(runtime/"latest_verification.json").write_text(json.dumps(dict(passed=all(x["exit_code"]==0 for x in steps) and len(steps)==len(commands),steps=steps),indent=2),encoding="utf-8")
sys.exit(0 if len(steps)==len(commands) and all(x["exit_code"]==0 for x in steps) else 1)
