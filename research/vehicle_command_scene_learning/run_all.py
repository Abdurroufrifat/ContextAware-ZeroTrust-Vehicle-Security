from cs_common import *
import subprocess,time
runtime=R/"runtime";runtime.mkdir(exist_ok=True);steps=[]
commands=[("package integrity",[sys.executable,str(R/"verify_package.py")]),
 ("independent48600decision replay",[sys.executable,str(R/"verify_study.py")]),
 ("independent report",[sys.executable,str(R/"verify_report.py")]),
 ("previous complete evidence chain",[sys.executable,str(LEARNED/"run_all.py")]),
 ("final package integrity",[sys.executable,str(R/"verify_package.py")])]
for i,(name,command) in enumerate(commands):
 started=time.perf_counter();x=subprocess.run(command,cwd=P,text=True,capture_output=True)
 checked(runtime/f"verification_{i}.log").write_text(x.stdout+x.stderr,encoding="utf-8")
 steps.append(dict(name=name,exit_code=x.returncode,seconds=time.perf_counter()-started))
 print(name+": "+("PASS" if x.returncode==0 else "FAIL"),flush=True)
 if x.returncode:print((x.stdout+x.stderr)[-4000:]);break
passed=len(steps)==len(commands) and all(x["exit_code"]==0 for x in steps)
checked(runtime/"latest_verification.json").write_text(json.dumps(dict(passed=passed,steps=steps),indent=2),encoding="utf-8")
sys.exit(0 if passed else 1)
