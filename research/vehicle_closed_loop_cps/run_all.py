from cps_common import R,P
import os,subprocess,sys,json,time
runtime=R/"runtime";runtime.mkdir(exist_ok=True)
env=os.environ.copy();env["PYTHONPATH"]=os.pathsep.join([str(P/"src"),str(P/"scripts"),env.get("PYTHONPATH","")])
checks=[("package integrity",[sys.executable,str(R/"verify_package.py")]),
        ("independent closed-loop verification",[sys.executable,str(R/"verify_study.py")]),
        ("independent report verification",[sys.executable,str(R/"verify_report.py")]),
        ("new tests",[sys.executable,"-m","pytest",str(R/"test_control_core.py"),str(R/"test_cps_session.py"),str(R/"test_simulator_insertion.py"),str(R/"test_report_core.py"),"-q"]),
        ("project tests",[sys.executable,"-m","pytest"]),
        ("previous frozen evidence",[sys.executable,str(R.parent/"vehicle_command_scene_learning"/"verify_package.py")]),
        ("final package integrity",[sys.executable,str(R/"verify_package.py")])]
results=[]
for i,(name,cmd) in enumerate(checks):
    started=time.monotonic();r=subprocess.run(cmd,cwd=P,capture_output=True,text=True,env=env)
    (runtime/("check_"+str(i)+".log")).write_text(r.stdout+r.stderr,encoding="utf-8")
    results.append({"name":name,"exit_code":r.returncode,"seconds":time.monotonic()-started})
    print(name+": "+("PASS" if r.returncode==0 else "FAIL"),flush=True)
    if r.returncode:print((r.stdout+r.stderr)[-3000:]);break
passed=len(results)==len(checks) and all(r["exit_code"]==0 for r in results)
(runtime/"latest_verification.json").write_text(json.dumps({"passed":passed,"checks":results},indent=2),encoding="utf-8")
raise SystemExit(0 if passed else 1)
