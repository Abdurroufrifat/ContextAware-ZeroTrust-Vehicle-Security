from pathlib import Path
import subprocess,sys,json,time,os
P=Path(__file__).resolve().parent
runtime=P/"runtime";runtime.mkdir(exist_ok=True)
checks=[("package integrity",[sys.executable,str(P/"verify_package.py")]),
        ("pairing tests",[sys.executable,"-m","pytest",str(P/"test_assay_core.py"),"-q"]),
        ("raw message extraction",[sys.executable,str(P/"verify_raw_extraction.py")]),
        ("independent causal metrics",[sys.executable,str(P/"verify_assay.py")]),
        ("original project tests",[sys.executable,"-m","pytest"]),
        ("previous frozen command-scene study",[sys.executable,str(P.parent/"vehicle_command_scene_learning"/"verify_package.py")]),
        ("final package integrity",[sys.executable,str(P/"verify_package.py")])]
env=os.environ.copy()
env["PYTHONPATH"]=os.pathsep.join([str(P.parent.parent/"src"),str(P.parent.parent/"scripts"),env.get("PYTHONPATH","")])
results=[]
for i,(name,cmd) in enumerate(checks):
 start=time.monotonic()
 r=subprocess.run(cmd,cwd=P.parent.parent,capture_output=True,text=True,env=env)
 (runtime/("check_"+str(i)+".log")).write_text(r.stdout+r.stderr,encoding="utf-8")
 results.append({"name":name,"exit_code":r.returncode,"seconds":time.monotonic()-start})
 print(name+": "+("PASS" if r.returncode==0 else "FAIL"),flush=True)
 if r.returncode:
  print((r.stdout+r.stderr)[-2500:]);break
passed=len(results)==len(checks) and all(x["exit_code"]==0 for x in results)
(runtime/"latest_verification.json").write_text(json.dumps({"passed":passed,"checks":results},indent=2),encoding="utf-8")
raise SystemExit(0 if passed else 1)
