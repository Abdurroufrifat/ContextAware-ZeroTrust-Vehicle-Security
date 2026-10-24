"""Single user-facing entry for the current tested project and its scientific limits."""
from pathlib import Path
import subprocess,sys,json
HD=Path(__file__).resolve().parent;P=HD.parent.parent;BR=HD.parent/"vehicle_boreas_rt_reference_study"
runtime=HD/"runtime";runtime.mkdir(exist_ok=True)
checks=[([str(BR/"run_all.py"),"--tests"],"project"),([str(HD/"verify_handoff.py")],"diagnosis")]
runs=[]
for args,name in checks:
    run=subprocess.run([sys.executable,*args],cwd=P,capture_output=True,text=True)
    (runtime/(name+".log")).write_text(run.stdout+run.stderr,encoding="utf-8")
    runs.append({"name":name,"exit_code":run.returncode,"log":str(runtime/(name+".log"))})
    print(name+": "+("PASS" if run.returncode==0 else "FAIL"),flush=True)
    if run.returncode:print(run.stdout+run.stderr);break
tests_path=BR/"runtime"/"recheck"/"test_validation.json"
tests=json.loads(tests_path.read_text()) if tests_path.exists() and all(x["exit_code"]==0 for x in runs) else None
status={"software_checks_passed":all(x["exit_code"]==0 for x in runs) and len(runs)==2,
"runs":runs,"tests":tests,"external_recorded_reference_comparison_completed":True,
"demonstrated_novel_algorithm":False,"physical_control_validation_completed":False,"hard_true_speed_certificate_established":False,
"journal_readiness_established":False,"full_research_goal_complete":False,
"open_research_requirements":["A distinct security/estimation contribution with a demonstrated advantage over appropriate published methods","Reference-error and clock uncertainty qualification for any claimed physical bound","Independent physical closed-loop validation if real-vehicle policy/control safety is claimed"]}
(runtime/"STATUS.json").write_text(json.dumps(status,indent=2)+"\n",encoding="utf-8")
print("Software status: "+("verified" if status["software_checks_passed"] else "failed"),flush=True)
print("Research status: incomplete; novelty and physical-control validation are unproven.",flush=True)
sys.exit(0 if status["software_checks_passed"] else 1)
