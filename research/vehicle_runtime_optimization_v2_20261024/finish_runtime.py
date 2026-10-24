"""Single-owner continuation of the isolated runtime study."""
from runtime_common import S,P,R,B,sha,load
from pathlib import Path
import os,json,sys,subprocess,datetime
def write(path,value):
 temp=path.with_suffix(path.suffix+".tmp");temp.write_text(json.dumps(value,indent=2),encoding="utf-8");os.replace(temp,path)
def main():
 if (S/"RUNTIME_COMPLETE.json").exists():print("RUNTIME_ALREADY_COMPLETE",flush=True);return
 owner=S/"runtime/finish.lock";fd=os.open(owner,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,json.dumps({"pid":os.getpid()}).encode());os.close(fd)
 old=P/"research/vehicle_runtime_optimization_20261024"
 def run(stage,arguments):
  write(S/"runtime/progress.json",{"stage":stage,"status":"RUNNING","utc":datetime.datetime.now(datetime.timezone.utc).isoformat()})
  log=S/"runtime"/(stage+".log");print("START",stage,flush=True)
  with log.open("w",encoding="utf-8") as f:rc=subprocess.call([sys.executable,*arguments],cwd=S,stdout=f,stderr=subprocess.STDOUT)
  if rc:write(S/"runtime/progress.json",{"stage":stage,"status":"FAILED","log":str(log),"returncode":rc});raise RuntimeError(log.read_text(encoding="utf-8")[-4000:])
  print("PASS",stage,flush=True)
 try:
  if not (S/"input_lock.json").exists():
   if (S/"BENCHMARK_PROTOCOL.json").exists():raise ValueError("Partial freeze requires inspection")
   run("final_unit_tests",["-m","unittest","-v","test_fast_runtime"])
   cfg=load(R/"PROTOCOL_V2.json");trials=load(B/"results.json")["trials"];joint=[t for t in trials if t["method"]=="joint"];seed=cfg["seeds"][1]
   profiles=list(cfg["profiles"])
   ids=[next(t["trial_id"] for t in joint if t["seed"]==seed and t["case"]==case and t["profile"]==profiles[j%len(profiles)]) for j,case in enumerate(cfg["primary_cases"])]
   assert len(joint)==216 and len(ids)==12 and len(set(ids))==12
   protocol={"benchmark_trial_ids":ids,"equivalence_trial_ids":[t["trial_id"] for t in joint],"benchmark_seed":seed,"variants":["original","pool_native","pool_fast"],"order":"Rotate variants by trial index modulo3; no timing-based selection","scope":"Sequential paired replay of12cases, profiles cyclically assigned; seed2467 distinct from development. All216joint traces for equivalence in3isolated processes; those contended timings are excluded from paired timing. Original vehicle experiments preserved.","tuning":"Adapter sources frozen after focused tests and before paired timing. No model, threshold, packet verification or FULL transaction synchronization changes.","single_owner_required":True,"source_diagnostic_float_tolerance":1e-12,"exact_policy_command_card_evidence":True}
   write(S/"BENCHMARK_PROTOCOL.json",protocol)
   bindings=load(R/"confirmation_v2_lock.json");bindings.update(load(R/"package_manifest.json")["files"])
   paths=[R/"package_manifest.json",R/"final_seal.json",S/"BENCHMARK_PROTOCOL.json",S/"runtime/final_unit_tests.log"]
   paths+=[p for p in S.iterdir() if p.is_file() and p.suffix in (".py",".md",".json")]
   for p in paths:
    if p.name!="input_lock.json":bindings[str(p.relative_to(P))]=sha(p)
   bindings[str((old/"CANDIDATE_STATUS.json").relative_to(P))]=sha(old/"CANDIDATE_STATUS.json")
   write(S/"input_lock.json",bindings)
  if not (S/"study_results.json").exists():run("replay_study",[str(S/"runtime_study.py")])
  else:print("Saved replay evidence reused; no repetition.",flush=True)
  run("runtime_audit",[str(S/"verify_runtime.py")])
  write(S/"runtime/progress.json",{"stage":"complete","status":"COMPLETE_FOR_RUNTIME_REPLAY_SCOPE","equivalent_trials":216,"equivalent_decisions":43200})
  print("RUNTIME_STUDY_COMPLETE",flush=True)
 except BaseException as e:
  progress=load(S/"runtime/progress.json") if (S/"runtime/progress.json").exists() else {}
  if progress.get("status")!="FAILED" and progress.get("stage")!="FAILED":write(S/"runtime/progress.json",{"stage":"pipeline","status":"FAILED","error":repr(e)})
  raise
 finally:owner.unlink(missing_ok=True)
if __name__=="__main__":main()
