"""Seal only after complete numerical, replay, report, viewer and test evidence."""
from cv_common import B,P,sha,dump
import json
def main():
 result=json.loads((B/"results.json").read_text())
 evidence=json.loads((B/"verification.json").read_text()); replay=json.loads((B/"policy_replay.json").read_text())
 report=json.loads((B/"report_verification.json").read_text());tests=json.loads((B/"runtime/preflight_tests.json").read_text())
 assert tests["verification_lock_digest"]==sha(B/"verification_lock.json")
 assert len(result["trials"])==1512 and evidence["passed"] and replay["passed"] and report["passed"] and tests["passed"]
 assert evidence["counts"]["decisions"]==replay["decisions"]==302400
 assert all(e["input_results_sha256"]==sha(B/"results.json") for e in (evidence,replay,report))
 for lockname in ("experiment_lock.json","verification_lock.json"):
  for name,h in json.loads((B/lockname).read_text()).items():assert sha(P/name)==h,(lockname,name)
 status=json.loads((B/"assembly_status.json").read_text());status["status"]="COMPLETE_FOR_COMPARATIVE_SIMULATION_SCOPE"
 status["report_verified"]=True;status["tests_verified"]=True;status["original_frozen_bindings_unchanged"]=True
 
 files=[p for p in B.rglob("*") if p.is_file() and "runtime" not in p.relative_to(B).parts and "__pycache__" not in p.parts and ".pytest_cache" not in p.parts and p.name not in ("package_manifest.json","development_manifest.json","final_seal.json")]
 files+=[P/"COACT_CPS_COMPARATIVE.py",P/"COMPARATIVE_SIMULATION_START.md"]
 dump(B/"package_manifest.json",{"scope":"Verified1512-trial comparative simulation extension; original540 preserved","files":{p.relative_to(P).as_posix():sha(p) for p in sorted(files)}})
 runtime=[p for p in (B/"runtime").rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.name not in ("pipeline_console.log","pipeline_state.json","pipeline.lock","seal_release.log")]
 dump(B/"development_manifest.json",{"scope":"Preserved pilot, failed development attempts and verification logs; live pipeline control files excluded","files":{p.relative_to(P).as_posix():sha(p) for p in sorted(runtime)}})
 for manifest in ("package_manifest.json","development_manifest.json"):
  contents=json.loads((B/manifest).read_text())["files"]
  assert all(sha(P/n)==h for n,h in contents.items())
 status["package_manifest_sha256"]=sha(B/"package_manifest.json")
 status["development_manifest_sha256"]=sha(B/"development_manifest.json")
 status["verification_sha256"]=sha(B/"verification.json")
 status["policy_replay_sha256"]=sha(B/"policy_replay.json")
 status["report_verification_sha256"]=sha(B/"report_verification.json")
 dump(B/"final_seal.json",status)
 print("COMPARATIVE_RELEASE_SEALED",len(files),len(runtime),flush=True)
if __name__=="__main__":main()
