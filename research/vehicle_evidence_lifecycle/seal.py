from lc_common import *
import json
def main():
 root=R/"confirmation_v2";result=json.loads((root/"results.json").read_text());audit=json.loads((root/"verification.json").read_text());report=json.loads((root/"report_verification.json").read_text());tests=json.loads((R/"runtime/final_tests.json").read_text())
 assert len(result["trials"])==1080 and audit["passed"] and report["passed"] and tests["passed"]
 assert audit["results_sha256"]==report["results_sha256"]==sha(root/"results.json")
 assert tests["confirmation_lock_digest"]==sha(R/"confirmation_v2_lock.json")
 assert all(sha(P/n)==h for n,h in json.loads((R/"confirmation_v2_lock.json").read_text()).items())
 files=[p for p in R.rglob("*") if p.is_file() and "__pycache__" not in p.parts and ".pytest_cache" not in p.parts and "runtime" not in p.relative_to(R).parts and p.name not in ("final_seal.json","package_manifest.json")]
 files += [P/"COACT_CPS_LIFECYCLE.py"]
 dump(R/"package_manifest.json",{"scope":"Joint lifecycle simulation extension, conditional evidence and preserved development","files":{str(p.relative_to(P)):sha(p) for p in sorted(files)}})
 bindings=json.loads((R/"package_manifest.json").read_text())["files"];assert all(sha(P/n)==h for n,h in bindings.items())
 dump(R/"final_seal.json",{"status":"COMPLETE_FOR_LIFECYCLE_SIMULATION_SCOPE","confirmation_trials":1080,"confirmation_decisions":216000,"development_trials_v2":60,"tests_verified":True,"audit_verified":True,"report_viewer_verified":True,"old_results_preserved":True,"qualified_safety_counts":{k:audit["counts"][k] for k in ("qualified_gap_failures","qualified_containment_failures","qualified_certificate_counterexamples","point_certified_errors")},"results_sha256":sha(root/"results.json"),"package_manifest_sha256":sha(R/"package_manifest.json"),"confirmation_lock_sha256":sha(R/"confirmation_v2_lock.json"),"native_method_superiority_claim":False,"first_ever_originality_certified":False,"journal_acceptance_certified":False,"physical_hardware_scope":False,"human_effectiveness_study":False,"manuscript_written":False})
 print("LIFECYCLE_RELEASE_SEALED",len(files),flush=True)
if __name__=="__main__":main()
