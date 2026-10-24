from learned_common import *
install_guard()
import importlib.metadata,sys
manifest=json.loads((R/"file_manifest.json").read_text());assert manifest
for name,h in manifest.items():assert sha(P/name)==h,("package",name)
for lockname in ("model_lock.json","experiment_lock.json"):
 lock=json.loads((R/lockname).read_text())
 for name,h in lock.items():assert sha(P/name)==h,(lockname,name)
deps=json.loads((R/"dependency_bindings.json").read_text())
for name,h in deps["source_files"].items():assert sha(P/name)==h,("dependency",name)
assert hashlib.sha256(BINARY.read_bytes()).hexdigest()==deps["sumo_sha256"]
selection=json.loads((R/"model_selection.json").read_text())
for name,h in selection["input_bindings"].items():assert sha(P/name)==h,("training_input",name)
assert set(selection["trial_ids"]["train"]).isdisjoint(selection["trial_ids"]["validation"])
assert selection["test_data_accessed"] is False
env=json.loads((R/"environment.json").read_text())
assert env["python"]==sys.version
for name,version in env["packages"].items():assert importlib.metadata.version(name)==version,(name,version)
tests=json.loads((R/"test_validation.json").read_text());assert all(v["exit_code"]==0 for v in tests.values())
assert tests["focused"]["counts"]["passed"]==538 and tests["core"]["counts"]["passed"]==731 and tests["core"]["counts"]["skipped"]==4
assert json.loads((R/"verification.json").read_text())["controlled_decisions_reconstructed"]==32400
assert json.loads((R/"report_verification.json").read_text())["passed"]


application=json.loads((R/"application_replay.json").read_text())
assert application["passed"] and application["trials"]==54 and application["decisions"]==10800 and application["truth_fields_passed_to_api"] is False
for name,h in application["application_source_sha256"].items():assert sha(P/name)==h,("application_source",name)

print("LEARNED_SHIELD_PACKAGE_VERIFIED",len(manifest),"files",len(deps["source_files"]),"dependencies",flush=True)
