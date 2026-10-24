from shared import *
from release_core import artifact_path,verify_manifest,file_sha,exact_sensitivity
import numpy as np,math,shutil
assert not (R/"release_summary.json").exists(),"Release snapshot already exists; do not overwrite"
names=["PROTOCOL.md","release_core.py","test_release_core.py","build_release.py","shared.py"]
write("experiment_lock.json",{name:sha(R/name) for name in names})
modules={"vehicle_evidence_freeze":"sha256.json","vehicle_reference_uncertainty":"sha256.json",
 "vehicle_matched_evidence":"file_manifest.json","vehicle_neural_baseline":"file_manifest.json",
 "vehicle_neural_adaptation":"file_manifest.json","vehicle_inertial_consistency":"file_manifest.json",
 "vehicle_reference_qualification":"file_manifest.json","vehicle_optical_timing":"file_manifest.json"}
inputs={};index=[]
def track(path):
    relative=str(Path(path).resolve().relative_to(P.resolve()))
    inputs[relative]=file_sha(artifact_path(P,relative))
for module,manifest in modules.items():
    root=R.parent/module;items=verify_manifest(root,manifest);track(root/manifest)
    for item in items:
        relative=str(Path(item["path"]).relative_to(P));inputs[relative]=item["sha256"]
    index.append({"module":module,"manifest":manifest,"files_verified":len(items),"scope":"artifact identity only; no physical source appraisal"})
    print("VERIFIED",module,len(items),flush=True)
N=R.parent/"vehicle_neural_baseline"
for item in json.loads((N/"source_inventory.json").read_text())["extracted_source"]:
    path=artifact_path(N/"author_source",item["path"])
    assert file_sha(path)==item["sha256"];track(path)
for item in json.loads((N/"model_admission_audit.json").read_text())["models"]:
    path=artifact_path(N,item["weight_file"]);assert file_sha(path)==item["weight_sha256"];track(path)
C=R.parent/"vehicle_crossmodal_doppler"
for name,digest in json.loads((R.parent/"vehicle_optical_timing/summary.json").read_text())["unchanged_prior_hashes"].items():
    path=artifact_path(C,name);assert file_sha(path)==digest;track(path)
for name,backup in [("CURRENT_PROJECT_STATUS.md","PROJECT_STATUS_BEFORE_RELEASE.md"),("FINAL_PROJECT_README.md","FINAL_README_BEFORE_RELEASE.md")]:
    track(P/name);shutil.copyfile(P/name,R/backup)
Q=R.parent/"vehicle_reference_qualification"
old_rows=json.loads((Q/"sensitivity_rows.json").read_text());arrays={}
for seq in ["Library01","Library02"]:arrays[seq]=np.load(Q/(seq+"_reference_audit.npz"),allow_pickle=False)
exact_rows=[];changes=[]
for number,old in enumerate(old_rows):
    d=arrays[old["sequence"]];mask=np.ones(len(d["queries"]),bool)
    if old["group"]!="all":
        start=int(old["group"].split("_")[1]);mask[:]=False;mask[start:start+1200]=True
    p=d["original_prediction"] if old["scope"]=="original_radar" else d["current_prediction"]
    admitted=d["original_admitted"] if old["scope"]=="original_radar" else d["current_admitted"]
    row=exact_sensitivity(p[mask],d["reference"][mask],old["epsilon_assumption_mps"],admitted[mask])
    keys=["scored_subset_upper_within_0p5","conditional_full_admitted_set_upper_within_0p5","definitely_above2","possibly_above2"]
    changed={key:{"historical_float":old[key],"exact":row[key]} for key in keys if old[key]!=row[key]}
    identity={"row":number,"sequence":old["sequence"],"group":old["group"],"scope":old["scope"],"epsilon_assumption_mps":old["epsilon_assumption_mps"]}
    exact_rows.append({**identity,**row})
    if changed:changes.append({**identity,"changes":changed})
    if number%50==0:print("EXACT_ROWS",number,flush=True)
write("exact_sensitivity_rows.json",exact_rows)
boundary=exact_sensitivity([.25],[0.],math.nextafter(.25,math.inf))
write("NUMERICAL_CORRECTION.json",{"historical_file_preserved":"research/vehicle_reference_qualification/reference_core.py",
 "historical_float_upper":.5,"historical_conditional_flag":True,"exact_boundary_result":boundary,
 "actual_sensitivity_rows_rechecked":len(exact_rows),"actual_rows_with_changed_flags_or_counts":len(changes),"changes":changes,
 "authoritative_numeric_sidecar":"exact_sensitivity_rows.json","measured_policy_outputs_changed":False,
 "exact_semantics":"rational arithmetic for stored numeric inputs; no physical uncertainty inferred"})
cohorts=[
 {"id":"historic_979_scene_development","source":"vehicle_evidence_freeze/FINAL_STATUS.json","level":"verified frozen report metadata; full raw events not rerun",
  "failure":"source-compromise large disagreements and unqualified physical source retained"},
 {"id":"external_Civic_reference_audit","source":"vehicle_reference_uncertainty/README.md","level":"prior external-data audit artifact; not a new rerun",
  "failure":"original64 GNSS disagreements above2m/s and unknown physical reference bound retained"},
 {"id":"HeRCULES_Library01_02_query_replay","source":"vehicle_optical_timing/summary.json","level":"existing fixed query-grid reports/source arrays",
  "failure":"originalLibrary01 302-query/301-score P951.1022975m/s and reference unknowns retained"},
 {"id":"HeRCULES_neural_component_comparison","source":"vehicle_neural_adaptation/evaluation_summary.json","level":"original adapted neural comparator,3seeds; not native-paper reproduction",
  "failure":"61-frame temporal candidate P950.560221m/s; prior development transfer not pristine holdout"},
 {"id":"HeRCULES_IMU_differential_diagnostic","source":"vehicle_inertial_consistency/summary.json","level":"frozen-schedule/interface diagnostic",
  "failure":"constant/slow coordinated offsets pass; additional latency not charged to original schedule"}]
write("evidence_index.json",{"modules":index,"cohorts":cohorts,"pooled_cross_cohort_metric_produced":False,
 "actual_reference_bound_established":False,"physical_witness_enabled":False,"promotion_allowed":False})
track(R.parent/"vehicle_evidence_freeze/FINAL_VALIDATION_REQUIREMENTS.json")
for relative,digest in inputs.items():
    path=artifact_path(P,relative);assert file_sha(path)==digest,relative
write("input_lock.json",inputs)
write("release_summary.json",{"snapshot_built":True,"source_modules_verified":len(modules),
 "manifest_entries_verified":sum(item["files_verified"] for item in index),"all_input_files_tracked":len(inputs),
 "nested_native_models_verified":3,"exact_sensitivity_rows":len(exact_rows),
 "changed_sensitivity_rows":len(changes),"measured_policy_outputs_changed":False,
 "datasets_or_estimators_rerun":False,"physical_validation_complete":False,"journal_readiness_established":False,
 "overall_project_complete":False,"promotion_allowed":False})
print(json.dumps(read("release_summary.json"),indent=2))
