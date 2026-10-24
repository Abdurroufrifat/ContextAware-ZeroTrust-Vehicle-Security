from shared import *
import numpy as np,csv,shutil
from reference_core import parse_status,latest_status,sensitivity
sys.path.insert(0,str(M))
from evidence_core import metrics
I=R.parent/"vehicle_inertial_consistency"
names=["PROTOCOL.md","reference_core.py","test_reference_core.py","reference_registry.json","run_audit.py","shared.py"]
lock={name:sha(R/name) for name in names}
if (R/"experiment_lock.json").exists():assert read("experiment_lock.json")==lock
else:write("experiment_lock.json",lock)
assert not (R/"summary.json").exists(),"Preserve frozen audit outcomes"
old={}
for folder,files in [(H,["Library01_replay.npz","Library02_replay.npz","failed_segments.json","download_lock.json"]),
 (I,["Library01_inertial.npz","Library02_inertial.npz","summary.json","completion.json"])]:
    for name in files:old[str(folder/name)]=sha(folder/name)
for item in json.loads((H/"download_lock.json").read_text())["files"]:
    assert item["sequence"] in {"Library01","Library02"}
    path=H/item["relative_path"];assert sha(path)==item["sha256"];old[str(path)]=item["sha256"]
status=P/"CURRENT_PROJECT_STATUS.md";old[str(status)]=sha(status)
shutil.copyfile(status,R/"PROJECT_STATUS_BEFORE_REFERENCE_AUDIT.md");write("input_lock.json",old)
summary={"scope":"post-hoc reference qualification and hypothetical error-budget sensitivity",
 "actual_independent_absolute_reference_found":False,"verified_reference_error_bound_mps":None,
 "physical_witness_admitted":False,"reference_gap_resolved":False,"sequences":{}}
status_rows=[];sensitivity_rows=[]
for seq in ["Library01","Library02"]:
    folder=H/"data"/seq/"sensor_data"
    raw=[line.split(",") for line in (folder/"inspva.csv").read_text().splitlines() if line.strip()]
    assert all(len(row)==11 for row in raw)
    times=np.array([int(row[0]) for row in raw],np.int64);codes=np.array([parse_status(row[10]) for row in raw])
    assert np.all(np.diff(times)>0)
    h=np.load(H/(seq+"_replay.npz"),allow_pickle=False);d=np.load(I/(seq+"_inertial.npz"),allow_pickle=False)
    queries=h["queries"];np.testing.assert_array_equal(queries,d["queries"])
    tagged,age=latest_status(queries,times,codes)
    cuts=np.r_[0,np.flatnonzero(np.diff(codes)!=0)+1,len(codes)];runs=[]
    for a,b in zip(cuts[:-1],cuts[1:]):
        runs.append({"code":int(codes[a]),"rows":int(b-a),"first_ns":int(times[a]),"last_ns":int(times[b-1]),
                     "duration_between_first_last_s":float((times[b-1]-times[a])*1e-9)})
    result={"ins_rows":len(times),"all_rows_have_11_columns":True,
      "ins_status_counts":{str(s):int(sum(codes==s)) for s in np.unique(codes)},
      "query_status_counts":{str(s):int(sum(tagged==s)) for s in np.unique(tagged)},
      "sensor_files_in_scoped_folder":sorted(p.name for p in folder.iterdir() if p.is_file()),
      "per_record_uncertainty_in_INS_export":False,"status_runs":runs,"groups":{}}
    groups={"all":np.ones(len(queries),bool)}
    for start in range(0,len(queries),1200):
        mask=np.zeros(len(queries),bool);mask[start:start+1200]=True;groups["segment_"+str(start)]=mask
    scopes={"original_radar":(h["request_prediction"],h["ready"]),
            "current_inertial_mask":(d["query_prediction"],d["new_accepted"])}
    reference=h["request_reference"]
    for group,group_mask in groups.items():
        entry={"queries":int(group_mask.sum()),"status_counts":{str(s):int(sum(group_mask&(tagged==s))) for s in np.unique(tagged)},"scopes":{}}
        for scope,(prediction,admitted) in scopes.items():
            whole=metrics(prediction[group_mask],reference[group_mask],admitted[group_mask])
            strat={}
            for code in sorted(set([-1,3,6])|set(map(int,np.unique(tagged)))):
                mask=group_mask&(tagged==code)
                m=metrics(prediction[mask],reference[mask],admitted[mask]);strat[str(code)]=m
                status_rows.append({"sequence":seq,"group":group,"scope":scope,"status_code":code,**m})
            entry["scopes"][scope]={"unchanged_all_status_metrics":whole,"post_hoc_by_status":strat}
            for epsilon in [None,0.,.05,.1,.2,.5,1.]:
                row=sensitivity(prediction[group_mask],reference[group_mask],epsilon,admitted[group_mask])
                sensitivity_rows.append({"sequence":seq,"group":group,"scope":scope,**row})
        result["groups"][group]=entry
    np.savez_compressed(R/(seq+"_reference_audit.npz"),queries=queries,ins_times=times,ins_status=codes,query_status=tagged,status_age_ns=age,
      original_prediction=h["request_prediction"],original_admitted=h["ready"],current_prediction=d["query_prediction"],
      current_admitted=d["new_accepted"],reference=reference)
    summary["sequences"][seq]=result
    tail=result["groups"][list(groups)[-1]]
    print(seq,json.dumps({"ins_status_counts":result["ins_status_counts"],"whole_query_status_counts":result["query_status_counts"],
       "last_segment_queries":tail["queries"],"last_segment_status_counts":tail["status_counts"],
       "last_segment_original":tail["scopes"]["original_radar"]["unchanged_all_status_metrics"]}),flush=True)
for filename,rows in [("status_stratified_metrics.csv",status_rows),("hypothetical_error_budgets.csv",sensitivity_rows)]:
    with (R/filename).open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
write("sensitivity_rows.json",sensitivity_rows);write("summary.json",summary)
write("REFERENCE_EVIDENCE_REQUIREMENTS.json",{
 "purpose":"Evidence needed for stronger physical absolute-speed claims; not a mandatory requirement of every journal study",
 "missing_in_inspected_export":["timestamped navigation velocity uncertainty/quality logs such as INSSTDEV or INSPVAX if recorded",
 "receiver output-point/lever-arm/datum configuration and processing lineage","clock-conversion and calibration uncertainty documentation",
 "independent absolute-motion validation with explicit coverage and error model"],
 "required_coverage":"Both development sequences, including original Library01 final302-query interval and original missing endpoints",
 "uncertainty_note":"Standard deviations require a justified probability/error model; they are not automatically uniform deterministic bounds",
 "existing_reference_replacement_authorized":False,"status3_is_not_error_bound":True,
 "acceptable_alternative":"Retain explicit proxy-only development claims and refrain from physical certification; existing stop rules remain unchanged",
 "requested_additional_sensor_download":False,"reserved_payloads_to_open":False})
print("REFERENCE_AUDIT_COMPLETE",flush=True)
