from shared import *
import numpy as np
from inertial_core import compare_changes
from diagnostic_details import additive_forgery,explain_frames
verify_lock()
names=["diagnostic_details.py","test_diagnostic_details.py","run_details.py","CLARIFICATIONS.md"]
lock={name:sha(R/name) for name in names}
if (R/"detail_experiment_lock.json").exists():assert read("detail_experiment_lock.json")==lock
else:write("detail_experiment_lock.json",lock)
assert not (R/"details_summary.json").exists(),"Preserve frozen supplemental results"
old={name:sha(R/name) for name in ["summary.json","Library01_inertial.npz","Library02_inertial.npz"]}
summary={"initial_outcomes_preserved":old,"clarification":"Original forged case is full coordinated replacement; supplemental forgery adds only the matching step","sequences":{}}
def counts(values):
    names,totals=np.unique(values,return_counts=True)
    return {str(n):int(k) for n,k in zip(names,totals)}
for seq in ["Library01","Library02"]:
    data=np.load(R/(seq+"_inertial.npz"),allow_pickle=False);h=np.load(H/(seq+"_replay.npz"),allow_pickle=False);o=np.load(T/(seq+"_replay.npz"),allow_pickle=False)
    reasons=explain_frames(data["stamps"],data["past_index"],data["quality"],data["supported"],data["accepted"])
    base_reasons=[]
    for i,index in enumerate(o["index"]):
        if not h["ready"][i]:name="radar_not_ready"
        elif index<0:name="optical_missing"
        elif not o["quality"][index]:name="optical_quality"
        elif not 0<=o["optical_age_ns"][i]<=100_000_000:name="optical_stale"
        elif abs(o["radar_query_speed"][i]-o["optical_query_speed"][i])>1:name="scalar_disagreement"
        else:name="accepted"
        base_reasons.append(name)
    base_reasons=np.array(base_reasons,dtype="U64")
    np.testing.assert_array_equal(base_reasons=="accepted",data["base_accepted"])
    query_reasons=base_reasons.copy()
    for i,index in enumerate(data["query_radar_index"]):
        if base_reasons[i]=="accepted":
            query_reasons[i]=reasons[index] if index>=0 else "missing_radar_frame"
    np.testing.assert_array_equal(query_reasons=="accepted",data["new_accepted"])
    t=data["stamps"];mid=int(t[0])+int(t[-1]-t[0])//2;bias=np.zeros_like(data["velocity_imu_origin_xy"]);bias[t>=mid,0]=3
    delta=additive_forgery(data["delta_imu"],bias,data["past_index"])
    v=data["velocity_imu_origin_xy"]+bias
    row=compare_changes(v,data["past_index"],delta,data["supported"])
    np.testing.assert_allclose(row["residual"],data["residual"],atol=1e-12,rtol=0,equal_nan=True)
    np.testing.assert_array_equal(row["accepted"],data["accepted"])
    np.savez_compressed(R/(seq+"_details.npz"),frame_reasons=reasons,base_query_reasons=base_reasons,
      new_query_reasons=query_reasons,additive_forged_delta=delta,additive_forgery_residual=row["residual"],additive_forgery_accepted=row["accepted"])
    result={"frame_reason_counts":counts(reasons),"old_query_reason_counts":counts(base_reasons),"new_query_reason_counts":counts(query_reasons),
      "additive_forgery_supported":int(row["supported"].sum()),"additive_forgery_accepted":int(row["accepted"].sum()),
      "additive_forgery_preserves_nominal_mismatch":True}
    assert sum(result["frame_reason_counts"].values())==len(t)
    assert sum(result["new_query_reason_counts"].values())==len(data["queries"])
    summary["sequences"][seq]=result;print(seq,json.dumps(result),flush=True)
for name,digest in old.items():assert sha(R/name)==digest
write("details_summary.json",summary);print("DETAILS_COMPLETE",flush=True)
