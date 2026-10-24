from shared import *
import numpy as np,torch,time,csv
from neural_core import SpeedSet,pack
sys.path.insert(0,str(M))
from evidence_core import metrics,paired_mae_ci
verify_lock()
assert not (R/"evaluation_summary.json").exists(),"Evaluation already frozen; no retuning"
audit=read("data_audit.json");assert sha(R/"evaluation.npz")==audit["data_hashes"]["evaluation.npz"]
training=read("training_summary.json");data=np.load(R/"evaluation.npz",allow_pickle=False);fs=frames(data)
y=data["reference"];roles=data["roles"];sequence=data["sequence"];stamps=data["stamps"]
torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
pred={name:data[name].copy() for name in ["candidate","author_3d_ransac","ols_3d"]}
pred["train_median"]=np.full(len(y),training["train_median_mps"])
timing={};load_audit=[]
for run in training["runs"]:
    path=R/run["checkpoint"];assert sha(path)==run["checkpoint_sha256"]
    checkpoint=torch.load(path,map_location="cpu",weights_only=True);model=SpeedSet()
    model.load_state_dict(checkpoint["state_dict"],strict=True);model.eval();values=[];elapsed=[];padding=[]
    warm_x,warm_m=pack([fs[0]])
    with torch.inference_mode():
        for _ in range(10):model(warm_x,warm_m)
        for f in fs:
            start=time.perf_counter_ns();x,mask=pack([f]);padding.append(time.perf_counter_ns()-start)
            start=time.perf_counter_ns();out=model(x,mask);elapsed.append(time.perf_counter_ns()-start);values.append(float(out[0]))
    assert all(torch.equal(v,checkpoint["state_dict"][k]) for k,v in model.state_dict().items())
    name="neural_seed_"+str(run["seed"]);pred[name]=np.array(values)
    timing[name]={"forward_only_p95_ms":float(np.quantile(elapsed,.95)*1e-6),
                 "padding_tensor_p95_ms":float(np.quantile(padding,.95)*1e-6),"frames":len(values),"threads":1}
    load_audit.append({"seed":run["seed"],"strict_weights_only_loaded":True,"inference_state_unchanged":True})
groups={"Library01_temporal_test":roles=="temporal_test","Library02_development_transfer":roles=="transfer",
        "Library01_training_diagnostic":roles=="train","Library01_validation_selection":roles=="validation",
        "Library01_purged_diagnostic":roles=="purge",
        "Library01_original_failed_window_sparse_cohort":(sequence==0)&(stamps>=1724812166350000000)&(stamps<=1724812181400000000)}
out={"scope":"development proxy-disagreement comparison; adapted neural baseline, not native-paper reproduction",
 "source_training_summary_sha256":sha(R/"training_summary.json"),"load_audit":load_audit,"groups":{},
 "timing":timing,"final_holdout_used":False,"policy_changed":False,"reference_gap_resolved":False}
rows=[]
for group,mask in groups.items():
    common=mask&np.isfinite(y)
    for values in pred.values():common&=np.isfinite(values)
    result={"attempts":int(mask.sum()),"common_scored":int(common.sum()),
        "metrics":{name:metrics(values[mask],y[mask]) for name,values in pred.items()},"candidate_minus_neural_mae_intervals":{}}
    if group in {"Library01_temporal_test","Library02_development_transfer"}:
        for run in training["runs"]:
            name="neural_seed_"+str(run["seed"])
            result["candidate_minus_neural_mae_intervals"][name]=paired_mae_ci(
                abs(pred["candidate"][common]-y[common]),abs(pred[name][common]-y[common]),5,1000,20261021+run["seed"])
        result["neural_across_seed_mae_mean"]=float(np.mean([result["metrics"]["neural_seed_"+str(s)]["mae_mps"] for s in training["seeds"]]))
    out["groups"][group]=result
    for name,m in result["metrics"].items():rows.append({"group":group,"model":name,**m})
    print(group,json.dumps(result),flush=True)
np.savez_compressed(R/"predictions.npz",stamps=stamps,frame_ids=data["frame_ids"],row_ids=data["row_ids"],sequence=sequence,
 roles=roles,reference=y,**pred)
with (R/"matched_metrics.csv").open("w",newline="",encoding="utf-8") as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
out["predictions_sha256"]=sha(R/"predictions.npz");write("evaluation_summary.json",out);print("EVALUATION_COMPLETE",flush=True)
