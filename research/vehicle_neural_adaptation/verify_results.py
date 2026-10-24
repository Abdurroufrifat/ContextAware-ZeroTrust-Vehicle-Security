from shared import *
import numpy as np,torch
verify_lock()
audit=read("data_audit.json");training=read("training_summary.json");summary=read("evaluation_summary.json")
for path,digest in audit["old_hashes"].items():
    if Path(path)==P/"CURRENT_PROJECT_STATUS.md":
        assert sha(R/"PROJECT_STATUS_BEFORE_ADAPTATION.md")==digest
        expected=read("completion.json")["current_status_sha256"] if (R/"completion.json").exists() else digest
        assert sha(path)==expected,path
    else:assert sha(path)==digest,path
for name,digest in audit["data_hashes"].items():assert sha(R/name)==digest,name
data=np.load(R/"evaluation.npz",allow_pickle=False);tv=np.load(R/"train_validation.npz",allow_pickle=False)
p=np.load(R/"predictions.npz",allow_pickle=False);fs=frames(data);trainfs=frames(tv)
roles=data["roles"];stamps=data["stamps"];sequence=data["sequence"];y=data["reference"]
assert len(roles)==564 and sum(sequence==0)==316 and sum(sequence==1)==248
for seqid,seq in enumerate(["Library01","Library02"]):
    ix=np.flatnonzero(sequence==seqid);old=np.load(H/(seq+"_replay.npz"),allow_pickle=False)
    matched=np.load(M/(seq+"_baselines.npz"),allow_pickle=False)
    np.testing.assert_array_equal(data["frame_ids"][ix],old["cohort_indices"])
    np.testing.assert_array_equal(stamps[ix],old["stamps"][old["cohort_indices"]])
    np.testing.assert_array_equal(y[ix],old["reference_primary"][old["cohort_indices"]])
    for name in ["candidate","author_3d_ransac","ols_3d"]:np.testing.assert_array_equal(data[name][ix],matched[name])
    if seqid==0:
        rel=stamps[ix]-stamps[ix[0]];duration=int(rel[-1]);b1=duration*3//5;b2=duration*4//5;g=2_000_000_000
        masks={"train":rel<b1-g,"validation":(rel>=b1+g)&(rel<b2-g),"temporal_test":rel>=b2+g}
        for name,mask in masks.items():np.testing.assert_array_equal(roles[ix]==name,mask)
        assert np.all(roles[ix][~np.logical_or.reduce(list(masks.values()))]=="purge")
    else:assert np.all(roles[ix]=="transfer")
expected=np.flatnonzero(np.isin(roles,["train","validation"]))
np.testing.assert_array_equal(tv["row_ids"],expected);assert np.all(tv["sequence"]==0)
for j,i in enumerate(expected):
    np.testing.assert_array_equal(trainfs[j],fs[i]);np.testing.assert_array_equal(tv["reference"][j],y[i])
assert sum(~np.isfinite(y))==2
assert not training["evaluation_labels_opened"] and len(training["runs"])==3
torch.set_num_threads(1);verified=0;max_error=0.
for run in training["runs"]:
    seed=run["seed"];history=read("history_"+str(seed)+".json")
    assert len(history)==150 and sha(R/("history_"+str(seed)+".json"))==run["history_sha256"]
    best=min(history,key=lambda row:(row["validation_mae"],row["epoch"]))
    assert best["epoch"]==run["selected_epoch"] and best["validation_mae"]==run["validation_mae"]
    path=R/run["checkpoint"];assert sha(path)==run["checkpoint_sha256"]
    checkpoint=torch.load(path,map_location="cpu",weights_only=True)
    assert checkpoint["seed"]==seed and checkpoint["selected_epoch"]==best["epoch"]
    w={name:tensor.numpy() for name,tensor in checkpoint["state_dict"].items()}
    # Independent NumPy implementation, no production SpeedSet/pack/metrics imports.
    out=[]
    for f in fs:
        a=np.maximum(f@w["points.0.weight"].T+w["points.0.bias"],0)
        a=np.maximum(a@w["points.2.weight"].T+w["points.2.bias"],0)
        v=np.r_[a.mean(axis=0),a.max(axis=0)]
        v=np.maximum(v@w["head.0.weight"].T+w["head.0.bias"],0)
        z=(v@w["head.2.weight"].T+w["head.2.bias"])[0]
        out.append(float(np.logaddexp(np.float32(0),z)))
    out=np.array(out);actual=p["neural_seed_"+str(seed)]
    np.testing.assert_allclose(actual,out,atol=2e-5,rtol=2e-6)
    max_error=max(max_error,float(np.max(abs(actual-out))));verified+=1
    vi=(roles=="validation")&np.isfinite(y)
    assert abs(float(np.mean(abs(actual[vi]-y[vi])))-best["validation_mae"])<1e-5
np.testing.assert_array_equal(p["roles"],roles);np.testing.assert_array_equal(p["reference"],y)
for name in ["candidate","author_3d_ransac","ols_3d"]:np.testing.assert_array_equal(p[name],data[name])
assert np.all(p["train_median"]==np.median(y[(roles=="train")&np.isfinite(y)]))
assert read("raw_feature_verification.json")["passed"]
for group,result in summary["groups"].items():
    masks={"Library01_temporal_test":roles=="temporal_test","Library02_development_transfer":roles=="transfer",
      "Library01_training_diagnostic":roles=="train","Library01_validation_selection":roles=="validation",
      "Library01_purged_diagnostic":roles=="purge",
      "Library01_original_failed_window_sparse_cohort":(sequence==0)&(stamps>=1724812166350000000)&(stamps<=1724812181400000000)}
    ix=masks[group];assert result["attempts"]==sum(ix)
    for name,m in result["metrics"].items():
        admitted=ix&np.isfinite(p[name]);scored=admitted&np.isfinite(y);e=abs(p[name][scored]-y[scored])
        assert m["attempts"]==sum(ix) and m["admitted"]==sum(admitted) and m["scored"]==sum(scored)
        assert m["unknown_reference"]==sum(admitted&~np.isfinite(y)) and m["above2"]==sum(e>2)
        assert abs(m["mae_mps"]-float(e.mean()))<1e-12
        assert abs(m["p95_mps"]-float(np.percentile(e,95)))<1e-12
        assert abs(m["max_mps"]-float(max(e)))<1e-12
    for name,ci in result["candidate_minus_neural_mae_intervals"].items():
        ix2=ix&np.isfinite(y)&np.isfinite(p[name])&np.isfinite(p["candidate"])
        delta=abs(p["candidate"][ix2]-y[ix2])-abs(p[name][ix2]-y[ix2])
        assert abs(ci["difference"]-float(delta.mean()))<1e-12
        rng=np.random.default_rng(ci["seed"]);values=[];n=len(delta)
        for _ in range(1000):
            starts=rng.integers(0,n,size=(n+4)//5);order=[(int(start)+j)%n for start in starts for j in range(5)][:n]
            values.append(float(delta[order].mean()))
        lo,hi=np.quantile(values,[.025,.975]);np.testing.assert_allclose([ci["low"],ci["high"]],[lo,hi],atol=1e-12,rtol=0)
assert sha(R/"predictions.npz")==summary["predictions_sha256"]
out={"passed":True,"attempts_verified":564,"neural_checkpoint_numpy_prediction_replays":verified,
 "numpy_torch_max_abs_difference":max_error,"temporal_test_attempts":int(sum(roles=="temporal_test")),
 "development_transfer_attempts":int(sum(roles=="transfer")),"unknown_references_preserved":2,
 "all_metrics_and_block_intervals_recomputed":True,"validation_only_selection_verified":True,
 "separate_training_file_verified":True,"original_results_unchanged":True,"final_holdout_used":False,
 "reference_gap_resolved":False,"overall_project_complete":False}
write("independent_verification.json",out);print(json.dumps(out,indent=2))
