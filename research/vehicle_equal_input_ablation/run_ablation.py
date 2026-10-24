from common import *
install_guard()
os.environ.update(OMP_NUM_THREADS="1",MKL_NUM_THREADS="1")
import numpy as np,csv
from equal_input_core import fit_motion
sys.path.insert(0,str(H))
from hercules_core import load_archive,decode_radar,latest_imu,quaternion_matrix,fit_planar
assert not (R/"results.json").exists(),"frozen results already exist"
inputs={}
def record(path,expected=None):
    digest=sha(path)
    if expected is not None:assert digest==expected,str(path)
    inputs[str(checked(path).relative_to(P))]=digest
    return path
data_audit=json.loads((N/"data_audit.json").read_text())
record(N/"data_audit.json")
record(N/"evaluation.npz",data_audit["data_hashes"]["evaluation.npz"])
source_lock=json.loads((H/"download_lock.json").read_text())
for entry in source_lock["files"]:
    assert entry["sequence"] in ("Library01","Library02")
    record(H/entry["relative_path"],entry["sha256"])
for folder,names in [(H,["hercules_core.py","download_lock.json","Library01_replay.npz","Library02_replay.npz","failed_segments.json"]),
                     (M,["Library01_baselines.npz","Library02_baselines.npz","baseline_summary.json"]),
                     (P,["CURRENT_PROJECT_STATUS.md","FINAL_PROJECT_README.md"])]:
    for name in names:record(folder/name)
for name in ("PROTOCOL.md","common.py","equal_input_core.py","test_equal_input.py","run_ablation.py"):
    record(R/name)
write("input_lock.json",inputs)
membership=np.load(N/"evaluation.npz",allow_pickle=False)
methods={"planar_ols":(2,False),"planar_huber20":(2,True),"spatial_ols":(3,False),"spatial_huber20":(3,True)}
n=len(membership["stamps"]);assert n==564
pred={name:np.full(n,np.nan) for name in methods}
condition={name:np.full(n,np.nan) for name in methods}
stationarity={name:np.full(n,np.nan) for name in methods}
weighted_condition={name:np.full(n,np.nan) for name in methods}
reasons={name:np.full(n,"not_attempted",dtype="U24") for name in methods}
features=[];offsets=[0];prior_candidate=np.full(n,np.nan);prior_ols=np.full(n,np.nan)
max_candidate_difference=0.;max_prior_ols_difference=0.
def rotation(path):
    return np.array([float(x) for x in path.read_text().splitlines()[1].split(":",1)[1].split()]).reshape(3,4)[:,:3]
for seqid,seq in enumerate(("Library01","Library02")):
    directory=H/"data"/seq
    old=np.load(M/(seq+"_baselines.npz"),allow_pickle=False)
    original=np.load(H/(seq+"_replay.npz"),allow_pickle=False)
    cohort=original["cohort_indices"]
    rows=np.flatnonzero(membership["sequence"]==seqid)
    assert len(rows)==len(cohort)
    np.testing.assert_array_equal(membership["frame_ids"][rows],cohort)
    np.testing.assert_array_equal(membership["stamps"][rows],old["stamps"])
    np.testing.assert_array_equal(membership["reference"][rows],old["reference"])
    raw=load_archive(directory/"Radar/Continental.tar.gz")
    imu=[x.split(",") for x in (directory/"sensor_data/xsens_imu.csv").read_text().splitlines() if x.strip()]
    times=np.array([int(x[0]) for x in imu],np.int64)
    quat=np.array([[float(v) for v in x[1:5]] for x in imu])
    calibration=rotation(directory/"Calibration/IMU_LiDAR.txt")@rotation(directory/"Calibration/Continental_LiDAR.txt").T
    for k,row in enumerate(rows):
        stamp=int(membership["stamps"][row]); pc=decode_radar(raw[stamp])
        xyz=np.column_stack([pc[x] for x in ("x","y","z")]).astype(float)
        d=pc["v"].astype(float); distance=pc["range"].astype(float)
        keep=np.isfinite(xyz).all(axis=1)&np.isfinite(d)&np.isfinite(distance)&(distance>=3)&(distance<=100)
        xyz=xyz[keep];d=d[keep];norm=np.linalg.norm(xyz,axis=1);keep=norm>0
        rays=xyz[keep]/norm[keep,None];d=d[keep]
        imu_index=latest_imu(stamp,times);assert imu_index is not None
        rot=quaternion_matrix(quat[imu_index])@calibration
        a=rays@rot.T
        frame=np.column_stack([a,d]);features.append(frame);offsets.append(offsets[-1]+len(frame))
        original_candidate=fit_planar(rays,d,rot)
        assert original_candidate["quality"]
        for name,(dim,robust) in methods.items():
            result=fit_motion(a,d,dim,robust);reasons[name][row]=result["reason"]
            if result["speed"] is not None:
                pred[name][row]=result["speed"];condition[name][row]=result["condition"]
                weighted_condition[name][row]=result["weighted_condition"];stationarity[name][row]=result["stationarity_inf"]
        np.testing.assert_allclose(pred["planar_huber20"][row],old["candidate"][k],atol=1e-10,rtol=0)
        plain_original=float(np.linalg.norm(np.linalg.lstsq(rays,-d,rcond=None)[0]))
        np.testing.assert_allclose(plain_original,old["ols_3d"][k],atol=1e-10,rtol=0)
        max_candidate_difference=max(max_candidate_difference,abs(pred["planar_huber20"][row]-old["candidate"][k]))
        max_prior_ols_difference=max(max_prior_ols_difference,abs(plain_original-old["ols_3d"][k]))
        prior_candidate[row]=old["candidate"][k];prior_ols[row]=old["ols_3d"][k]
    print(seq,"completed",len(rows),"frames",flush=True)
reference=membership["reference"]
groups={"Library01_sparse_development":membership["sequence"]==0,
        "Library02_development_transfer":membership["sequence"]==1,
        "Library01_temporal_development":membership["roles"]=="temporal_test"}
def stats(p,y):
    finite=np.isfinite(p);known=finite&np.isfinite(y);errors=abs(p[known]-y[known])
    return {"attempts":len(p),"numeric_estimates":int(finite.sum()),"invalid_estimates":int((~finite).sum()),
            "scored":int(known.sum()),"unknown_reference":int((finite&~np.isfinite(y)).sum()),
            "mae_mps":float(np.mean(errors)) if len(errors) else None,
            "p95_mps":float(np.quantile(errors,.95)) if len(errors) else None,
            "max_mps":float(np.max(errors)) if len(errors) else None,
            "above2":int(np.sum(errors>2))}
table=[];paired=[];diagnostics={}
for group,mask in groups.items():
    common=mask&np.isfinite(reference)
    for values in pred.values():common&=np.isfinite(values)
    for name,values in pred.items():
        table.append({"group":group,"method":name,**stats(values[mask],reference[mask])})
        if name!="planar_huber20":
            delta=abs(pred["planar_huber20"][common]-reference[common])-abs(values[common]-reference[common])
            paired.append({"group":group,"candidate":"planar_huber20","comparator":name,
                           "attempts":int(mask.sum()),"common_scored":int(common.sum()),
                           "candidate_minus_comparator_mae":float(delta.mean()) if len(delta) else None})
    diagnostics[group]={name:{"condition_over100":int(np.sum(condition[name][mask]>100)),
        "max_condition":float(np.nanmax(condition[name][mask])),
        "stationarity_inf_p95":float(np.nanquantile(stationarity[name][mask],.95)),
        "stationarity_inf_max":float(np.nanmax(stationarity[name][mask])),
        "reason_counts":{str(v):int(np.sum(reasons[name][mask]==v)) for v in np.unique(reasons[name][mask])}} for name in methods}
for filename,records in (("metrics.csv",table),("paired_mae.csv",paired)):
    with (R/filename).open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
np.savez_compressed(R/"frames_and_predictions.npz",features=np.concatenate(features),offsets=np.array(offsets),
    stamps=membership["stamps"],frame_ids=membership["frame_ids"],sequence=membership["sequence"],roles=membership["roles"],reference=reference,
    historical_candidate=prior_candidate,historical_unrotated_ols=prior_ols,**pred,
    **{name+"_condition":v for name,v in condition.items()},**{name+"_weighted_condition":v for name,v in weighted_condition.items()},
    **{name+"_stationarity":v for name,v in stationarity.items()},**{name+"_reason":v for name,v in reasons.items()})
results={"scope":"Equal supplied radar/IMU observations; local fixed component ablations; proxy disagreement only",
         "methods":{name:{"dimensions":v[0],"huber":v[1],"iterations":20 if v[1] else 0} for name,v in methods.items()},
         "attempts":n,"metrics":table,"paired_mae":paired,"diagnostics":diagnostics,
         "prior_candidate_max_difference":max_candidate_difference,"prior_unrotated_ols_max_difference":max_prior_ols_difference,
         "transformed_spatial_ols_max_difference_from_historical":float(np.max(abs(pred["spatial_ols"]-prior_ols))),
         "independent_sequences":2,"fresh_final_holdout":False,"physical_reference_qualified":False,
         "published_full_framework_reproduced":False,"novelty_established":False,"promotion_allowed":False}
assert inputs=={name:sha(P/name) for name in inputs}
write("results.json",results)
print(json.dumps({"metrics":table,"paired_mae":paired,"max_candidate_difference":max_candidate_difference},indent=2),flush=True)
