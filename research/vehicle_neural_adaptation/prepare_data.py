from shared import *
import numpy as np,time,shutil
sys.path.insert(0,str(H))
from hercules_core import load_archive,decode_radar,latest_imu,quaternion_matrix,fit_planar
from neural_core import point_features,split_labels
verify_lock()
assert not (R/"data_audit.json").exists(),"Preparation already frozen; do not overwrite"
old={}
for folder,names in [(H,["hercules_core.py","download_lock.json","Library01_replay.npz","Library02_replay.npz","failed_segments.json"]),
 (M,["Library01_baselines.npz","Library02_baselines.npz","baseline_summary.json","completion.json"]),
 (R.parent/"vehicle_neural_baseline",["completion.json","EVALUATION_SCOPE.json","model_admission_audit.json"])]:
    for name in names:old[str(folder/name)]=sha(folder/name)
old[str(P/"CURRENT_PROJECT_STATUS.md")]=sha(P/"CURRENT_PROJECT_STATUS.md")
shutil.copyfile(P/"CURRENT_PROJECT_STATUS.md",R/"PROJECT_STATUS_BEFORE_ADAPTATION.md")
for e in json.loads((H/"download_lock.json").read_text())["files"]:
    assert e["sequence"] in {"Library01","Library02"}
    path=H/e["relative_path"];assert sha(path)==e["sha256"]
    old[str(path)]=e["sha256"]
all_features=[];stamps=[];ids=[];seqids=[];roles=[];refs=[];prep=[];counts=[]
pred={name:[] for name in ["candidate","author_3d_ransac","ols_3d"]}
def rotation(path):
    return np.array([float(v) for v in path.read_text().splitlines()[1].split(":",1)[1].split()]).reshape(3,4)[:,:3]
for seqid,seq in enumerate(["Library01","Library02"]):
    D=H/"data"/seq;data=np.load(H/(seq+"_replay.npz"),allow_pickle=False)
    matched=np.load(M/(seq+"_baselines.npz"),allow_pickle=False);cohort=data["cohort_indices"]
    np.testing.assert_array_equal(cohort,matched["indices"])
    np.testing.assert_array_equal(data["stamps"][cohort],matched["stamps"])
    payload=load_archive(D/"Radar/Continental.tar.gz")
    rows=[line.split(",") for line in (D/"sensor_data/xsens_imu.csv").read_text().splitlines() if line.strip()]
    mt=np.array([int(x[0]) for x in rows],np.int64);q=np.array([[float(v) for v in x[1:5]] for x in rows])
    calibration=rotation(D/"Calibration/IMU_LiDAR.txt")@rotation(D/"Calibration/Continental_LiDAR.txt").T
    labels=split_labels(data["stamps"][cohort],seq)
    for k,i in enumerate(cohort):
        stamp=int(data["stamps"][i]);start=time.perf_counter_ns();pc=decode_radar(payload[stamp])
        xyz=np.column_stack([pc[name] for name in ["x","y","z"]]).astype(float);dop=pc["v"].astype(float);ran=pc["range"].astype(float)
        keep=np.isfinite(xyz).all(axis=1)&np.isfinite(dop)&np.isfinite(ran)&(ran>=3)&(ran<=100)
        xyz=xyz[keep];dop=dop[keep];norm=np.linalg.norm(xyz,axis=1);keep=norm>0
        rays=xyz[keep]/norm[keep,None];dop=dop[keep];j=latest_imu(stamp,mt);assert j is not None
        rot=quaternion_matrix(q[j])@calibration
        f=point_features(rays,dop,rot);prep.append(time.perf_counter_ns()-start)
        check=fit_planar(rays,dop,rot);assert check["quality"]
        np.testing.assert_allclose(check["speed"],matched["candidate"][k],atol=1e-10,rtol=0)
        all_features.append(f);stamps.append(stamp);ids.append(int(i));seqids.append(seqid);roles.append(str(labels[k]))
        refs.append(float(data["reference_primary"][i]));counts.append(len(f))
        for name in pred:pred[name].append(float(matched[name][k]))
    print(seq,"prepared",len(cohort),"frames",flush=True)
stamps=np.array(stamps,np.int64);roles=np.array(roles);refs=np.array(refs);seqids=np.array(seqids,np.int8)
def export(name,indices,include_baselines):
    fs=[all_features[i] for i in indices];offsets=np.r_[0,np.cumsum([len(f) for f in fs])]
    content=dict(features=np.concatenate(fs),offsets=offsets,stamps=stamps[indices],roles=roles[indices],
        reference=refs[indices],frame_ids=np.array(ids)[indices],sequence=seqids[indices],row_ids=indices)
    if include_baselines:content.update({key:np.array(values)[indices] for key,values in pred.items()})
    np.savez_compressed(R/name,**content)
training=np.flatnonzero(np.isin(roles,["train","validation"]))
assert np.all(seqids[training]==0)
export("train_validation.npz",training,False);export("evaluation.npz",np.arange(len(roles)),True)
summary={"attempts":len(roles),"role_counts":{s:int(sum(roles==s)) for s in np.unique(roles)},
 "known_reference_by_role":{s:int(sum((roles==s)&np.isfinite(refs))) for s in np.unique(roles)},
 "training_file_only_train_validation":True,"no_point_subsampling":True,"point_count_min":min(counts),"point_count_max":max(counts),
 "feature_preparation_p95_ms":float(np.quantile(prep,.95)*1e-6),"old_hashes":old,
 "data_hashes":{name:sha(R/name) for name in ["train_validation.npz","evaluation.npz"]},
 "training_targets":"frozen pose speed magnitude proxy","final_holdout_used":False}
write("data_audit.json",summary);print(json.dumps({k:v for k,v in summary.items() if k!="old_hashes"},indent=2),flush=True)
