from shared import *
import numpy as np,time,shutil
sys.path[:0]=[str(H),str(M)]
from hercules_core import decode_radar,load_archive,fit_planar,quaternion_matrix,latest_imu
from inertial_core import acceleration_world_xy,imu_origin_velocity,innovations,compare_changes
from evidence_core import metrics
names=["PROTOCOL.md","test_inertial_core.py","inertial_core.py","shared.py","run_study.py"]
lock={name:sha(R/name) for name in names}
if (R/"experiment_lock.json").exists():assert read("experiment_lock.json")==lock
else:write("experiment_lock.json",lock)
assert not (R/"summary.json").exists(),"Frozen outcomes already exist"
old={}
for folder,names_in in [(H,["hercules_core.py","Library01_replay.npz","Library02_replay.npz","failed_segments.json"]),
 (T,["Library01_replay.npz","Library02_replay.npz","summary.json"]),
 (R.parent/"vehicle_neural_adaptation",["completion.json","evaluation_summary.json"])]:
    for name in names_in:old[str(folder/name)]=sha(folder/name)
status=P/"CURRENT_PROJECT_STATUS.md";old[str(status)]=sha(status)
shutil.copyfile(status,R/"PROJECT_STATUS_BEFORE_INERTIAL.md")
for item in json.loads((H/"download_lock.json").read_text())["files"]:
    assert item["sequence"] in {"Library01","Library02"}
    path=H/item["relative_path"];assert sha(path)==item["sha256"];old[str(path)]=item["sha256"]
write("input_lock.json",old)
def extrinsic(path):
    return np.array([float(v) for v in path.read_text().splitlines()[1].split(":",1)[1].split()]).reshape(3,4)
def report(pred,ref,mask):
    row=metrics(pred,ref,mask)
    row["numeric_pass"]=bool(row["availability"]>=.95 and row["scored"]>0 and row["p95_mps"]<=.5 and row["above2"]==0)
    row["complete_reference_support"]=row["unknown_reference"]==0
    return row
summary={"scope":"reference-free causal inertial-stage diagnostic; frozen-schedule query mask, not deployment proof",
 "sequences":{},"reference_gap_resolved":False,"overall_project_complete":False,"physical_witness_admitted":False}
for seq in ["Library01","Library02"]:
    folder=H/"data"/seq;radar=np.load(H/(seq+"_replay.npz"),allow_pickle=False);opt=np.load(T/(seq+"_replay.npz"),allow_pickle=False)
    t=radar["stamps"];rows=[line.split(",") for line in (folder/"sensor_data/xsens_imu.csv").read_text().splitlines() if line.strip()]
    mt=np.array([int(row[0]) for row in rows],np.int64)
    q=np.array([[float(v) for v in row[1:5]] for row in rows]);gyro=np.array([[float(v) for v in row[8:11]] for row in rows])
    acceleration=np.array([[float(v) for v in row[11:14]] for row in rows])
    raw_world_acc=acceleration_world_xy(q,acceleration)
    IL=extrinsic(folder/"Calibration/IMU_LiDAR.txt");RL=extrinsic(folder/"Calibration/Continental_LiDAR.txt")
    calibration=IL[:,:3]@RL[:,:3].T;lever=IL[:,3]-IL[:,:3]@RL[:,:3].T@RL[:,3]
    payload=load_archive(folder/"Radar/Continental.tar.gz")
    velocity=np.full((len(t),2),np.nan);quality=np.zeros(len(t),bool);imu_index=np.full(len(t),-1,np.int64)
    for i,stamp in enumerate(t):
        pc=decode_radar(payload[int(stamp)]);xyz=np.column_stack([pc[name] for name in ["x","y","z"]]).astype(float)
        d=pc["v"].astype(float);ran=pc["range"].astype(float)
        mask=np.isfinite(xyz).all(axis=1)&np.isfinite(d)&np.isfinite(ran)&(ran>=3)&(ran<=100)
        xyz=xyz[mask];d=d[mask];norm=np.linalg.norm(xyz,axis=1);mask=norm>0;rays=xyz[mask]/norm[mask,None];d=d[mask]
        j=latest_imu(int(stamp),mt);assert j is not None;imu_index[i]=j
        result=fit_planar(rays,d,quaternion_matrix(q[j])@calibration)
        velocity[i]=result["velocity"];quality[i]=result["quality"]
    np.testing.assert_allclose(np.linalg.norm(velocity,axis=1),radar["speed"],atol=1e-10,rtol=0)
    np.testing.assert_array_equal(quality,radar["quality"])
    start=time.perf_counter_ns();anchored=imu_origin_velocity(velocity,q[imu_index],gyro[imu_index],lever)
    origin_correction_ns=time.perf_counter_ns()-start
    out=innovations(anchored,t,mt,raw_world_acc,quality)
    qtime=radar["queries"];ri=radar["latest"];safe=np.maximum(ri,0)
    np.testing.assert_array_equal(qtime,opt["queries"])
    np.testing.assert_array_equal(radar["speed"][safe],opt["radar_query_speed"])
    base=opt["accepted"];added=(ri>=0)&out["accepted"][safe];new=base&added
    ref=opt["primary"];pred=opt["radar_query_speed"]
    result={"frames":len(t),"requests":len(qtime),"lever_IMU_to_radar_m":lever.tolist(),
      "frame_pairs_supported":int(out["supported"].sum()),"inertial_frame_accepted":int(out["accepted"].sum()),
      "nominal_innovation_p95_mps":float(np.quantile(out["residual"][out["supported"]],.95)),
      "check_only_p95_ms":float(np.quantile(out["check_ns"],.95)*1e-6),"origin_correction_total_ms":origin_correction_ns*1e-6,
      "original_guard":report(pred,ref,base),"with_inertial_diagnostic":report(pred,ref,new),
      "base_admissions_removed":int((base&~new).sum()),"segments":[],"attacks":{}}
    for begin in range(0,len(qtime),1200):
        sl=slice(begin,min(begin+1200,len(qtime)))
        result["segments"].append({"start":begin,"requests":len(qtime[sl]),
          "original_guard":report(pred[sl],ref[sl],base[sl]),"with_inertial_diagnostic":report(pred[sl],ref[sl],new[sl])})
    mid=int(t[0])+int(t[-1]-t[0])//2
    zeros=np.zeros_like(anchored);constant=zeros+np.array([3.,0])
    step=zeros.copy();step[t>=mid,0]=3.
    ramp=zeros.copy();ramp[:,0]=3*np.clip((t-mid)*1e-9/10,0,1)
    attack_arrays={};past=np.maximum(out["past_index"],0)
    for name,bias in [("nominal",zeros),("constant_world_x_plus3",constant),("midpoint_step_plus3",step),("slow_10s_ramp_plus3",ramp),("step_with_forged_imu_delta",step)]:
        av=anchored+bias;delta=out["delta_imu"].copy()
        if name=="step_with_forged_imu_delta":delta=av-av[past]
        row=compare_changes(av,out["past_index"],delta,out["supported"])
        speed=np.linalg.norm(velocity+bias,axis=1)
        s={"scope":"inertial stage only; no optical source attack/injection reproduced","supported":int(row["supported"].sum()),
           "accepted":int(row["accepted"].sum()),"rejected":int((row["supported"]&~row["accepted"]).sum()),
           "proxy_metrics_on_full_frame_denominator":metrics(speed,radar["reference_primary"],row["accepted"])}
        for period,window in [("onset_1p1s",(t>=mid)&(t<=mid+1_100_000_000)),("persistent_after_onset",t>mid+1_100_000_000)]:
            support=row["supported"]&window
            s[period]={"supported":int(support.sum()),"accepted":int((support&row["accepted"]).sum())}
        result["attacks"][name]=s
        attack_arrays.update({name+"__residual":row["residual"],name+"__accepted":row["accepted"],name+"__speed":speed})
    np.testing.assert_allclose(attack_arrays["constant_world_x_plus3__residual"],out["residual"],atol=1e-12,rtol=0,equal_nan=True)
    np.savez_compressed(R/(seq+"_inertial.npz"),stamps=t,velocity_world_xy=velocity,velocity_imu_origin_xy=anchored,
      imu_times=mt,world_acceleration_xy=raw_world_acc,imu_index=imu_index,quaternion=q,gyro=gyro,acceleration_imu=acceleration,lever=lever,
      queries=qtime,query_radar_index=ri,reference=ref,query_prediction=pred,base_accepted=base,new_accepted=new,
      quality=quality,frame_reference=radar["reference_primary"],**out,**attack_arrays)
    summary["sequences"][seq]=result
    print(seq,json.dumps({k:result[k] for k in ["frames","frame_pairs_supported","nominal_innovation_p95_mps","original_guard","with_inertial_diagnostic"]}),flush=True)
write("summary.json",summary);print("INERTIAL_STUDY_COMPLETE",flush=True)
