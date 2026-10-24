from shared import *
import numpy as np
from scipy.spatial.transform import Rotation
verify_lock();summary=read("summary.json");total=0;segments=0;max_delta_error=0.
for path,digest in read("input_lock.json").items():
    if Path(path)==P/"CURRENT_PROJECT_STATUS.md":
        assert sha(R/"PROJECT_STATUS_BEFORE_INERTIAL.md")==digest
        expected=read("completion.json")["current_status_sha256"] if (R/"completion.json").exists() else digest
        assert sha(path)==expected
    else:assert sha(path)==digest,path
def check_metrics(values,y,mask,row):
    valid=mask&np.isfinite(values);scored=valid&np.isfinite(y);error=abs(values[scored]-y[scored])
    assert row["attempts"]==len(y) and row["admitted"]==sum(valid) and row["scored"]==sum(scored)
    assert row["unknown_reference"]==sum(valid&~np.isfinite(y)) and row["above2"]==sum(error>2)
    assert abs(row["availability"]-float(valid.mean()))<1e-12
    for key,val in [("mae_mps",float(error.mean()) if len(error) else None),("p95_mps",float(np.quantile(error,.95)) if len(error) else None),("max_mps",float(error.max()) if len(error) else None)]:
        if val is None:assert row[key] is None
        else:assert abs(row[key]-val)<1e-12
for seq in ["Library01","Library02"]:
    data=np.load(R/(seq+"_inertial.npz"),allow_pickle=False);old=np.load(H/(seq+"_replay.npz"),allow_pickle=False)
    opt=np.load(T/(seq+"_replay.npz"),allow_pickle=False);result=summary["sequences"][seq]
    t=data["stamps"];mt=data["imu_times"];acc=data["world_acceleration_xy"];past=data["past_index"];v=data["velocity_imu_origin_xy"]
    np.testing.assert_array_equal(t,old["stamps"]);np.testing.assert_allclose(np.linalg.norm(data["velocity_world_xy"],axis=1),old["speed"],atol=1e-10,rtol=0)
    folder=H/"data"/seq
    lines=[line.split(",") for line in (folder/"sensor_data/xsens_imu.csv").read_text().splitlines() if line.strip()]
    np.testing.assert_array_equal(mt,np.array([int(line[0]) for line in lines],np.int64))
    q=np.array([[float(x) for x in line[1:5]] for line in lines]);a=np.array([[float(x) for x in line[11:14]] for line in lines]);g=np.array([[float(x) for x in line[8:11]] for line in lines])
    np.testing.assert_array_equal(q,data["quaternion"]);np.testing.assert_array_equal(a,data["acceleration_imu"]);np.testing.assert_array_equal(g,data["gyro"])
    world=Rotation.from_quat(q).as_matrix();expected_acc=np.einsum("nij,nj->ni",world,a)[:,:2]
    np.testing.assert_allclose(acc,expected_acc,atol=1e-12,rtol=0)
    ji=np.searchsorted(mt,t,side="right")-1;np.testing.assert_array_equal(ji,data["imu_index"]);assert np.all((t-mt[ji])<=100_000_000)
    def tr(name):return np.array([float(x) for x in (folder/"Calibration"/name).read_text().splitlines()[1].split(":",1)[1].split()]).reshape(3,4)
    il=tr("IMU_LiDAR.txt");rl=tr("Continental_LiDAR.txt");lever=il[:,3]-il[:,:3]@rl[:,:3].T@rl[:,3]
    np.testing.assert_allclose(lever,data["lever"],atol=1e-12,rtol=0)
    expected_v=data["velocity_world_xy"]-np.einsum("nij,nj->ni",world[ji],np.cross(g[ji],lever))[:,:2]
    np.testing.assert_allclose(v,expected_v,atol=1e-12,rtol=0)
    expected_past=np.searchsorted(t,t-1_000_000_000,side="right")-1;np.testing.assert_array_equal(past,expected_past)
    ok=np.zeros(len(t),bool);integrated=np.full((len(t),2),np.nan)
    for i,j in enumerate(past):
        if j<0 or not old["quality"][i] or not old["quality"][j] or t[i]-t[j]>1_100_000_000:continue
        cursor=int(t[j]);end=int(t[i]);sensor=int(np.searchsorted(mt,cursor,side="right"))-1
        if sensor<0:continue
        delta=np.zeros(2);valid=True
        while cursor<end:
            next_time=min(end,int(mt[sensor+1])) if sensor+1<len(mt) else end
            if next_time-int(mt[sensor])>100_000_000:valid=False;break
            delta+=expected_acc[sensor]*(next_time-cursor)*1e-9
            cursor=next_time
            if sensor+1<len(mt) and cursor==int(mt[sensor+1]):sensor+=1
        if valid:ok[i]=True;integrated[i]=delta
    np.testing.assert_array_equal(ok,data["supported"])
    np.testing.assert_allclose(integrated,data["delta_imu"],atol=1e-12,rtol=0,equal_nan=True)
    max_delta_error=max(max_delta_error,float(np.max(abs(integrated[ok]-data["delta_imu"][ok]))))
    resid=np.full(len(t),np.nan);resid[ok]=np.linalg.norm(v[ok]-v[past[ok]]-integrated[ok],axis=1)
    np.testing.assert_allclose(resid,data["residual"],atol=1e-12,rtol=0,equal_nan=True)
    accept=ok&(resid<=1);np.testing.assert_array_equal(accept,data["accepted"])
    ri=old["latest"];base=opt["accepted"];new=base&(ri>=0)&accept[np.maximum(ri,0)]
    np.testing.assert_array_equal(base,data["base_accepted"]);np.testing.assert_array_equal(new,data["new_accepted"])
    np.testing.assert_array_equal(data["reference"],old["request_reference"])
    for mask,row in [(base,result["original_guard"]),(new,result["with_inertial_diagnostic"])]:
        check_metrics(data["query_prediction"],data["reference"],mask,row)
    for row in result["segments"]:
        start=row["start"];end=start+row["requests"]
        for mask,key in [(base,"original_guard"),(new,"with_inertial_diagnostic")]:
            check_metrics(data["query_prediction"][start:end],data["reference"][start:end],mask[start:end],row[key])
        segments+=1
    mid=int(t[0])+int(t[-1]-t[0])//2;constant=np.tile([3.,0],(len(t),1));step=np.zeros_like(v);step[t>=mid,0]=3
    ramp=np.zeros_like(v);ramp[:,0]=3*np.clip((t-mid)*1e-9/10,0,1);safe=np.maximum(past,0)
    for name,bias in [("nominal",np.zeros_like(v)),("constant_world_x_plus3",constant),("midpoint_step_plus3",step),("slow_10s_ramp_plus3",ramp),("step_with_forged_imu_delta",step)]:
        av=v+bias;delta=integrated if name!="step_with_forged_imu_delta" else av-av[safe]
        e=np.full(len(t),np.nan);e[ok]=np.linalg.norm(av[ok]-av[past[ok]]-delta[ok],axis=1);admit=ok&(e<=1)
        np.testing.assert_allclose(e,data[name+"__residual"],atol=1e-12,rtol=0,equal_nan=True)
        np.testing.assert_array_equal(admit,data[name+"__accepted"])
        row=result["attacks"][name];assert row["supported"]==sum(ok) and row["accepted"]==sum(admit)
        check_metrics(data[name+"__speed"],data["frame_reference"],admit,row["proxy_metrics_on_full_frame_denominator"])
    np.testing.assert_allclose(data["constant_world_x_plus3__residual"],resid,atol=1e-12,rtol=0,equal_nan=True)
    assert np.all(data["step_with_forged_imu_delta__accepted"]==ok)
    total+=len(t)
assert segments==11
out={"passed":True,"radar_frames_verified":total,"retained_segments_verified":segments,"all_imu_integrals_independently_recomputed":True,
 "max_integral_difference_mps":max_delta_error,"nominal_and_attack_metrics_verified":True,"constant_offset_invariance_verified":True,
 "old_results_preserved":True,"reserved_payloads_accessed":False,"reference_gap_resolved":False,"physical_witness_admitted":False}
write("independent_verification.json",out);print(json.dumps(out,indent=2))
