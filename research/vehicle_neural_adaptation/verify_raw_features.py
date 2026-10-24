from shared import *
import numpy as np,tarfile
from scipy.spatial.transform import Rotation
verify_lock()
data=np.load(R/"evaluation.npz",allow_pickle=False);fs=frames(data);checked=0
dtype=np.dtype([("x","<f4"),("y","<f4"),("z","<f4"),("v","<f4"),("range","<f4"),("rcs","u1"),("az","<f4"),("el","<f4")])
assert dtype.itemsize==29
def calibration(path):
    values=[float(v) for v in path.read_text().splitlines()[1].split(":",1)[1].split()]
    return np.array(values).reshape(3,4)[:,:3]
for seqid,seq in enumerate(["Library01","Library02"]):
    indices=np.flatnonzero(data["sequence"]==seqid)
    selected=indices[[0,len(indices)//2,-1]];wanted={int(data["stamps"][i]):int(i) for i in selected}
    folder=H/"data"/seq
    text=[row.split(",") for row in (folder/"sensor_data/xsens_imu.csv").read_text().splitlines() if row.strip()]
    times=np.array([int(row[0]) for row in text],dtype=np.int64)
    q=np.array([[float(v) for v in row[1:5]] for row in text])
    fixed=calibration(folder/"Calibration/IMU_LiDAR.txt")@calibration(folder/"Calibration/Continental_LiDAR.txt").T
    with tarfile.open(folder/"Radar/Continental.tar.gz","r|gz") as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(".bin"):continue
            stamp=int(Path(member.name).stem)
            if stamp not in wanted:continue
            buf=archive.extractfile(member).read();assert len(buf)%29==0
            points=np.frombuffer(buf,dtype=dtype)
            xyz=np.column_stack([points["x"],points["y"],points["z"]]).astype(float);d=points["v"].astype(float);r=points["range"].astype(float)
            valid=np.isfinite(xyz).all(axis=1)&np.isfinite(d)&np.isfinite(r)&(r>=3)&(r<=100)
            xyz=xyz[valid];d=d[valid];norm=np.sqrt((xyz**2).sum(axis=1));valid=norm>0
            unit=xyz[valid]/norm[valid,None];d=d[valid]
            j=int(np.searchsorted(times,stamp,side="right"))-1
            assert j>=0 and 0<=stamp-int(times[j])<=100_000_000
            rotate=Rotation.from_quat(q[j]).as_matrix()@fixed
            expected=np.column_stack([unit@rotate.T,-d/20]).astype(np.float32)
            np.testing.assert_array_equal(expected,fs[wanted[stamp]])
            checked+=1
assert checked==6
out={"passed":True,"raw_frame_feature_oracles":checked,"both_development_sequences_checked":True,
 "all_points_preserved_in_sampled_frames":True,"model_feature_functions_imported":False,"protected_payloads_opened":False}
write("raw_feature_verification.json",out);print(json.dumps(out,indent=2))
