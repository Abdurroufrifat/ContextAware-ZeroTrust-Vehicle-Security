"""Fixed source-stress study; reference used only after estimates."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
from scipy.spatial.transform import Rotation
R=Path(__file__).resolve().parent;H=R.parent/"vehicle_hercules_development"
sys.path.insert(0,str(H))
from hercules_core import decode_radar,quaternion_matrix,fit_planar,load_archive,latest_imu
for n,h in json.loads((H/"experiment_lock.json").read_text()).items():
 assert hashlib.sha256((H/n).read_bytes()).hexdigest()==h,n
before={n:hashlib.sha256((H/n).read_bytes()).hexdigest() for n in ["hercules_core.py","Library01_replay.npz","Library02_replay.npz","development_summary.json"]}
cases=["nominal","points10pct_plus10","points20pct_plus10","points40pct_plus10","coherent_speed_plus3","orientation_pitch30","imu_missing","quaternion_zero","imu_stale100ms_plus1ns","replay1s_original_timestamp","replay1s_forged_timestamp"]
def transform(p):
 return np.array([float(x) for x in p.read_text().splitlines()[1].split(":",1)[1].split()]).reshape(3,4)[:,:3]
def geometry(b):
 pc=decode_radar(b);xyz=np.column_stack([pc[k] for k in ["x","y","z"]]).astype(float);d=pc["v"].astype(float);rg=pc["range"].astype(float)
 norm=np.linalg.norm(xyz,axis=1);mask=np.all(np.isfinite(xyz),axis=1)&np.isfinite(d)&np.isfinite(rg)&(rg>=3)&(rg<=100)&(norm>0)
 return xyz[mask]/norm[mask,None],d[mask]
rows=[]
for seq in ["Library01","Library02"]:
 D=H/"data"/seq;saved=np.load(H/(seq+"_replay.npz"),allow_pickle=False);t=saved["stamps"]
 data=load_archive(D/"Radar"/"Continental.tar.gz")
 mr=[l.split(",") for l in (D/"sensor_data"/"xsens_imu.csv").read_text().splitlines() if l.strip()]
 mt=np.array([int(x[0]) for x in mr],dtype=np.int64);quats=np.array([[float(v) for v in x[1:5]] for x in mr])
 ir=transform(D/"Calibration"/"IMU_LiDAR.txt")@transform(D/"Calibration"/"Continental_LiDAR.txt").T
 for i in range(0,len(t),20):
  stamp=int(t[i]);j=latest_imu(stamp,mt);assert j is not None
  dirs,doppler=geometry(data[stamp]);rotation=quaternion_matrix(quats[j])@ir
  nominal=fit_planar(dirs,doppler,rotation)
  assert nominal["quality"]==bool(saved["quality"][i])
  assert abs(nominal["speed"]-saved["speed"][i])<1e-10
  for case in cases:
   row={"sequence":seq,"index":i,"case":case,"request_stamp_ns":stamp,"claimed_stamp_ns":stamp,"true_source_stamp_ns":stamp,"imu_stamp_ns":int(mt[j]),"nominal_speed_mps":nominal["speed"],"quality_passed":False,"reason":"unknown"}
   try:
    n=dirs;d=doppler.copy();rot=rotation
    if case=="imu_missing":raise ValueError("missing_context")
    if case=="quaternion_zero":quaternion_matrix([0,0,0,0])
    if case=="imu_stale100ms_plus1ns":
     row["imu_stamp_ns"]=stamp-100_000_001
     if stamp-row["imu_stamp_ns"]>100_000_000:raise ValueError("expired_context")
    if case.startswith("replay1s"):
     old=int(np.searchsorted(t,stamp-1_000_000_000,side="right"))-1
     if old<0:raise ValueError("no_replay_history")
     row["true_source_stamp_ns"]=int(t[old])
     row["claimed_stamp_ns"]=int(t[old]) if case.endswith("original_timestamp") else stamp
     if stamp-row["claimed_stamp_ns"]>100_000_000:raise ValueError("expired_radar")
     n,d=geometry(data[int(t[old])])
    if case.startswith("points"):
     frac=int(case.split("pct")[0].removeprefix("points"));indices=np.arange(len(d))
     d[(indices%10)<frac//10]+=10.
    if case=="coherent_speed_plus3":
     v=np.array(nominal["velocity"]);delta=3*v/np.linalg.norm(v) if np.linalg.norm(v)>1e-12 else np.array([3.,0.])
     A=(n@rotation.T)[:,:2];d=d-A@delta
    if case=="orientation_pitch30":rot=Rotation.from_euler("y",30,degrees=True).as_matrix()@rotation
    result=fit_planar(n,d,rot);row.update(quality_passed=result["quality"],speed_mps=result["speed"],reason=result["reason"],inlier_fraction=result.get("inlier_fraction"),inlier_rmse=result.get("inlier_rmse"))
   except ValueError as ex:row["reason"]=str(ex)
   row["reference_primary_mps"]=float(saved["reference_primary"][i]) if np.isfinite(saved["reference_primary"][i]) else None
   row["reference_INS_mps"]=float(saved["reference_ins"][i]) if np.isfinite(saved["reference_ins"][i]) else None
   rows.append(row)
 print(seq,"frames",len(range(0,len(t),20)),"cases",len(cases),flush=True)
def report(group):
 accepted=[x for x in group if x["quality_passed"] and x.get("speed_mps") is not None]
 out={"attempts":len(group),"quality_passed":len(accepted),"quality_fraction":len(accepted)/len(group),"reasons":{}}
 for x in group:out["reasons"][x["reason"]]=out["reasons"].get(x["reason"],0)+1
 shifts=[abs(x["speed_mps"]-x["nominal_speed_mps"]) for x in accepted];out["quality_passed_nominal_shift_above2"]=sum(x>2 for x in shifts);out["nominal_shift_p95_mps"]=float(np.quantile(shifts,.95)) if shifts else None
 for key,label in [("reference_primary_mps","primary"),("reference_INS_mps","INS")]:
  err=[abs(x["speed_mps"]-x[key]) for x in accepted if x[key] is not None]
  out[label]={"scored":len(err),"p95_mps":float(np.quantile(err,.95)) if err else None,"above2":sum(x>2 for x in err)}
 return out
result={"scope":"candidate quality checks only; source compromise and corrupted context, not policy authorization or physical certification","cohort_frames":sum(x["case"]=="nominal" for x in rows),"cases":{c:report([x for x in rows if x["case"]==c]) for c in cases},"by_sequence":{seq:{c:report([x for x in rows if x["case"]==c and x["sequence"]==seq]) for c in cases} for seq in ["Library01","Library02"]},"unchanged_input_hashes":before,"production_changed":False,"Library03_payloads_accessed":False}
for n,h in before.items():assert hashlib.sha256((H/n).read_bytes()).hexdigest()==h,n
(R/"source_stress_rows.json").write_text(json.dumps(rows,indent=2,allow_nan=False),encoding="utf-8")
(R/"source_stress_summary.json").write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
print(json.dumps({"cohort_frames":result["cohort_frames"],"cases":result["cases"]},indent=2),flush=True)
