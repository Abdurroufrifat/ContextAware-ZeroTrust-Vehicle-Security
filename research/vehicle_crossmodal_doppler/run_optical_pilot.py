from pathlib import Path
import json,sys,hashlib,time,ast,struct
import numpy as np
R=Path(__file__).resolve().parent;H=R.parent/"vehicle_hercules_development";sys.path.insert(0,str(H))
from hercules_core import latest_imu
from crossmodal_core import decode_optical,estimate_optical,agree
names=["EXPERIMENT_PROTOCOL.md","crossmodal_core.py","test_crossmodal_core.py","run_optical_pilot.py","pilot_data_lock.json"]
frozen={n:hashlib.sha256((R/n).read_bytes()).hexdigest() for n in names}
lock=R/"experiment_lock.json"
if lock.exists():assert json.loads(lock.read_text())==frozen
else:lock.write_text(json.dumps(frozen,indent=2),encoding="utf-8")
source=H/"official_source"/"HeRCULES_Pointcloud_Toolbox"/"bin2pcd.py"
tree=ast.parse(source.read_text());func=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="bin_to_pcd")
oracle={"np":np,"struct":struct};exec(compile(ast.Module(body=[func],type_ignores=[]),str(source),"exec"),oracle)
def tr(p):
 return np.array([float(x) for x in p.read_text().splitlines()[1].split(":",1)[1].split()]).reshape(3,4)
rows=[];out={}
for seq,group in json.loads((R/"pilot_data_lock.json").read_text()).items():
 D=H/"data"/seq;radar=np.load(H/(seq+"_replay.npz"),allow_pickle=False);T=tr(D/"Calibration"/"Continental_LiDAR.txt");I=tr(D/"Calibration"/"IMU_LiDAR.txt")[:,:3]
 mr=[l.split(",") for l in (D/"sensor_data"/"xsens_imu.csv").read_text().splitlines() if l.strip()]
 mt=np.array([int(x[0]) for x in mr],dtype=np.int64);gyro=np.array([[float(v) for v in x[8:11]] for x in mr]);local=[]
 for ix,e in enumerate(group["frames"]):
  b=(R/e["relative_path"]).read_bytes();assert hashlib.sha256(b).hexdigest()==e["sha256"];stamp=e["stamp_ns"]
  row={"sequence":seq,"stamp_ns":stamp,"quality":False,"agreement":False}
  start=time.perf_counter_ns()
  try:
   j=latest_imu(stamp,mt)
   if j is None:raise ValueError("stale_or_missing_gyro")
   result=estimate_optical(b,T[:,:3],T[:,3],I,gyro[j]);row.update(result)
   k=int(np.searchsorted(radar["stamps"],stamp,side="right"))-1
   if k<0:raise ValueError("no_radar_history")
   age=stamp-int(radar["stamps"][k]);row.update(radar_index=k,radar_age_ns=age,radar_speed=float(radar["speed"][k]))
   row["agreement"]=agree(row.get("radar_speed"),row.get("speed"),age,0,bool(radar["quality"][k]),row["quality"])
  except ValueError as ex:row["error"]=str(ex)
  row["compute_ms"]=(time.perf_counter_ns()-start)*1e-6
  if ix in [0,len(group["frames"])//2,len(group["frames"])-1]:
   xyz,extra,_=oracle["bin_to_pcd"](str(R/e["relative_path"]),"Aeva");pc=decode_optical(b)
   assert np.array_equal(xyz,np.column_stack([pc[k] for k in ["x","y","z"]]))
   assert np.array_equal(extra[:,1],pc["velocity"]);assert np.array_equal(extra[:,2],pc["time_offset"]);row["official_oracle_checked"]=True
  local.append(row);rows.append(row)
 valid=[x for x in local if x["quality"] and x.get("speed") is not None and "radar_speed" in x]
 diff=[abs(x["speed"]-x["radar_speed"]) for x in valid]
 out[seq]={"attempts":len(local),"optical_quality_passed":sum(x["quality"] for x in local),"agreement_passed":sum(x["agreement"] for x in local),"agreement_fraction":sum(x["agreement"] for x in local)/len(local),"difference_p95_mps":float(np.quantile(diff,.95)) if diff else None,"difference_max_mps":float(max(diff)) if diff else None,"compute_p95_ms":float(np.quantile([x["compute_ms"] for x in local],.95)),"admit_full_stream":len(local)>=20 and sum(x["agreement"] for x in local)/len(local)>=.95}
 print(seq,json.dumps(out[seq]),flush=True)
(R/"pilot_predictions.json").write_text(json.dumps(rows,indent=2,allow_nan=False),encoding="utf-8")
(R/"pilot_summary.json").write_text(json.dumps({"sequences":out,"full_stream_trigger":all(s["admit_full_stream"] for s in out.values()),"Library03_payloads_accessed":False,"scope":"sparse source-admission pilot, not continuous availability or final evaluation"},indent=2),encoding="utf-8")
