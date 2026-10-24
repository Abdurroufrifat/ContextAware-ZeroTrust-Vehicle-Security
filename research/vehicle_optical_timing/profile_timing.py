"""Bounded timing diagnosis, no scoring-reference reads or threshold changes."""
from pathlib import Path
import sys,json,time,hashlib,tarfile
import numpy as np
R=Path(__file__).resolve().parent;C=R.parent/"vehicle_crossmodal_doppler";H=R.parent/"vehicle_hercules_development"
sys.path[:0]=[str(C),str(H)]
import crossmodal_core as core
from hercules_core import latest_imu
def tr(p):return np.array([float(x) for x in p.read_text().splitlines()[1].split(":",1)[1].split()]).reshape(3,4)
original=core.fit_velocity;fit_ns=[]
def timed_fit(A,d):
 t=time.perf_counter_ns();v=original(A,d);fit_ns.append(time.perf_counter_ns()-t);return v
core.fit_velocity=timed_fit
out={};rows=[]
for seq in ["Library01","Library02"]:
 D=H/"data"/seq;T=tr(D/"Calibration"/"Continental_LiDAR.txt");I=tr(D/"Calibration"/"IMU_LiDAR.txt")[:,:3]
 mr=[x.split(",") for x in (D/"sensor_data"/"xsens_imu.csv").read_text().splitlines() if x.strip()]
 mt=np.array([int(x[0]) for x in mr],dtype=np.int64);gyro=np.array([[float(v) for v in x[8:11]] for x in mr])
 old=np.load(C/(seq+"_crossmodal_replay.npz"),allow_pickle=False);stamps=old["stamps"];q=old["queries"]
 lookup={int(x):i for i,x in enumerate(stamps)};local=[]
 with tarfile.open(C/(seq+"_Aeva.tar.gz"),"r|gz") as archive:
  for m in archive:
   if m.isdir():continue
   stamp=int(Path(m.name).stem);i=lookup[stamp]
   t0=time.perf_counter_ns();b=archive.extractfile(m).read();t1=time.perf_counter_ns()
   j=latest_imu(stamp,mt);assert j is not None
   result=core.estimate_optical(b,T[:,:3],T[:,3],I,gyro[j]);t2=time.perf_counter_ns()
   assert result["quality"]==old["quality"][i] and abs(result["speed"]-old["speed"][i])<1e-12
   row={"sequence":seq,"frame":i,"stamp_ns":stamp,"archive_read_ns":t1-t0,"estimate_ns":t2-t1,"fit_ns":fit_ns[-1],"total_ns":t2-t0}
   local.append(row);rows.append(row)
   if len(local)==64:break
 zero=np.searchsorted(stamps,q,side="right")-1;safe=np.maximum(zero,0);zero_fresh=(zero>=0)&(q-stamps[safe]<=100_000_000)
 trial=[]
 for service in range(0,25_000_001,250_000):
  done=stamps+service;k=np.searchsorted(done,q,side="right")-1;z=np.maximum(k,0)
  trial.append({"constant_service_ms":service*1e-6,"fresh_fraction":float(np.mean((k>=0)&(q-stamps[z]<=100_000_000)))})
 good=[x["constant_service_ms"] for x in trial if x["fresh_fraction"]>=.95]
 out[seq]={"sample_frames":len(local),"archive_read_p95_ms":float(np.quantile([x["archive_read_ns"] for x in local],.95)*1e-6),"estimate_p95_ms":float(np.quantile([x["estimate_ns"] for x in local],.95)*1e-6),"fit_p95_ms":float(np.quantile([x["fit_ns"] for x in local],.95)*1e-6),"old_service_p95_ms":float(np.quantile(old["compute_ns"],.95)*1e-6),"old_admitted_fraction":float(old["accepted"].mean()),"zero_service_freshness_ceiling":float(zero_fresh.mean()),"largest_grid_constant_service_for95_ms":max(good) if good else None,"constant_service_sensitivity":trial}
 print(seq,json.dumps({k:v for k,v in out[seq].items() if k!="constant_service_sensitivity"}),flush=True)
(R/"profile_rows.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")
(R/"diagnosis.json").write_text(json.dumps({"sequences":out,"reference_labels_used":False,"unchanged_predictions_verified":128,"constant_service_scope":"counterfactual timing feasibility only; not measured performance","Library03_payloads_accessed":False},indent=2),encoding="utf-8")
