from pathlib import Path
import sys,os,json,time,hashlib,importlib.util
os.environ.update(OMP_NUM_THREADS="1",MKL_NUM_THREADS="1")
import numpy as np
R=Path(__file__).resolve().parent;H=R.parent/"vehicle_hercules_development";Q=R.parent/"vehicle_independent_source_qualification";T=R.parent/"vehicle_optical_timing";C=R.parent/"vehicle_crossmodal_doppler"
sys.path.insert(0,str(H))
from hercules_core import load_archive,decode_radar,latest_imu,quaternion_matrix,fit_planar
from evidence_core import metrics,paired_mae_ci
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
names=["PROTOCOL.md","evidence_core.py","test_evidence_core.py","reproduce_baselines.py","run_matched_analysis.py"]
frozen={n:sha(R/n) for n in names};lock=R/"experiment_lock.json"
if lock.exists():assert json.loads(lock.read_text())==frozen
else:lock.write_text(json.dumps(frozen,indent=2),encoding="utf-8")
authors=json.loads((Q/"author_source_lock.json").read_text());path=Q/authors["files"]["src/utils/methods.py"]["local"]
assert sha(path)==authors["files"]["src/utils/methods.py"]["sha256"]
spec=importlib.util.spec_from_file_location("author_native",path);native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
inputs={}
for folder,names_in in [(H,["hercules_core.py","Library01_replay.npz","Library02_replay.npz","development_summary.json","paired_comparison.json","failed_segments.json"]),(T,["Library01_replay.npz","Library02_replay.npz","summary.json","RESEARCH_DECISION.json"]),(C,["crossmodal_core.py","crossmodal_replay.py"]),(R.parent/"vehicle_hercules_reference_diagnosis",["Library01_reference_diagnostics.npz","Library02_reference_diagnostics.npz","reference_diagnosis.json"])]:
 for n in names_in:inputs[str(folder/n)]=sha(folder/n)
for e in json.loads((H/"download_lock.json").read_text())["files"]:
 assert e["sequence"] in ["Library01","Library02"] and sha(H/e["relative_path"])==e["sha256"]
def tr(p):return np.array([float(x) for x in p.read_text().splitlines()[1].split(":",1)[1].split()]).reshape(3,4)[:,:3]
out={}
for seqid,seq in enumerate(["Library01","Library02"]):
 D=H/"data"/seq;a=np.load(H/(seq+"_replay.npz"),allow_pickle=False);ids=a["cohort_indices"]
 payload=load_archive(D/"Radar"/"Continental.tar.gz");mr=[x.split(",") for x in (D/"sensor_data"/"xsens_imu.csv").read_text().splitlines() if x.strip()]
 mt=np.array([int(x[0]) for x in mr],dtype=np.int64);quat=np.array([[float(v) for v in x[1:5]] for x in mr])
 rotation=tr(D/"Calibration"/"IMU_LiDAR.txt")@tr(D/"Calibration"/"Continental_LiDAR.txt").T
 predicted={name:np.full(len(ids),np.nan) for name in ["candidate","author_3d_ransac","ols_3d"]};times={name:np.zeros(len(ids),dtype=np.int64) for name in predicted}
 qmask=np.zeros(len(ids),bool);prep=np.zeros(len(ids),dtype=np.int64)
 for k,i in enumerate(ids):
  stamp=int(a["stamps"][i]);b=payload[stamp];start=time.perf_counter_ns();pc=decode_radar(b)
  xyz=np.column_stack([pc[x] for x in ["x","y","z"]]).astype(float);d=pc["v"].astype(float);r=pc["range"].astype(float)
  valid=np.all(np.isfinite(xyz),axis=1)&np.isfinite(d)&np.isfinite(r)&(r>=3)&(r<=100);xyz=xyz[valid];d=d[valid];norm=np.linalg.norm(xyz,axis=1);valid=norm>0
  A=xyz[valid]/norm[valid,None];d=d[valid];j=latest_imu(stamp,mt);assert j is not None
  rot=quaternion_matrix(quat[j])@rotation;theta=np.arctan2(A[:,1],A[:,0]);phi=np.arctan2(A[:,2],np.hypot(A[:,0],A[:,1]));prep[k]=time.perf_counter_ns()-start
  estimator=native.AlternativeMethod(R=np.eye(3),trials=300)
  funcs={"candidate":lambda:fit_planar(A,d,rot),"author_3d_ransac":lambda:float(estimator.estimate(d,theta,phi)),"ols_3d":lambda:float(np.linalg.norm(np.linalg.lstsq(A,-d,rcond=None)[0]))}
  order=list(funcs);order=order[k%3:]+order[:k%3]
  for name in order:
   start=time.perf_counter_ns();v=funcs[name]();times[name][k]=time.perf_counter_ns()-start
   if name=="candidate":qmask[k]=v["quality"];v=v["speed"]
   predicted[name][k]=v
 np.testing.assert_allclose(predicted["candidate"],a["speed"][ids],atol=1e-10,rtol=0);np.testing.assert_array_equal(qmask,a["quality"][ids])
 np.testing.assert_allclose(predicted["author_3d_ransac"],a["native_prediction"],atol=1e-10,rtol=0)
 np.testing.assert_allclose(predicted["ols_3d"],a["ols_prediction"],atol=1e-10,rtol=0)
 ref=a["reference_primary"][ids];common=np.isfinite(ref)
 for p in predicted.values():common&=np.isfinite(p)
 s={"cohort_attempts":len(ids),"paired_scored":int(common.sum()),"all_attempt_metrics":{name:metrics(p,ref) for name,p in predicted.items()},"common_metrics":{name:metrics(p,ref,common) for name,p in predicted.items()},"fit_only_p95_ms":{name:float(np.quantile(t,.95)*1e-6) for name,t in times.items()},"shared_prep_p95_ms":float(np.quantile(prep,.95)*1e-6),"paired_mae_intervals":{}}
 for name in ["author_3d_ransac","ols_3d"]:s["paired_mae_intervals"][name]=paired_mae_ci(abs(predicted["candidate"][common]-ref[common]),abs(predicted[name][common]-ref[common]),5,1000,20261021+seqid)
 np.savez_compressed(R/(seq+"_baselines.npz"),indices=ids,stamps=a["stamps"][ids],reference=ref,common=common,quality=qmask,prep_ns=prep,**predicted,**{name+"_fit_ns":t for name,t in times.items()})
 out[seq]=s;print(seq,json.dumps(s),flush=True)
(R/"baseline_summary.json").write_text(json.dumps({"sequences":out,"author_commit":authors["commit"],"reproduced":"classical AlternativeMethod comparator, not the deep model","unchanged_input_hashes":inputs},indent=2),encoding="utf-8")
print("BASELINE_REPRODUCTION_FINISHED",flush=True)
