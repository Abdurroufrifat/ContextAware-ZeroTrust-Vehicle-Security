"""Frozen full development replay: optical predictor never reads evaluation labels."""
from pathlib import Path
import sys,os,json,hashlib,time,tarfile,ast,struct
os.environ.update(OMP_NUM_THREADS="1",MKL_NUM_THREADS="1")
import numpy as np
R=Path(__file__).resolve().parent;H=R.parent/"vehicle_hercules_development";sys.path.insert(0,str(H))
from hercules_core import latest_imu
from crossmodal_core import estimate_optical,decode_optical
from crossmodal_replay import latest_completed,make_cases,metrics,interpolate
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(8*1024**2),b""):h.update(b)
 return h.hexdigest()
names=["EXPERIMENT_PROTOCOL.md","crossmodal_core.py","test_crossmodal_core.py","crossmodal_replay.py","test_crossmodal_replay.py","run_full_stream.py","full_data_lock.json"]
frozen={n:sha(R/n) for n in names};lock=R/"full_experiment_lock.json"
if lock.exists():assert json.loads(lock.read_text())==frozen
else:lock.write_text(json.dumps(frozen,indent=2),encoding="utf-8")
assert all(sha(R/n)==v for n,v in json.loads((R/"experiment_lock.json").read_text()).items())
downloads=json.loads((R/"full_data_lock.json").read_text())
source=H/"official_source"/"HeRCULES_Pointcloud_Toolbox"/"bin2pcd.py"
tree=ast.parse(source.read_text());func=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="bin_to_pcd")
oracle={"np":np,"struct":struct};exec(compile(ast.Module(body=[func],type_ignores=[]),str(source),"exec"),oracle)
runtime=R/"runtime";runtime.mkdir(exist_ok=True)
def tr(p):return np.array([float(x) for x in p.read_text().splitlines()[1].split(":",1)[1].split()]).reshape(3,4)
out={};proofs=[];membership={}
for seq in ["Library01","Library02"]:
 D=H/"data"/seq;entry=downloads[seq];archive_path=R/entry["file"]
 assert archive_path.stat().st_size==entry["bytes"] and sha(archive_path)==entry["sha256"]
 T=tr(D/"Calibration"/"Continental_LiDAR.txt");I=tr(D/"Calibration"/"IMU_LiDAR.txt")[:,:3]
 mr=[x.split(",") for x in (D/"sensor_data"/"xsens_imu.csv").read_text().splitlines() if x.strip()]
 mt=np.array([int(x[0]) for x in mr],dtype=np.int64);gyro=np.array([[float(v) for v in x[8:11]] for x in mr])
 stamps=np.array([int(x) for x in (R/(seq+"_aeva_stamp.csv")).read_text().splitlines() if x.strip()],dtype=np.int64)
 assert np.all(np.diff(stamps)>0);lookup={int(s):i for i,s in enumerate(stamps)}
 n=len(stamps);speeds=np.full(n,np.nan);quality=np.zeros(n,bool);compute=np.zeros(n,dtype=np.int64)
 fractions=np.full(n,np.nan);conditions=fractions.copy();imu_age=np.full(n,-1,dtype=np.int64)
 seen=set();members=[];errors=[];points=0;oracle_checks=[]
 with tarfile.open(archive_path,"r|gz") as archive:
  for member in archive:
   if member.isdir():continue
   name=Path(member.name)
   assert member.isfile() and not name.is_absolute() and ".." not in name.parts and name.suffix==".bin" and member.size<=32*1024**2
   stamp=int(name.stem);assert stamp in lookup and stamp not in seen;seen.add(stamp);i=lookup[stamp]
   tic=time.perf_counter_ns();b=archive.extractfile(member).read();assert len(b)==member.size
   try:
    j=latest_imu(stamp,mt)
    if j is None:raise ValueError("missing_or_stale_gyro")
    imu_age[i]=stamp-int(mt[j]);result=estimate_optical(b,T[:,:3],T[:,3],I,gyro[j])
    speeds[i]=result["speed"] if result["speed"] is not None else np.nan;quality[i]=result["quality"]
    fractions[i]=result.get("inlier_fraction",np.nan);conditions[i]=result.get("condition",np.nan)
   except ValueError as ex:errors.append({"frame":i,"reason":str(ex)})
   compute[i]=time.perf_counter_ns()-tic
   members.append({"stamp_ns":stamp,"bytes":len(b),"sha256":hashlib.sha256(b).hexdigest()});points+=len(b)//29
   if i in [0,n//2,n-1]:
    pc=decode_optical(b);temp=runtime/(seq+"_optical_oracle.bin");temp.write_bytes(b)
    xyz,extra,_=oracle["bin_to_pcd"](str(temp),"Aeva")
    assert np.array_equal(xyz,np.column_stack([pc[k] for k in ["x","y","z"]]))
    assert np.array_equal(extra[:,1],pc["velocity"]) and np.array_equal(extra[:,2],pc["time_offset"])
    oracle_checks.append({"frame":i,"points":len(pc)})
    if j is not None and quality[i]:
     # Encode coherent Doppler forgery into every finite nonzero-range raw record.
     vr=np.array(result["radar_origin_velocity"]);delta=T[:,:3].T@(3*vr/np.linalg.norm(vr))
     forged=pc.copy();xyz=np.column_stack([pc[k] for k in ["x","y","z"]]).astype(float);length=np.linalg.norm(xyz,axis=1)
     good=np.all(np.isfinite(xyz),axis=1)&np.isfinite(pc["velocity"])&(length>0)
     forged["velocity"][good]=pc["velocity"][good]-(xyz[good]/length[good,None])@delta
     attack=estimate_optical(forged.tobytes(),T[:,:3],T[:,3],I,gyro[j])
     proofs.append({"sequence":seq,"frame":i,"nominal_quality":bool(quality[i]),"attack_quality":attack["quality"],"nominal_speed":float(speeds[i]),"attack_speed":attack["speed"],"speed_shift":attack["speed"]-float(speeds[i]),"encoded_float32_records":int(good.sum())})
   if len(seen)%500==0:print(seq,"processed",len(seen),"/",n,flush=True)
 assert seen==set(map(int,stamps));membership[seq]=members
 completion=np.empty(n,dtype=np.int64)
 for i,s in enumerate(stamps):completion[i]=max(int(s),int(completion[i-1]) if i else int(s))+int(compute[i])
 # The following stage consumes saved predictor outputs only. References are scoring-only.
 radar=np.load(H/(seq+"_replay.npz"),allow_pickle=False);q=radar["queries"]
 k=latest_completed(completion,q);safe=np.maximum(k,0);oa=q-stamps[safe]
 optical=speeds[safe];optical_ready=(k>=0)&quality[safe]&(oa>=0)&(oa<=100_000_000)
 rk=radar["latest"];rsafe=np.maximum(rk,0);rv=radar["speed"][rsafe];ra=radar["age_ns"]
 pair=optical_ready&radar["ready"]
 cases=make_cases(q,rv,optical,pair,ra,oa);accepted=cases["nominal"]["accepted"];prediction=cases["nominal"]["prediction"]
 primary=radar["request_reference"];secondary=interpolate(q,radar["stamps"],radar["reference_ins"])
 summary={"optical_frames":n,"points":points,"optical_quality":int(quality.sum()),"optical_quality_fraction":float(quality.mean()),"requests":len(q),"paired_fresh_quality":int(pair.sum()),"optical_fresh_quality":int(optical_ready.sum()),"accepted":int(accepted.sum()),"availability":float(accepted.mean()),"primary":metrics(prediction,primary),"secondary_INS":metrics(prediction,secondary),"compute_p95_ms":float(np.quantile(compute,.95)*1e-6),"compute_max_ms":float(max(compute)*1e-6),"oracle_checks":oracle_checks,"errors":errors,"segments":[],"cases":{}}
 for begin in range(0,len(q),1200):
  sl=slice(begin,min(begin+1200,len(q)));m=metrics(prediction[sl],primary[sl]);a=float(accepted[sl].mean())
  summary["segments"].append({"start":begin,"requests":len(q[sl]),"availability":a,"primary":m,"numeric_pass":bool(a>=.95 and m["scored"]>0 and m["p95_mps"]<=.5 and m["above2"]==0),"complete_reference_support":m["unknown_reference"]==0})
 attack_arrays={}
 for case,c in cases.items():
  summary["cases"][case]={"supported":int(c["supported"].sum()),"unsupported":int((~c["supported"]).sum()),"accepted":int(c["accepted"].sum()),"acceptance_on_supported":float(c["accepted"].sum()/c["supported"].sum()) if c["supported"].any() else None,"primary":metrics(c["prediction"],primary),"secondary_INS":metrics(c["prediction"],secondary)}
  for field in ["supported","accepted","prediction"]:attack_arrays[case+"__"+field]=c[field]
 summary["numeric_pass"]=bool(summary["availability"]>=.95 and summary["primary"]["scored"]>0 and summary["primary"]["p95_mps"]<=.5 and summary["primary"]["above2"]==0)
 np.savez_compressed(R/(seq+"_crossmodal_replay.npz"),stamps=stamps,speed=speeds,quality=quality,compute_ns=compute,completion_ns=completion,inlier_fraction=fractions,condition=conditions,imu_age_ns=imu_age,queries=q,optical_index=k,optical_age_ns=oa,optical_query_speed=optical,radar_query_speed=rv,radar_age_ns=ra,pair=pair,accepted=accepted,prediction=prediction,primary=primary,secondary=secondary,**attack_arrays)
 out[seq]=summary;(R/"full_summary.json").write_text(json.dumps({"sequences":out,"physical_witness_admitted":False,"Library03_payloads_accessed":False,"timing_scope":"measured binary read/decode/fit on nominal dataset stamp arrivals; excludes certified sensor publication/network stack"},indent=2,allow_nan=False),encoding="utf-8")
 print(seq,json.dumps({k:summary[k] for k in ["optical_frames","optical_quality_fraction","availability","primary","compute_p95_ms","numeric_pass"]}),flush=True)
(R/"optical_member_lock.json").write_text(json.dumps(membership,indent=2),encoding="utf-8")
(R/"raw_optical_forgery_proofs.json").write_text(json.dumps(proofs,indent=2),encoding="utf-8")
print("FULL_REPLAY_FINISHED",flush=True)
