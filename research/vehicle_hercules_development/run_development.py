"""Frozen two-sequence development replay; no Library03/protected payload access."""
from pathlib import Path
import os,sys,json,hashlib,tarfile,struct,ast,time,importlib.util
os.environ.update(OMP_NUM_THREADS="1",MKL_NUM_THREADS="1")
import numpy as np
from hercules_core import decode_radar,quaternion_matrix,latest_imu,fit_planar,load_archive
R=Path(__file__).resolve().parent
OLD=R.parent/"vehicle_independent_source_qualification"
lock=json.loads((OLD/"author_source_lock.json").read_text())
entry=lock["files"]["src/utils/methods.py"];mp=OLD/entry["local"]
assert hashlib.sha256(mp.read_bytes()).hexdigest()==entry["sha256"]
spec=importlib.util.spec_from_file_location("native_methods",mp);native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
download=json.loads((R/"download_lock.json").read_text())
for e in download["files"]:
 assert e["sequence"] in ["Library01","Library02"] and "sha256" in e
 assert hashlib.sha256((R/e["relative_path"]).read_bytes()).hexdigest()==e["sha256"]
# Independent struct-record oracle from official toolbox, without Open3D side effects.
source=R/"official_source"/"HeRCULES_Pointcloud_Toolbox"/"bin2pcd.py"
tree=ast.parse(source.read_text());func=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="bin_to_pcd")
oracle={"np":np,"struct":struct};exec(compile(ast.Module(body=[func],type_ignores=[]),str(source),"exec"),oracle)
def load_csv(path):
 rows=[l.split(",") for l in path.read_text().splitlines() if l.strip()]
 return np.array([int(x[0]) for x in rows],dtype=np.int64),rows
def transform(path):
 s=path.read_text().splitlines()[1].split(":",1)[1]
 return np.array([float(x) for x in s.split()]).reshape(3,4)
def metrics(pred,ref):
 p=np.asarray(pred,dtype=float);y=np.asarray(ref,dtype=float)
 valid=np.isfinite(p)&np.isfinite(y);a=abs(p[valid]-y[valid])
 return {"predictions":int(np.isfinite(p).sum()),"scored":int(valid.sum()),"mae_mps":float(np.mean(a)) if len(a) else None,"p95_mps":float(np.quantile(a,.95)) if len(a) else None,"max_mps":float(np.max(a)) if len(a) else None,"above_2_mps":int(sum(a>2))}
def interpolate(query,times,values):
 q=np.asarray(query,dtype=np.int64);t=np.asarray(times,dtype=np.int64);y=np.asarray(values,dtype=float)
 result=np.full(len(q),np.nan);j=np.searchsorted(t,q)
 exact=(j<len(t))&(q==t[np.minimum(j,len(t)-1)])
 result[exact]=y[j[exact]]
 ok=(j>0)&(j<len(t))&~exact
 idx=np.flatnonzero(ok);lo=j[idx]-1;hi=j[idx]
 good=(t[hi]-t[lo]<=200_000_000)&np.isfinite(y[lo])&np.isfinite(y[hi])
 idx=idx[good];lo=lo[good];hi=hi[good]
 result[idx]=y[lo]+(y[hi]-y[lo])*(q[idx]-t[lo])/(t[hi]-t[lo])
 return result
summaries={};oracle_checks=[]
for seq in ["Library01","Library02"]:
 D=R/"data"/seq
 stamps=np.array([int(x) for x in (D/"sensor_data"/"continental_stamp.csv").read_text().splitlines()],dtype=np.int64)
 assert np.all(np.diff(stamps)>0)
 imu_t,imu_rows=load_csv(D/"sensor_data"/"xsens_imu.csv")
 imu_q=np.array([[float(x) for x in r[1:5]] for r in imu_rows]);assert np.all(np.diff(imu_t)>0)
 ins_t,ins_rows=load_csv(D/"sensor_data"/"inspva.csv")
 ins_v=np.array([[float(x) for x in r[4:7]] for r in ins_rows])
 ins_speed=np.linalg.norm(ins_v,axis=1);status=np.array([r[10].strip()=="status: 3" for r in ins_rows]);ins_speed[~status]=np.nan
 gt_rows=[l.split() for l in (D/"PR_GT"/"Continental_gt.txt").read_text().splitlines()]
 gt_t=np.array([int(r[0]) for r in gt_rows],dtype=np.int64);gt_pos=np.array([[float(x) for x in r[1:4]] for r in gt_rows])
 assert np.array_equal(stamps,gt_t)
 gt_speed=np.full(len(gt_t),np.nan);span=gt_t[2:]-gt_t[:-2]
 good=(span>0)&(span<=200_000_000)
 gt_speed[1:-1][good]=np.linalg.norm(gt_pos[2:]-gt_pos[:-2],axis=1)[good]/(span[good]*1e-9)
 radar_from_lidar=transform(D/"Calibration"/"Continental_LiDAR.txt")[:,:3]
 imu_from_lidar=transform(D/"Calibration"/"IMU_LiDAR.txt")[:,:3]
 imu_from_radar=imu_from_lidar@radar_from_lidar.T
 speeds=np.full(len(stamps),np.nan);quality=np.zeros(len(stamps),dtype=bool);compute=np.zeros(len(stamps),dtype=np.int64)
 fractions=np.full(len(stamps),np.nan);conditions=fractions.copy();imu_age=np.full(len(stamps),-1,dtype=np.int64)
 errors=[];cohort=[];points_total=0;geometric_errors=[]
 payloads=load_archive(D/"Radar"/"Continental.tar.gz")
 with tarfile.open(D/"Radar"/"Continental.tar.gz","r:gz") as archive:
  files={}
  for member in archive.getmembers():
   if member.isfile():
    name=Path(member.name)
    assert not name.is_absolute() and ".." not in name.parts and name.suffix==".bin" and member.size<1024*1024
    stamp=int(name.stem);assert stamp not in files;files[stamp]=member
  assert set(files)==set(map(int,stamps))
  for i,stamp in enumerate(stamps):
   b=payloads[int(stamp)]
   tic=time.perf_counter_ns()
   dirs=np.empty((0,3));vr=np.empty(0)
   try:
    pc=decode_radar(b);xyz=np.column_stack([pc[x] for x in ["x","y","z"]]).astype(float)
    vr=pc["v"].astype(float);r=pc["range"].astype(float)
    mask=np.all(np.isfinite(xyz),axis=1)&np.isfinite(vr)&np.isfinite(r)&(r>=3)&(r<=100)
    xyz=xyz[mask];vr=vr[mask];norm=np.linalg.norm(xyz,axis=1)
    valid=norm>0;dirs=xyz[valid]/norm[valid,None];vr=vr[valid];points_total+=len(pc)
    j=latest_imu(int(stamp),imu_t)
    if j is None:raise ValueError("IMU unavailable or stale")
    imu_age[i]=int(stamp)-int(imu_t[j])
    rotation=quaternion_matrix(imu_q[j])@imu_from_radar
    result=fit_planar(dirs,vr,rotation);speeds[i]=result["speed"] if result["speed"] is not None else np.nan
    quality[i]=result["quality"];fractions[i]=result.get("inlier_fraction",np.nan);conditions[i]=result.get("condition",np.nan)
   except Exception as ex:errors.append({"index":i,"error":str(ex)})
   compute[i]=time.perf_counter_ns()-tic
   if i%20==0:
    cohort.append((i,dirs.copy(),vr.copy()))
   if i in [0,len(stamps)//2,len(stamps)-1]:
    temp=R/"runtime"/(seq+"_oracle.bin");temp.parent.mkdir(exist_ok=True);temp.write_bytes(b)
    op,extra,_=oracle["bin_to_pcd"](str(temp),"Continental")
    assert np.array_equal(op,np.column_stack([pc[x] for x in ["x","y","z"]]))
    assert np.array_equal(extra[:,0],pc["v"]);oracle_checks.append({"sequence":seq,"frame":i,"points":len(pc)})
    goodxyz=np.column_stack([pc[x] for x in ["x","y","z"]]).astype(float)
    angles=np.column_stack([np.cos(pc["elevation"])*np.cos(pc["azimuth"]),np.cos(pc["elevation"])*np.sin(pc["azimuth"]),np.sin(pc["elevation"])])
    geometric_errors.append(float(np.max(np.linalg.norm(goodxyz-pc["range"][:,None]*angles,axis=1))))
   if i%1000==0:print(seq,"candidate frames",i,"/",len(stamps),flush=True)
 # Comparator computation is outside measured candidate jobs.
 native_pred=[];ols_pred=[];indices=[];native_failures=[]
 for i,dirs,vr in cohort:
  indices.append(i)
  ols_pred.append(float(np.linalg.norm(np.linalg.lstsq(dirs,-vr,rcond=None)[0])) if len(vr)>=3 else np.nan)
  try:
   theta=np.arctan2(dirs[:,1],dirs[:,0]);phi=np.arctan2(dirs[:,2],np.hypot(dirs[:,0],dirs[:,1]))
   estimator=native.AlternativeMethod(R=np.eye(3),trials=300)
   native_pred.append(float(estimator.estimate(vr,theta,phi)))
  except Exception as ex:native_pred.append(np.nan);native_failures.append({"frame":i,"error":str(ex)})
 indices=np.array(indices)
 nearest=np.searchsorted(ins_t,stamps);nearest=np.clip(nearest,0,len(ins_t)-1)
 prior=np.maximum(nearest-1,0);use_prior=abs(ins_t[prior]-stamps)<=abs(ins_t[nearest]-stamps);nearest[use_prior]=prior[use_prior]
 ins_match=ins_speed[nearest].copy();ins_match[abs(ins_t[nearest]-stamps)>20_000_000]=np.nan
 # Simulate serial measured service time on nominal publisher stamp arrivals.
 completion=np.empty(len(stamps),dtype=np.int64)
 for i,stamp in enumerate(stamps):completion[i]=max(int(stamp),int(completion[i-1]) if i else int(stamp))+int(compute[i])
 queries=np.arange(((int(stamps[0])+49_999_999)//50_000_000)*50_000_000,(int(stamps[-1])//50_000_000)*50_000_000+1,50_000_000,dtype=np.int64)
 latest=np.searchsorted(completion,queries,side="right")-1;safe=np.maximum(latest,0)
 age=queries-stamps[safe];ready=(latest>=0)&quality[safe]&(age>=0)&(age<=100_000_000)
 request_pred=np.where(ready,speeds[safe],np.nan);request_ref=interpolate(queries,stamps,gt_speed)
 req_metrics=metrics(request_pred,request_ref);segments=[]
 for start in range(0,len(queries),1200):
  sl=slice(start,min(start+1200,len(queries)));mtr=metrics(request_pred[sl],request_ref[sl])
  segments.append({"start_request":start,"requests":len(queries[sl]),"ready_fraction":float(np.mean(ready[sl])),**mtr})
 seq_summary={"frames":len(stamps),"duration_seconds":float((stamps[-1]-stamps[0])*1e-9),"points":points_total,"quality_fraction":float(np.mean(quality)),"candidate_all_frames_primary":metrics(np.where(quality,speeds,np.nan),gt_speed),"candidate_all_frames_INS_secondary":metrics(np.where(quality,speeds,np.nan),ins_match),"common_cohort":{"frames":len(indices),"candidate_planar_huber":metrics(np.where(quality[indices],speeds[indices],np.nan),gt_speed[indices]),"author_3d_ransac":metrics(native_pred,gt_speed[indices]),"ols_3d":metrics(ols_pred,gt_speed[indices])},"requests":{"count":len(queries),"ready":int(ready.sum()),"ready_fraction":float(np.mean(ready)),"missing_reference_while_ready":int(sum(ready&~np.isfinite(request_ref))),**req_metrics,"segments_60s":segments},"timing":{"candidate_compute_p95_ms":float(np.quantile(compute,.95)*1e-6),"maximum_radar_gap_ms":float(np.max(np.diff(stamps))*1e-6),"raw_stamp_clock_and_arrival_certified":False},"parser_geometry_max_errors_m":geometric_errors,"candidate_failures":errors,"native_failures":native_failures,"physical_witness_admitted":False}
 seq_summary["numeric_request_gates"]={"availability":seq_summary["requests"]["ready_fraction"]>=.95,"p95":req_metrics["p95_mps"] is not None and req_metrics["p95_mps"]<=.5,"zero_recorded_above_2":req_metrics["above_2_mps"]==0 and req_metrics["scored"]>0,"all_ready_reference_scored":seq_summary["requests"]["missing_reference_while_ready"]==0}
 np.savez_compressed(R/(seq+"_replay.npz"),stamps=stamps,speed=speeds,quality=quality,compute_ns=compute,completion_ns=completion,imu_age_ns=imu_age,inlier_fraction=fractions,condition=conditions,reference_primary=gt_speed,reference_ins=ins_match,cohort_indices=indices,native_prediction=native_pred,ols_prediction=ols_pred,queries=queries,latest=latest,age_ns=age,ready=ready,request_prediction=request_pred,request_reference=request_ref)
 summaries[seq]=seq_summary
 (R/"development_summary.json").write_text(json.dumps({"sequences":summaries,"oracle_checks":oracle_checks,"Library03_payloads_accessed":False,"production_changed":False},indent=2,allow_nan=False),encoding="utf-8")
 print(seq,json.dumps(seq_summary),flush=True)
print("DEVELOPMENT_RUN_FINISHED",flush=True)
