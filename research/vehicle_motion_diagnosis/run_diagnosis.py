"""Controlled training/calibration diagnosis; no estimator or policy promotion."""
import bisect,hashlib,json,sys,time
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from controls import select_anchors,warp_to_sensor,discrepancy
R=Path(__file__).resolve().parent;STREAM=R.parent/"vehicle_lidar_stream";PACKET=R.parent/"vehicle_packet_map"
sys.path.insert(0,str(STREAM));sys.path.insert(0,str(PACKET))
from run_stream_benchmark import BAGS,extrinsic,CAN
from stream_decoder import decode,prepare
from packet_core import CONFIG
from kiss_icp.config import KISSConfig
from kiss_icp.kiss_icp import KissICP
from kiss_icp.mapping import VoxelHashMap
from kiss_icp.voxelization import voxel_down_sample
from rosbags.highlevel import AnyReader
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def normals(target):
 _,ids=cKDTree(target).query(target,k=min(16,len(target)))
 a=target[ids];a=a-a.mean(axis=1,keepdims=True)
 values,vectors=np.linalg.eigh(np.einsum("nki,nkj->nij",a,a)/a.shape[1])
 return vectors[:,:,0]
def plane(source,target,n):
 distance,index=cKDTree(target).query(source)
 valid=distance<=1.5
 residual=np.abs(np.sum((source[valid]-target[index[valid]])*n[index[valid]],axis=1))
 return dict(inliers=int(valid.sum()),p95_m=float(np.quantile(residual,.95)) if len(residual) else None)
def fit(source,target,tag):
 model=KissICP(KISSConfig(**CONFIG))
 # Default map contents are passed explicitly; source is frozen before known transforms.
 model.local_map.add_points(target)
 start=time.perf_counter_ns()
 pose=model.registration.align_points_to_map(points=source,voxel_map=model.local_map,initial_guess=np.eye(4),max_correspondance_distance=6.,kernel=2.)
 return pose,(time.perf_counter_ns()-start)/1e6,model.local_map.point_cloud()
def main():
 protocol=json.loads((R/"protocol_v1.json").read_text());freeze=json.loads((R/"protocol_freeze.json").read_text())
 assert sha(R/"protocol_v1.json")==freeze["sha256"]
 assert all(sha(R/n)==h for n,h in freeze["implementation_hashes"].items())
 runtime=R/"runtime"/"fixed_controls_v1";runtime.mkdir(parents=True)
 assert not (R/"diagnosis_summary.json").exists()
 cr=CAN/json.loads((CAN/"experiment_summary.json").read_text())["runtime_relative_path"]
 metadata=json.loads((cr/"online_inputs.json").read_text())
 anchors=select_anchors(metadata);assert len(anchors)==18 and all(a["role"] in ("train","calibration") for a in anchors)
 (R/"anchors.json").write_text(json.dumps(anchors,indent=2))
 cal=prepare(json.loads((STREAM/"factory_calibration.json").read_text()))
 decoded=defaultdict(list);source_audits=[]
 for name,bag in BAGS.items():
  aa=[a for a in anchors if a["recording"]==name];last=max(a["now_us"]*1000 for a in aa)
  with AnyReader([bag]) as reader:
   con=[c for c in reader.connections if c.topic=="/bus/vls128_roof/eth_scan/bus_to_host"]
   for c,receipt,raw in reader.messages(connections=con):
    if receipt>last:break
    relevant=[a for a in aa if a["now_us"]*1000-202000000<=receipt<=a["now_us"]*1000]
    if not relevant:continue
    msg=reader.deserialize(raw,c.msgtype);header=msg.header.stamp.sec*1000000000+msg.header.stamp.nanosec
    data=decode(msg.payload,header,cal)
    valid=np.isfinite(data["xyz"]).all(axis=1)&(data["distance"]>0)&(data["distance"]<100)
    item=(receipt,header,data["xyz"][valid],data["time_ns"][valid],hashlib.sha256(bytes(msg.payload)).hexdigest())
    for a in relevant:decoded[name,a["role"],a["index"]].append(item)
  print(name,"training/calibration source collected",flush=True)
 rows=[];anchor_records=[];temporal=[]
 for ai,a in enumerate(anchors):
  key=(a["recording"],a["role"],a["index"]);packets=decoded[key];now=a["now_us"]*1000
  def cloud(lo,hi):
   xx=[];tt=[]
   for receipt,header,xyz,times,h in packets:
    if receipt>hi:continue
    mask=(times>=lo)&(times<=hi)
    if mask.any():xx.append(xyz[mask]);tt.append(times[mask])
   return np.vstack(xx),np.concatenate(tt)
  parent,pt=cloud(now-100000000,now);previous,pr=cloud(now-200000000,now-100000000)
  keep=pt>=now-20000000;current=parent[keep];ct=pt[keep]
  assert len(current)>150 and len(parent)>150 and len(previous)>150
  # These arrays are exact subsets in the same measured sensor coordinates.
  source=voxel_down_sample(voxel_down_sample(current,.5),1.5)
  down=voxel_down_sample(parent,.5)
  vm=VoxelHashMap(1.,100.,20);vm.update(down,np.eye(4));mapped=vm.point_cloud()
  vm2=VoxelHashMap(1.,100.,20);vm2.update(source,np.eye(4));preserving=vm2.point_cloud()
  n=normals(mapped)
  floor={tag:discrepancy(source,target,np.eye(4)) for tag,target in [("raw_parent",parent),("voxel_half_m",down),("default_map",mapped),("source_preserving_map",preserving)]}
  floor["default_map_point_plane"]=plane(source,mapped,n)
  file=f"anchor_{ai:02d}.npz";np.savez_compressed(runtime/file,parent=parent,parent_times=pt,current=current,current_times=ct,previous=previous,previous_times=pr,source=source,voxel=down,mapped=mapped,preserving=preserving,normals=n)
  rec=dict(anchor=ai,**a,now_ns=now,source_points=len(source),parent_points=len(parent),current_points=len(current),previous_points=len(previous),floor_at_exact_identity=floor,input_file=file,input_sha256=sha(runtime/file),packets=[[x[0],x[1],x[4]] for x in packets])
  anchor_records.append(rec)
  for shift,yaw in [(0.,0.),(.04,0.),(.16,.005)]:
   truth=np.eye(4);truth[:3,:3]=Rotation.from_euler("z",yaw).as_matrix();truth[0,3]=shift
   moved=warp_to_sensor(source,truth)
   for tag,target in [("default_map",mapped),("source_preserving_map",preserving)]:
    pose,elapsed,actual=fit(moved,target,tag)
    # Re-insertion can discard map points; preserve actual fitting targets explicitly.
    snapshot=f"fit_{len(rows):03d}.npz";np.savez_compressed(runtime/snapshot,source=moved,target=actual)
    delta=np.linalg.inv(truth)@pose
    rows.append(dict(anchor=ai,recording=a["recording"],role=a["role"],case="same_parent_known_pose",diagnostic_only=True,current_points_in_target=True,
      target_stage=tag,truth_pose=truth.tolist(),estimated_pose=pose.tolist(),
      translation_error_m=float(np.linalg.norm(delta[:3,3])),rotation_error_rad=float(Rotation.from_matrix(delta[:3,:3]).magnitude()),
      equivalent_twenty_ms_speed_error_mps=float(np.linalg.norm(delta[:3,3])/.02),
      at_known_pose=discrepancy(moved,actual,truth),at_estimated_pose=discrepancy(moved,actual,pose),
      fit_ms=elapsed,snapshot_file=snapshot,snapshot_sha256=sha(runtime/snapshot)))
  if a["recording"]=="04_crossTraffic":
   prevdown=voxel_down_sample(previous,.5);vmp=VoxelHashMap(1.,100.,20);vmp.update(prevdown,np.eye(4))
   pose,elapsed,target=fit(source,vmp.point_cloud(),"temporal")
   snapshot=f"temporal_{ai:02d}.npz";np.savez_compressed(runtime/snapshot,source=source,target=target)
   dt=(int(ct.max())-int(pr.max()))/1e9;mount=extrinsic(a["recording"])
   rear=mount@pose@np.linalg.inv(mount)
   temporal.append(dict(anchor=ai,recording=a["recording"],role=a["role"],case="earlier_scan_stationary_control",current_points_in_target=False,
    estimated_pose=pose.tolist(),dt_seconds=dt,midpoint_ns=(int(ct.max())+int(pr.max()))//2,
    speed_mps=float(np.linalg.norm(rear[:3,3])/dt),rear_from_lidar=mount.tolist(),
    identity_residual=discrepancy(source,target,np.eye(4)),estimated_residual=discrepancy(source,target,pose),
    snapshot_file=snapshot,snapshot_sha256=sha(runtime/snapshot),fit_ms=elapsed))
  print("anchor",ai,a["recording"],a["role"],"true-pose map P95",floor["default_map"]["p95_m"],flush=True)
 (runtime/"anchor_records.json").write_text(json.dumps(anchor_records,indent=2))
 (runtime/"known_pose_results.json").write_text(json.dumps(rows,indent=2))
 (runtime/"temporal_results_unscored.json").write_text(json.dumps(temporal,indent=2))
 # References used only after all controlled fits and input snapshots are frozen.
 refs=json.loads((cr/"offline_references.json").read_text())
 for row in temporal:
  ref=refs[row["recording"]];stamps=[int(r[0])*1000 for r in ref];j=bisect.bisect_left(stamps,row["midpoint_ns"]);value=None
  if 0<j<len(stamps) and stamps[j]-stamps[j-1]<=50000000:
   value=ref[j-1][1]+(ref[j][1]-ref[j-1][1])*(row["midpoint_ns"]-stamps[j-1])/(stamps[j]-stamps[j-1])
  row["offline_speed_reference_mps"]=value
  row["speed_discrepancy_mps"]=abs(row["speed_mps"]-value) if value is not None else None
 (runtime/"temporal_results_scored.json").write_text(json.dumps(temporal,indent=2))
 by_stage={}
 for stage in ("raw_parent","voxel_half_m","default_map","source_preserving_map"):
  q=[a["floor_at_exact_identity"][stage] for a in anchor_records]
  by_stage[stage]=dict(unique_anchors=len(q),residual_guard_rejections_at_exact_pose=sum(not x["residual_screen_passed"] for x in q),
    median_p95_m=float(np.median([x["p95_m"] for x in q])),maximum_p95_m=max(x["p95_m"] for x in q))
 fits={}
 for stage in ("default_map","source_preserving_map"):
  q=[x for x in rows if x["target_stage"]==stage];errors=[x["equivalent_twenty_ms_speed_error_mps"] for x in q]
  fits[stage]=dict(cases=len(q),unique_anchors=18,translation_equivalent_p95_mps=float(np.quantile(errors,.95)),
   translation_equivalent_maximum_mps=max(errors),translation_equivalent_above_2_mps=sum(e>2 for e in errors),
   guard_rejections_at_exact_known_pose=sum(not x["at_known_pose"]["residual_screen_passed"] for x in q))
 summary=dict(protocol_sha256=freeze["sha256"],runtime_relative_path=str(runtime.relative_to(R)),
  roles=["train","calibration"],unique_anchors=18,known_pose_cases=len(rows),temporal_stationary_cases=len(temporal),
  identity_floor_by_stage=by_stage,known_pose_fits=fits,
  normal_residual_median_p95_m=float(np.median([a["floor_at_exact_identity"]["default_map_point_plane"]["p95_m"] for a in anchor_records])),
  diagnostic_only=True,physical_speed_authorized=False,production_policy_changed=False,promotion_allowed=False,
  evaluation_inputs_decoded=False,final_holdouts_accessed=False,
  limitation="same-parent controls deliberately share observations and cannot validate online independence, runtime freshness, generalization or a physical error certificate; no uncertainty calibration is claimed")
 (R/"diagnosis_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
 print("FINAL",json.dumps(summary,indent=2),flush=True)
if __name__=="__main__":main()
