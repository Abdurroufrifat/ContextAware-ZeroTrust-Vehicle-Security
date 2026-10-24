import bisect,hashlib,json,sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R.parent/"vehicle_lidar_stream"))
from run_stream_benchmark import BAGS,CAN
from stream_decoder import decode,prepare
from rosbags.highlevel import AnyReader
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def check_metric(source,target,pose,expected):
 p=source@pose[:3,:3].T+pose[:3,3];d,_=cKDTree(target).query(p);keep=d<=1.5
 q=float(np.percentile(d[keep],95)) if keep.any() else None
 assert expected["inliers"]==int(keep.sum()) and abs(expected["overlap"]-keep.mean())<1e-12
 assert q is not None and abs(q-expected["p95_m"])<1e-10
 assert expected["residual_screen_passed"]==bool(keep.sum()>=100 and keep.mean()>=.5 and q<=.3)
def surface(source,target,pose,expected):
 tree=cKDTree(target);d,ids=tree.query(source@pose[:3,:3].T+pose[:3,3]);_,near=tree.query(target,k=16)
 neighbors=target[near];centers=neighbors-neighbors.mean(axis=1,keepdims=True)
 cov=(centers.transpose(0,2,1)@centers)/16
 eigen,vec=np.linalg.eigh(cov);normals=vec[:,:,0]
 p=source@pose[:3,:3].T+pose[:3,3];keep=d<=1.5;n=normals[ids[keep]]
 normal_error=np.einsum("ij,ij->i",p[keep]-target[ids[keep]],n)
 a=np.concatenate((n,np.cross(p[keep],n)),axis=1);sv=np.linalg.svd(a,compute_uv=False)
 assert expected["jacobian_rank"]==np.linalg.matrix_rank(a)
 np.testing.assert_allclose(expected["jacobian_singular_values"],sv,rtol=1e-6,atol=1e-8)
 for key,values in [("euclidean_p95_m",d[keep]),("plane_p95_m",np.abs(normal_error)),("tangential_p95_m",np.sqrt(np.maximum(0,d[keep]**2-normal_error**2)))]:
  assert abs(expected[key]-np.percentile(values,95))<1e-7
def main():
 s=json.loads((R/"diagnosis_summary.json").read_text());run=R/s["runtime_relative_path"]
 frozen=json.loads((R/"protocol_freeze.json").read_text())
 assert sha(R/"protocol_v1.json")==frozen["sha256"]==s["protocol_sha256"]
 assert all(sha(R/n)==h for n,h in frozen["implementation_hashes"].items())
 sf=json.loads((R/"surface_protocol_freeze.json").read_text())
 assert sf["code_sha256"]==sha(R/"surface_diagnosis.py") and sf["protocol_sha256"]==sha(R/"surface_protocol_v1.json")
 anchors=json.loads((run/"anchor_records.json").read_text());fits=json.loads((run/"known_pose_results.json").read_text());temporal=json.loads((run/"temporal_results_scored.json").read_text())
 cr=CAN/json.loads((CAN/"experiment_summary.json").read_text())["runtime_relative_path"]
 online=json.loads((cr/"online_inputs.json").read_text())
 permitted={}
 for name in BAGS:
  for role in ("train","calibration"):
   a=sorted([x for x in online if x["recording"]==name and x["role"]==role],key=lambda x:x["now_us"])
   for f in (.25,.5,.75):
    x=a[int((len(a)-1)*f)];permitted[name,role,x["index"]]=x["now_us"]*1000
 assert len(anchors)==len(permitted)==18 and len(fits)==108 and len(temporal)==6
 for a in anchors:
  assert permitted[a["recording"],a["role"],a["index"]]==a["now_ns"]
  assert sha(run/a["input_file"])==a["input_sha256"]
  assert all(a["now_ns"]-202000000<=p[0]<=a["now_ns"] for p in a["packets"])
  with np.load(run/a["input_file"]) as z:
   assert len(z["source"])==a["source_points"]
   assert (z["parent_times"]<=a["now_ns"]).all() and (z["parent_times"]>=a["now_ns"]-100000000).all()
   assert (z["current_times"]>=a["now_ns"]-20000000).all()
   assert (z["previous_times"]<=a["now_ns"]-100000000).all() and (z["previous_times"]>=a["now_ns"]-200000000).all()
   np.testing.assert_array_equal(z["current"],z["parent"][z["parent_times"]>=a["now_ns"]-20000000])
   for tag,key in [("raw_parent","parent"),("voxel_half_m","voxel"),("default_map","mapped"),("source_preserving_map","preserving")]:
    check_metric(z["source"],z[key],np.eye(4),a["floor_at_exact_identity"][tag])
   dist,ids=cKDTree(z["mapped"]).query(z["source"]);keep=dist<=1.5
   normal=z["normals"][ids[keep]];errors=np.abs(np.einsum("ij,ij->i",z["source"][keep]-z["mapped"][ids[keep]],normal))
   assert abs(np.percentile(errors,95)-a["floor_at_exact_identity"]["default_map_point_plane"]["p95_m"])<1e-10
 for f in fits:
  assert f["role"] in ("train","calibration") and sha(run/f["snapshot_file"])==f["snapshot_sha256"]
  truth=np.array(f["truth_pose"]);pose=np.array(f["estimated_pose"])
  with np.load(run/f["snapshot_file"]) as z:
   check_metric(z["source"],z["target"],truth,f["at_known_pose"]);check_metric(z["source"],z["target"],pose,f["at_estimated_pose"])
   with np.load(run/anchors[f["anchor"]]["input_file"]) as a:np.testing.assert_allclose(z["source"]@truth[:3,:3].T+truth[:3,3],a["source"],atol=1e-12)
  delta=np.linalg.solve(truth,pose)
  assert abs(np.linalg.norm(delta[:3,3])-f["translation_error_m"])<1e-12
  assert abs(f["translation_error_m"]/.02-f["equivalent_twenty_ms_speed_error_mps"])<1e-12
  assert abs(Rotation.from_matrix(delta[:3,:3]).magnitude()-f["rotation_error_rad"])<1e-12
 refs=json.loads((cr/"offline_references.json").read_text());surfaces=json.loads((R/"surface_results.json").read_text())
 for row,sm in zip(temporal,surfaces):
  assert row["anchor"]==sm["anchor"] and sha(run/row["snapshot_file"])==row["snapshot_sha256"]
  with np.load(run/row["snapshot_file"]) as z:
   check_metric(z["source"],z["target"],np.array(row["estimated_pose"]),row["estimated_residual"])
   surface(z["source"],z["target"],np.array(row["estimated_pose"]),sm["at_fitted_pose"])
   surface(z["source"],z["target"],np.eye(4),sm["at_identity"])
  mount=np.array(row["rear_from_lidar"]);delta=mount@np.array(row["estimated_pose"])@np.linalg.inv(mount)
  assert abs(np.linalg.norm(delta[:3,3])/row["dt_seconds"]-row["speed_mps"])<1e-10
  values=refs[row["recording"]];stamps=[int(v[0])*1000 for v in values];i=bisect.bisect_left(stamps,row["midpoint_ns"])
  assert 0<i<len(stamps) and stamps[i]-stamps[i-1]<=50000000
  expected=values[i-1][1]+(values[i][1]-values[i-1][1])*(row["midpoint_ns"]-stamps[i-1])/(stamps[i]-stamps[i-1])
  assert abs(expected-row["offline_speed_reference_mps"])<1e-10
  assert abs(abs(expected-row["speed_mps"])-row["speed_discrepancy_mps"])<1e-10
 for stage,m in s["identity_floor_by_stage"].items():
  a=[x["floor_at_exact_identity"][stage] for x in anchors]
  assert m["residual_guard_rejections_at_exact_pose"]==sum(not x["residual_screen_passed"] for x in a)
  assert abs(m["median_p95_m"]-np.median([x["p95_m"] for x in a]))<1e-12
 for stage,m in s["known_pose_fits"].items():
  a=[x for x in fits if x["target_stage"]==stage];e=[x["equivalent_twenty_ms_speed_error_mps"] for x in a]
  assert len(a)==m["cases"] and abs(np.percentile(e,95)-m["translation_equivalent_p95_mps"])<1e-12
 # Independently re-read the original training/calibration packet support for every anchor.
 cal=prepare(json.loads((R.parent/"vehicle_lidar_stream"/"factory_calibration.json").read_text()));checked=0
 for name,bag in BAGS.items():
  aa=[a for a in anchors if a["recording"]==name];wanted={tuple(p) for a in aa for p in a["packets"]};receipts={p[0] for p in wanted};got={}
  with AnyReader([bag]) as reader:
   con=[c for c in reader.connections if c.topic=="/bus/vls128_roof/eth_scan/bus_to_host"]
   for c,receipt,raw in reader.messages(connections=con):
    if receipt>max(receipts):break
    if receipt not in receipts:continue
    msg=reader.deserialize(raw,c.msgtype);header=msg.header.stamp.sec*1000000000+msg.header.stamp.nanosec
    identity=(receipt,header,hashlib.sha256(bytes(msg.payload)).hexdigest())
    if identity not in wanted:continue
    d=decode(msg.payload,header,cal);good=np.isfinite(d["xyz"]).all(axis=1)&(d["distance"]>0)&(d["distance"]<100)
    got[identity]=(d["xyz"][good],d["time_ns"][good])
  for a in aa:
   with np.load(run/a["input_file"]) as z:
    for key,lo,hi in [("parent",a["now_ns"]-100000000,a["now_ns"]),("previous",a["now_ns"]-200000000,a["now_ns"]-100000000)]:
     xx=[];tt=[]
     for receipt,header,digest in a["packets"]:
      if receipt>hi:continue
      xyz,times=got[(receipt,header,digest)];mask=(times>=lo)&(times<=hi)
      if mask.any():xx.append(xyz[mask]);tt.append(times[mask])
     np.testing.assert_array_equal(np.vstack(xx),z[key]);np.testing.assert_array_equal(np.concatenate(tt),z[key+"_times"])
   checked+=1
 result=dict(all_passed=True,training_calibration_anchors_checked=18,known_pose_cases_checked=108,temporal_cases_checked=6,surface_cases_checked=len(surfaces),
  source_anchors_redecoded_with_previously_validated_decoder=checked,evaluation_payloads_decoded=False,protected_holdouts_accessed=False,
  physical_certificate_established=False,limitation="diagnostic controls share source observations by design; no observer generalization or physical uncertainty certificate is established")
 (R/"independent_verification.json").write_text(json.dumps(result,indent=2)+"\n")
 print(json.dumps(result,indent=2))
if __name__=="__main__":main()
