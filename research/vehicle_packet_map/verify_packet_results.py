"""Independent audit of scheduler, accepted-state history, frozen geometry and scores."""
import bisect,hashlib,json,sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R.parent/"vehicle_lidar_stream"))
from run_stream_benchmark import BAGS,extrinsic,CAN
from stream_decoder import decode,prepare
from rosbags.highlevel import AnyReader
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def reference(rows,ns):
 if ns is None:return None
 stamps=[int(x[0])*1000 for x in rows];at=bisect.bisect_left(stamps,ns)
 if at<len(stamps) and stamps[at]==ns:return float(rows[at][1])
 if at==0 or at==len(stamps) or stamps[at]-stamps[at-1]>50000000:return None
 ratio=(ns-stamps[at-1])/(stamps[at]-stamps[at-1])
 return float((1-ratio)*rows[at-1][1]+ratio*rows[at][1])
def main():
 summary=json.loads((R/"benchmark_summary.json").read_text());run=R/summary["runtime_relative_path"]
 protocol=json.loads((R/"protocol_v1.json").read_text());freeze=json.loads((R/"protocol_freeze.json").read_text())
 assert summary["protocol_sha256"]==freeze["sha256"]==sha(R/"protocol_v1.json")
 assert all(sha(R/n)==v for n,v in freeze["implementation_hashes"].items())
 jobs=[json.loads(x) for x in (run/"jobs.jsonl").read_text().splitlines()]
 queries=json.loads((run/"queries.json").read_text());scores=json.loads((run/"offline_scores.json").read_text())
 assert len(queries)==len(scores)==283
 geometry_checked=0;source_checked=0;groups={};snapshot_count=0
 cal=prepare(json.loads((R.parent/"vehicle_lidar_stream"/"factory_calibration.json").read_text()))
 for name,bag in BAGS.items():
  assert name in protocol["development_recordings"]
  group=[j for j in jobs if j["recording"]==name];groups[name]=group
  mount=extrinsic(name);last=None;state_end=None;state_first=None;state_pose=np.eye(4);updates=0;epoch=0
  for i,j in enumerate(group):
   assert j["job_index"]==i
   assert j["start_ns"]==max(j["due_before_ns"],j["worker_before_ns"])
   assert j["completed_ns"]==j["start_ns"]+j["elapsed_ns"]
   assert j["worker_before_ns"]==(last["completed_ns"] if last else 0)
   if last:assert j["due_before_ns"]==last["start_ns"]+20000000
   reset=bool(state_end is not None and j["start_ns"]-state_end>200000000)
   assert reset==j["reset"]
   if reset:state_end=None;state_first=None;state_pose=np.eye(4);updates=0;epoch+=1
   assert j["epoch"]==epoch and j["state_end_before_ns"]==state_end and j["previous_first_firing_ns"]==state_first
   assert j["span_ns"]==(100000000 if state_end is None else 20000000)
   packetids=[]
   for index,receipt,header,digest in j["packets"]:
    assert j["start_ns"]-j["span_ns"]-2000000<=receipt<=j["start_ns"]
    assert len(digest)==64;packetids.append(index)
   assert packetids==sorted(set(packetids))
   r=j["result"];assert r["prior_updates"]==updates
   if j["points"]:
    assert j["start_ns"]-j["span_ns"]<=j["first_firing_ns"]<=j["end_ns"]<=j["start_ns"]
   if "geometry" in r:
    assert r["previous_end_ns"]==state_end
    np.testing.assert_allclose(r["previous_pose"],state_pose,atol=1e-12)
    g=r["geometry"]
    screen=bool(g["correspondences"]>=100 and g["overlap"]>=.5 and g["residual_p95_m"] is not None and g["residual_p95_m"]<=.3 and g["condition"] is not None and g["condition"]<=1000)
    assert g["passed"]==screen
    assert r["quality_passed"]==bool(screen and r["motion"] is not None)
   if r["motion"]:
    delta=mount@np.linalg.solve(state_pose,np.array(r["pose"]))@np.linalg.inv(mount)
    dt=(j["end_ns"]-state_end)/1e9
    assert .005<=dt<=.2 and abs(np.linalg.norm(delta[:3,3])/dt-r["motion"]["speed_mps"])<1e-8
    assert r["motion"]["midpoint_ns"]==(j["end_ns"]+state_end)//2
    assert j["lower_support_midpoint_ns"]==(state_first+j["first_firing_ns"])//2
   if r["bootstrap"] or r["quality_passed"]:
    state_end=j["end_ns"];state_first=j["first_firing_ns"];state_pose=np.array(r["pose"]);updates+=1
   if "snapshot_file" in j:
    path=run/j["snapshot_file"];assert sha(path)==j["snapshot_sha256"]
    with np.load(path) as a:
     points=a["points"];times=a["times"]
     assert len(points)==len(times)==j["points"]
     assert hashlib.sha256(points.astype("<f8").tobytes()+times.astype("<i8").tobytes()).hexdigest()==j["input_sha256"]
     assert np.isfinite(points).all() and (times<=j["start_ns"]).all() and (times>=j["start_ns"]-j["span_ns"]).all()
     if "source" in a:
      source=a["source"];target=a["target"];pose=np.array(r["pose"])
      moved=source@pose[:3,:3].T+pose[:3,3];distance,index=cKDTree(target).query(moved)
      keep=distance<=1.5;g=r["geometry"]
      assert int(keep.sum())==g["correspondences"] and abs(keep.mean()-g["overlap"])<1e-12
      if keep.sum()>=100:
       assert abs(np.percentile(distance[keep],95)-g["residual_p95_m"])<1e-10
       a6=[]
       for x,y,z in moved[keep]:a6.extend([[1,0,0,0,z,-y],[0,1,0,-z,0,x],[0,0,1,y,-x,0]])
       singular=np.linalg.svd(np.array(a6),compute_uv=False)
       cond=float(singular[0]/singular[-1]) if singular[-1]>1e-12 else None
       if cond is None:assert g["condition"] is None
       else:assert abs(cond-g["condition"])<1e-7
      geometry_checked+=1
    snapshot_count+=1
   last=j
  snapshots=[j for j in group if "snapshot_file" in j]
  sample=[snapshots[k] for k in sorted({0,len(snapshots)//2,len(snapshots)-1})]
  wanted={p[0]:p for j in sample for p in j["packets"]};decoded={}
  with AnyReader([bag]) as reader:
   con=[c for c in reader.connections if c.topic=="/bus/vls128_roof/eth_scan/bus_to_host"]
   for index,(c,receipt,raw) in enumerate(reader.messages(connections=con)):
    if index>max(wanted):break
    if index not in wanted:continue
    msg=reader.deserialize(raw,c.msgtype);header=msg.header.stamp.sec*1000000000+msg.header.stamp.nanosec
    item=wanted[index];assert receipt==item[1] and header==item[2] and hashlib.sha256(bytes(msg.payload)).hexdigest()==item[3]
    try:decoded[index]=decode(msg.payload,header,cal)
    except ValueError:decoded[index]=None
  for j in sample:
   xs=[];tt=[]
   for p in j["packets"]:
    data=decoded[p[0]]
    if data is None:continue
    times=data["time_ns"];mask=(times<=j["start_ns"])&(times>=j["start_ns"]-j["span_ns"])&np.isfinite(data["xyz"]).all(axis=1)&(data["distance"]>0)&(data["distance"]<100)
    if mask.any():xs.append(data["xyz"][mask]);tt.append(times[mask])
   points=np.vstack(xs) if xs else np.empty((0,3));times=np.concatenate(tt) if tt else np.empty(0,dtype=np.int64)
   assert hashlib.sha256(points.astype("<f8").tobytes()+times.astype("<i8").tobytes()).hexdigest()==j["input_sha256"]
   source_checked+=1
 for q,s in zip(queries,scores):
  for k in q:assert s[k]==q[k]
  available=[j for j in groups[q["recording"]] if j["completed_ns"]<=q["now_ns"]]
  last=available[-1] if available else None
  assert q["selected_job"]==(last["job_index"] if last else None)
  motion=last["result"]["motion"] if last else None
  assert q["numeric"]==bool(motion)
  quality=bool(last and last["result"]["quality_passed"])
  timing=bool(motion and last["lower_support_midpoint_ns"] is not None and 0<=q["now_ns"]-motion["midpoint_ns"]<=100000000 and 0<=q["now_ns"]-last["lower_support_midpoint_ns"]<=100000000)
  assert q["quality"]==quality and q["timing"]==timing and q["joint_eligible"]==bool(quality and timing)
 cr=CAN/json.loads((CAN/"experiment_summary.json").read_text())["runtime_relative_path"]
 refs=json.loads((cr/"offline_references.json").read_text())
 for q in scores:
  for kind,key in [("query","now_ns"),("midpoint","midpoint_ns")]:
   value=reference(refs[q["recording"]],q[key])
   actual=q[kind+"_reference"]
   assert (value is None and actual is None) or abs(value-actual)<1e-10
   err=abs(q["speed_mps"]-value) if value is not None and q["numeric"] else None
   actual=q[kind+"_error_mps"]
   assert (err is None and actual is None) or abs(err-actual)<1e-10
 for gate,m in summary["metrics"].items():
  group=[q for q in scores if q[gate]]
  assert len(group)==m["count"] and abs(100*len(group)/283-m["coverage_pct"])<1e-10
  for kind in ("query","midpoint"):
   err=[q[kind+"_error_mps"] for q in group if q[kind+"_error_mps"] is not None];a=m[kind]
   assert a["scored"]==len(err) and a["unscorable"]==len(group)-len(err)
   assert a["errors_above_2_mps"]==sum(x>2 for x in err)
   assert (a["p95_error_mps"] is None and not err) or abs(a["p95_error_mps"]-np.percentile(err,95))<1e-10
 result=dict(all_passed=True,jobs_checked=len(jobs),queries_checked=283,offline_scores_checked=283,
  snapshots_checked=snapshot_count,geometry_snapshots_independently_checked=geometry_checked,source_windows_redecoded_with_validated_decoder=source_checked,
  protected_holdouts_accessed=False,physical_certificate_established=False,
  limitation="all metadata/state/arithmetic and scores checked; input sources sampled, not every raw payload; geometry is local fixed-correspondence screen")
 (R/"independent_verification.json").write_text(json.dumps(result,indent=2)+"\n")
 print(json.dumps(result,indent=2))
if __name__=="__main__":main()
