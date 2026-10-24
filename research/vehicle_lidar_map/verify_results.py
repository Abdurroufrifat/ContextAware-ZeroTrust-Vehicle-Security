"""Independent arithmetic, causality, source-sample and aggregate verification."""
import bisect,hashlib,json,sys
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R.parent/"vehicle_lidar_stream"))
from run_stream_benchmark import BAGS,extrinsic,CAN
from rosbags.highlevel import AnyReader
def main():
 summary=json.loads((R/"benchmark_summary.json").read_text());run=R/summary["runtime_relative_path"]
 frames=[json.loads(s) for s in (run/"frames.jsonl").read_text().splitlines()]
 queries=json.loads((run/"queries.json").read_text());scores=json.loads((run/"offline_scores.json").read_text())
 assert len(queries)==len(scores)==283
 ph=hashlib.sha256((R/"protocol_v1.json").read_bytes()).hexdigest()
 assert ph==summary["protocol_sha256"]==json.loads((R/"protocol_freeze.json").read_text())["sha256"]
 checked=0;sampled=0;groups={}
 for name,bag in BAGS.items():
  group=[f for f in frames if f["recording"]==name];groups[name]=group
  mount=extrinsic(name);prior=None
  for i,f in enumerate(group):
   assert f["scan_index"]==f["prior_scan_count"]==i
   assert f["first_firing_ns"]<=f["end_ns"]<=f["receipt_ns"]<=f["completed_ns"]
   assert f["completed_ns"]==max(f["receipt_ns"],f["worker_before_ns"])+f["elapsed_ns"]
   assert f["worker_before_ns"]==(prior["completed_ns"] if prior else 0)
   pose=np.array(f["pose"]);np.testing.assert_allclose(pose[3],[0,0,0,1],atol=1e-8)
   np.testing.assert_allclose(pose[:3,:3].T@pose[:3,:3],np.eye(3),atol=1e-6)
   if prior:
    np.testing.assert_array_equal(f["previous_pose"],prior["pose"])
    assert f["previous_end_ns"]==prior["end_ns"]
    if f["motion"]:
     delta=mount@np.linalg.solve(np.array(prior["pose"]),pose)@np.linalg.inv(mount)
     dt=(f["end_ns"]-prior["end_ns"])/1e9
     assert .05<=dt<=.2
     assert abs(np.linalg.norm(delta[:3,3])/dt-f["motion"]["speed_mps"])<1e-8
     assert f["motion"]["midpoint_ns"]==(f["end_ns"]+prior["end_ns"])//2
   else:assert f["motion"] is None
   prior=f;checked+=1
  targets={0,len(group)//2,len(group)-1}
  with AnyReader([bag]) as reader:
   con=[c for c in reader.connections if c.topic=="/sensor/lidar/vls128_roof/velodyne_points"]
   for i,(c,receipt,raw) in enumerate(reader.messages(connections=con)):
    if i>max(targets):break
    if i not in targets:continue
    msg=reader.deserialize(raw,c.msgtype);fs={f.name:f for f in msg.fields}
    a={k:np.ndarray((msg.height,msg.width),dtype=(">" if msg.is_bigendian else "<")+"f4",buffer=msg.data,offset=fs[k].offset,strides=(msg.row_step,msg.point_step)).reshape(-1).astype(float) for k in ("x","y","z","time","distance")}
    header=msg.header.stamp.sec*1000000000+msg.header.stamp.nanosec
    valid=np.logical_and.reduce([np.isfinite(a[k]) for k in a])&(a["distance"]>0)&(a["time"]>=-.001)&(a["time"]<=.2)&(a["time"]<=(receipt-header)/1e9)
    xyz=np.stack([a[k][valid] for k in ("x","y","z")],axis=1)
    times=np.rint(a["time"][valid]*1e9).astype(np.int64)+header
    digest=hashlib.sha256(xyz.astype("<f8").tobytes()+times.astype("<i8").tobytes()).hexdigest()
    f=group[i]
    assert f["receipt_ns"]==receipt and f["input_sha256"]==digest
    assert f["end_ns"]==int(times.max()) and f["first_firing_ns"]==int(times.min()) and f["points"]==len(times)
    sampled+=1
 for q in queries:
  eligible_frames=[f for f in groups[q["recording"]] if f["receipt_ns"]<=q["now_ns"] and f["completed_ns"]<=q["now_ns"]]
  latest=eligible_frames[-1] if eligible_frames else None
  assert q["selected_scan"]==(latest["scan_index"] if latest else None)
  motion=latest["motion"] if latest else None
  assert q["available_numeric"]==bool(motion)
  assert q["timing_eligible"]==bool(motion and 0<=q["now_ns"]-motion["midpoint_ns"]<=100000000)
  if motion:assert q["speed_mps"]==motion["speed_mps"]
 cr=CAN/json.loads((CAN/"experiment_summary.json").read_text())["runtime_relative_path"]
 refs=json.loads((cr/"offline_references.json").read_text())
 for q,s in zip(queries,scores):
  for k in q:assert q[k]==s[k]
  a=refs[q["recording"]];tt=[int(v[0])*1000 for v in a]
  for kind,key in [("query","now_ns"),("midpoint","midpoint_ns")]:
   t=q[key];value=None
   if t is not None:
    j=bisect.bisect_left(tt,t)
    if j<len(tt) and tt[j]==t:value=a[j][1]
    elif 0<j<len(tt) and tt[j]-tt[j-1]<=50000000:
     value=a[j-1][1]+(a[j][1]-a[j-1][1])*(t-tt[j-1])/(tt[j]-tt[j-1])
   expected=abs(q["speed_mps"]-value) if value is not None and q["speed_mps"] is not None else None
   actual=s[kind+"_error_mps"]
   assert (expected is None and actual is None) or (expected is not None and actual is not None and abs(expected-actual)<1e-8)
 for gate,m in summary["metrics"].items():
  group=[s for s in scores if s[gate]]
  assert len(group)==m["count"] and abs(m["coverage_pct"]-100*len(group)/283)<1e-10
  for k in ("query","midpoint"):
   e=[s[k+"_error_mps"] for s in group if s[k+"_error_mps"] is not None];got=m[k]
   assert got["scored"]==len(e) and got["errors_above_2_mps"]==sum(x>2 for x in e)
   assert (got["p95_error_mps"] is None and not e) or abs(got["p95_error_mps"]-np.percentile(e,95))<1e-10
 result=dict(all_passed=True,frames_checked=checked,queries_checked=len(queries),offline_rows_checked=len(scores),source_clouds_independently_checked=sampled,protected_holdouts_accessed=False,limitations="source hashes checked on 9 sampled clouds; all pose/time/queue arithmetic and aggregate scores checked; no physical timing certificate")
 (R/"independent_verification.json").write_text(json.dumps(result,indent=2)+"\n")
 print(json.dumps(result,indent=2))
if __name__=="__main__":main()
