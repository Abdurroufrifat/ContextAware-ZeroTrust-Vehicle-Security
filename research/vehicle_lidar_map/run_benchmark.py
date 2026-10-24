"""Fixed full-scene scan-to-map benchmark. Online estimation is isolated from references."""
import bisect,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
from kiss_icp.config import KISSConfig
from kiss_icp.kiss_icp import KissICP
from map_core import interval_motion,eligible,select_available,firing_mask
R=Path(__file__).resolve().parent
S=R.parent/"vehicle_lidar_stream";sys.path.insert(0,str(S))
from run_stream_benchmark import BAGS,extrinsic,CAN
from rosbags.highlevel import AnyReader
def extract(msg,receipt):
 h=msg.header.stamp;header=int(h.sec)*1000000000+int(h.nanosec)
 if not 0<=int(h.nanosec)<1000000000: raise ValueError("header")
 if msg.width<1 or msg.height<1 or msg.row_step<msg.width*msg.point_step or len(msg.data)<msg.row_step*msg.height: raise ValueError("layout")
 fields={}
 for f in msg.fields:
  if f.name in fields: raise ValueError("duplicate")
  fields[f.name]=f
 out={}
 for name in ("x","y","z","time","distance"):
  f=fields[name]
  if f.datatype!=7 or f.count!=1 or f.offset<0 or f.offset+4>msg.point_step: raise ValueError("field")
  out[name]=np.ndarray((msg.height,msg.width),dtype=(">" if msg.is_bigendian else "<")+"f4",buffer=msg.data,offset=f.offset,strides=(msg.row_step,msg.point_step)).reshape(-1).astype(float)
 good=np.isfinite(out["distance"])&(out["distance"]>0)
 for name in ("x","y","z","time"):good &=np.isfinite(out[name])
 offsets=out["time"]
 if np.any((offsets[good]<-.001)|(offsets[good]>.2)): raise ValueError("offset model")
 keep=good&firing_mask(offsets,header,receipt)
 xyz=np.column_stack([out[k][keep] for k in ("x","y","z")])
 times=np.rint(offsets[keep]*1e9).astype(np.int64)+header
 if len(times)<100 or times.max()==times.min(): raise ValueError("insufficient firing support")
 fingerprint=hashlib.sha256(xyz.astype("<f8").tobytes()+times.astype("<i8").tobytes()).hexdigest()
 return xyz,offsets[keep],dict(header_ns=header,first_firing_ns=int(times.min()),end_ns=int(times.max()),points=len(times),future_points_excluded=int((good&~keep).sum()),input_sha256=fingerprint)
def interp(ref,ns):
 stamps=[int(v)*1000 for v in ref[:,0]]
 at=bisect.bisect_left(stamps,ns)
 if at<len(stamps) and stamps[at]==ns:return float(ref[at,1])
 if not 0<at<len(stamps):return None
 gap=stamps[at]-stamps[at-1]
 if gap>50000000:return None
 return float(ref[at-1,1]+(ref[at,1]-ref[at-1,1])*(ns-stamps[at-1])/gap)
def main():
 protocol=json.loads((R/"protocol_v1.json").read_text())
 ph=hashlib.sha256((R/"protocol_v1.json").read_bytes()).hexdigest()
 assert ph==json.loads((R/"protocol_freeze.json").read_text())["sha256"]
 cr=CAN/json.loads((CAN/"experiment_summary.json").read_text())["runtime_relative_path"]
 queries=[dict(recording=r["recording"],index=r["index"],now_ns=r["now_us"]*1000) for r in json.loads((cr/"online_inputs.json").read_text()) if r["role"]=="evaluation"]
 assert len(queries)==283
 runtime=R/"runtime"/("run_"+os.urandom(8).hex());runtime.mkdir()
 (R/"active_run.json").write_text(json.dumps(dict(runtime=str(runtime))))
 allframes=[];queryrows=[];record_results={}
 with (runtime/"frames.jsonl").open("w") as f:
  for name,bag in BAGS.items():
   assert name in protocol["development_recordings"]
   model=KissICP(KISSConfig(**protocol["configuration"]))
   mount=extrinsic(name);worker=0;previous=None;frames=[]
   qq=[q for q in queries if q["recording"]==name];last_query=max(q["now_ns"] for q in qq)
   with AnyReader([bag]) as reader:
    con=[c for c in reader.connections if c.topic=="/sensor/lidar/vls128_roof/velodyne_points"]
    assert len(con)==1
    for c,receipt,raw in reader.messages(connections=con):
     if receipt>last_query:break
     start=time.perf_counter_ns()
     msg=reader.deserialize(raw,c.msgtype)
     xyz,offsets,info=extract(msg,receipt)
     prior_pose=model.last_pose.copy()
     prior_end=previous["end_ns"] if previous else None
     model.register_frame(xyz,offsets)
     pose=model.last_pose.copy()
     reason="bootstrap";motion=None
     if previous:
      try:motion=interval_motion(prior_pose,pose,prior_end,info["end_ns"],mount);reason="numeric_output"
      except ValueError as exc:reason=str(exc)
     elapsed=time.perf_counter_ns()-start
     completed=max(receipt,worker)+elapsed
     row=dict(recording=name,scan_index=len(frames),receipt_ns=receipt,worker_before_ns=worker,
       elapsed_ns=elapsed,completed_ns=completed,queue_wait_ns=max(0,worker-receipt),
       previous_end_ns=prior_end,previous_pose=prior_pose.tolist(),pose=pose.tolist(),
       rear_from_lidar=mount.tolist(),motion=motion,reason=reason,**info,
       prior_scan_count=len(frames),acquisition_age_certified=False,physical_speed_authorized=False)
     worker=completed;previous=row;frames.append(row);allframes.append(row)
     f.write(json.dumps(row,allow_nan=False)+"\n");f.flush()
     if len(frames)%50==0: print(name,"clouds",len(frames),"processing_ms",round(elapsed/1e6,2),flush=True)
   for q in qq:
    row=select_available(frames,q["now_ns"])
    motion=row["motion"] if row else None
    timed=bool(motion and eligible(motion["midpoint_ns"],row["completed_ns"],q["now_ns"]))
    queryrows.append(dict(**q,selected_scan=row["scan_index"] if row else None,
      speed_mps=motion["speed_mps"] if motion else None,midpoint_ns=motion["midpoint_ns"] if motion else None,
      completed_ns=row["completed_ns"] if row else None,receipt_ns=row["receipt_ns"] if row else None,
      available_numeric=bool(motion),timing_eligible=timed,
      inferred_midpoint_age_ms=(q["now_ns"]-motion["midpoint_ns"])/1e6 if motion else None,
      original_header_age_ms=(q["now_ns"]-row["header_ns"])/1e6 if row else None,
      physical_speed_authorized=False))
   record_results[name]=dict(clouds=len(frames),queries=len(qq),numeric=sum(q["available_numeric"] for q in queryrows if q["recording"]==name),
      timing_eligible=sum(q["timing_eligible"] for q in queryrows if q["recording"]==name),
      processing_p95_ms=float(np.quantile([r["elapsed_ns"] for r in frames],.95)/1e6),
      queue_wait_max_ms=max(r["queue_wait_ns"] for r in frames)/1e6)
   print(name,json.dumps(record_results[name]),flush=True)
 (runtime/"queries.json").write_text(json.dumps(queryrows,indent=2))
 # No references are loaded until all online results have been persisted.
 refs={k:np.asarray(v,dtype=float) for k,v in json.loads((cr/"offline_references.json").read_text()).items()}
 scores=[]
 for q in queryrows:
  a=interp(refs[q["recording"]],q["now_ns"])
  b=interp(refs[q["recording"]],q["midpoint_ns"]) if q["midpoint_ns"] is not None else None
  scores.append(dict(**q,query_reference=a,midpoint_reference=b,
   query_error_mps=abs(q["speed_mps"]-a) if a is not None and q["speed_mps"] is not None else None,
   midpoint_error_mps=abs(q["speed_mps"]-b) if b is not None and q["speed_mps"] is not None else None))
 (runtime/"offline_scores.json").write_text(json.dumps(scores,indent=2))
 metrics={}
 for gate in ("available_numeric","timing_eligible"):
  group=[q for q in scores if q[gate]]
  m=dict(count=len(group),queries=len(scores),coverage_pct=100*len(group)/len(scores))
  for point in ("query","midpoint"):
   errors=[q[point+"_error_mps"] for q in group if q[point+"_error_mps"] is not None]
   m[point]=dict(scored=len(errors),unscorable=len(group)-len(errors),
     p95_error_mps=float(np.quantile(errors,.95)) if errors else None,
     max_error_mps=max(errors) if errors else None,errors_above_2_mps=sum(e>2 for e in errors))
  metrics[gate]=m
 result=dict(protocol_sha256=ph,runtime_relative_path=str(runtime.relative_to(R)),records=record_results,metrics=metrics,
  development_only=True,configuration_variants=1,reference_used_online=False,CAN_values_used_online=False,
  acquisition_age_certified=False,physical_speed_authorized=False,production_policy_changed=False,promotion_allowed=False,
  policy_nominal_coverage_unchanged_pct=79.15194346289752,
  limitation="numeric timing screen is not a geometric quality or physical freshness certificate")
 (R/"benchmark_summary.json").write_text(json.dumps(result,indent=2)+"\n")
 print("FINAL",json.dumps(result,indent=2),flush=True)
if __name__=="__main__":main()
