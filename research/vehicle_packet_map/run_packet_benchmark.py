"""Causal latest-window packet replay; references loaded after all online outputs."""
import bisect,hashlib,json,os,sys,time
from collections import deque,Counter
from pathlib import Path
import numpy as np
from packet_core import PacketMap,scheduled_start,window_mask
R=Path(__file__).resolve().parent;STREAM=R.parent/"vehicle_lidar_stream"
sys.path.insert(0,str(STREAM))
from run_stream_benchmark import BAGS,extrinsic,CAN
from stream_decoder import decode,prepare
from rosbags.highlevel import AnyReader
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def interpolate(ref,ns):
 if ns is None:return None
 stamps=[int(x[0])*1000 for x in ref];j=bisect.bisect_left(stamps,ns)
 if j<len(stamps) and stamps[j]==ns:return float(ref[j][1])
 if not 0<j<len(stamps) or stamps[j]-stamps[j-1]>50000000:return None
 return float(ref[j-1][1]+(ref[j][1]-ref[j-1][1])*(ns-stamps[j-1])/(stamps[j]-stamps[j-1]))
def main():
 protocol=json.loads((R/"protocol_v1.json").read_text());freeze=json.loads((R/"protocol_freeze.json").read_text())
 assert sha(R/"protocol_v1.json")==freeze["sha256"]
 assert all(sha(R/name)==value for name,value in freeze["implementation_hashes"].items())
 cr=CAN/json.loads((CAN/"experiment_summary.json").read_text())["runtime_relative_path"]
 queries=[dict(recording=r["recording"],index=r["index"],now_ns=r["now_us"]*1000) for r in json.loads((cr/"online_inputs.json").read_text()) if r["role"]=="evaluation"]
 assert len(queries)==283
 runtime=R/"runtime"/("run_"+os.urandom(8).hex());runtime.mkdir(parents=True)
 cal=prepare(json.loads((STREAM/"factory_calibration.json").read_text()))
 rows=[];outputs=[];stats={}
 with (runtime/"jobs.jsonl").open("w") as log:
  for name,bag in BAGS.items():
   assert name in protocol["development_recordings"]
   qq=[q for q in queries if q["recording"]==name];last_query=max(q["now_ns"] for q in qq)
   mount=extrinsic(name);model=PacketMap();buffer=deque();jobs=[];worker=0;due=None;epoch=0;prior_first=None;packet_index=0;decoded_ids=set();dropped=0
   def run_job(at):
    nonlocal worker,due,model,epoch,prior_first
    clock=time.perf_counter_ns()
    reset=bool(model.end_ns is not None and at-model.end_ns>200000000)
    if reset:model=PacketMap();epoch+=1;prior_first=None
    span=100000000 if model.end_ns is None else 20000000
    selected=[p for p in buffer if at-span-2000000<=p[0]<=at]
    xs=[];ts=[];errors=Counter();packet_records=[];future=0
    for receipt,raw,c,index in selected:
     msg=reader.deserialize(raw,c.msgtype)
     header=msg.header.stamp.sec*1000000000+msg.header.stamp.nanosec
     packet_records.append([index,receipt,header,hashlib.sha256(bytes(msg.payload)).hexdigest()])
     decoded_ids.add(index)
     try:
      data=decode(msg.payload,header,cal)
      times=data["time_ns"];mask=window_mask(times,at,span)
      mask &=np.isfinite(data["xyz"]).all(axis=1)&(data["distance"]>0)&(data["distance"]<100)
      future+=int((times>at).sum())
      if mask.any():xs.append(data["xyz"][mask]);ts.append(times[mask])
     except ValueError as exc:errors[str(exc)]+=1
    points=np.vstack(xs) if xs else np.empty((0,3))
    times=np.concatenate(ts) if ts else np.empty(0,dtype=np.int64)
    first=int(times.min()) if len(times) else None;end=int(times.max()) if len(times) else None
    previous_first=prior_first;state_end_before=model.end_ns
    if len(times)>=150 and end>first:
     offsets=(times-first)/1e9
     result=model.step(points,offsets,end,mount)
    else:result=dict(bootstrap=False,quality_passed=False,reason="insufficient_firing_support",motion=None,prior_updates=model.updates)
    elapsed=time.perf_counter_ns()-clock;completed=at+elapsed
    lower=(previous_first+first)//2 if previous_first is not None and first is not None and result["motion"] else None
    row=dict(recording=name,job_index=len(jobs),start_ns=at,worker_before_ns=worker,due_before_ns=due,
     completed_ns=completed,elapsed_ns=elapsed,span_ns=span,epoch=epoch,reset=reset,
     first_firing_ns=first,end_ns=end,previous_first_firing_ns=previous_first,state_end_before_ns=state_end_before,
     lower_support_midpoint_ns=lower,points=len(times),packets=packet_records,
     decode_failures=dict(errors),future_points_excluded=future,rear_from_lidar=mount.tolist(),result=result,
     input_sha256=hashlib.sha256(points.astype("<f8").tobytes()+times.astype("<i8").tobytes()).hexdigest(),
     acquisition_age_certified=False,physical_speed_authorized=False)
    if result["bootstrap"] or result["quality_passed"]:prior_first=first
    # Fixed audit sample chosen without references or error outcomes.
    if len(jobs)<2 or len(jobs)%100==0:
     file=f"{name}_{len(jobs)}.npz"
     arrays=dict(points=points,times=times)
     if "geometry" in result:arrays.update(source=model.last_source,target=model.last_target)
     np.savez_compressed(runtime/file,**arrays);row["snapshot_file"]=file;row["snapshot_sha256"]=sha(runtime/file)
    jobs.append(row);rows.append(row);log.write(json.dumps(row,allow_nan=False)+"\n");log.flush()
    worker=completed;due=at+20000000
    if len(jobs)%100==0:print(name,"jobs",len(jobs),"elapsed_ms",round(elapsed/1e6,2),"quality",sum(j["result"]["quality_passed"] for j in jobs),flush=True)
   with AnyReader([bag]) as reader:
    con=[c for c in reader.connections if c.topic=="/bus/vls128_roof/eth_scan/bus_to_host"]
    assert len(con)==1
    for c,receipt,raw in reader.messages(connections=con):
     if due is None:due=receipt+100000000
     while scheduled_start(due,worker)<=min(receipt,last_query):run_job(scheduled_start(due,worker))
     if receipt>last_query:break
     buffer.append((receipt,raw,c,packet_index));packet_index+=1
     # Bounded raw pending storage; old pending packets are never decoded retroactively.
     while buffer and buffer[0][0]<receipt-220000000:buffer.popleft()
    while due is not None and scheduled_start(due,worker)<=last_query:run_job(scheduled_start(due,worker))
   for q in qq:
    available=[j for j in jobs if j["completed_ns"]<=q["now_ns"]]
    latest=available[-1] if available else None
    result=latest["result"] if latest else None;motion=result["motion"] if result else None
    midpoint=motion["midpoint_ns"] if motion else None
    lower=latest["lower_support_midpoint_ns"] if latest else None
    timing=bool(motion and lower is not None and 0<=q["now_ns"]-midpoint<=100000000 and 0<=q["now_ns"]-lower<=100000000)
    quality=bool(result and result["quality_passed"])
    outputs.append(dict(**q,selected_job=latest["job_index"] if latest else None,
      numeric=bool(motion),quality=quality,timing=timing,joint_eligible=bool(quality and timing),
      speed_mps=motion["speed_mps"] if motion else None,midpoint_ns=midpoint,
      lower_support_midpoint_ns=lower,completed_ns=latest["completed_ns"] if latest else None,
      physical_speed_authorized=False))
   group=[q for q in outputs if q["recording"]==name]
   stats[name]=dict(jobs=len(jobs),packets_received=packet_index,unique_packets_decoded=len(decoded_ids),
    packets_not_decoded=packet_index-len(decoded_ids),queries=len(group),geometry_passed=sum(q["quality"] for q in group),
    joint_eligible=sum(q["joint_eligible"] for q in group),state_resets=sum(j["reset"] for j in jobs),
    processing_p95_ms=float(np.quantile([j["elapsed_ns"] for j in jobs],.95)/1e6))
   print(name,json.dumps(stats[name]),flush=True)
 (runtime/"queries.json").write_text(json.dumps(outputs,indent=2))
 # Offline reference access starts only after every online output is persisted.
 refs=json.loads((cr/"offline_references.json").read_text());scores=[]
 for q in outputs:
  a=interpolate(refs[q["recording"]],q["now_ns"]);b=interpolate(refs[q["recording"]],q["midpoint_ns"])
  scores.append(dict(**q,query_reference=a,midpoint_reference=b,
    query_error_mps=abs(q["speed_mps"]-a) if q["numeric"] and a is not None else None,
    midpoint_error_mps=abs(q["speed_mps"]-b) if q["numeric"] and b is not None else None))
 (runtime/"offline_scores.json").write_text(json.dumps(scores,indent=2))
 metrics={}
 for gate in ("numeric","quality","timing","joint_eligible"):
  group=[q for q in scores if q[gate]];m=dict(count=len(group),queries=283,coverage_pct=100*len(group)/283)
  for k in ("query","midpoint"):
   errors=[q[k+"_error_mps"] for q in group if q[k+"_error_mps"] is not None]
   m[k]=dict(scored=len(errors),unscorable=len(group)-len(errors),p95_error_mps=float(np.quantile(errors,.95)) if errors else None,max_error_mps=max(errors) if errors else None,errors_above_2_mps=sum(e>2 for e in errors))
  metrics[gate]=m
 summary=dict(protocol_sha256=freeze["sha256"],runtime_relative_path=str(runtime.relative_to(R)),records=stats,metrics=metrics,
  development_only=True,variants=1,reference_used_online=False,CAN_values_used_online=False,physical_speed_authorized=False,
  acquisition_age_certified=False,production_policy_changed=False,promotion_allowed=False,policy_nominal_coverage_unchanged_pct=79.15194346289752,
  limitation="geometry screens and encoded time do not certify physical error, authenticity or freshness")
 (R/"benchmark_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
 print("FINAL",json.dumps(summary,indent=2),flush=True)
if __name__=="__main__":main()
