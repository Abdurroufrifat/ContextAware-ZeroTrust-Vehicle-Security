"""Collect only separate temporal train/calibration pairs; labels loaded after features frozen."""
import hashlib,json,sys,time
from collections import deque
from pathlib import Path
import numpy as np
from quality_core import split_roles,extract_features,FEATURES
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R.parent/"vehicle_lidar_stream"))
sys.path.insert(0,str(R.parent/"vehicle_motion_diagnosis"))
from run_stream_benchmark import BAGS,extrinsic,CAN
from stream_decoder import decode,prepare
from run_diagnosis import fit
from kiss_icp.mapping import VoxelHashMap
from kiss_icp.voxelization import voxel_down_sample
from rosbags.highlevel import AnyReader
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 frozen=json.loads((R/"protocol_freeze.json").read_text())
 assert sha(R/"protocol_v1.json")==frozen["sha256"]
 assert all(sha(R/n)==h for n,h in frozen["implementation_hashes"].items())
 runtime=R/"runtime"/"fixed_pairs_v1";runtime.mkdir(parents=True)
 assert not (R/"pair_summary.json").exists()
 cr=CAN/json.loads((CAN/"experiment_summary.json").read_text())["runtime_relative_path"]
 anchors=split_roles(json.loads((cr/"online_inputs.json").read_text()))
 (R/"split_metadata.json").write_text(json.dumps(anchors,indent=2))
 cal=prepare(json.loads((R.parent/"vehicle_lidar_stream"/"factory_calibration.json").read_text()))
 rows=[]
 with (runtime/"pairs.jsonl").open("w") as log:
  for name,bag in BAGS.items():
   aa=[a for a in anchors if a["recording"]==name];i=0;buffer=deque();mount=extrinsic(name)
   def process(anchor):
    now=anchor["now_us"]*1000;start=time.perf_counter_ns()
    def cloud(lo,hi):
     xx=[];tt=[]
     for receipt,xyz,ts in buffer:
      if receipt>hi:continue
      mask=(ts>=lo)&(ts<=hi)
      if mask.any():xx.append(xyz[mask]);tt.append(ts[mask])
     return (np.vstack(xx),np.concatenate(tt)) if xx else (np.empty((0,3)),np.empty(0,dtype=np.int64))
    previous,pt=cloud(now-200000000,now-100000000);current,ct=cloud(now-20000000,now)
    row=dict(**anchor,valid=False,reason="insufficient_support",features=None,speed_mps=None,
      source_points=0,target_points=0,previous_end_ns=int(pt.max()) if len(pt) else None,
      current_end_ns=int(ct.max()) if len(ct) else None,midpoint_ns=None,pose=None,
      current_first_ns=int(ct.min()) if len(ct) else None,previous_first_ns=int(pt.min()) if len(pt) else None,
      physical_speed_authorized=False,acquisition_age_certified=False)
    if min(len(previous),len(current))>=150:
     source=voxel_down_sample(voxel_down_sample(current,.5),1.5)
     down=voxel_down_sample(previous,.5);vm=VoxelHashMap(1.,100.,20);vm.update(down,np.eye(4))
     pose,fit_ms,target=fit(source,vm.point_cloud(),"temporal_quality")
     dt=(int(ct.max())-int(pt.max()))/1e9
     rear=mount@pose@np.linalg.inv(mount);speed=float(np.linalg.norm(rear[:3,3])/dt)
     row.update(source_points=len(source),target_points=len(target),dt_seconds=dt,pose=pose.tolist(),rear_from_lidar=mount.tolist(),
       speed_mps=speed,midpoint_ns=(int(ct.max())+int(pt.max()))//2,fit_ms=fit_ms)
     try:
      if not .05<=dt<=.2 or not np.isfinite(speed) or speed>60:raise ValueError("implausible_motion")
      features=extract_features(source,target,pose,dt,speed)
      row.update(valid=True,reason="numeric_features",features=features)
     except ValueError as exc:row["reason"]=str(exc)
     # Deterministic audit sample, independent of references and errors.
     same=[r for r in rows if r["recording"]==name]
     if len(same)%50==0 or not any(r["quality_role"]==anchor["quality_role"] for r in same):
      file=f"{name}_{anchor['index']}.npz";np.savez_compressed(runtime/file,source=source,target=target,current_times=ct,previous_times=pt)
      row.update(snapshot_file=file,snapshot_sha256=sha(runtime/file))
    row["diagnostic_processing_ns"]=time.perf_counter_ns()-start
    rows.append(row);log.write(json.dumps(row,allow_nan=False)+"\n");log.flush()
    if len(rows)%100==0:print("pairs",len(rows),"record",name,flush=True)
   with AnyReader([bag]) as reader:
    con=[c for c in reader.connections if c.topic=="/bus/vls128_roof/eth_scan/bus_to_host"]
    for c,receipt,raw in reader.messages(connections=con):
     while i<len(aa) and aa[i]["now_us"]*1000<=receipt:process(aa[i]);i+=1
     if i>=len(aa):break
     # Decode only packets capable of supporting the next allowed anchor.
     now=aa[i]["now_us"]*1000
     if receipt<now-202000000:continue
     msg=reader.deserialize(raw,c.msgtype);header=msg.header.stamp.sec*1000000000+msg.header.stamp.nanosec
     d=decode(msg.payload,header,cal);good=np.isfinite(d["xyz"]).all(axis=1)&(d["distance"]>0)&(d["distance"]<100)
     buffer.append((receipt,d["xyz"][good],d["time_ns"][good]))
     while buffer and buffer[0][0]<receipt-220000000:buffer.popleft()
    while i<len(aa):process(aa[i]);i+=1
   print(name,"all training/calibration pairs saved",flush=True)
 summary=dict(protocol_sha256=frozen["sha256"],runtime_relative_path=str(runtime.relative_to(R)),pairs=len(rows),valid=sum(r["valid"] for r in rows),
  feature_names=FEATURES,quality_roles={k:sum(r["quality_role"]==k for r in rows) for k in ("fit","calibrate","check")},
  evaluation_payloads_decoded=False,reference_values_used=False,production_policy_changed=False,promotion_allowed=False,
  limitation="fixed independent temporal-pair observation model, not the prior continuous packet-map observer; diagnostic timing is not end-to-end freshness")
 (R/"pair_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
 print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
