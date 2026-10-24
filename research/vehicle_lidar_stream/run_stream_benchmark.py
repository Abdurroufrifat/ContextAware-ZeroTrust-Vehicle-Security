"""Causal CPU-queue replay of the frozen LiDAR-only interval motion prototype."""
import hashlib,json,math,os,sys,time
from pathlib import Path
from collections import Counter,deque
import numpy as np
from scipy.spatial.transform import Rotation
from stream_decoder import decode,prepare
from motion import downsample,estimate
ROOT=Path(__file__).resolve().parent;SCAN=ROOT.parent/"vehicle_scanario_reference";RADAR=ROOT.parent/"vehicle_radar_context";CAN=ROOT.parent/"vehicle_context_policy"
sys.path.append(str(SCAN/".venv"/"Lib"/"site-packages"))
from rosbags.highlevel import AnyReader
BAGS={"07_turning":SCAN/"raw_data"/"development_07_turning.bag",
    "04_crossTraffic":SCAN/"raw_data"/"04_crossTraffic"/"development.bag",
    "14_countryRoad":SCAN/"raw_data"/"14_countryRoad"/"development.bag"}
def extrinsic(name):
    transforms=json.loads((RADAR/"runtime"/name/"transforms.json").read_text())
    nodes={x["child_frame_id"].lstrip("/"):x for x in transforms}
    def pose(frame):
        if frame=="base_link": return np.eye(4)
        t=nodes[frame];q=t["transform"]["rotation"];p=t["transform"]["translation"]
        mat=np.eye(4);mat[:3,:3]=Rotation.from_quat([q[k] for k in ("x","y","z","w")]).as_matrix()
        mat[:3,3]=[p[k] for k in ("x","y","z")]
        return pose(t["header"]["frame_id"].lstrip("/"))@mat
    return np.linalg.inv(pose("vehicle/rear_axis"))@pose("sensor/lidar/vls128_roof")
def main():
    protocol=json.loads((ROOT/"protocol_v1.json").read_text())
    ph=hashlib.sha256((ROOT/"protocol_v1.json").read_bytes()).hexdigest()
    assert ph==json.loads((ROOT/"protocol_freeze.json").read_text())["sha256"]
    assert all(x["all_passed"] for x in json.loads((ROOT/"decoder_validation.json").read_text()).values())
    notes=dict(frozen_before_motion_outcomes=True,CPU_model="single worker: each packet deserialization/decoding and each query preprocessing/registration is serialized; bag file I/O is excluded",
        snapshot="only packets whose actual receipt and simulated decode completion precede the fixed query",
        timestamp="encoded bridge-epoch firing times; constant-motion interval estimate, not instantaneous attestation",
        previous_observations_are_history_not_fresh_acquisitions=True,convergence_translation_m=.0001,convergence_yaw_rad=.00001,
        offline_INS_not_loaded_until_all_motion_estimates_saved=True)
    npth=ROOT/"implementation_notes_v1.json"
    if npth.exists(): assert json.loads(npth.read_text())==notes
    else: npth.write_text(json.dumps(notes,indent=2)+"\n")
    cr=CAN/json.loads((CAN/"experiment_summary.json").read_text())["runtime_relative_path"]
    online=[r for r in json.loads((cr/"online_inputs.json").read_text()) if r["role"]=="evaluation"]
    # Whitelist query identifiers/times only: no candidate or CAN context enters estimation.
    queries=[dict(recording=r["recording"],index=r["index"],now_ns=r["now_us"]*1000) for r in online]
    runtime=ROOT/"runtime"/("run_"+os.urandom(8).hex());runtime.mkdir()
    cal=prepare(json.loads((ROOT/"factory_calibration.json").read_text()))
    outcomes=[];records={}
    with (runtime/"estimates.jsonl").open("w") as output:
        for name,bag in BAGS.items():
            assert not any(x in str(bag).lower() for x in ("urban02","rav4","vehicle_final_evaluation"))
            qq=[q for q in queries if q["recording"]==name];qi=0;buffer=deque();worker=0
            packet_count=0;failures=Counter();decode_ns=[];queue_lag=[];mount=extrinsic(name)
            def process(q,worker):
                start=time.perf_counter_ns();now=q["now_ns"]
                current=[];previous=[];ct=[];pt=[];latest_available=0
                for item in buffer:
                    if item["available_ns"]>now or item["receipt_ns"]>now: continue
                    xyz=item["xyz"];times=item["time_ns"]
                    cm=(times<=now)&(times>=now-20000000)
                    pm=(times<=now-100000000)&(times>=now-120000000)
                    if cm.any(): current.append(xyz[cm]);ct.append(times[cm]);latest_available=max(latest_available,item["available_ns"])
                    if pm.any(): previous.append(xyz[pm]);pt.append(times[pm]);latest_available=max(latest_available,item["available_ns"])
                cp=np.vstack(current) if current else np.empty((0,3));pp=np.vstack(previous) if previous else np.empty((0,3))
                ctime=np.concatenate(ct) if ct else np.empty(0,dtype="int64")
                ptime=np.concatenate(pt) if pt else np.empty(0,dtype="int64")
                cp,ctime=downsample(cp,ctime);pp,ptime=downsample(pp,ptime)
                measured=lower=dt=None
                if len(ctime) and len(ptime):
                    ccenter=now+round(float(np.median(ctime-now)));pcenter=now+round(float(np.median(ptime-now)))
                    measured=(ccenter+pcenter)//2;lower=(int(ctime.min())+int(ptime.min()))//2;dt=(ccenter-pcenter)/1e9
                    result=estimate(cp,pp,dt,mount)
                else: result=dict(accepted=False,reason="no_causally_available_lidar_windows",speed_mps=None,yaw_rate=None)
                algorithm_ns=time.perf_counter_ns()-start
                completed=max(now,worker)+algorithm_ns
                deadline=bool(result["accepted"] and measured is not None and lower is not None and
                    0<=completed-measured<=100000000 and 0<=completed-lower<=100000000)
                file=f"{name}_{q['index']}.npz"
                np.savez_compressed(runtime/file,current=cp,previous=pp,current_times=ctime,previous_times=ptime,rear_from_lidar=mount)
                row=dict(**q,estimate=result,current_points=len(cp),previous_points=len(pp),dt_seconds=dt,
                    velocity_midpoint_ns=measured,lower_support_midpoint_ns=lower,algorithm_ns=algorithm_ns,
                    worker_before_ns=worker,completed_ns=completed,queue_wait_ns=max(0,worker-now),
                    latest_input_available_ns=latest_available,deadline_eligible=deadline,
                    input_file=file,input_sha256=hashlib.sha256((runtime/file).read_bytes()).hexdigest(),
                    acquisition_age_certified=False,instantaneous_speed_certified=False,physical_speed_authorized=False)
                output.write(json.dumps(row,allow_nan=False)+"\n");output.flush();outcomes.append(row)
                return completed
            with AnyReader([bag]) as reader:
                selected=[c for c in reader.connections if c.topic=="/bus/vls128_roof/eth_scan/bus_to_host"]
                for conn,receipt,raw in reader.messages(connections=selected):
                    while qi<len(qq) and qq[qi]["now_ns"]<=receipt:
                        worker=process(qq[qi],worker);qi+=1
                    if qi>=len(qq): break
                    start=time.perf_counter_ns()
                    msg=reader.deserialize(raw,conn.msgtype)
                    stamp=msg.header.stamp.sec*1000000000+msg.header.stamp.nanosec
                    try:
                        decoded=decode(msg.payload,stamp,cal)
                        distance=decoded["distance"];mask=(distance>=1.5)&(distance<=60)
                        xyz=decoded["xyz"][mask];times=decoded["time_ns"][mask]
                        elapsed=time.perf_counter_ns()-start
                        worker=max(receipt,worker)+elapsed
                        buffer.append(dict(receipt_ns=receipt,available_ns=worker,xyz=xyz,time_ns=times))
                    except ValueError as exc:
                        failures[str(exc)]+=1;elapsed=time.perf_counter_ns()-start;worker=max(receipt,worker)+elapsed
                    decode_ns.append(elapsed);queue_lag.append(worker-receipt);packet_count+=1
                    while buffer and buffer[0]["receipt_ns"]<receipt-200000000: buffer.popleft()
                while qi<len(qq): worker=process(qq[qi],worker);qi+=1
            group=[r for r in outcomes if r["recording"]==name]
            records[name]=dict(queries=len(group),packets_decoded=packet_count,decode_failures=dict(failures),
                geometry_accepted=sum(r["estimate"]["accepted"] for r in group),deadline_eligible=sum(r["deadline_eligible"] for r in group),
                reasons=dict(Counter(r["estimate"]["reason"] for r in group)),
                packet_decode_p95_us=float(np.quantile(decode_ns,.95)/1000),
                maximum_decoder_queue_ms=max(queue_lag)/1e6,
                registration_p95_ms=float(np.quantile([r["algorithm_ns"] for r in group],.95)/1e6))
            print(name,json.dumps(records[name]),flush=True)
    # Offline scoring starts here, after every estimate and input has been frozen.
    with (cr/"packet_events.jsonl").open() as f: nominal={(e["recording"],e["index"]):e for e in map(json.loads,f) if e["case"]=="nominal"}
    metrics={}
    for gate in ("geometry_accepted","deadline_eligible"):
        accepted=[r for r in outcomes if r["estimate"]["accepted"] and (gate=="geometry_accepted" or r["deadline_eligible"])]
        errors=[abs(r["estimate"]["speed_mps"]-nominal[r["recording"],r["index"]]["reference"])
            for r in accepted if nominal[r["recording"],r["index"]]["reference"] is not None]
        metrics[gate]=dict(accepted=len(accepted),queries=len(outcomes),availability_pct=100*len(accepted)/len(outcomes),
            scored=len(errors),unscorable_accepted=len(accepted)-len(errors),
            p95_query_time_discrepancy_mps=float(np.quantile(errors,.95)) if errors else None,
            maximum_query_time_discrepancy_mps=max(errors) if errors else None,discrepancy_above_2_mps=sum(x>2 for x in errors))
    result=dict(protocol_sha256=ph,runtime_relative_path=str(runtime.relative_to(ROOT)),records=records,metrics=metrics,
        query_count=len(outcomes),development_only=True,production_policy_changed=False,promotion_allowed=False,
        INS_used_online=False,CAN_values_used_online=False,acquisition_age_certified=False,
        source_time_model="original bridge epoch + firing offsets; historical observation intervals explicitly retained",
        algorithm="planar interval registration; not a published-method reproduction",policy_nominal_coverage_unchanged_pct=79.15194346289752)
    (ROOT/"stream_benchmark_summary.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(metrics,indent=2),flush=True)
if __name__=="__main__": main()
