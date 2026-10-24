"""Full-record metadata-only LiDAR point-time feasibility; no coordinates, fitting or INS."""
import hashlib,json,os,struct
from pathlib import Path
from collections import Counter
import numpy as np
from rosbags.highlevel import AnyReader
from lidar_timing import firing_times,fresh_mask,MODEL
ROOT=Path(__file__).resolve().parent;SCAN=ROOT.parent/"vehicle_scanario_reference";CAN=ROOT.parent/"vehicle_context_policy"
BAGS={"07_turning":SCAN/"raw_data"/"development_07_turning.bag",
    "04_crossTraffic":SCAN/"raw_data"/"04_crossTraffic"/"development.bag",
    "14_countryRoad":SCAN/"raw_data"/"14_countryRoad"/"development.bag"}
def main():
    source=json.loads((ROOT/"velodyne_source_provenance.json").read_text())
    protocol=dict(version=1,development_only=True,records=list(BAGS),time_model=MODEL,
        source_tree_sha=source["tree_sha"],cloud_topic="/sensor/lidar/vls128_roof/velodyne_points",
        raw_topic="/bus/vls128_roof/eth_scan/bus_to_host",max_age_ns=100000000,
        offset_model_seconds=[-.001,.2],valid_return="finite positive distance and finite firing offset",
        availability="whole cloud must have arrived by query; each encoded point timestamp is checked separately",
        minimum_points_for_timing_feasibility=30,original_header_preserved=True,
        raw_sensor_clock_certified=False,actual_recording_driver_build_attested=False,
        source_model_is_bridge_epoch_plus_firing_offset=True,coordinates_inspected=False,INS_inputs_allowed=False,
        velocity_estimation=False,promotion_allowed=False,old_radar_coverage_unchanged=True)
    pp=ROOT/"lidar_point_timing_protocol_v1.json"
    if pp.exists(): assert json.loads(pp.read_text())==protocol
    else: pp.write_text(json.dumps(protocol,indent=2)+"\n")
    freeze=ROOT/"lidar_point_timing_freeze.json"
    ph=hashlib.sha256(pp.read_bytes()).hexdigest()
    if freeze.exists(): assert json.loads(freeze.read_text())["sha256"]==ph
    else: freeze.write_text(json.dumps(dict(sha256=ph,frozen_before_point_audit=True),indent=2)+"\n")
    cr=CAN/json.loads((CAN/"experiment_summary.json").read_text())["runtime_relative_path"]
    online=[r for r in json.loads((cr/"online_inputs.json").read_text()) if r["role"]=="evaluation"]
    run=ROOT/"runtime"/("lidar_timing_"+os.urandom(8).hex());run.mkdir()
    results={};total_scans=total_valid=raw_count=0
    with (run/"cloud_timing.jsonl").open("w") as out:
        for name,bag in BAGS.items():
            assert not any(x in str(bag).lower() for x in ("urban02","rav4","vehicle_final_evaluation"))
            queries=[dict(index=r["index"],now_ns=r["now_us"]*1000,fresh_points=0,fresh_clouds=0,
                oldest_used_ns=None,newest_used_ns=None) for r in online if r["recording"]==name]
            scans=valid=raw=0;errors=Counter();delays=[];point_age_at_receipt=[]
            with AnyReader([bag]) as reader:
                selected=[c for c in reader.connections if c.topic in (protocol["cloud_topic"],protocol["raw_topic"])]
                expected={c.topic:c.msgcount for c in selected}
                for c,receipt,data in reader.messages(connections=selected):
                    if c.topic==protocol["raw_topic"]:
                        seq,sec,ns,size=struct.unpack_from("<4I",data)
                        assert ns<1000000000
                        delays.append(receipt-sec*1000000000-ns);raw+=1;continue
                    msg=reader.deserialize(data,c.msgtype);scans+=1
                    header=msg.header.stamp.sec*1000000000+msg.header.stamp.nanosec
                    try: stamps=firing_times(msg,MODEL)
                    except ValueError as exc:
                        errors[str(exc)]+=1
                        out.write(json.dumps(dict(recording=name,scan=scans,receipt_ns=receipt,header_ns=header,
                            status="declined",reason=str(exc)))+"\n");continue
                    valid+=len(stamps)
                    audit=dict(recording=name,scan=scans,receipt_ns=receipt,header_ns=header,status="decoded",
                        original_point_count=int(msg.width*msg.height),valid_points=len(stamps),
                        minimum_point_ns=int(stamps.min()) if len(stamps) else None,
                        maximum_point_ns=int(stamps.max()) if len(stamps) else None,query_counts={})
                    if len(stamps):
                        point_age_at_receipt.append(receipt-int(stamps.max()))
                        for q in queries:
                            now=q["now_ns"]
                            if receipt>now or now>int(stamps.max())+100000000: continue
                            mask=fresh_mask(stamps,receipt,now);count=int(mask.sum())
                            audit["query_counts"][str(q["index"])]=count
                            if count:
                                used=stamps[mask];lo=int(used.min());hi=int(used.max())
                                q["fresh_points"]+=count;q["fresh_clouds"]+=1
                                q["oldest_used_ns"]=lo if q["oldest_used_ns"] is None else min(q["oldest_used_ns"],lo)
                                q["newest_used_ns"]=hi if q["newest_used_ns"] is None else max(q["newest_used_ns"],hi)
                    out.write(json.dumps(audit)+"\n")
                assert scans==expected[protocol["cloud_topic"]] and raw==expected[protocol["raw_topic"]]
            a=np.array(point_age_at_receipt)/1e6;rd=np.array(delays)/1e6
            results[name]=dict(scans=scans,valid_points=valid,raw_packets=raw,decoder_declines=dict(errors),
                evaluation_queries=len(queries),queries_with_30_fresh_points=sum(q["fresh_points"]>=30 for q in queries),
                minimum_fresh_points=min(q["fresh_points"] for q in queries),
                newest_point_age_at_cloud_receipt_ms=dict(min=float(a.min()),median=float(np.median(a)),max=float(a.max())),
                raw_packet_header_delay_ms=dict(min=float(rd.min()),median=float(np.median(rd)),max=float(rd.max())),
                queries=queries)
            total_scans+=scans;total_valid+=valid;raw_count+=raw
            print(name,json.dumps({k:v for k,v in results[name].items() if k!="queries"}),flush=True)
    result=dict(protocol_sha256=ph,runtime_relative_path=str(run.relative_to(ROOT)),records=results,
        total_scans=total_scans,total_valid_points=total_valid,raw_packets=raw_count,
        time_model_feasible_queries=sum(r["queries_with_30_fresh_points"] for r in results.values()),
        total_queries=len(online),acquisition_age_certified=False,driver_build_attested=False,
        time_association_only=True,velocity_or_security_coverage_established=False,
        production_policy_changed=False,promotion_allowed=False)
    (ROOT/"lidar_point_timing_audit.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="records"},indent=2))
if __name__=="__main__": main()
