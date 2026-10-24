"""Independent source reread: encoded point offsets, causal availability and raw packet feasibility."""
import hashlib,json,struct
from pathlib import Path
import numpy as np
from rosbags.highlevel import AnyReader
ROOT=Path(__file__).resolve().parent;SCAN=ROOT.parent/"vehicle_scanario_reference"
BAGS={"07_turning":SCAN/"raw_data"/"development_07_turning.bag",
    "04_crossTraffic":SCAN/"raw_data"/"04_crossTraffic"/"development.bag",
    "14_countryRoad":SCAN/"raw_data"/"14_countryRoad"/"development.bag"}
def main():
    r=json.loads((ROOT/"lidar_point_timing_audit.json").read_text())
    spec=json.loads((ROOT/"lidar_point_timing_protocol_v1.json").read_text())
    assert hashlib.sha256((ROOT/"lidar_point_timing_protocol_v1.json").read_bytes()).hexdigest()==r["protocol_sha256"]
    source=json.loads((ROOT/"velodyne_source_provenance.json").read_text())
    for item in source["files"]:
        assert hashlib.sha256((ROOT/"vendor_velodyne"/item["path"]).read_bytes()).hexdigest()==item["sha256"]
    with (ROOT/r["runtime_relative_path"]/"cloud_timing.jsonl").open() as f:
        saved={(x["recording"],x["scan"]):x for x in map(json.loads,f)}
    total=valid_total=raw_total=queries_checked=0;raw_results={}
    for name,bag in BAGS.items():
        assert not any(x in str(bag).lower() for x in ("urban02","rav4","vehicle_final_evaluation"))
        meta=r["records"][name]
        queries=[dict(index=q["index"],now_ns=q["now_ns"],fresh_points=0,fresh_clouds=0,
            oldest_used_ns=None,newest_used_ns=None) for q in meta["queries"]]
        scans=valid=0;raw_headers=[];raw_receipts=[]
        with AnyReader([bag]) as reader:
            selected=[c for c in reader.connections if c.topic in (spec["cloud_topic"],spec["raw_topic"])]
            for connection,receipt,raw in reader.messages(connections=selected):
                if connection.topic==spec["raw_topic"]:
                    _,seconds,nanos,_=struct.unpack_from("<4I",raw)
                    assert nanos<1000000000
                    raw_headers.append(seconds*1000000000+nanos);raw_receipts.append(receipt);continue
                m=reader.deserialize(raw,connection.msgtype);scans+=1
                record=saved[name,scans];assert record["status"]=="decoded"
                assert record["receipt_ns"]==receipt
                header=int(m.header.stamp.sec)*1000000000+int(m.header.stamp.nanosec)
                assert header==record["header_ns"]
                values={}
                for field in m.fields:
                    if field.name not in ("time","distance"): continue
                    dt=np.dtype((">" if m.is_bigendian else "<")+{7:"f4",8:"f8"}[field.datatype])
                    values[field.name]=np.ndarray(shape=(m.height,m.width),dtype=dt,buffer=m.data,
                        offset=field.offset,strides=(m.row_step,m.point_step)).ravel().astype("float64")
                t=values["time"];distance=values["distance"]
                offsets=t[np.logical_and.reduce((np.isfinite(t),np.isfinite(distance),distance>0))]
                assert np.all(offsets>=-.001) and np.all(offsets<=.2)
                stamps=np.round(offsets*1e9).astype("int64")+header;valid+=len(stamps)
                assert len(stamps)==record["valid_points"]
                assert int(stamps.min())==record["minimum_point_ns"] and int(stamps.max())==record["maximum_point_ns"]
                recomputed={}
                for q in queries:
                    now=q["now_ns"]
                    if receipt>now: continue
                    used=stamps[np.logical_and(stamps>=now-100000000,stamps<=now)]
                    if now<=int(stamps.max())+100000000: recomputed[str(q["index"])]=len(used)
                    if len(used):
                        q["fresh_points"]+=len(used);q["fresh_clouds"]+=1
                        lo=int(used.min());hi=int(used.max())
                        q["oldest_used_ns"]=lo if q["oldest_used_ns"] is None else min(q["oldest_used_ns"],lo)
                        q["newest_used_ns"]=hi if q["newest_used_ns"] is None else max(q["newest_used_ns"],hi)
                assert recomputed==record["query_counts"]
        assert queries==meta["queries"]
        assert scans==meta["scans"] and valid==meta["valid_points"] and len(raw_headers)==meta["raw_packets"]
        raw_headers=np.array(raw_headers,dtype="int64");raw_receipts=np.array(raw_receipts,dtype="int64")
        checks=[]
        for q in queries:
            now=q["now_ns"]
            mask=(raw_receipts<=now)&(raw_headers<=now)&(raw_headers>=now-100000000)
            used=raw_headers[mask]
            checks.append(dict(index=q["index"],eligible_raw_packets=int(mask.sum()),
                latest_raw_header_age_ms=float((now-int(used.max()))/1e6) if len(used) else None))
        raw_results[name]=dict(queries=len(queries),queries_with_fresh_raw_packets=sum(c["eligible_raw_packets"]>0 for c in checks),
            minimum_eligible_raw_packets=min(c["eligible_raw_packets"] for c in checks),
            maximum_latest_header_age_ms=max(c["latest_raw_header_age_ms"] for c in checks if c["latest_raw_header_age_ms"] is not None),
            query_checks=checks,geometry_not_decoded=True,source_acquisition_age_certified=False)
        total+=scans;valid_total+=valid;raw_total+=len(raw_headers);queries_checked+=len(queries)
        print(name,"verified",scans,"clouds;",raw_results[name]["queries_with_fresh_raw_packets"],"raw-eligible queries",flush=True)
    assert total==r["total_scans"] and valid_total==r["total_valid_points"] and raw_total==r["raw_packets"]
    result=dict(all_passed=True,independently_reread_scans=total,verified_valid_point_times=valid_total,
        raw_packets_inspected=raw_total,queries_checked=queries_checked,
        processed_point_time_feasible_queries=r["time_model_feasible_queries"],
        raw_packet_time_feasible_queries=sum(x["queries_with_fresh_raw_packets"] for x in raw_results.values()),
        raw_results=raw_results,stale_or_future_points_admitted=False,acquisition_age_certified=False,
        velocity_accuracy_established=False,production_promotion_allowed=False)
    (ROOT/"lidar_timing_independent_verification.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="raw_results"},indent=2))
if __name__=="__main__": main()
