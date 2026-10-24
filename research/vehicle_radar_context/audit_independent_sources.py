"""Development-only timing feasibility; header metadata and numeric lidar time fields only."""
import bisect,dataclasses,hashlib,json,os,struct
from pathlib import Path
from collections import defaultdict
import numpy as np
from rosbags.highlevel import AnyReader
ROOT=Path(__file__).resolve().parent;SCAN=ROOT.parent/"vehicle_scanario_reference";CAN=ROOT.parent/"vehicle_context_policy"
def header(raw):
    if len(raw)<16: raise ValueError("Truncated ROS1 Header")
    seq,sec,nsec,n=struct.unpack_from("<4I",raw)
    if nsec>=1000000000 or n>4096 or len(raw)<16+n: raise ValueError("Malformed ROS1 Header")
    return sec*1000000000+nsec,bytes(raw[16:16+n]).decode("utf-8")
def stats(v):
    a=np.array(v,dtype=float)/1e6
    return dict(count=len(v),min_ms=float(a.min()),median_ms=float(np.median(a)),p95_ms=float(np.quantile(a,.95)),
        max_ms=float(a.max()),negative_count=int((a<0).sum()),within_100ms_count=int(((a>=0)&(a<=100)).sum()))
def main():
    assert header(struct.pack("<4I",1,20,123,3)+b"abc")== (20000000123,"abc")
    bags={"07_turning":SCAN/"raw_data"/"development_07_turning.bag",
        "04_crossTraffic":SCAN/"raw_data"/"04_crossTraffic"/"development.bag",
        "14_countryRoad":SCAN/"raw_data"/"14_countryRoad"/"development.bag"}
    topics={"/sensor/radar/umrr/detections","/sensor/lidar/vls128_roof/velodyne_points",
        "/bus/umrr/eth_measurements/bus_to_host","/bus/vls128/eth_scan/bus_to_host"}
    protocol=dict(development_only=True,records=list(bags),selected_topics=sorted(topics),
        inspected_data="ROS1 header metadata; lidar PointCloud2 field schema and numeric time field only",
        max_age_ns=100000000,retime_allowed=False,INS_inputs_allowed=False,image_pixels_inspected=False,
        no_velocity_or_policy_fitted=True,reserved_access_allowed=False)
    p=ROOT/"source_timing_protocol.json"
    if p.exists(): assert json.loads(p.read_text())==protocol
    else: p.write_text(json.dumps(protocol,indent=2)+"\n")
    freeze=dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),frozen_before_timing_audit=True)
    fp=ROOT/"source_timing_protocol_freeze.json"
    if fp.exists(): assert json.loads(fp.read_text())==freeze
    else: fp.write_text(json.dumps(freeze,indent=2)+"\n")
    cr=CAN/json.loads((CAN/"experiment_summary.json").read_text())["runtime_relative_path"]
    online=[r for r in json.loads((cr/"online_inputs.json").read_text()) if r["role"]=="evaluation"]
    run=ROOT/"runtime"/("timing_"+os.urandom(8).hex());run.mkdir()
    results={};allrows=[]
    with (run/"headers.jsonl").open("w") as out:
        for name,bag in bags.items():
            assert not any(s in str(bag).lower() for s in ("urban02","rav4","vehicle_final_evaluation"))
            records=defaultdict(list);schema=None;inventory=[]
            with AnyReader([bag]) as reader:
                inventory=[dict(topic=c.topic,msgtype=c.msgtype,count=c.msgcount) for c in reader.connections
                    if not c.topic.startswith("/other/")]
                selected=[c for c in reader.connections if c.topic in topics]
                for c,receipt,raw in reader.messages(connections=selected):
                    stamp,frame=header(raw)
                    item=dict(recording=name,topic=c.topic,frame=frame,receipt_ns=receipt,stamp_ns=stamp)
                    records[c.topic+"|"+frame].append(item);out.write(json.dumps(item)+"\n");allrows.append(item)
                    if schema is None and c.topic=="/sensor/lidar/vls128_roof/velodyne_points":
                        m=reader.deserialize(raw,c.msgtype)
                        schema=dict(width=m.width,height=m.height,point_step=m.point_step,row_step=m.row_step,
                            fields=[dict(name=f.name,offset=f.offset,datatype=f.datatype,count=f.count) for f in m.fields],
                            time_fields={},coordinates_inspected=False)
                        for f in m.fields:
                            if f.name.lower() not in ("time","t","timestamp","offset_time"): continue
                            typ={2:"u1",4:"u2",6:"u4",7:"f4",8:"f8"}.get(f.datatype)
                            if typ is None or f.count!=1: continue
                            dtype=np.dtype((">" if m.is_bigendian else "<")+typ)
                            values=np.ndarray((m.height,m.width),dtype=dtype,buffer=m.data,
                                offset=f.offset,strides=(m.row_step,m.point_step)).reshape(-1)
                            finite=values[np.isfinite(values)]
                            schema["time_fields"][f.name]=dict(min=float(finite.min()) if len(finite) else None,
                                max=float(finite.max()) if len(finite) else None,nonfinite=int(len(values)-len(finite)),
                                unit_and_origin_validated=False)
            queries=[r for r in online if r["recording"]==name]
            groups={}
            for key,rr in records.items():
                delays=[r["receipt_ns"]-r["stamp_ns"] for r in rr]
                coverage=[]
                for q in queries:
                    now=q["now_us"]*1000
                    eligible=[r for r in rr if r["receipt_ns"]<=now and 0<=now-r["stamp_ns"]<=100000000]
                    coverage.append(bool(eligible))
                groups[key]=dict(receipt_delay=stats(delays),evaluation_queries=len(queries),
                    queries_with_original_header_fresh_evidence=sum(coverage),source_time_certified=False)
            results[name]=dict(groups=groups,lidar_first_scan_schema=schema,topic_inventory=inventory)
            print(name,json.dumps(dict(groups=groups,lidar_schema=schema)),flush=True)
    report=dict(protocol=protocol,protocol_sha256=freeze["sha256"],runtime_relative_path=str(run.relative_to(ROOT)),
        records=results,header_records=len(allrows),source_acquisition_age_certified=False,
        production_policy_changed=False,promotion_allowed=False)
    (ROOT/"independent_source_timing_audit.json").write_text(json.dumps(report,indent=2)+"\n")
    print("Header audit complete:",len(allrows),flush=True)
if __name__=="__main__": main()
