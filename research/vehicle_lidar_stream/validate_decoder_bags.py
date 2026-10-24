"""Compare raw decoding to published numeric clouds, without any INS/CAN input."""
import json
from pathlib import Path
import numpy as np
from rosbags.highlevel import AnyReader
from stream_decoder import decode,prepare
ROOT=Path(__file__).resolve().parent;SCAN=ROOT.parent/"vehicle_scanario_reference"
BAGS={"07_turning":SCAN/"raw_data"/"development_07_turning.bag",
    "04_crossTraffic":SCAN/"raw_data"/"04_crossTraffic"/"development.bag",
    "14_countryRoad":SCAN/"raw_data"/"14_countryRoad"/"development.bag"}
def fields(m):
    out={}
    for f in m.fields:
        if f.name not in ("x","y","z","distance","time","laser_id","firing_bin"): continue
        dt=np.dtype(("<" if not m.is_bigendian else ">")+{2:"u1",4:"u2",7:"f4"}[f.datatype])
        out[f.name]=np.ndarray((m.height,m.width),dtype=dt,buffer=m.data,offset=f.offset,
            strides=(m.row_step,m.point_step)).reshape(-1)
    return out
def main():
    cal=prepare(json.loads((ROOT/"factory_calibration.json").read_text()));report={}
    for name,bag in BAGS.items():
        packets=[];clouds=[];start=None
        with AnyReader([bag]) as reader:
            selected=[c for c in reader.connections if c.topic in ("/bus/vls128_roof/eth_scan/bus_to_host","/sensor/lidar/vls128_roof/velodyne_points")]
            for c,receipt,raw in reader.messages(connections=selected):
                if start is None: start=receipt
                msg=reader.deserialize(raw,c.msgtype)
                stamp=msg.header.stamp.sec*1000000000+msg.header.stamp.nanosec
                if c.topic.endswith("bus_to_host"): packets.append((stamp,bytes(msg.payload)))
                else: clouds.append((stamp,fields(msg)))
                if receipt-start>350000000 and len(clouds)>=2: break
        index={p[0]:i for i,p in enumerate(packets)}
        header,data=next((h,d) for h,d in clouds if h in index)
        first=index[header];valid=np.isfinite(data["x"])&np.isfinite(data["distance"])&(data["distance"]>0)
        xyz=np.column_stack([data[k][valid] for k in ("x","y","z")]);laser=data["laser_id"][valid]
        firing=data["firing_bin"][valid].astype(int);times=header+np.rint(data["time"][valid].astype(float)*1e9).astype("int64")
        decoded={};expected_xyz=np.zeros_like(xyz);expected_time=np.zeros(len(times),dtype="int64");matched=np.zeros(len(times),dtype=bool)
        for packet_index in np.unique(firing//3):
            location=first+packet_index
            if location>=len(packets): continue
            stamp,payload=packets[location];r=decode(payload,stamp,cal)
            mapping=np.full(384,-1,dtype=int);mapping[r["point_index"]]=np.arange(len(r["xyz"]))
            ids=np.flatnonzero(firing//3==packet_index);points=(firing[ids]%3)*128+laser[ids]
            source_ids=mapping[points]
            okay=source_ids>=0;ids=ids[okay];source_ids=source_ids[okay]
            expected_xyz[ids]=r["xyz"][source_ids];expected_time[ids]=r["time_ns"][source_ids];matched[ids]=True
        err=np.linalg.norm(expected_xyz[matched]-xyz[matched],axis=1)
        terr=np.abs(expected_time[matched]-times[matched])
        status=dict(valid_cloud_points=len(xyz),matched_raw_points=int(matched.sum()),
            maximum_geometry_discrepancy_m=float(err.max()),p99_geometry_discrepancy_m=float(np.quantile(err,.99)),
            maximum_timestamp_discrepancy_ns=int(terr.max()),geometry_over_25mm=int((err>.025).sum()),
            timestamp_over_50ns=int((terr>50).sum()),first_cloud_header_ns=header)
        status["all_passed"]=bool(matched.all() and not status["geometry_over_25mm"] and not status["timestamp_over_50ns"])
        report[name]=status;print(name,json.dumps(status),flush=True)
    (ROOT/"decoder_validation.json").write_text(json.dumps(report,indent=2)+"\n")
    if not all(x["all_passed"] for x in report.values()): raise RuntimeError("Source decoder validation failed; do not run motion evaluation")
if __name__=="__main__": main()
