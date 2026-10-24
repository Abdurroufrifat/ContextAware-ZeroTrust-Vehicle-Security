"""Read development packet schema and pin factory calibration before motion results."""
import base64,hashlib,json,shutil,struct,urllib.request
from pathlib import Path
import numpy as np
from rosbags.highlevel import AnyReader
from ruamel.yaml import YAML
ROOT=Path(__file__).resolve().parent;RADAR=ROOT.parent/"vehicle_radar_context";SCAN=ROOT.parent/"vehicle_scanario_reference"
(ROOT/"runtime").mkdir(exist_ok=True)
(ROOT/".gitignore").write_text("runtime/\n")
meta=json.loads((RADAR/"velodyne_driver_metadata.json").read_text())
files=[x for x in meta["tree"] if x["type"]=="blob" and "128" in x["path"].lower() and x["path"].endswith((".yaml",".yml"))]
print("CALIBRATION_FILES",json.dumps([x["path"] for x in files]),flush=True)
selected=next(x for x in files if x["path"].endswith("VLS128.yaml"))
commit=next(b["commit"]["sha"] for b in meta["branches"] if b["name"]==meta["branch"])
url="https://raw.githubusercontent.com/UniBwTAS/velodyne/"+commit+"/"+selected["path"]
with urllib.request.urlopen(urllib.request.Request(url,headers={"User-Agent":"COACT-dev"}),timeout=30) as f: content=f.read()
assert hashlib.sha1(b"blob "+str(len(content)).encode()+b"\0"+content).hexdigest()==selected["sha"]
(ROOT/"factory_VLS128.yaml").write_bytes(content)
cal=YAML(typ="safe").load(content.decode())
(ROOT/"factory_calibration.json").write_text(json.dumps(cal,indent=2)+"\n")
provenance=dict(repository=meta["repository"],tree_sha=meta["tree_sha"],path=selected["path"],
    blob_sha=selected["sha"],sha256=hashlib.sha256(content).hexdigest(),sensor_unit_calibration_attested=False)
(ROOT/"source_provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
for filename in ("velodyne_pointcloud/src/lib/rawdata.cc","velodyne_pointcloud/include/velodyne_pointcloud/rawdata.h"):
    src=RADAR/"vendor_velodyne"/filename
    dst=ROOT/"vendor_reference"/filename;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
bags={"07_turning":SCAN/"raw_data"/"development_07_turning.bag",
    "04_crossTraffic":SCAN/"raw_data"/"04_crossTraffic"/"development.bag",
    "14_countryRoad":SCAN/"raw_data"/"14_countryRoad"/"development.bag"}
results={}
for name,bag in bags.items():
    packets=[];cloud=None
    with AnyReader([bag]) as reader:
        selected=[c for c in reader.connections if c.topic in ("/bus/vls128_roof/eth_scan/bus_to_host","/sensor/lidar/vls128_roof/velodyne_points")]
        for c,receipt,raw in reader.messages(connections=selected):
            if c.topic.endswith("bus_to_host") and len(packets)<4:
                msg=reader.deserialize(raw,c.msgtype);p=bytes(msg.payload)
                packets.append(dict(stamp_ns=msg.header.stamp.sec*1000000000+msg.header.stamp.nanosec,receipt_ns=receipt,
                    payload=p.hex(),length=len(p),mode=p[1204],model=p[1205],
                    blocks=[struct.unpack_from("<HH",p,k*100) for k in range(12)]))
            if c.topic.endswith("velodyne_points") and cloud is None:
                msg=reader.deserialize(raw,c.msgtype);data={}
                for f in msg.fields:
                    if f.name not in ("x","y","z","distance","laser_id","ring","time"): continue
                    dt=np.dtype(("<" if not msg.is_bigendian else ">")+{2:"u1",4:"u2",7:"f4"}[f.datatype])
                    data[f.name]=np.ndarray((msg.height,msg.width),dtype=dt,buffer=msg.data,offset=f.offset,
                        strides=(msg.row_step,msg.point_step)).reshape(-1)
                valid=np.isfinite(data["distance"])&(data["distance"]>0)&np.isfinite(data["x"])
                idx=np.flatnonzero(valid);laser=data["laser_id"][idx]
                vert=np.arctan2(data["z"][idx],np.hypot(data["x"][idx],data["y"][idx]))
                lookup={int(x["laser_id"]):float(x["vert_correction"]) for x in cal["lasers"]}
                diffs=np.array([v-lookup[int(l)] for v,l in zip(vert,laser)])
                cloud=dict(valid_points=len(idx),frame=msg.header.frame_id,
                    factory_vertical_max_discrepancy_rad=float(np.max(np.abs(diffs))),
                    sample=[{k:float(v[i]) for k,v in data.items()} for i in idx[:12]])
            if len(packets)==4 and cloud is not None: break
    (ROOT/"runtime"/(name+"_packet_sample.json")).write_text(json.dumps(packets,indent=2)+"\n")
    results[name]=dict(packets=[{k:v for k,v in p.items() if k!="payload"} for p in packets],cloud=cloud)
    print(name,json.dumps(results[name]),flush=True)
(ROOT/"source_probe.json").write_text(json.dumps(results,indent=2)+"\n")
