"""Separate online CAN inputs from offline INS comparisons before model fitting."""
import bisect,hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parent
SCAN=ROOT.parent/"vehicle_scanario_reference"
def sensor_us(msg):
    t=msg["header"]["stamp"]
    return (t["sec"]*10**9+t["nanosec"])//1000
def read_source(folder):
    path=folder/"raw_data"/"extracted"/"ego_streams.jsonl"
    expected=json.loads((folder/"extraction_summary.json").read_text())["extracted_sha256"]
    with path.open("rb") as inp:
        if hashlib.file_digest(inp,"sha256").hexdigest()!=expected: raise ValueError("Changed input extraction")
    with path.open() as inp: rows=[json.loads(line) for line in inp]
    return rows,expected
def online_rows(raw,name,spec):
    can=[r for r in raw if r["topic"]=="actuator/vehicle/feedback"]
    first,last=can[0]["receipt_ns"]//1000,can[-1]["receipt_ns"]//1000
    span=last-first
    split=spec["split"]
    out=[]
    for i,r in enumerate(can):
        if i%split["stride"]: continue
        m=r["message"]; now=r["receipt_ns"]//1000
        delta=now-first
        if delta<span*split["train_fraction"]: role="train"
        elif delta>=span*split["train_fraction"]+split["purge_seconds"]*10**6 and delta<span*split["calibration_end_fraction"]: role="calibration"
        elif delta>=span*split["calibration_end_fraction"]+split["purge_seconds"]*10**6: role="evaluation"
        else: role="purged"
        wheels={w["id"]:w["speed"] for w in m["wheel_speed"]}
        candidate=wheels[0]["mean"] if 0 in wheels and wheels[0]["availability"]["data"] in (1,2) else None
        features=None
        names={"yaw":"onboard_yaw_rate_esp","steering":"onboard_steering_angle","acceleration":"acceleration"}
        if all(j in wheels and wheels[j]["availability"]["data"] in (1,2) for j in (3,4)) and all(m[k]["availability"]["data"] in (1,2) for k in names.values()):
            features={key:float(m[field]["mean"]) for key,field in names.items()}
            features["peer"]=round(1000*(wheels[3]["mean"]+wheels[4]["mean"])/2)/1000
        if candidate is not None: candidate=round(candidate*1000)/1000
        out.append(dict(recording=name,index=i,role=role,now_us=now,measured_us=sensor_us(m),candidate=candidate,features=features))
    return out
def offline_reference(raw):
    transforms={}
    for r in raw:
        if r["topic"]=="tf_static":
            for t in r["message"]["transforms"]:
                if t["child_frame_id"] in ("sensor/ins/oxts","vehicle/rear_axis"):
                    assert t["header"]["frame_id"]=="base_link"
                    q=t["transform"]["rotation"]
                    assert (q["x"],q["y"],q["z"],q["w"])==(0.,0.,0.,1.)
                    transforms[t["child_frame_id"]]=t["transform"]["translation"]
    offset=[transforms["vehicle/rear_axis"][k]-transforms["sensor/ins/oxts"][k] for k in ("x","y","z")]
    values=[]
    for r in raw:
        if r["topic"]!="sensor/ins/oxts_rt3000/gps/odom": continue
        m=r["message"]
        assert m["child_frame_id"]=="sensor/ins/oxts"
        v,w=m["twist"]["twist"]["linear"],m["twist"]["twist"]["angular"]
        x,y,z=offset
        vx=v["x"]+w["y"]*z-w["z"]*y
        vy=v["y"]+w["z"]*x-w["x"]*z
        values.append((sensor_us(m),math.hypot(vx,vy)))
    if any(values[i][0]>=values[i+1][0] for i in range(len(values)-1)): raise ValueError("Non-increasing INS timestamps")
    return values
def interpolate(values,timestamp):
    times=[v[0] for v in values]
    j=bisect.bisect_left(times,timestamp)
    if j<len(values) and times[j]==timestamp: return values[j][1]
    if j==0 or j==len(values) or times[j]-times[j-1]>50_000: return None
    fraction=(timestamp-times[j-1])/(times[j]-times[j-1])
    return values[j-1][1]+fraction*(values[j][1]-values[j-1][1])
def build(spec,runtime=None):
    folders={"07_turning":SCAN,"04_crossTraffic":SCAN/"raw_data"/"04_crossTraffic","14_countryRoad":SCAN/"raw_data"/"14_countryRoad"}
    online=[];references={};hashes={}
    for name in spec["records"]:
        raw,source_hash=read_source(folders[name])
        online.extend(online_rows(raw,name,spec))
        references[name]=offline_reference(raw)
        hashes[name]=source_hash
    runtime=Path(runtime) if runtime is not None else ROOT/"runtime"
    runtime.mkdir(parents=True,exist_ok=True)
    (runtime/"online_inputs.json").write_text(json.dumps(online,allow_nan=False))
    (runtime/"offline_references.json").write_text(json.dumps(references,allow_nan=False))
    return online,references,hashes
if __name__=="__main__": build(json.loads((ROOT/"protocol_v1.json").read_text()))
