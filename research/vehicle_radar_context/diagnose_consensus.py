"""Development-only root-cause diagnostics. Does not modify previous results."""
import json,math,hashlib
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from radar_motion import design,mounting,select_frames
ROOT=Path(__file__).resolve().parent
def rows(p):
    with p.open() as f: return [json.loads(l) for l in f]
summary=json.loads((ROOT/"radar_experiment_summary.json").read_text())
runtime=ROOT/summary["runtime_relative_path"]
can=ROOT.parent/"vehicle_context_policy"
cr=can/json.loads((can/"experiment_summary.json").read_text())["runtime_relative_path"]
events=rows(cr/"packet_events.jsonl")
nom={(e["recording"],e["index"]):e for e in events if e["case"]=="nominal"}
print("EVENT_SCHEMA",json.dumps({k:v for k,v in next(iter(nom.values())).items() if "wire" not in k})[:6500])
est=[e for e in rows(runtime/"radar_estimates.jsonl") if e["track"]=="raw_header_association"]
out=[]
for name in summary["source_frame_hashes"]:
    folder=ROOT/"runtime"/name
    frames=rows(folder/"frames.jsonl");cal=mounting(json.loads((folder/"transforms.json").read_text()))
    group=[]
    for e in [x for x in est if x["recording"]==name]:
        selected=select_frames(frames,e["now_us"]*1000,e["track"])
        a,b,s=design(selected,cal);f=e["estimate"];ev=nom[name,e["index"]]
        err=abs(f.get("speed_mps",0)-ev["reference"]) if f["accepted"] and ev["reference"] is not None else None
        state=np.asarray(f["state"]) if f["accepted"] else None
        mask=np.abs(a@state-b)<=.25 if state is not None else np.zeros(len(b),dtype=bool)
        item=dict(recording=name,index=e["index"],now_us=e["now_us"],accepted=f["accepted"],reason=f["reason"],
            reference=ev["reference"],speed=f.get("speed_mps"),error=err,inlier_fraction=f.get("inlier_fraction"),
            sensor_counts=dict(Counter(s.tolist())),sensor_inliers=dict(Counter(s[mask].tolist())),
            sensor_zero_radial_fraction={str(k):float(np.mean(np.abs(b[s==k])<=.25)) for k in sorted(set(s))},
            state=f["state"],matrix_condition=f.get("matrix_condition"))
        group.append(item);out.append(item)
    good=[x for x in group if x["accepted"]]
    bad=[x for x in group if x["error"] is not None and x["error"]>2]
    rejected=[x for x in group if not x["accepted"]]
    print("RECORD",name,"accepted",len(good),"rejected",len(rejected),"bad",len(bad))
    print("BAD",json.dumps(bad))
    print("REJECT_EXAMPLES",json.dumps(rejected[:2]))
    for x in bad:
        ix=group.index(x)
        print("BAD_NEIGHBORS",json.dumps(group[max(0,ix-2):ix+3]))
(ROOT/"consensus_diagnosis.json").write_text(json.dumps(dict(development_only=True,source_runtime=summary["runtime_relative_path"],queries=out),indent=2)+"\n")
