import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
r=json.loads((ROOT/"lidar_point_timing_audit.json").read_text())
with (ROOT/r["runtime_relative_path"]/"cloud_timing.jsonl").open() as f: scans=[json.loads(l) for l in f]
for name,m in r["records"].items():
    ss=[s for s in scans if s["recording"]==name]
    qq=m["queries"];base=qq[0]["now_ns"]
    print(name,"relative-to-first-query seconds")
    print("query range",0,(qq[-1]["now_ns"]-base)/1e9)
    print("cloud receipt range",(ss[0]["receipt_ns"]-base)/1e9,(ss[-1]["receipt_ns"]-base)/1e9)
    print("cloud point time range",(ss[0]["minimum_point_ns"]-base)/1e9,(ss[-1]["maximum_point_ns"]-base)/1e9)
    if any(q["fresh_points"]==0 for q in qq):
        for q in [v for v in qq if v["fresh_points"]==0][:3]:
            before=[s for s in ss if s["receipt_ns"]<=q["now_ns"]]
            latest=max(before,key=lambda s:s["receipt_ns"])
            print("missing",q["index"],"latest_cloud_age_ms",(q["now_ns"]-latest["receipt_ns"])/1e6,
                "latest_point_age_ms",(q["now_ns"]-latest["maximum_point_ns"])/1e6)
