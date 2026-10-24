"""Explain rejections without changing acceptance gates or native parameters."""
import json
from pathlib import Path
from collections import Counter
import numpy as np
ROOT=Path(__file__).resolve().parent
summary=json.loads((ROOT/"comparison_summary.json").read_text())
with (ROOT/summary["runtime_relative_path"]/"estimates.jsonl").open() as f: rows=[json.loads(l) for l in f]
CAN=ROOT.parent/"vehicle_context_policy"
cr=CAN/json.loads((CAN/"experiment_summary.json").read_text())["runtime_relative_path"]
with (cr/"packet_events.jsonl").open() as f: refs={(e["recording"],e["index"]):e["reference"] for e in map(json.loads,f) if e["case"]=="nominal"}
result={}
for method in ("ICP","PLANE_ICP","GICP"):
    rr=[r for r in rows if r["method"]==method and r["estimate"].get("converged")]
    failures=Counter();raw_errors=[];per_record={}
    for r in rr:
        e=r["estimate"]
        for key,bad in [("native_inliers_below_100",e["native_inliers"]<100),
            ("host_inliers_below_100",e["inliers"]<100),("overlap_below_50pct",e["overlap"]<.5),
            ("euclidean_p95_above_30cm",e["euclidean_inlier_p95_m"] is None or e["euclidean_inlier_p95_m"]>.3)]:
            if bad: failures[key]+=1
        v=float(np.linalg.norm(np.array(e["rear_pose"])[:2,3])/r["dt_seconds"])
        ref=refs[r["recording"],r["index"]]
        if ref is not None: raw_errors.append(abs(v-ref))
    for name in ("07_turning","04_crossTraffic","14_countryRoad"):
        values=[r["estimate"] for r in rr if r["recording"]==name]
        per_record[name]=dict(converged=len(values),
            median_p95_m=float(np.median([e["euclidean_inlier_p95_m"] for e in values])) if values else None,
            median_overlap=float(np.median([e["overlap"] for e in values])) if values else None,
            example=values[0] if values else None)
        if values:
            per_record[name]["example"]={k:v for k,v in values[0].items() if k not in ("hessian",)}
    result[method]=dict(native_converged=len(rr),quality_failure_counts=dict(failures),
        ungated_diagnostic_p95_error_mps=float(np.quantile(raw_errors,.95)) if raw_errors else None,
        ungated_diagnostic_large_errors=sum(x>2 for x in raw_errors),per_record=per_record,
        policy_acceptance_unchanged=True)
    print(method,json.dumps(result[method]),flush=True)
(ROOT/"native_rejection_diagnosis.json").write_text(json.dumps(result,indent=2)+"\n")
