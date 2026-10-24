import json
from collections import Counter
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;s=json.loads((R/"benchmark_summary.json").read_text());run=R/s["runtime_relative_path"]
rows=[json.loads(x) for x in (run/"jobs.jsonl").read_text().splitlines()]
out={}
for name in s["records"]:
 group=[r for r in rows if r["recording"]==name];reasons=Counter();residuals=[];times=[];passing=0;query=[q for q in json.loads((run/"queries.json").read_text()) if q["recording"]==name]
 for r in group:
  a=r["result"];times.append(r["elapsed_ns"]/1e6)
  if a["bootstrap"]:reasons["bootstrap"]+=1;continue
  g=a.get("geometry")
  if g:
   if g["residual_p95_m"] is not None:residuals.append(g["residual_p95_m"])
   if g["correspondences"]<100:reasons["insufficient_correspondences"]+=1
   if g["overlap"]<.5:reasons["overlap_below_half"]+=1
   if g["residual_p95_m"] is not None and g["residual_p95_m"]>.3:reasons["residual_p95_above_point3m"]+=1
   if g["condition"] is None or g["condition"]>1000:reasons["unobservable_or_ill_conditioned"]+=1
  if a["motion"] is None:reasons["no_numeric_motion"]+=1
  passing+=a["quality_passed"]
 ages=[(q["now_ns"]-q["midpoint_ns"])/1e6 for q in query if q["numeric"]]
 out[name]=dict(jobs=len(group),geometry_passed_jobs=passing,multi_label_reasons=dict(reasons),
  geometry_residual_p95_median_m=float(np.median(residuals)) if residuals else None,
  processing_median_ms=float(np.median(times)),processing_p95_ms=float(np.quantile(times,.95)),
  numeric_query_age_median_ms=float(np.median(ages)) if ages else None,
  numeric_query_age_max_ms=max(ages) if ages else None)
result=dict(records=out,observed=True,thresholds_changed=False,
 conclusion="faster window updates can provide recent numeric hypotheses, but independent-source eligibility still requires geometry and timing jointly; sparse map sampling, scene dynamics, deskew assumptions and errors are not separable from these residuals alone",
 no_success_claim_from_zero_accepts=True)
(R/"rejection_diagnosis.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))
