import json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;s=json.loads((R/"benchmark_summary.json").read_text());run=R/s["runtime_relative_path"]
frames=[json.loads(x) for x in (run/"frames.jsonl").read_text().splitlines()]
queries=json.loads((run/"queries.json").read_text())
records={};possible=0
for name in s["records"]:
 ff=[f for f in frames if f["recording"]==name];qq=[q for q in queries if q["recording"]==name];ages=[];passing=0
 for q in qq:
  candidates=[f for f in ff if f["receipt_ns"]<=q["now_ns"] and f["motion"]]
  if not candidates:continue
  f=candidates[-1];age=(q["now_ns"]-f["motion"]["midpoint_ns"])/1e6
  ages.append(age);passing+=0<=age<=100
 possible+=passing
 records[name]=dict(queries=len(qq),zero_compute_timing_upper_bound=passing,
  zero_compute_age_min_ms=min(ages),zero_compute_age_p95_ms=float(np.quantile(ages,.95)),
  actual_age_min_ms=min(q["inferred_midpoint_age_ms"] for q in qq if q["available_numeric"]),
  actual_age_p95_ms=float(np.quantile([q["inferred_midpoint_age_ms"] for q in qq if q["available_numeric"]],.95)))
result=dict(records=records,zero_compute_timing_upper_bound=possible,queries=283,
 zero_compute_coverage_upper_bound_pct=100*possible/283,actual_timing_eligible=0,
 interpretation="hypothetical timing-only ceiling for this fixed full-cloud interval definition and fixed request schedule, assuming zero processing and no queue; not a new estimator run or an accuracy claim",
 consequence="compute acceleration alone cannot meet 95% if this ceiling is below 95%; acquisition/update granularity must also change; no thresholds relaxed")
(R/"timing_feasibility.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))
