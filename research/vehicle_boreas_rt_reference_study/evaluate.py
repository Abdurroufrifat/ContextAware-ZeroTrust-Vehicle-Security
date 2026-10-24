"""Evaluate frozen physical-scan fits; reference is used only after prediction freeze."""
from common import *
install_guard()
import numpy as np,collections,csv
from boreas_core import reference_speed
from boreas_reference_io import parse_reference,comparison_completed
assert not (R/"results.json").exists(),"Results already frozen"
pred=json.loads((R/"predictions.json").read_text());membership=json.loads((R/"membership.json").read_text())
assert pred["reference_used_for_fit"] is False and len(pred["rows"])==288 and pred["frames"]==96
lock=json.loads((R/"input_lock.json").read_text());assert all(sha(P/n)==digest for n,digest in lock.items())
write("prediction_freeze.json",{"predictions_sha256":sha(R/"predictions.json"),"source_lock_sha256":sha(R/"input_lock.json"),
"data_lock_sha256":sha(R/"data_lock.json"),"evaluation_code_sha256":sha(R/"evaluate.py"),"reference_io_code_sha256":sha(R/"boreas_reference_io.py"),"reference_io_tests_sha256":sha(R/"test_boreas_reference_io.py")})
data_lock={e["key"]:e for e in json.loads((R/"data_lock.json").read_text())["objects"]}
references={};reference_status={}
for seq in membership["sequences"]:
 path=checked(R/"data"/seq/"applanix"/"aeva_poses.csv")
 key=seq+"/applanix/aeva_poses.csv";record=data_lock[key]
 try:
  if record["status"]!="downloaded" or sha(path)!=record["sha256"]:raise ValueError("Reference integrity unavailable")
  t,v=parse_reference(path.read_bytes())
  references[seq]=(t,v);reference_status[seq]="integrity_checked"
 except (ValueError,OSError):references[seq]=None;reference_status[seq]="reference_integrity_or_schema_unavailable"
rows=[]
for item in pred["rows"]:
 seq=item["key"].split("/")[0];stamp=int(Path(item["key"]).stem);solution=item["solution"]
 for age in (0,50000,100000):
  query=stamp+round(solution["end_offset_s"]*1e6)+age if solution["end_offset_s"] is not None else None
  ref=reference_speed(*references[seq],query) if query is not None and references[seq] is not None else None
  vector=item["query_vectors"][str(age)];speed=float(np.linalg.norm(vector)) if vector is not None else None
  scored=speed is not None and ref is not None;error=abs(speed-ref) if scored else None
  rows.append({"frame_id":item["frame_id"],"key":item["key"],"sequence":seq,"method":item["method"],"age_us":age,
    "query_us":query,"status":solution["status"],"reason":solution["reason"],"prediction_mps":speed,"reference_mps":ref,
    "reference_supported":ref is not None,"reference_status":reference_status[seq],"scored":scored,"absolute_disagreement_mps":error,"hard_true_error_certified":False})
assert len(rows)==864
groups=collections.defaultdict(list)
for row in rows:groups[(row["sequence"],row["method"],row["age_us"])].append(row)
metrics=[]
for (seq,method,age),group in groups.items():
 assert len(group)==32;errors=np.array([r["absolute_disagreement_mps"] for r in group if r["scored"]])
 metrics.append({"sequence":seq,"method":method,"age_us":age,"selected_frames":32,"estimated":sum(r["prediction_mps"] is not None for r in group),
 "scored":len(errors),"availability":len(errors)/32,"mae_mps":float(errors.mean()) if len(errors) else None,
 "rmse_mps":float(np.sqrt(np.mean(errors**2))) if len(errors) else None,"p95_mps":float(np.percentile(errors,95)) if len(errors) else None,
 "above_05_count":int((errors>.5).sum()),"above_2_count":int((errors>2).sum())})
paired=[]
for seq in membership["sequences"]:
 for age in (0,50000,100000):
  a={r["frame_id"]:r for r in rows if (r["sequence"],r["method"],r["age_us"])==(seq,"huber3",age) and r["scored"]}
  b={r["frame_id"]:r for r in rows if (r["sequence"],r["method"],r["age_us"])==(seq,"huber6time",age) and r["scored"]}
  ids=sorted(a.keys()&b.keys());delta=np.array([b[i]["absolute_disagreement_mps"]-a[i]["absolute_disagreement_mps"] for i in ids])
  paired.append({"sequence":seq,"age_us":age,"paired_frames":len(ids),"huber6_minus_huber3_mean_abs_error_mps":float(delta.mean()) if len(ids) else None,
    "huber6_lower_error_count":int((delta<0).sum()),"huber3_lower_error_count":int((delta>0).sum())})
write("results.json",{"rows":rows,"metrics":metrics,"paired":paired,"possible_comparisons":864,"selected_frames":96,"sequences":3,
 "documented_separate_reference_instrument":True,"independent_physical_reference_comparison_completed":comparison_completed(rows,864),
 "reference_supported_comparisons":sum(r["reference_supported"] for r in rows),"scored_comparisons":sum(r["scored"] for r in rows),
 "hard_true_speed_error_bound_established":False,"statistical_error_independence_established":False,
 "clock_mapping_independently_qualified":False,"specific_novelty_established":False,"promotion_allowed":False})
for name,content in (("comparisons.csv",rows),("metrics.csv",metrics),("paired.csv",paired)):
 with (R/name).open("w",newline="",encoding="utf-8") as f:
  w=csv.DictWriter(f,fieldnames=list(content[0]));w.writeheader();w.writerows(content)
print(json.dumps({"rows":len(rows),"scored":sum(r["scored"] for r in rows),"paired":paired},indent=2))
