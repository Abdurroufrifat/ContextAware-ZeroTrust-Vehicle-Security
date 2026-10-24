"""Independent QR/struct/csv replay of the recorded-reference study; no estimator imports."""
from common import *
install_guard()
import numpy as np,scipy.linalg as la,struct,csv,io,bisect,math,collections,xml.etree.ElementTree as ET
membership=json.loads((R/"membership.json").read_text());data=json.loads((R/"data_lock.json").read_text())
pred=json.loads((R/"predictions.json").read_text());result=json.loads((R/"results.json").read_text())
lock=json.loads((R/"input_lock.json").read_text());freeze=json.loads((R/"prediction_freeze.json").read_text())
assert all(sha(P/n)==h for n,h in lock.items())
assert freeze["predictions_sha256"]==sha(R/"predictions.json") and freeze["evaluation_code_sha256"]==sha(R/"evaluate.py")
assert freeze["source_lock_sha256"]==sha(R/"input_lock.json") and freeze["data_lock_sha256"]==sha(R/"data_lock.json")
assert freeze["reference_io_code_sha256"]==sha(R/"boreas_reference_io.py") and freeze["reference_io_tests_sha256"]==sha(R/"test_boreas_reference_io.py")
frames=[e for e in membership["objects"] if e["kind"]=="frame"];assert len(frames)==96 and len({e["key"] for e in frames})==96
assert len(membership["sequences"])==3 and len(data["objects"])==132
assert membership["protocol_sha256"]==sha(R/"PROTOCOL.md") and membership["official_test_sequences_excluded"] is True
ns={"s":"http://s3.amazonaws.com/doc/2006-03-01/"}
for seq in membership["sequences"]:
 root=ET.fromstring((R/(seq+"_selected_listing.xml")).read_bytes())
 candidates=[e.find("s:Key",ns).text for e in root.findall("s:Contents",ns)]
 initial=ET.fromstring((R/(seq+"_aeva_listing.xml")).read_bytes())
 start=int(Path(initial.find("s:Contents",ns).find("s:Key",ns).text).stem)
 selected=[e for e in frames if e["key"].startswith(seq+"/")];assert len(selected)==32
 for index,e in enumerate(selected):
  target=start+120000000+index*2000000;assert e["target_us"]==target
  assert e["key"]==next(k for k in candidates if int(Path(k).stem)>=target)
bykey={e["key"]:e for e in data["objects"]}
assert set(bykey)=={e["key"] for e in membership["objects"]}
for e in data["objects"]:
 assert e["status"]=="downloaded" and checked(R/"data"/e["key"]).stat().st_size==e["bytes"]
 assert sha(R/"data"/e["key"])==e["sha256"]
def close(a,b):
 assert math.isclose(float(a),float(b),rel_tol=1e-8,abs_tol=2e-6),(a,b)
def solve(x,y,robust):
 beta,_,rank,_=la.lstsq(x,y,lapack_driver="gelsy")
 if robust:
  for iteration in range(20):
   residual=np.abs(y-x.dot(beta));w=np.sqrt(np.where(residual<=.3,1.,.3/np.maximum(residual,1e-12)))
   beta=la.lstsq(w[:,None]*x,w*y,lapack_driver="gelsy")[0]
 return beta
predictions={(r["frame_id"],r["method"]):r for r in pred["rows"]}
assert len(predictions)==len(pred["rows"])==288
computed={};maximum_beta_difference=0.;reference={}
for seq in membership["sequences"]:
 raw=checked(R/"data"/seq/"applanix"/"aeva_poses.csv").read_text()
 records=list(csv.DictReader(io.StringIO(raw)));t=[int(r["GPSTime"]) for r in records]
 assert all(b>a for a,b in zip(t,t[1:]))
 v=[tuple(float(r[k]) for k in ("vel_east","vel_north","vel_up")) for r in records];reference[seq]=(t,v)
for fid,e in enumerate(frames):
 raw=checked(R/"data"/e["key"]).read_bytes();records=list(struct.iter_unpack("<8fQ",raw));p=np.array([r[:8] for r in records],dtype=float);p[:,7]/=1000000000.
 flags_count=sum(r[8]!=0 for r in records)
 timed=np.isfinite(p[:,7]);end=float(p[timed,7].max());start=float(p[timed,7].min())
 finite=np.isfinite(p).all(axis=1);ranges=np.sqrt((p[:,:3]**2).sum(axis=1))
 eligible=np.nonzero(finite&(ranges>=3)&(ranges<=100))[0]
 selected=eligible[np.linspace(0,len(eligible)-1,min(10000,len(eligible)),dtype=np.int64)] if len(eligible) else []
 for method in ("ols3","huber3","huber6time"):
  saved=predictions[(fid,method)];s=saved["solution"]
  assert saved["key"]==e["key"] and saved["raw_points"]==len(p) and saved["flags_nonzero"]==flags_count
  assert s["n_eligible"]==len(eligible);close(s["end_offset_s"],end)
  reason="unsupported_scan_timing" if start<0 or end>.2 else "too_few_points" if len(eligible)<30 else None
  beta=None
  if reason is None:
   q=p[selected];u=q[:,:3]/np.sqrt((q[:,:3]**2).sum(axis=1))[:,None]
   x=np.column_stack([u[:,j] for j in range(3)]+([u[:,j]*(q[:,7]-end)/.1 for j in range(3)] if method=="huber6time" else []))
   y=-q[:,3];condition=float(np.linalg.cond(x));close(s["condition_number"],condition)
   assert s["n_used"]==len(q)
   if np.linalg.matrix_rank(x)<x.shape[1] or (method=="huber6time" and condition>100000):reason="unobservable_design"
   else:
    beta=solve(x,y,method!="ols3")
    maximum_beta_difference=max(maximum_beta_difference,float(np.max(np.abs(beta-np.array(s["beta"])))))
    np.testing.assert_allclose(beta,s["beta"],rtol=1e-8,atol=2e-6)
  assert s["status"]==("estimated" if beta is not None else "rejected") and s["reason"]==("fit" if beta is not None else reason)
  for age in (0,50000,100000):
   vector=None if beta is None else beta[:3]+(beta[3:]*age/100000 if len(beta)==6 else 0)
   sv=saved["query_vectors"][str(age)]
   if vector is None:assert sv is None
   else:np.testing.assert_allclose(vector,sv,rtol=1e-8,atol=2e-6)
   query=int(Path(e["key"]).stem)+int(round(end*1000000))+age
   t,v=reference[e["key"].split("/")[0]];index=bisect.bisect_left(t,query);ref=None
   if index<len(t) and t[index]==query:ref=math.sqrt(sum(x*x for x in v[index]))
   elif 0<index<len(t) and t[index]-t[index-1]<=200000:
    f=(query-t[index-1])/(t[index]-t[index-1]);z=[a+(b-a)*f for a,b in zip(v[index-1],v[index])]
    ref=math.sqrt(sum(x*x for x in z))
   speed=None if vector is None else float(la.norm(vector));error=None if speed is None or ref is None else abs(speed-ref)
   computed[(fid,method,age)]={"query":query,"speed":speed,"ref":ref,"error":error}
rows=result["rows"];assert len(rows)==len(computed)==864 and result["possible_comparisons"]==864 and result["selected_frames"]==96 and result["sequences"]==3
ids=[(r["frame_id"],r["method"],r["age_us"]) for r in rows];assert len(set(ids))==864 and set(ids)==set(computed)
for row in rows:
 c=computed[(row["frame_id"],row["method"],row["age_us"])]
 assert row["key"]==frames[row["frame_id"]]["key"] and row["sequence"]==row["key"].split("/")[0]
 assert row["query_us"]==c["query"] and row["hard_true_error_certified"] is False
 for name,expected in (("prediction_mps",c["speed"]),("reference_mps",c["ref"]),("absolute_disagreement_mps",c["error"])):
  if expected is None:assert row[name] is None
  else:close(row[name],expected)
 assert row["reference_supported"] is (c["ref"] is not None) and row["scored"] is (c["error"] is not None)
assert result["reference_supported_comparisons"]==sum(x["reference_supported"] for x in rows)
assert result["scored_comparisons"]==sum(x["scored"] for x in rows)
assert result["independent_physical_reference_comparison_completed"] is (len(rows)==864 and all(x["reference_supported"] for x in rows) and any(x["scored"] for x in rows))
assert rows==json.loads((R/"pre_reference_io_fix"/"results.json").read_text())["rows"]
metrics=result["metrics"];assert len(metrics)==27 and len({(m["sequence"],m["method"],m["age_us"]) for m in metrics})==27
for m in metrics:
 selected=[computed[(fid,m["method"],m["age_us"])] for fid,e in enumerate(frames) if e["key"].startswith(m["sequence"]+"/")]
 errors=[c["error"] for c in selected if c["error"] is not None];assert len(selected)==32 and m["selected_frames"]==32 and m["scored"]==len(errors)
 assert m["estimated"]==sum(c["speed"] is not None for c in selected);close(m["availability"],len(errors)/32)
 if errors:
  close(m["mae_mps"],math.fsum(errors)/len(errors));close(m["rmse_mps"],math.sqrt(math.fsum(x*x for x in errors)/len(errors)))
  ordered=sorted(errors);pos=(len(errors)-1)*.95;i=int(pos);q=ordered[i]+(ordered[min(i+1,len(errors)-1)]-ordered[i])*(pos-i);close(m["p95_mps"],q)
 assert m["above_05_count"]==sum(x>.5 for x in errors) and m["above_2_count"]==sum(x>2 for x in errors)
assert len(result["paired"])==9 and len({(x["sequence"],x["age_us"]) for x in result["paired"]})==9
for pair in result["paired"]:
 values=[]
 for fid,e in enumerate(frames):
  if e["key"].startswith(pair["sequence"]+"/"):
   a=computed[(fid,"huber3",pair["age_us"])]["error"];b=computed[(fid,"huber6time",pair["age_us"])]["error"]
   if a is not None and b is not None:values.append(b-a)
 assert pair["paired_frames"]==len(values);close(pair["huber6_minus_huber3_mean_abs_error_mps"],math.fsum(values)/len(values))
 assert pair["huber6_lower_error_count"]==sum(x<0 for x in values) and pair["huber3_lower_error_count"]==sum(x>0 for x in values)
assert all(result[k] is False for k in ("hard_true_speed_error_bound_established","statistical_error_independence_established","clock_mapping_independently_qualified","specific_novelty_established","promotion_allowed"))
before=json.loads((R/"pre_review"/"predictions.json").read_text())["rows"]
changed=sum(a["query_vectors"]!=b["query_vectors"] for a,b in zip(before,pred["rows"]))
report={"passed":True,"physical_frames":96,"data_objects_hashed":132,"fits_recomputed_by_independent_QR":288,
"comparisons_verified":864,"metric_cells":27,"paired_cells":9,"max_coefficient_difference":maximum_beta_difference,
"prediction_rows_changed_after_review":changed,"documented_separate_reference_instrument":True,
"hard_true_speed_error_bound_established":False,"specific_novelty_established":False,"promotion_allowed":False}
write("verification.json",report);print(json.dumps(report,indent=2),flush=True)
