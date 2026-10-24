"""Independent saved-output arithmetic, timing and provenance replay."""
from pathlib import Path
import hashlib,json
import numpy as np
R=Path(__file__).resolve().parent
def metric(p,y):
 p=np.asarray(p);y=np.asarray(y);mask=np.isfinite(p)&np.isfinite(y);e=abs(p[mask]-y[mask])
 return {"predictions":int(np.isfinite(p).sum()),"scored":int(mask.sum()),"mae_mps":float(np.mean(e)) if len(e) else None,"p95_mps":float(np.percentile(e,95)) if len(e) else None,"max_mps":float(np.max(e)) if len(e) else None,"above_2_mps":int(sum(e>2))}
def same(a,b):
 for k in a:
  if isinstance(a[k],float):assert abs(a[k]-b[k])<1e-10,(k,a[k],b[k])
  else:assert a[k]==b[k],(k,a[k],b[k])
s=json.loads((R/"development_summary.json").read_text());assert set(s["sequences"])=={"Library01","Library02"}
for n,h in json.loads((R/"experiment_lock.json").read_text()).items():assert hashlib.sha256((R/n).read_bytes()).hexdigest()==h,n
files=json.loads((R/"download_lock.json").read_text())["files"]
for e in files:
 assert e["sequence"] in {"Library01","Library02"}
 assert hashlib.sha256((R/e["relative_path"]).read_bytes()).hexdigest()==e["sha256"]
assert not (R/"data"/"Library03").exists()
comparisons={};replayed={}
for seq,reported in s["sequences"].items():
 a=np.load(R/(seq+"_replay.npz"),allow_pickle=False)
 t=a["stamps"];q=a["queries"];last=np.searchsorted(a["completion_ns"],q,side="right")-1;safe=np.maximum(last,0)
 age=q-t[safe];ready=(last>=0)&a["quality"][safe]&(age>=0)&(age<=100_000_000)
 np.testing.assert_array_equal(last,a["latest"]);np.testing.assert_array_equal(age,a["age_ns"]);np.testing.assert_array_equal(ready,a["ready"])
 assert np.all(np.diff(q)==50_000_000) and np.all(np.diff(t)>0)
 assert np.all(a["completion_ns"]>=t+a["compute_ns"])
 assert np.all(np.diff(a["completion_ns"])>0)
 valid_imu=a["imu_age_ns"]>=0
 assert np.all(a["imu_age_ns"][valid_imu]<=100_000_000)
 pred=np.where(ready,a["speed"][safe],np.nan)
 np.testing.assert_allclose(pred,a["request_prediction"],equal_nan=True)
 same(metric(pred,a["request_reference"]),{k:reported["requests"][k] for k in metric(pred,a["request_reference"])})
 assert int(sum(ready))==reported["requests"]["ready"] and len(q)==reported["requests"]["count"]
 same(metric(np.where(a["quality"],a["speed"],np.nan),a["reference_primary"]),reported["candidate_all_frames_primary"])
 same(metric(np.where(a["quality"],a["speed"],np.nan),a["reference_ins"]),reported["candidate_all_frames_INS_secondary"])
 for segment in reported["requests"]["segments_60s"]:
  start=segment["start_request"];sl=slice(start,start+segment["requests"])
  same(metric(pred[sl],a["request_reference"][sl]),segment)
  assert abs(np.mean(ready[sl])-segment["ready_fraction"])<1e-12
 # Common intersection reported explicitly; no comparison of differently selected denominators.
 idx=a["cohort_indices"];x=a["speed"][idx];truth=a["reference_primary"][idx]
 pair=a["quality"][idx]&np.isfinite(x)&np.isfinite(truth)&np.isfinite(a["native_prediction"])&np.isfinite(a["ols_prediction"])
 comparisons[seq]={"cohort_frames":len(idx),"paired_scored_frames":int(sum(pair)),"scope":"post-run paired descriptive report; candidate quality conditions on subset; all cohort counts remain available","candidate":metric(x[pair],truth[pair]),"author_3d_ransac":metric(a["native_prediction"][pair],truth[pair]),"ols_3d":metric(a["ols_prediction"][pair],truth[pair])}
 replayed[seq]={"frames":len(t),"requests":len(q),"ready":int(sum(ready)),"numeric_gates":reported["numeric_request_gates"]}
tests=json.loads((R/"test_validation.json").read_text());assert tests["core"]["exit_code"]==tests["pilot"]["exit_code"]==0 and tests["denials"]==[]
assert s["Library03_payloads_accessed"] is False and s["production_changed"] is False
(R/"paired_comparison.json").write_text(json.dumps(comparisons,indent=2),encoding="utf-8")
out={"passed":True,"downloaded_inputs_rehashed":len(files),"sequences_replayed":replayed,"official_struct_oracle_checked_frames":len(s["oracle_checks"]),"tests":tests,"Library03_payloads_accessed":False,"physical_witness_admitted":False,"scope":"provenance and independent arithmetic/timing replay, not physical source certification"}
(R/"independent_verification.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
print(json.dumps({k:v for k,v in out.items() if k!="tests"},indent=2))
